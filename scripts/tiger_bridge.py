"""Tiger Data bridge for Hat Detect: stores the hat's unified hazard events in Tiger Cloud.

The Pi (QNX) has no Postgres client and never holds the database password. Its events
reach the laptop as `EVENT {json}` lines over SSH (scripts/laptop_companion.py), and this
module upserts them into the `detections` hypertable, one row per event_id: an event's
open, update and closed records all land on the same row, and re-sending any of them is
harmless (an older record never overwrites a newer one).

Reliability: TigerWriter.add() never blocks the hat's alerts. If Tiger Cloud is
unreachable, events are kept in .tiger_spool.jsonl and replayed when the connection comes
back. A record the database rejects as invalid goes to .tiger_rejected.jsonl instead of
blocking everything behind it.

Setup: put the service's connection string in .env (see .env.example):
    TIGER_DATABASE_URL=postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require

Commands (run from the project root):
    python scripts/tiger_bridge.py init          # create/migrate tables, rollup, policies (safe to re-run)
    python scripts/tiger_bridge.py check         # connection + row counts
    python scripts/tiger_bridge.py ingest-pi     # load the Pi's ~/hatdetect/events.jsonl (no duplicates)
    python scripts/tiger_bridge.py ingest FILE   # load a local events.jsonl
    python scripts/tiger_bridge.py demo          # replace the demo-01 sample data (for the dashboard)
    python scripts/tiger_bridge.py purge-demo    # remove the demo-01 sample data
"""
import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPOOL = os.path.join(ROOT, ".tiger_spool.jsonl")
REJECTS = os.path.join(ROOT, ".tiger_rejected.jsonl")
SCHEMA = os.path.join(ROOT, "db", "schema.sql")

FUSION_SOURCE = {"AUDIO_ONLY": "audio", "VISION_ONLY": "video", "MULTIMODAL": "combined"}
STATUS_RANK = {"open": 0, "update": 1, "closed": 2}
SEVERITIES = {"low", "medium", "high"}

COLUMNS = ("time, device_id, source, hazard, severity, label, score, position, bbox, alerted, "
           "event_id, fusion, status, last_time, audio_group, audio_label, audio_score, audio_count, "
           "video_group, video_label, video_score, video_count, correlation_window, explanation")
_UPDATED = [c.strip() for c in COLUMNS.split(",") if c.strip() not in ("time", "event_id", "device_id", "alerted")]
_RANK_SQL = "(CASE {t}.status WHEN 'closed' THEN 2 WHEN 'update' THEN 1 ELSE 0 END)"
INSERT_SQL = f"""
INSERT INTO detections ({COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::uuid, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id, time) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in _UPDATED)},
    alerted = detections.alerted OR EXCLUDED.alerted
WHERE {_RANK_SQL.format(t="detections")} <= {_RANK_SQL.format(t="EXCLUDED")}
"""


def database_url():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass
    url = os.environ.get("TIGER_DATABASE_URL", "").strip()
    return url if url and "<password>" not in url else None


def connect(url, autocommit=False):
    import psycopg
    return psycopg.connect(url, autocommit=autocommit, connect_timeout=10,
                           application_name="hatdetect-bridge")


def is_event(ev):
    """Unified event record (fusion.py), as opposed to a pre-fusion raw detection line."""
    return isinstance(ev, dict) and "event_id" in ev


def to_row(ev):
    """Unified event record -> detections row. Raises ValueError if it is malformed."""
    try:
        fusion, status, severity = ev["fusion"], ev["status"], ev["severity"]
        if fusion not in FUSION_SOURCE or status not in STATUS_RANK or severity not in SEVERITIES:
            raise ValueError(f"bad fusion/status/severity {fusion!r}/{status!r}/{severity!r}")
        a, v = ev.get("audio") or {}, ev.get("video") or {}
        scores = [float(p["score"]) for p in (a, v) if p]
        if not scores:
            raise ValueError("event has neither an audio nor a video part")
        corr = ev.get("correlation") or {}
        bbox = v.get("bbox")
        return (ev["time"], ev["device_id"], FUSION_SOURCE[fusion], ev["hazard"], severity, ev["label"],
                max(scores), v.get("position"), json.dumps(bbox) if bbox is not None else None,
                bool(ev.get("alerted")), str(uuid.UUID(ev["event_id"])), fusion, status, ev.get("last_time"),
                a.get("group"), a.get("label"), a.get("score"), a.get("detections"),
                v.get("group"), v.get("label"), v.get("score"), v.get("detections"),
                corr.get("window_seconds"), corr.get("explanation"))
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed event: {type(e).__name__} {e}") from None


def latest_per_event(events):
    """Collapse a batch to one record per event_id: the most advanced status, latest wins ties."""
    best = {}
    for ev in events:
        key = ev.get("event_id")
        cur = best.get(key)
        if cur is None or STATUS_RANK.get(ev.get("status"), -1) >= STATUS_RANK.get(cur.get("status"), -1):
            best[key] = ev
    return list(best.values())


def is_offline(e):
    """Connection problems (retry later) vs. data problems (reject the record)."""
    if isinstance(e, (OSError, TimeoutError, ConnectionError)):
        return True
    try:
        import psycopg
    except ImportError:
        return False
    return isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError))


def insert_events(conn, events):
    """Upsert unified events. Returns (rows written, rejected [(event, reason)]).
    Connection errors propagate (caller buffers and retries); invalid records are returned
    as rejected and never block the rest of the batch."""
    rows, rejected = [], []
    for ev in latest_per_event(e for e in events if is_event(e)):
        try:
            rows.append((ev, to_row(ev)))
        except ValueError as e:
            rejected.append((ev, str(e)))
    if not rows:
        return 0, rejected
    try:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, [r for _, r in rows])
        conn.commit()
        return len(rows), rejected
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        if is_offline(e):
            raise
    written = 0                      # the batch had a bad row: find it, keep the rest
    for ev, row in rows:
        try:
            with conn.cursor() as cur:
                cur.execute(INSERT_SQL, row)
            conn.commit()
            written += 1
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            if is_offline(e):
                raise
            rejected.append((ev, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"))
    return written, rejected


def write_rejects(rejected, path=REJECTS):
    with open(path, "a") as f:
        for ev, reason in rejected:
            f.write(json.dumps({"reason": reason, "event": ev}) + "\n")


class TigerWriter:
    """Background writer used by the laptop companion. add() never blocks or raises:
    events are batched (up to 1 s) and upserted; if Tiger Cloud is unreachable they go to
    the spool file, which is replayed (before newer events) once a retry connects."""

    def __init__(self, url, log=print, connect_fn=None, spool=SPOOL, rejects=REJECTS, retry_seconds=10.0):
        self.url, self.log = url, log
        self._connect = connect_fn or connect
        self.spool_path, self.rejects_path, self.retry_seconds = spool, rejects, retry_seconds
        self.q = queue.Queue()
        self.conn = None
        self.sent = 0
        self.spooled = 0
        self.rejected = 0
        self._inflight = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, ev):
        self.q.put(ev)

    def _ensure_conn(self):
        if self.conn is None or self.conn.closed:
            self.conn = self._connect(self.url)
            self.log("[tiger] connected to Tiger Cloud")
            self._replay_spool()

    def _write(self, events):
        n, bad = insert_events(self.conn, events)
        if bad:
            write_rejects(bad, self.rejects_path)
            self.rejected += len(bad)
            self.log(f"[tiger] rejected {len(bad)} invalid events ({bad[0][1]}); "
                     f"kept in {os.path.basename(self.rejects_path)}")
        return n

    def _replay_spool(self):
        if not os.path.exists(self.spool_path):
            return
        with open(self.spool_path) as f:
            events = []
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if events:
            n = self._write(events)
            self.log(f"[tiger] sent {n} buffered events")
        os.remove(self.spool_path)

    def _spool(self, batch):
        with open(self.spool_path, "a") as f:
            for e in batch:
                f.write(json.dumps(e) + "\n")
        self.spooled += len(batch)

    def _run(self):
        next_retry = 0.0
        while not (self._stop.is_set() and self.q.empty()):
            batch = []
            try:
                batch.append(self.q.get(timeout=0.5))
                deadline = time.monotonic() + 1.0
                while len(batch) < 200 and time.monotonic() < deadline:
                    try:
                        batch.append(self.q.get(timeout=max(0.0, deadline - time.monotonic())))
                    except queue.Empty:
                        break
            except queue.Empty:
                if self.conn is None and os.path.exists(self.spool_path) and time.monotonic() >= next_retry:
                    batch = []           # idle with a backlog: try to reconnect and replay it
                else:
                    continue
            if time.monotonic() < next_retry:
                self._spool(batch)
                continue
            self._inflight = batch
            try:
                self._ensure_conn()
                if batch:
                    self.sent += self._write(batch)
                self._inflight = []
            except Exception as e:  # noqa: BLE001 - offline, or an unexpected bug: never lose events
                self.log(f"[tiger] offline ({type(e).__name__}: {(str(e).splitlines() or [''])[0][:120]}); "
                         f"buffering events, retrying in {self.retry_seconds:g} s")
                self._inflight = []
                if batch:
                    self._spool(batch)
                try:
                    if self.conn:
                        self.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self.conn = None
                next_retry = time.monotonic() + self.retry_seconds

    def close(self, timeout=10):
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            # Still stuck connecting: keep the in-flight batch and anything queued for next time.
            # (If the insert does finish, re-sending them later is harmless: upserts by event_id.)
            left = list(self._inflight)
            while True:
                try:
                    left.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if left:
                self._spool(left)
        elif self.conn:
            self.conn.close()
        self.log(f"[tiger] {self.sent} event records stored" +
                 (f", buffered in {os.path.basename(self.spool_path)}" if os.path.exists(self.spool_path) else "") +
                 (f", {self.rejected} rejected" if self.rejected else ""))


# ---------------------------------------------------------------- CLI commands

def split_sql(text):
    """Split a SQL script into statements, respecting quotes and -- comments."""
    out, cur, i, quote = [], [], 0, False
    while i < len(text):
        c = text[i]
        if not quote and text.startswith("--", i):
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
            continue
        if c == "'":
            quote = not quote
        if c == ";" and not quote:
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt)
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def cmd_init(url, args):
    with open(SCHEMA) as f:
        stmts = split_sql(f.read())
    with connect(url, autocommit=True) as conn:   # continuous aggregates can't run in a transaction
        for s in stmts:
            first = " ".join(s.split())[:70]
            conn.execute(s)
            print(f"  ok  {first}")
    print("Schema ready.")


def cmd_check(url, args):
    with connect(url) as conn:
        v = conn.execute("SELECT version()").fetchone()[0]
        ts = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
        print("server:", v.split(",")[0])
        print("timescaledb:", ts[0] if ts else "not installed")
        exists = conn.execute("SELECT to_regclass('detections') IS NOT NULL").fetchone()[0]
        if not exists:
            print("detections table missing: run `python scripts/tiger_bridge.py init`")
            return
        migrated = conn.execute("""SELECT count(*) FROM information_schema.columns
                                   WHERE table_name = 'detections' AND column_name = 'event_id'""").fetchone()[0]
        if not migrated:
            print("detections table is the pre-fusion schema: run `python scripts/tiger_bridge.py init`")
            return
        rows = conn.execute("""SELECT device_id, count(*) FILTER (WHERE event_id IS NOT NULL),
                                      count(*) FILTER (WHERE event_id IS NOT NULL AND fusion = 'MULTIMODAL'),
                                      count(*) FILTER (WHERE event_id IS NOT NULL AND alerted),
                                      count(*) FILTER (WHERE event_id IS NULL), max(time)
                               FROM detections GROUP BY device_id ORDER BY 6 DESC""").fetchall()
        if not rows:
            print("detections: empty")
        for d, n, m, a, legacy, last in rows:
            print(f"  {d:<10} {n:>6} events ({m} multimodal, {a} alerted)  "
                  f"{legacy:>6} pre-fusion rows  last {last:%Y-%m-%d %H:%M:%S %Z}")


def read_jsonl(lines):
    """Parse EVENT lines / JSONL. Returns (unified events, number of pre-fusion lines skipped)."""
    events, legacy = [], 0
    for line in lines:
        line = line.strip()
        if line.startswith("EVENT "):
            line = line[6:]
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if is_event(ev):
            events.append(ev)
        else:
            legacy += 1
    return events, legacy


def _ingest(url, events, legacy, what):
    with connect(url) as conn:
        before = conn.execute("SELECT count(*) FROM detections WHERE event_id IS NOT NULL").fetchone()[0]
        n, bad = insert_events(conn, events)
        after = conn.execute("SELECT count(*) FROM detections WHERE event_id IS NOT NULL").fetchone()[0]
    if bad:
        write_rejects(bad)
    print(f"Read {len(events)} event records from {what}: {after - before} new events, "
          f"{n} rows written (existing events updated, duplicates skipped).")
    if legacy:
        print(f"Skipped {legacy} pre-fusion raw detection lines (that format is no longer stored).")
    if bad:
        print(f"Rejected {len(bad)} invalid records -> {os.path.basename(REJECTS)} (first: {bad[0][1]})")


def cmd_ingest(url, args):
    with open(args.file) as f:
        events, legacy = read_jsonl(f)
    _ingest(url, events, legacy, args.file)


def cmd_ingest_pi(url, args):
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, "cat ~/hatdetect/events.jsonl"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        sys.exit(f"Could not read events from the Pi: {out.stderr.strip()}")
    events, legacy = read_jsonl(out.stdout.splitlines())
    _ingest(url, events, legacy, "the Pi")


DEMO_DEVICE = "demo-01"
# (weight, [(kind, group, label, frames, step_s)...]) incidents on a site; multimodal ones
# put the sound and the object close together so the real fusion rules combine them.
DEMO_INCIDENTS = [
    (30, [("video", "PERSON", "person", (3, 25), 0.2)]),
    (10, [("video", "VEHICLE", "truck", (5, 30), 0.2)]),
    (8, [("video", "VEHICLE", "car", (5, 20), 0.2)]),
    (9, [("audio", "ALARM", "Reversing beeps", (2, 6), 0.5)]),
    (5, [("audio", "HORN", "Vehicle horn, car horn, honking", (1, 3), 0.5)]),
    (6, [("audio", "HORN", "Air horn, truck horn", (1, 3), 0.5), ("video", "VEHICLE", "truck", (5, 20), 0.2)]),
    (2, [("audio", "SIREN", "Siren", (4, 10), 0.5), ("video", "VEHICLE", "car", (5, 15), 0.2)]),
    (4, [("audio", "VOICE", "Shout", (1, 3), 0.5)]),
    (2, [("audio", "VOICE", "Yell", (1, 2), 0.5), ("video", "PERSON", "person", (5, 15), 0.2)]),
    (2, [("audio", "IMPACT", "Smash, crash", (1, 2), 0.5)]),
    (1, [("audio", "IMPACT", "Bang", (1, 2), 0.5), ("video", "PERSON", "person", (5, 10), 0.2)]),
    (1, [("audio", "ALARM", "Smoke detector, smoke alarm", (4, 10), 0.5)]),
    (1, [("video", "SHARP", "knife", (2, 6), 0.2)]),
]


def demo_events(hours, seed=42, now=None):
    """Sample shift data made the way the hat makes it: simulated raw detections through
    the real fusion engine (hat_config.json rules), so the events look like live ones."""
    sys.path.insert(0, ROOT)
    import fusion
    with open(os.path.join(ROOT, "hat_config.json")) as f:
        cfg = json.load(f)
    engine = fusion.FusionEngine(cfg.get("fusion"), cfg["alerts"]["severity"], DEMO_DEVICE)
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    span = hours * 3600
    detections, t = [], 0.0
    weights = [w for w, _ in DEMO_INCIDENTS]
    while True:
        t += rng.expovariate(1 / 45)                          # an incident every ~45 s
        if t >= span - 60:
            break
        _, parts = rng.choices(DEMO_INCIDENTS, weights)[0]
        pos = rng.choice(["left", "centre", "right"])
        for i, (kind, group, label, (lo, hi), step) in enumerate(parts):
            base, offset = rng.uniform(0.5, 0.92), (rng.uniform(0, 1.2) if i else 0.0)
            for k in range(rng.randint(lo, hi)):
                ts = t + offset + k * step
                score = min(0.99, max(0.35, base + rng.uniform(-0.06, 0.06)))
                box = [rng.randint(0, 300), rng.randint(0, 200), rng.randint(320, 640), rng.randint(240, 480)]
                detections.append(fusion.Detection(kind, group, label, round(score, 3), ts,
                                                   (start + timedelta(seconds=ts)).isoformat(timespec="milliseconds"),
                                                   pos if kind == "video" else None, box if kind == "video" else None))
    detections.sort(key=lambda d: d.t)
    records, i, tick = [], 0, 0.0
    while i < len(detections):
        while i < len(detections) and detections[i].t <= tick:
            records += engine.add(detections[i])
            i += 1
        records += engine.tick(tick)
        tick += 0.25
    return records + engine.flush()


def cmd_demo(url, args):
    """Sample shift data so the dashboard has something to show before the hat runs."""
    records = demo_events(args.hours)
    with connect(url) as conn:
        old = conn.execute("DELETE FROM detections WHERE device_id = %s", (DEMO_DEVICE,)).rowcount
        conn.commit()
        n, bad = insert_events(conn, records)
    print(f"Added {n} sample events for {DEMO_DEVICE} over the last {args.hours:g} h"
          + (f" (replaced {old} old demo rows)" if old else "") + ".")
    if bad:
        print(f"{len(bad)} sample events were rejected: {bad[0][1]}")


def cmd_purge_demo(url, args):
    with connect(url) as conn:
        n = conn.execute("DELETE FROM detections WHERE device_id = %s", (DEMO_DEVICE,)).rowcount
        conn.commit()
    with connect(url, autocommit=True) as conn:   # drop the rolled-up minutes for the demo data too
        conn.execute("CALL refresh_continuous_aggregate('detections_per_minute', NULL, NULL)")
    print(f"Removed {n} demo detections.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("check")
    p = sub.add_parser("ingest")
    p.add_argument("file")
    p = sub.add_parser("ingest-pi")
    p.add_argument("--host", default="qnxpi")
    p = sub.add_parser("demo")
    p.add_argument("--hours", type=float, default=8)
    sub.add_parser("purge-demo")
    args = ap.parse_args()

    url = database_url()
    if not url:
        sys.exit("TIGER_DATABASE_URL is not set. Copy .env.example to .env and paste your Tiger Cloud "
                 "service's connection string.")
    {"init": cmd_init, "check": cmd_check, "ingest": cmd_ingest, "ingest-pi": cmd_ingest_pi,
     "demo": cmd_demo, "purge-demo": cmd_purge_demo}[args.cmd](url, args)


if __name__ == "__main__":
    main()
