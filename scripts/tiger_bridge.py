"""Tiger Data bridge for Hat Detect: puts every detection from the hat into Tiger Cloud.

The Pi (QNX) has no Postgres client and never holds the database password. Its events
reach the laptop as `EVENT {json}` lines over SSH (scripts/laptop_companion.py), and this
module inserts them into the `detections` hypertable. If Tiger Cloud is unreachable,
events are kept in .tiger_spool.jsonl and sent when the connection comes back.

Setup: put the service's connection string in .env (see .env.example):
    TIGER_DATABASE_URL=postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require

Commands (run from the project root):
    python scripts/tiger_bridge.py init          # create tables, rollup, policies (safe to re-run)
    python scripts/tiger_bridge.py check         # connection + row counts
    python scripts/tiger_bridge.py ingest-pi     # load the Pi's ~/hatdetect/events.jsonl (dedups)
    python scripts/tiger_bridge.py ingest FILE   # load a local events.jsonl
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPOOL = os.path.join(ROOT, ".tiger_spool.jsonl")
SCHEMA = os.path.join(ROOT, "db", "schema.sql")

# Same severities as hat_config.json; used when an event doesn't carry one.
SEVERITY = {"COMBINED": "high", "ALARM": "high", "SIREN": "high", "IMPACT": "high", "FIRE": "high",
            "HORN": "medium", "VOICE": "medium", "VEHICLE": "medium", "SHARP": "medium", "PERSON": "low"}

INSERT_SQL = """
INSERT INTO detections (time, device_id, source, hazard, severity, label, score, position, bbox, alerted)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (device_id, source, label, time) DO NOTHING
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


def to_row(ev):
    """Pi event dict -> detections row. Accepts the hat's `group` field as the hazard."""
    from psycopg.types.json import Jsonb
    hazard = ev.get("hazard") or ev.get("group") or "UNKNOWN"
    bbox = ev.get("bbox")
    return (ev["time"], ev.get("device_id", "hat-01"), ev["source"], hazard,
            ev.get("severity") or SEVERITY.get(hazard, "low"), ev["label"], float(ev["score"]),
            ev.get("position"), Jsonb(bbox) if bbox is not None else None, bool(ev.get("alerted")))


def insert_events(conn, events):
    rows = [to_row(e) for e in events]
    with conn.cursor() as cur:
        cur.executemany(INSERT_SQL, rows)
    conn.commit()
    return len(rows)


class TigerWriter:
    """Background writer used by the laptop companion. add() never blocks or raises:
    events are batched (up to 1 s) and inserted; on failure they go to the spool file
    and are retried every 10 s."""

    def __init__(self, url, log=print):
        self.url, self.log = url, log
        self.q = queue.Queue()
        self.conn = None
        self.sent = 0
        self.spooled = 0
        self._inflight = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, ev):
        self.q.put(ev)

    def _ensure_conn(self):
        if self.conn is None or self.conn.closed:
            self.conn = connect(self.url)
            self.log("[tiger] connected to Tiger Cloud")
            self._replay_spool()

    def _replay_spool(self):
        if not os.path.exists(SPOOL):
            return
        with open(SPOOL) as f:
            events = [json.loads(line) for line in f if line.strip()]
        if events:
            n = insert_events(self.conn, events)
            self.log(f"[tiger] sent {n} buffered events")
        os.remove(SPOOL)

    def _spool(self, batch):
        with open(SPOOL, "a") as f:
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
                continue
            if time.monotonic() < next_retry:
                self._spool(batch)
                continue
            self._inflight = batch
            try:
                self._ensure_conn()
                self.sent += insert_events(self.conn, batch)
                self._inflight = []
            except Exception as e:  # noqa: BLE001 - network/db errors: buffer and retry later
                self.log(f"[tiger] offline ({type(e).__name__}: {str(e).splitlines()[0][:120]}); "
                         "buffering events, retrying in 10 s")
                self._inflight = []
                self._spool(batch)
                try:
                    if self.conn:
                        self.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self.conn = None
                next_retry = time.monotonic() + 10

    def close(self, timeout=10):
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            # Still stuck connecting: keep the in-flight batch and anything queued for next time.
            # (If the insert does finish, re-sending them later is harmless: duplicates are skipped.)
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
        self.log(f"[tiger] {self.sent} events stored" +
                 (f", {self.spooled} buffered in {os.path.basename(SPOOL)}" if os.path.exists(SPOOL) else ""))


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
        rows = conn.execute("""SELECT device_id, count(*), count(*) FILTER (WHERE alerted), max(time)
                               FROM detections GROUP BY device_id ORDER BY 4 DESC""").fetchall()
        if not rows:
            print("detections: empty")
        for d, n, a, last in rows:
            print(f"  {d:<10} {n:>7} detections  {a:>6} alerts  last {last:%Y-%m-%d %H:%M:%S %Z}")


def read_jsonl(lines):
    events = []
    for line in lines:
        line = line.strip()
        if line.startswith("EVENT "):
            line = line[6:]
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def cmd_ingest(url, args):
    with open(args.file) as f:
        events = read_jsonl(f)
    with connect(url) as conn:
        n = insert_events(conn, events)
    print(f"Sent {n} events from {args.file} (duplicates skipped).")


def cmd_ingest_pi(url, args):
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, "cat ~/hatdetect/events.jsonl"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        sys.exit(f"Could not read events from the Pi: {out.stderr.strip()}")
    events = read_jsonl(out.stdout.splitlines())
    with connect(url) as conn:
        before = conn.execute("SELECT count(*) FROM detections").fetchone()[0]
        insert_events(conn, events)
        after = conn.execute("SELECT count(*) FROM detections").fetchone()[0]
    print(f"Read {len(events)} events from the Pi; {after - before} were new.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("check")
    p = sub.add_parser("ingest")
    p.add_argument("file")
    p = sub.add_parser("ingest-pi")
    p.add_argument("--host", default="qnxpi")
    args = ap.parse_args()

    url = database_url()
    if not url:
        sys.exit("TIGER_DATABASE_URL is not set. Copy .env.example to .env and paste your Tiger Cloud "
                 "service's connection string.")
    {"init": cmd_init, "check": cmd_check, "ingest": cmd_ingest, "ingest-pi": cmd_ingest_pi}[args.cmd](url, args)


if __name__ == "__main__":
    main()
