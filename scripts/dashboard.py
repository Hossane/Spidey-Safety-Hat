"""Hat Detect safety dashboard: the hats' unified hazard events, live from Tiger Cloud.

Shows what the helmet detected and how the fusion engine interpreted it: headline counts
(all, high severity, multimodal, audio-only, vision-only), events over time, breakdowns,
a clickable event timeline with full details (both confidences, why audio and video were
combined, the recommended control), and the hazard register. Filters: time range or
custom window, hat, hazard, severity, event type, and audio/video source. The page polls
/api/data every 5 s.

Reads the `hazard_events` view, so rows from before fusion are not counted.
Runs a small local web server; the database password stays on this laptop.
    python scripts/dashboard.py            # then open http://localhost:8050
    python scripts/dashboard.py --port 9000
Reads TIGER_DATABASE_URL from .env (see scripts/tiger_bridge.py).
"""
import argparse
import json
import os
import sys
import re
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiger_bridge  # noqa: E402

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
RANGES = {"1h": 3600, "8h": 8 * 3600, "24h": 24 * 3600, "7d": 7 * 24 * 3600}
MAX_SPAN = 90 * 24 * 3600            # retention: there is nothing older
BUCKETS = [(3600, 60), (8 * 3600, 600), (24 * 3600, 1800), (7 * 24 * 3600, 6 * 3600)]  # span -> bucket s
SEVERITIES = ("high", "medium", "low")
FUSIONS = ("MULTIMODAL", "AUDIO_ONLY", "VISION_ONLY")
SOURCES = ("audio", "video")
LEGACY_HAZARDS = {"COMBINED"}        # pre-fusion register entry: shown only if it has events
RECENT_LIMIT = 100
EVENT_COLUMNS = ("event_id, time, last_time, device_id, hazard, severity, fusion, label, "
                 "audio_group, audio_label, audio_score, audio_count, "
                 "video_group, video_label, video_score, video_count, position, bbox, "
                 "correlation_window, explanation, alerted, status, description, control")


class Db:
    def __init__(self, url):
        self.url, self.conn, self.lock = url, None, threading.Lock()

    def query(self, sql, params):
        with self.lock:
            for attempt in (1, 2):
                try:
                    if self.conn is None or self.conn.closed:
                        self.conn = tiger_bridge.connect(self.url, autocommit=True)
                    return self.conn.execute(sql, params).fetchall()
                except Exception:  # noqa: BLE001 - reconnect once, then report
                    try:
                        if self.conn:
                            self.conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self.conn = None
                    if attempt == 2:
                        raise


def iso(t):
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds") if t else None


def _parse_time(text):
    t = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def parse_filters(q, now=None):
    """Query-string dict (values are lists, as from parse_qs) -> validated filters.
    Unknown or malformed values are ignored rather than trusted."""
    now = now or datetime.now(timezone.utc)
    get = lambda k: (q.get(k) or [""])[0].strip()  # noqa: E731
    rng = get("range") or "8h"
    since = until = None
    if rng == "custom":
        try:
            since, until = _parse_time(get("from")), _parse_time(get("to"))
        except ValueError:
            since = until = None
        if not since or not until or since >= until:
            rng = "8h"
            since = until = None
        else:
            until = min(until, now)
            since = max(since, until - timedelta(seconds=MAX_SPAN))
    if since is None:
        rng = rng if rng in RANGES else "8h"
        until, since = now, now - timedelta(seconds=RANGES[rng])
    span = (until - since).total_seconds()
    bucket = next((b for limit, b in BUCKETS if span <= limit), 24 * 3600)
    hazard = get("hazard")
    return {
        "range": rng, "since": since, "until": until, "bucket": bucket,
        "device": get("device") or None,
        "hazard": hazard if re.fullmatch(r"[A-Z_]{1,40}", hazard) else None,
        "severity": get("severity") if get("severity") in SEVERITIES else None,
        "fusion": get("fusion") if get("fusion") in FUSIONS else None,
        "source": get("source") if get("source") in SOURCES else None,
    }


def where(f):
    """SQL WHERE clause + params for the filters (all values bound, never interpolated)."""
    conds = ["time >= %(since)s", "time < %(until)s"]
    for key in ("device", "hazard", "severity", "fusion"):
        if f[key]:
            conds.append(f"{'device_id' if key == 'device' else key} = %({key})s")
    if f["source"]:
        conds.append(f"{f['source']}_group IS NOT NULL")
    return " AND ".join(conds), {k: f[k] for k in ("since", "until", "device", "hazard", "severity", "fusion")}


def bucket_starts(f):
    """Every bucket in the window (aligned like time_bucket: multiples of the bucket since
    the epoch, UTC), so empty periods show as zero instead of disappearing."""
    b = f["bucket"]
    start = int(f["since"].timestamp()) // b * b
    end = f["until"].timestamp()
    return [datetime.fromtimestamp(t, timezone.utc) for t in range(start, int(end) + 1, b) if t < end]


def gather(db, f):
    w, p = where(f)
    p = dict(p, bucket=f["bucket"])

    k = db.query(f"""
        SELECT count(*),
               count(*) FILTER (WHERE severity = 'high'),
               count(*) FILTER (WHERE fusion = 'MULTIMODAL'),
               count(*) FILTER (WHERE fusion = 'AUDIO_ONLY'),
               count(*) FILTER (WHERE fusion = 'VISION_ONLY'),
               count(*) FILTER (WHERE alerted),
               count(*) FILTER (WHERE audio_group IS NOT NULL),
               count(*) FILTER (WHERE video_group IS NOT NULL),
               count(DISTINCT device_id), max(time)
        FROM hazard_events WHERE {w}""", p)[0]
    timeline = db.query(f"""
        SELECT time_bucket(%(bucket)s * INTERVAL '1 second', time) AS b, fusion, count(*)::int
        FROM hazard_events WHERE {w} GROUP BY 1, 2""", p)
    by_hazard = db.query(f"""
        SELECT hazard, severity, count(*)::int, count(*) FILTER (WHERE alerted)::int, max(time), max(score)
        FROM hazard_events WHERE {w} GROUP BY 1, 2""", p)
    positions = db.query(f"""
        SELECT position, hazard, count(*)::int FROM hazard_events
        WHERE {w} AND position IS NOT NULL GROUP BY 1, 2""", p)
    recent = db.query(f"""
        SELECT {EVENT_COLUMNS} FROM hazard_events WHERE {w}
        ORDER BY time DESC LIMIT {RECENT_LIMIT}""", p)
    register = db.query("SELECT hazard, severity, source, description, control FROM hazard_types", {})
    devices = db.query("""SELECT device_id, max(time) FROM hazard_events
                          WHERE time > now() - INTERVAL '30 days' GROUP BY 1 ORDER BY 2 DESC""", {})
    return shape(f, k, timeline, by_hazard, positions, recent, register, devices)


def shape(f, k, timeline, by_hazard, positions, recent, register, devices):
    """Query rows -> the JSON the page renders (separate from gather() so it can be tested)."""
    starts = bucket_starts(f)
    idx = {int(b.timestamp()): i for i, b in enumerate(starts)}
    series = {fu: [0] * len(starts) for fu in FUSIONS}
    for b, fusion, n in timeline:
        i = idx.get(int(b.timestamp()))
        if i is not None and fusion in series:
            series[fusion][i] += n

    stats = {}
    for hazard, severity, n, alerted, last, peak in by_hazard:
        s = stats.setdefault(hazard, {"events": 0, "alerted": 0, "last_seen": None, "peak": None,
                                      "severity": severity})
        s["events"] += n
        s["alerted"] += alerted
        s["last_seen"] = max(filter(None, [s["last_seen"], last]), default=None)
        s["peak"] = max(filter(lambda v: v is not None, [s["peak"], peak]), default=None)
    sev_counts = {sv: 0 for sv in SEVERITIES}
    for hazard, severity, n, *_ in by_hazard:
        if severity in sev_counts:
            sev_counts[severity] += n

    rank = {sv: i for i, sv in enumerate(SEVERITIES)}
    reg = []
    for hazard, severity, source, description, control in register:
        s = stats.get(hazard, {})
        if hazard in LEGACY_HAZARDS and not s:
            continue
        reg.append({"hazard": hazard, "severity": severity, "source": source, "description": description,
                    "control": control, "events": s.get("events", 0), "alerted": s.get("alerted", 0),
                    "last_seen": iso(s.get("last_seen")),
                    "peak": round(s["peak"], 2) if s.get("peak") is not None else None})
    known = {r["hazard"] for r in reg}
    for hazard, s in stats.items():              # events for a hazard missing from the register
        if hazard not in known:
            reg.append({"hazard": hazard, "severity": s["severity"], "source": None, "description": None,
                        "control": None, "events": s["events"], "alerted": s["alerted"],
                        "last_seen": iso(s["last_seen"]), "peak": round(s["peak"], 2) if s["peak"] else None})
    reg.sort(key=lambda r: (rank.get(r["severity"], 3), -r["events"], r["hazard"]))

    cols = [c.strip() for c in EVENT_COLUMNS.split(",")]
    events = []
    for row in recent:
        e = dict(zip(cols, row))
        e["event_id"] = str(e["event_id"])
        e["time"], e["last_time"] = iso(e["time"]), iso(e["last_time"])
        if isinstance(e["bbox"], str):
            e["bbox"] = json.loads(e["bbox"])
        for c in ("audio_score", "video_score", "correlation_window"):
            if e[c] is not None:
                e[c] = round(float(e[c]), 3)
        events.append(e)

    total = k[0]
    return {
        "now": iso(datetime.now(timezone.utc)),
        "filters": {"range": f["range"], "since": iso(f["since"]), "until": iso(f["until"]),
                    "bucket_seconds": f["bucket"], "device": f["device"] or "", "hazard": f["hazard"] or "",
                    "severity": f["severity"] or "", "fusion": f["fusion"] or "", "source": f["source"] or ""},
        "kpis": {"events": total, "high": k[1], "multimodal": k[2], "audio_only": k[3], "vision_only": k[4],
                 "alerted": k[5], "with_audio": k[6], "with_video": k[7], "devices": k[8],
                 "last_event": iso(k[9])},
        "timeline": {"buckets": [iso(b) for b in starts], "series": series},
        "severity": [{"severity": sv, "events": sev_counts[sv]} for sv in SEVERITIES],
        "positions": [{"position": pos, "hazard": h, "count": n} for pos, h, n in positions],
        "register": reg,
        "events": events,
        "events_limit": RECENT_LIMIT,
        "devices": [{"id": d, "last": iso(t)} for d, t in devices],
    }


def make_handler(db):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                with open(HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif u.path == "/api/data":
                try:
                    data = gather(db, parse_filters(parse_qs(u.query)))
                    self._send(200, json.dumps(data), "application/json")
                except Exception as e:  # noqa: BLE001
                    msg = str(e).splitlines()[0][:300]
                    self._send(503, json.dumps({"error": f"{type(e).__name__}: {msg}"}), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def log_message(self, fmt, *args):  # keep the console quiet
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    url = tiger_bridge.database_url()
    if not url:
        sys.exit("TIGER_DATABASE_URL is not set. Copy .env.example to .env and paste your Tiger Cloud "
                 "service's connection string.")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Db(url)))
    print(f"Hat Detect safety dashboard on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
