"""Hat Detect safety dashboard: hazards detected by the hats, live from Tiger Cloud.

Runs a small local web server; the database password stays on this laptop.
    python scripts/dashboard.py            # then open http://localhost:8050
    python scripts/dashboard.py --port 9000
Reads TIGER_DATABASE_URL from .env (see scripts/tiger_bridge.py).
"""
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiger_bridge  # noqa: E402

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
RANGES = {  # name -> (lookback, bucket for the timeline, label format)
    "1h": ("1 hour", "1 minute", "%H:%M"),
    "8h": ("8 hours", "10 minutes", "%H:%M"),
    "24h": ("24 hours", "30 minutes", "%H:%M"),
    "7d": ("7 days", "6 hours", "%a %H:%M"),
}
# Every query filters to one hat (or all), and hides simulated hats (sim-...) unless asked.
SIM = "(%(sim)s OR device_id NOT LIKE 'sim-%%')"
DEV = "(%(dev)s::text IS NULL OR device_id = %(dev)s) AND " + SIM


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
    return t.astimezone(timezone.utc).isoformat() if t else None


def gather(db, rng, dev, sim=False):
    lookback, bucket, fmt = RANGES.get(rng, RANGES["8h"])
    p = {"lookback": lookback, "bucket": bucket, "dev": dev or None, "sim": sim}
    since = f"now() - %(lookback)s::interval"

    k = db.query(f"""
        SELECT count(*), count(*) FILTER (WHERE alerted),
               count(*) FILTER (WHERE alerted AND severity = 'high'),
               count(DISTINCT hazard), count(DISTINCT device_id), max(time)
        FROM detections WHERE time > {since} AND {DEV}""", p)[0]
    last_alert = db.query(f"""
        SELECT time, hazard, label, device_id FROM detections
        WHERE alerted AND time > {since} AND {DEV} ORDER BY time DESC LIMIT 1""", p)

    timeline = db.query(f"""
        SELECT time_bucket(%(bucket)s::interval, bucket) AS b, severity,
               sum(alerts)::int, sum(detections)::int
        FROM detections_per_minute WHERE bucket > {since} AND {DEV}
        GROUP BY b, severity ORDER BY b""", p)
    hazards = db.query(f"""
        SELECT hazard, severity, sum(detections)::int, sum(alerts)::int
        FROM detections_per_minute WHERE bucket > {since} AND {DEV}
        GROUP BY hazard, severity
        ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, 3 DESC""", p)
    positions = db.query(f"""
        SELECT hazard, position, count(*)::int FROM detections
        WHERE time > {since} AND {DEV} AND position IS NOT NULL AND alerted
        GROUP BY hazard, position""", p)
    register = db.query(f"""
        SELECT h.hazard, h.severity, h.source, h.description, h.control,
               coalesce(s.detections, 0), coalesce(s.alerts, 0), s.last_seen, s.peak
        FROM hazard_types h LEFT JOIN (
            SELECT hazard, count(*)::int AS detections, count(*) FILTER (WHERE alerted)::int AS alerts,
                   max(time) AS last_seen, max(score) AS peak
            FROM detections WHERE time > {since} AND {DEV} GROUP BY hazard) s USING (hazard)
        ORDER BY CASE h.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                 coalesce(s.alerts, 0) DESC, h.hazard""", p)
    recent = db.query(f"""
        SELECT time, device_id, hazard, severity, label, score, position, control
        FROM safety_alerts WHERE time > {since} AND {DEV}
        ORDER BY time DESC LIMIT 25""", p)
    devices = db.query(f"""
        SELECT device_id, w.name, max(m.bucket) FROM detections_per_minute m
        LEFT JOIN workers w USING (device_id)
        WHERE m.bucket > now() - interval '30 days' AND {SIM}
        GROUP BY device_id, w.name ORDER BY w.name NULLS LAST, device_id""", p)
    worker = db.query("SELECT device_id, name, role, zone, simulated FROM workers WHERE device_id = %(dev)s",
                      p) if dev else []

    buckets = sorted({r[0] for r in timeline})
    series = {s: [0] * len(buckets) for s in ("high", "medium", "low")}
    idx = {b: i for i, b in enumerate(buckets)}
    for b, sev, alerts, _ in timeline:
        series.setdefault(sev, [0] * len(buckets))[idx[b]] += alerts

    return {
        "now": iso(datetime.now(timezone.utc)), "range": rng, "device": dev or "",
        "kpis": {"detections": k[0], "alerts": k[1], "high_alerts": k[2], "hazard_types": k[3],
                 "devices": k[4], "last_event": iso(k[5]),
                 "last_alert": ({"time": iso(last_alert[0][0]), "hazard": last_alert[0][1],
                                 "label": last_alert[0][2], "device": last_alert[0][3]} if last_alert else None)},
        "timeline": {"labels": [b.astimezone().strftime(fmt) for b in buckets], "series": series},
        "hazards": [{"hazard": h, "severity": s, "detections": d, "alerts": a} for h, s, d, a in hazards],
        "positions": [{"hazard": h, "position": pos, "count": n} for h, pos, n in positions],
        "register": [{"hazard": r[0], "severity": r[1], "source": r[2], "description": r[3], "control": r[4],
                      "detections": r[5], "alerts": r[6], "last_seen": iso(r[7]),
                      "peak": round(r[8], 2) if r[8] is not None else None} for r in register],
        "recent": [{"time": iso(r[0]), "device": r[1], "hazard": r[2], "severity": r[3], "label": r[4],
                    "score": round(r[5], 2), "position": r[6], "control": r[7]} for r in recent],
        "devices": [{"id": d, "name": n or d, "last": iso(t)} for d, n, t in devices],
        "worker": ({"id": worker[0][0], "name": worker[0][1], "role": worker[0][2], "zone": worker[0][3],
                    "simulated": worker[0][4]} if worker else
                   ({"id": dev, "name": dev, "role": "Unregistered hat", "zone": "Unassigned",
                     "simulated": dev.startswith("sim-")} if dev else None)),
    }


def overview(db, rng, sim=False):
    """Site-wide view: one row per worker, plus totals and per-zone counts. Counts come from
    the per-minute rollup, so this stays fast as hats and history grow."""
    t0 = time.perf_counter()
    lookback, bucket, fmt = RANGES.get(rng, RANGES["8h"])
    p = {"lookback": lookback, "bucket": bucket, "sim": sim}
    since = "now() - %(lookback)s::interval"

    rows = db.query(f"""
        WITH s AS (
            SELECT device_id, sum(detections)::int AS det, sum(alerts)::int AS alerts,
                   sum(alerts) FILTER (WHERE severity = 'high')::int AS high,
                   sum(alerts) FILTER (WHERE severity = 'medium')::int AS medium,
                   sum(alerts) FILTER (WHERE severity = 'low')::int AS low, max(bucket) AS last_seen
            FROM detections_per_minute WHERE bucket > {since} AND {SIM} GROUP BY device_id),
        top AS (
            SELECT DISTINCT ON (device_id) device_id, hazard FROM (
                SELECT device_id, hazard, severity, sum(alerts) AS n FROM detections_per_minute
                WHERE bucket > {since} AND {SIM} AND alerts > 0 GROUP BY device_id, hazard, severity) x
            ORDER BY device_id, CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, n DESC),
        la AS (
            SELECT DISTINCT ON (device_id) device_id, time, hazard, severity FROM detections
            WHERE alerted AND time > {since} AND {SIM} ORDER BY device_id, time DESC)
        SELECT device_id, w.name, w.role, w.zone, coalesce(w.simulated, false),
               coalesce(s.det, 0), coalesce(s.alerts, 0), coalesce(s.high, 0), coalesce(s.medium, 0),
               coalesce(s.low, 0), s.last_seen, top.hazard, la.time, la.hazard, la.severity
        FROM workers w FULL JOIN s USING (device_id)
        LEFT JOIN top USING (device_id) LEFT JOIN la USING (device_id)
        WHERE {SIM}""", p)
    spark = db.query(f"""
        SELECT device_id, time_bucket(%(bucket)s::interval, bucket) AS b, severity, sum(alerts)::int
        FROM detections_per_minute WHERE bucket > {since} AND {SIM}
        GROUP BY device_id, b, severity ORDER BY b""", p)
    zones = db.query(f"""
        SELECT coalesce(w.zone, 'Unassigned'), m.severity, sum(m.alerts)::int
        FROM detections_per_minute m LEFT JOIN workers w USING (device_id)
        WHERE m.bucket > {since} AND {SIM} GROUP BY 1, 2""", p)
    total_rows = db.query(f"SELECT count(*) FROM detections WHERE {SIM}", p)[0][0]

    buckets = sorted({r[1] for r in spark})
    idx = {b: i for i, b in enumerate(buckets)}
    sparks = {}
    series = {s: [0] * len(buckets) for s in ("high", "medium", "low")}
    for d, b, sev, n in spark:
        sparks.setdefault(d, [0] * len(buckets))[idx[b]] += n
        series.setdefault(sev, [0] * len(buckets))[idx[b]] += n
    zmap = {}
    for z, sev, n in zones:
        zmap.setdefault(z, {"zone": z, "high": 0, "medium": 0, "low": 0})[sev] = n

    now = datetime.now(timezone.utc)
    workers = []
    for r in rows:
        d = r[0]
        workers.append({
            "id": d, "name": r[1] or d, "role": r[2] or "Unregistered hat", "zone": r[3] or "Unassigned",
            "simulated": bool(r[4]) or d.startswith("sim-"), "detections": r[5], "alerts": r[6],
            "high": r[7], "medium": r[8], "low": r[9], "risk": 5 * r[7] + 2 * r[8] + r[9],
            "last_seen": iso(r[10]), "active": bool(r[10] and (now - r[10]).total_seconds() < 600),
            "top_hazard": r[11],
            "last_alert": {"time": iso(r[12]), "hazard": r[13], "severity": r[14]} if r[12] else None,
            "spark": sparks.get(d, [0] * len(buckets))})
    workers.sort(key=lambda w: (-w["risk"], w["name"]))
    return {
        "now": iso(now), "range": rng, "sim": sim,
        "kpis": {"workers": len(workers), "active": sum(w["active"] for w in workers),
                 "alerts": sum(w["alerts"] for w in workers), "high_alerts": sum(w["high"] for w in workers),
                 "at_risk": sum(1 for w in workers if w["high"]),
                 "detections": sum(w["detections"] for w in workers),
                 "total_rows": total_rows, "query_ms": round((time.perf_counter() - t0) * 1000)},
        "timeline": {"labels": [b.astimezone().strftime(fmt) for b in buckets], "series": series},
        "zones": sorted(zmap.values(), key=lambda z: -(5 * z["high"] + 2 * z["medium"] + z["low"])),
        "workers": workers,
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
            elif u.path in ("/api/data", "/api/overview"):
                q = parse_qs(u.query)
                rng, sim = q.get("range", ["8h"])[0], q.get("sim", ["0"])[0] == "1"
                try:
                    data = (overview(db, rng, sim) if u.path == "/api/overview"
                            else gather(db, rng, q.get("device", [""])[0], sim))
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
