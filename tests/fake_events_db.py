"""In-memory stand-in for the `hazard_events` / `hazard_types` queries in scripts/dashboard.py.

Answers each of gather()'s queries from a list of unified events (by default the bridge's
demo events, made by the real fusion engine), applying the same filters as the SQL. Used
by tests/test_dashboard.py; the SQL text itself is checked against a real database.
"""
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tiger_bridge as tb  # noqa: E402

_REG = re.compile(r"\('(\w+)',\s*'(\w+)',\s*'(\w+)',\s*'((?:[^']|'')*)',\s*\n?\s*'((?:[^']|'')*)'\)")


def hazard_types():
    """(hazard, severity, source, description, control): the column order dashboard.py selects."""
    with open(os.path.join(ROOT, "db", "schema.sql")) as f:
        rows = [tuple(v.replace("''", "'") for v in m) for m in _REG.findall(f.read())]
    return [(h, sev, src, desc, ctl) for h, src, sev, desc, ctl in rows]   # INSERT order is source, severity


def to_view_rows(records):
    """Event records -> rows shaped like the hazard_events view (latest record per event)."""
    reg = {h[0]: h for h in hazard_types()}
    rows = []
    for ev in tb.latest_per_event(records):
        r = dict(zip([c.strip() for c in tb.COLUMNS.split(",")], tb.to_row(ev)))
        r["time"] = datetime.fromisoformat(r["time"])
        r["last_time"] = datetime.fromisoformat(r["last_time"]) if r["last_time"] else None
        h = reg.get(r["hazard"])
        r["description"], r["control"] = (h[3], h[4]) if h else (None, None)
        rows.append(r)
    return rows


class FakeEventsDb:
    def __init__(self, rows=None, now=None):
        self.now = now or datetime.now(timezone.utc)
        self.rows = rows if rows is not None else to_view_rows(tb.demo_events(8, now=self.now))
        self.register = hazard_types()
        self.queries = []

    def _match(self, sql, p):
        out = []
        for r in self.rows:
            if not (p["since"] <= r["time"] < p["until"]):
                continue
            if p.get("device") and "device_id = %(device)s" in sql and r["device_id"] != p["device"]:
                continue
            if any(p.get(k) and f"{k} = %({k})s" in sql and r[k] != p[k] for k in ("hazard", "severity", "fusion")):
                continue
            if "AND audio_group IS NOT NULL" in sql and r["audio_group"] is None:
                continue
            if "AND video_group IS NOT NULL" in sql and r["video_group"] is None:
                continue
            out.append(r)
        return out

    def query(self, sql, params):
        self.queries.append((sql, params))
        s = " ".join(sql.split())
        if "FROM hazard_types" in s:
            return list(self.register)
        if "now() - INTERVAL '30 days'" in s:
            last = {}
            for r in self.rows:
                if r["time"] > self.now - timedelta(days=30):
                    last[r["device_id"]] = max(last.get(r["device_id"], r["time"]), r["time"])
            return sorted(last.items(), key=lambda x: x[1], reverse=True)
        rows = self._match(s, params)
        if "count(DISTINCT device_id)" in s:
            c = Counter()
            for r in rows:
                c["high"] += r["severity"] == "high"
                c[r["fusion"]] += 1
                c["alerted"] += bool(r["alerted"])
                c["audio"] += r["audio_group"] is not None
                c["video"] += r["video_group"] is not None
            return [(len(rows), c["high"], c["MULTIMODAL"], c["AUDIO_ONLY"], c["VISION_ONLY"], c["alerted"],
                     c["audio"], c["video"], len({r["device_id"] for r in rows}),
                     max((r["time"] for r in rows), default=None))]
        if "time_bucket" in s:
            b = params["bucket"]
            c = Counter()
            for r in rows:
                t = int(r["time"].timestamp()) // b * b
                c[(datetime.fromtimestamp(t, timezone.utc), r["fusion"])] += 1
            return [(k[0], k[1], n) for k, n in c.items()]
        if "max(score)" in s:
            g = defaultdict(list)
            for r in rows:
                g[(r["hazard"], r["severity"])].append(r)
            return [(h, sv, len(v), sum(bool(x["alerted"]) for x in v), max(x["time"] for x in v),
                     max(x["score"] for x in v)) for (h, sv), v in g.items()]
        if "position IS NOT NULL" in s:
            c = Counter((r["position"], r["hazard"]) for r in rows if r["position"] is not None)
            return [(pos, h, n) for (pos, h), n in c.items()]
        if "ORDER BY time DESC LIMIT" in s:
            limit = int(re.search(r"LIMIT (\d+)", s).group(1))
            cols = re.search(r"SELECT (.*?) FROM hazard_events", s).group(1).split(", ")
            ordered = sorted(rows, key=lambda r: r["time"], reverse=True)[:limit]
            return [tuple(r[c] for c in cols) for r in ordered]
        raise AssertionError(f"unexpected query: {s[:120]}")
