"""Integration tests against a real PostgreSQL + TimescaleDB (CLAUDE.md tests 6 and 7, plus
the Phase 4 migration). Skipped unless HATDETECT_TEST_DATABASE_URL points at a LOCAL server.

The tests create (and finally drop) their own database, `hatdetect_it`, on that server, so
they never touch an existing one. Only localhost URLs are accepted, so Tiger Cloud can't
be used by accident.

    docker run -d --name hatdetect-tsdb -p 127.0.0.1:55432:5432 \\
        -e POSTGRES_PASSWORD=localtest timescale/timescaledb:latest-pg17
    HATDETECT_TEST_DATABASE_URL=postgres://postgres:localtest@127.0.0.1:55432/postgres \\
        .venv/bin/python -m unittest tests.test_integration_db -v
"""
import io
import os
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ADMIN_URL = os.environ.get("HATDETECT_TEST_DATABASE_URL", "")
DB_NAME = "hatdetect_it"
PRE_FUSION_COMMIT = "f9548d0"        # last commit with the raw-detection schema
try:
    import psycopg
except ImportError:
    psycopg = None

skip_reason = None
if not ADMIN_URL:
    skip_reason = "HATDETECT_TEST_DATABASE_URL not set"
elif psycopg is None:
    skip_reason = "psycopg not installed (pip install 'psycopg[binary]')"
elif urlsplit(ADMIN_URL).hostname not in ("localhost", "127.0.0.1", "::1"):
    skip_reason = "HATDETECT_TEST_DATABASE_URL must be a local server (these tests create and drop a database)"

if not skip_reason:
    import dashboard
    import tiger_bridge as tb
    from fake_events_db import FakeEventsDb, to_view_rows
    from test_fusion import CAR, HORN, det, repeat, run

NOW = datetime.now(timezone.utc).replace(microsecond=0)
URL = urlunsplit(urlsplit(ADMIN_URL)._replace(path="/" + DB_NAME)) if ADMIN_URL else ""

OLD_INSERT = """INSERT INTO detections (time, device_id, source, hazard, severity, label, score, position, bbox, alerted)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (device_id, source, label, time) DO NOTHING"""


def run_init():
    with redirect_stdout(io.StringIO()):
        tb.cmd_init(URL, SimpleNamespace())


@unittest.skipIf(skip_reason, skip_reason)
class RealDatabase(unittest.TestCase):
    """Tests run in definition order on one database (unittest sorts names: hence t1_, t2_...)."""

    @classmethod
    def setUpClass(cls):
        with psycopg.connect(ADMIN_URL, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")
            c.execute(f"CREATE DATABASE {DB_NAME}")
        with psycopg.connect(URL, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        cls.conn = psycopg.connect(URL, autocommit=True)

        # 1. The database as the previous version left it: old schema + raw detections, the
        #    oldest of them in compressed (columnstore) chunks.
        old = subprocess.run(["git", "-C", ROOT, "show", f"{PRE_FUSION_COMMIT}:db/schema.sql"],
                             capture_output=True, text=True)
        cls.has_old_schema = old.returncode == 0
        if cls.has_old_schema:
            for stmt in tb.split_sql(old.stdout):
                cls.conn.execute(stmt)
            rows = []
            for days in (20, 12, 9, 1):
                t = NOW - timedelta(days=days)
                rows += [(t + timedelta(seconds=i), "hat-01", "video", "PERSON", "low", "person", 0.8,
                          "left", "[1, 2, 3, 4]", i == 0) for i in range(5)]
            with cls.conn.cursor() as cur:
                cur.executemany(OLD_INSERT, rows)
            cls.legacy_rows = len(rows)
            chunks = cls.conn.execute(
                "SELECT show_chunks('detections', older_than => now() - INTERVAL '8 days')").fetchall()
            for (chunk,) in chunks:
                cls.conn.execute("SELECT compress_chunk(%s)", (chunk,))
            cls.compressed = len(chunks)

        # 2. Migrate: the new schema, twice (init must be safe to re-run).
        run_init()
        run_init()

        # 3. Demo shift data, made by the real fusion engine.
        cls.demo = tb.demo_events(8, now=NOW)
        with psycopg.connect(URL) as c:
            cls.demo_written, cls.demo_rejected = tb.insert_events(c, cls.demo)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        with psycopg.connect(ADMIN_URL, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {DB_NAME} WITH (FORCE)")

    def one(self, sql, *params):
        return self.conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------ migration (phase 4)

    def test_t1_migration_keeps_legacy_rows(self):
        if not self.has_old_schema:
            self.skipTest("pre-fusion schema not in git history")
        self.assertGreater(self.compressed, 0, "expected some compressed chunks to migrate")
        n, = self.one("SELECT count(*) FROM detections WHERE event_id IS NULL")
        self.assertEqual(n, self.legacy_rows)
        n, = self.one("SELECT count(*) FROM hazard_events WHERE device_id = 'hat-01' AND time < now() - INTERVAL '1 hour'")
        self.assertEqual(n, 0, "pre-fusion rows must not appear as events")
        idx = {r[0] for r in self.conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'detections'").fetchall()}
        self.assertIn("detections_event", idx)
        self.assertIn("detections_dedup_legacy", idx)
        self.assertNotIn("detections_dedup", idx)
        reg = dict(self.conn.execute("SELECT hazard, severity FROM hazard_types").fetchall())
        self.assertEqual(reg["VEHICLE_HAZARD"], "high")
        self.assertEqual(reg["WORKER_DISTRESS"], "high")

    # ------------------------------------------------------------ ingestion (phase 4)

    def test_t2_demo_events_one_row_each(self):
        events = tb.latest_per_event(self.demo)
        self.assertEqual(self.demo_rejected, [])
        n, = self.one("SELECT count(*) FROM detections WHERE device_id = 'demo-01'")
        self.assertEqual(n, len(events))
        n, = self.one("SELECT count(*) FROM hazard_events WHERE device_id = 'demo-01' AND status <> 'closed'")
        self.assertEqual(n, 0)

    def test_t3_resending_is_idempotent_and_never_downgrades(self):
        before = self.conn.execute("SELECT event_id, status, fusion, alerted FROM detections "
                                   "WHERE device_id = 'demo-01' ORDER BY event_id").fetchall()
        with psycopg.connect(URL) as c:
            tb.insert_events(c, self.demo)                                       # everything again
            for r in [r for r in self.demo if r["status"] == "open"][:50]:      # stale records, one by one
                tb.insert_events(c, [r])
        after = self.conn.execute("SELECT event_id, status, fusion, alerted FROM detections "
                                  "WHERE device_id = 'demo-01' ORDER BY event_id").fetchall()
        self.assertEqual(before, after)

    def test_t4_upgrade_updates_the_same_row(self):
        recs = run(repeat(CAR, 0.0, 3.0, 0.2) + [det(HORN, 2.5)])
        for r in recs:
            r["device_id"] = "it-upgrade"
        with psycopg.connect(URL) as c:
            for r in recs:
                tb.insert_events(c, [r])
        rows = self.conn.execute("SELECT fusion, hazard, status, audio_score, video_score, explanation "
                                 "FROM detections WHERE device_id = 'it-upgrade'").fetchall()
        self.assertEqual(len(rows), 1)
        fusion, hazard, status, a, v, why = rows[0]
        self.assertEqual((fusion, hazard, status), ("MULTIMODAL", "VEHICLE_HAZARD", "closed"))
        self.assertIsNotNone(a); self.assertIsNotNone(v); self.assertTrue(why)

    def test_t5_upsert_into_a_compressed_chunk(self):
        if not self.has_old_schema:
            self.skipTest("no compressed chunks without the old schema")
        recs = run([det(HORN, 0.0)])
        old = (NOW - timedelta(days=12, seconds=-30)).isoformat()      # inside a compressed chunk
        for r in recs:
            r["time"], r["last_time"], r["device_id"] = old, old, "it-compressed"
        with psycopg.connect(URL) as c:
            n1, bad1 = tb.insert_events(c, [recs[0]])                  # insert (open)
            n2, bad2 = tb.insert_events(c, recs)                       # upsert (closed)
            n3, bad3 = tb.insert_events(c, [recs[0]])                  # stale open: ignored
        self.assertEqual(bad1 + bad2 + bad3, [])
        rows = self.conn.execute("SELECT status FROM detections WHERE device_id = 'it-compressed'").fetchall()
        self.assertEqual(rows, [("closed",)])

    def test_t6_offline_then_reconnect_sends_once(self):          # CLAUDE.md test 6, real driver
        import tempfile
        tmp = tempfile.mkdtemp()
        spool, rejects = os.path.join(tmp, "spool.jsonl"), os.path.join(tmp, "rej.jsonl")
        dead = urlunsplit(urlsplit(URL)._replace(netloc=urlsplit(URL).netloc.rsplit(":", 1)[0] + ":1"))
        logs = []
        w = tb.TigerWriter(dead, log=logs.append, spool=spool, rejects=rejects, retry_seconds=0.5)
        recs = run([det(HORN, 0.0), det(CAR, 1.0)]) + run([det(HORN, 0.0)])
        for r in recs:
            r["device_id"] = "it-offline"
        t0 = time.perf_counter()
        for r in recs:
            w.add(r)
        self.assertLess(time.perf_counter() - t0, 0.05)
        deadline = time.time() + 10
        while not os.path.exists(spool) and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(os.path.exists(spool), logs)
        self.assertTrue(any("offline (OperationalError" in m for m in logs), logs)
        w.url = URL                                                    # connectivity returns
        deadline = time.time() + 15
        n = 0
        while time.time() < deadline:
            n, = self.one("SELECT count(*) FROM detections WHERE device_id = 'it-offline' AND status = 'closed'")
            if n == 2 and not os.path.exists(spool):
                break
            time.sleep(0.1)
        for r in recs:                                                 # same events again
            w.add(r)
        w.close(timeout=10)
        n, = self.one("SELECT count(*) FROM detections WHERE device_id = 'it-offline'")
        self.assertEqual(n, 2)
        self.assertFalse(os.path.exists(spool))

    def test_t7_ingest_skips_pre_fusion_lines(self):
        import json
        import tempfile
        recs = run([det(HORN, 0.0)])
        for r in recs:
            r["device_id"] = "it-ingest"
        path = os.path.join(tempfile.mkdtemp(), "events.jsonl")
        with open(path, "w") as f:
            f.write('{"time":"2026-11-21T10:00:00+00:00","source":"audio","label":"Siren","score":0.9}\n')
            for r in recs:
                f.write(json.dumps(r) + "\n")
        out = io.StringIO()
        with redirect_stdout(out):
            tb.cmd_ingest(URL, SimpleNamespace(file=path))
            tb.cmd_ingest(URL, SimpleNamespace(file=path))
        self.assertIn("1 new events", out.getvalue())
        self.assertIn("0 new events", out.getvalue())
        self.assertIn("Skipped 1 pre-fusion", out.getvalue())

    def test_t8_check_command(self):
        out = io.StringIO()
        with redirect_stdout(out):
            tb.cmd_check(URL, SimpleNamespace())
        self.assertIn("timescaledb: 2.", out.getvalue())
        self.assertRegex(out.getvalue(), r"demo-01\s+\d+ events \(\d+ multimodal")

    # ------------------------------------------------------------ dashboard SQL (phase 5)

    def test_t9_dashboard_sql_matches_the_reference(self):
        """Every dashboard query on the real database gives the same answer as the fake DB
        (which computes it in Python) for the same demo events."""
        db = dashboard.Db(URL)
        self.addCleanup(lambda: db.conn and db.conn.close())
        fake = FakeEventsDb(rows=to_view_rows(self.demo), now=NOW)
        cases = [{}, {"fusion": "MULTIMODAL"}, {"severity": "high"}, {"source": "audio"}, {"source": "video"},
                 {"hazard": "PERSON", "fusion": "VISION_ONLY"}, {"range": "1h"}, {"range": "24h"},
                 {"range": "custom", "from": (NOW - timedelta(hours=3)).isoformat(),
                  "to": (NOW - timedelta(hours=1)).isoformat()}]
        for extra in cases:
            with self.subTest(**extra):
                params = {"range": "8h", "device": "demo-01", **extra}
                f = dashboard.parse_filters({k: [v] for k, v in params.items()}, now=NOW)
                real, ref = dashboard.gather(db, f), dashboard.gather(fake, f)
                if not extra:
                    self.assertGreater(real["kpis"]["events"], 100)
                    self.assertGreater(real["kpis"]["multimodal"], 0)
                for key in ("kpis", "timeline", "severity", "register"):
                    a, b = real[key], ref[key]
                    if key == "kpis":
                        a, b = dict(a), dict(b)
                    self.assertEqual(a, b, key)
                self.assertEqual([e["event_id"] for e in real["events"]], [e["event_id"] for e in ref["events"]])
                self.assertEqual(real["events"], ref["events"])
                self.assertEqual(sorted(map(tuple, map(dict.values, real["positions"]))),
                                 sorted(map(tuple, map(dict.values, ref["positions"]))))


if __name__ == "__main__":
    unittest.main()
