"""Phase 4 tests: unified events into Tiger Cloud through scripts/tiger_bridge.py.

Uses an in-memory fake database with the same upsert rule as INSERT_SQL (key
(event_id, time); an older status never overwrites a newer one), so the bridge's
batching, offline spool, replay and rejection logic run without Postgres or psycopg.
The SQL itself is exercised against a real database by `tiger_bridge.py init/demo/check`.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tiger_bridge as tb  # noqa: E402
from test_fusion import CAR, HORN, det, final, repeat, run  # noqa: E402

COL = {c.strip(): i for i, c in enumerate(tb.COLUMNS.split(","))}


class FakeDataError(Exception):
    """Like psycopg.DataError: the row is bad, the connection is fine."""


class FakeDB:
    def __init__(self):
        self.rows = {}               # (event_id, time) -> row tuple
        self.online = True
        self.bad_ids = set()
        self.connects = 0

    def connect(self, url):
        if not self.online:
            raise ConnectionError("Tiger Cloud unreachable")
        self.connects += 1
        return FakeConn(self)

    def row(self, event_id):
        found = [r for (eid, _), r in self.rows.items() if eid == event_id]
        assert len(found) <= 1
        return dict(zip(COL, found[0])) if found else None


class FakeConn:
    def __init__(self, db):
        self.db, self.closed, self.pending = db, False, {}

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, rows):
        for r in rows:
            self.execute(sql, r)

    def execute(self, sql, row):
        assert sql is tb.INSERT_SQL
        if not self.db.online:
            raise ConnectionError("connection lost")
        if row[COL["event_id"]] in self.db.bad_ids:
            raise FakeDataError("invalid input syntax")
        key = (row[COL["event_id"]], row[COL["time"]])
        cur = self.pending.get(key) or self.db.rows.get(key)
        if cur is None:
            self.pending[key] = row
        elif tb.STATUS_RANK[cur[COL["status"]]] <= tb.STATUS_RANK[row[COL["status"]]]:
            new = list(row)
            new[COL["alerted"]] = cur[COL["alerted"]] or row[COL["alerted"]]
            self.pending[key] = tuple(new)

    def commit(self):
        self.db.rows.update(self.pending)
        self.pending = {}

    def rollback(self):
        self.pending = {}

    def close(self):
        self.closed = True


def wait_for(cond, timeout=5):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.02)
    return cond()


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = FakeDB()
        self.spool = os.path.join(self.tmp.name, "spool.jsonl")
        self.rejects = os.path.join(self.tmp.name, "rejected.jsonl")
        self.logs = []

    def writer(self, retry=0.2):
        w = tb.TigerWriter("fake://", log=self.logs.append, connect_fn=self.db.connect,
                           spool=self.spool, rejects=self.rejects, retry_seconds=retry)
        self.addCleanup(w.close, 2)
        return w

    def send(self, w, records):
        for r in records:
            w.add(r)


class TestStoringEvents(BridgeTest):
    def test_audio_only_event_stored_once(self):              # CLAUDE.md test 1 (storage)
        recs = run([det(HORN, 0.0, 0.94), det(HORN, 0.5, 0.8)])
        w = self.writer()
        self.send(w, recs)
        ev = final(recs)[0]
        self.assertTrue(wait_for(lambda: (self.db.row(ev["event_id"]) or {}).get("status") == "closed"))
        row = self.db.row(ev["event_id"])
        self.assertEqual(len(self.db.rows), 1)
        self.assertEqual((row["fusion"], row["source"], row["hazard"], row["severity"]),
                         ("AUDIO_ONLY", "audio", "HORN", "medium"))
        self.assertEqual((row["audio_score"], row["audio_count"], row["video_score"]), (0.94, 2, None))
        self.assertTrue(row["alerted"])

    def test_multimodal_event_is_one_row_with_both_scores(self):   # CLAUDE.md test 3 (storage)
        recs = run([det(HORN, 0.0, 0.94), det(CAR, 1.0, 0.91)])
        w = self.writer()
        self.send(w, recs)
        ev = final(recs)[0]
        self.assertTrue(wait_for(lambda: (self.db.row(ev["event_id"]) or {}).get("status") == "closed"))
        row = self.db.row(ev["event_id"])
        self.assertEqual(len(self.db.rows), 1)
        self.assertEqual((row["fusion"], row["source"], row["hazard"]), ("MULTIMODAL", "combined", "VEHICLE_HAZARD"))
        self.assertEqual((row["audio_label"], row["audio_score"]), ("Vehicle horn, car horn, honking", 0.94))
        self.assertEqual((row["video_label"], row["video_score"], row["position"]), ("car", 0.91, "left"))
        self.assertEqual(row["score"], 0.94)
        self.assertEqual(row["correlation_window"], 3.0)
        self.assertIn("heard 1.0 s before car", row["explanation"])
        self.assertEqual(json.loads(row["bbox"]), [10, 20, 200, 220])

    def test_upgraded_event_updates_the_same_row(self):
        recs = run(repeat(CAR, 0.0, 3.0, 0.2) + [det(HORN, 2.5)])
        self.assertEqual([r["status"] for r in recs], ["open", "update", "closed"])
        conn = self.db.connect("x")
        for r in recs:                                   # one record per batch, like live
            tb.insert_events(conn, [r])
        self.assertEqual(len(self.db.rows), 1)
        self.assertEqual(self.db.row(recs[0]["event_id"])["fusion"], "MULTIMODAL")


class TestIdempotency(BridgeTest):
    def test_resending_never_duplicates_or_downgrades(self):
        recs = run([det(HORN, 0.0), det(CAR, 1.0)]) + run([det(HORN, 0.0)])
        conn = self.db.connect("x")
        tb.insert_events(conn, recs)
        snapshot = dict(self.db.rows)
        tb.insert_events(conn, recs)                     # whole log again (ingest-pi twice)
        tb.insert_events(conn, [r for r in recs if r["status"] == "open"])   # stale records
        self.assertEqual(self.db.rows, snapshot)
        self.assertEqual({row[COL["status"]] for row in self.db.rows.values()}, {"closed"})

    def test_batch_collapses_to_latest_record_per_event(self):
        recs = run([det(HORN, 0.0)])
        self.assertEqual([r["status"] for r in tb.latest_per_event(recs)], ["closed"])


class TestDatabaseDisconnection(BridgeTest):            # CLAUDE.md test 6
    def test_offline_events_are_spooled_and_sent_once_back(self):
        self.db.online = False
        recs = run([det(HORN, 0.0), det(CAR, 1.0)]) + run([det(HORN, 0.0)])
        w = self.writer(retry=0.2)
        t0 = time.perf_counter()
        self.send(w, recs)
        self.assertLess(time.perf_counter() - t0, 0.05)  # add() never blocks the alert path
        self.assertTrue(wait_for(lambda: os.path.exists(self.spool)))
        self.assertEqual(self.db.rows, {})
        self.assertTrue(any("offline" in m for m in self.logs))

        self.db.online = True                             # connectivity returns
        self.assertTrue(wait_for(lambda: len(self.db.rows) == 2 and not os.path.exists(self.spool)))
        self.assertTrue(any("buffered events" in m for m in self.logs))

        self.send(w, recs)                                # the same events arrive again
        time.sleep(1.5)
        self.assertEqual(len(self.db.rows), 2)            # no duplicate records
        self.assertEqual(sorted(dict(zip(COL, r))["status"] for r in self.db.rows.values()),
                         ["closed", "closed"])

    def test_spool_left_from_last_run_is_replayed_when_idle(self):
        recs = run([det(HORN, 0.0)])
        with open(self.spool, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        self.writer()                                     # no new events arrive
        self.assertTrue(wait_for(lambda: len(self.db.rows) == 1 and not os.path.exists(self.spool)))

    def test_close_while_offline_keeps_events(self):
        self.db.online = False
        w = tb.TigerWriter("fake://", log=self.logs.append, connect_fn=self.db.connect,
                           spool=self.spool, rejects=self.rejects, retry_seconds=60)
        recs = run([det(HORN, 0.0)])
        self.send(w, recs)
        w.close(timeout=3)
        with open(self.spool) as f:
            self.assertEqual(len(f.readlines()), len(recs))


class TestRejection(BridgeTest):
    def test_malformed_record_does_not_block_the_batch(self):
        good = run([det(HORN, 0.0)])
        bad = dict(good[0], event_id="11111111-1111-1111-1111-111111111111", fusion="SOMETHING")
        w = self.writer()
        self.send(w, [bad] + good)
        self.assertTrue(wait_for(lambda: len(self.db.rows) == 1 and os.path.exists(self.rejects)))
        with open(self.rejects) as f:
            rej = [json.loads(line) for line in f]
        self.assertEqual(rej[0]["event"]["fusion"], "SOMETHING")
        self.assertFalse(os.path.exists(self.spool))      # not mistaken for being offline

    def test_row_the_database_rejects_is_isolated(self):
        recs = final(run([det(HORN, 0.0)]) + run([det(CAR, 0.0)]))
        self.db.bad_ids.add(recs[0]["event_id"])
        n, bad = tb.insert_events(self.db.connect("x"), recs)
        self.assertEqual((n, len(bad)), (1, 1))
        self.assertIn("FakeDataError", bad[0][1])
        self.assertIsNotNone(self.db.row(recs[1]["event_id"]))

    def test_pre_fusion_lines_are_skipped(self):
        new = run([det(HORN, 0.0)])
        lines = ['EVENT {"time":"2026-11-21T10:00:00+00:00","source":"audio","label":"Siren","score":0.9}',
                 "not json"] + ["EVENT " + json.dumps(r) for r in new]
        events, legacy = tb.read_jsonl(lines)
        self.assertEqual((len(events), legacy), (len(new), 1))


class TestDemo(unittest.TestCase):
    def test_demo_events_come_from_the_fusion_engine(self):
        recs = tb.demo_events(1, seed=1)
        events = tb.latest_per_event(recs)
        self.assertGreater(len(events), 20)
        self.assertEqual({e["status"] for e in events}, {"closed"})
        for e in events:
            tb.to_row(e)                                  # every one is insertable
        self.assertEqual(len({e["event_id"] for e in events}), len(events))
        self.assertTrue({e["fusion"] for e in events} <= set(tb.FUSION_SOURCE))


if __name__ == "__main__":
    unittest.main()
