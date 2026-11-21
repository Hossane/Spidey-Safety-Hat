"""Phase 5 tests (CLAUDE.md test 7): the dashboard's filters, metrics and event payload.

Runs scripts/dashboard.py's gather() against tests/fake_events_db.py (demo events from the
real fusion engine), and checks the numbers against the events themselves.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import threading
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dashboard  # noqa: E402
from fake_events_db import FakeEventsDb  # noqa: E402

NOW = datetime(2026, 11, 21, 20, 0, tzinfo=timezone.utc)
DB = FakeEventsDb(now=NOW)


def q(**kw):
    return {k: [v] for k, v in kw.items()}


def get(**kw):
    return dashboard.gather(DB, dashboard.parse_filters(q(**kw), now=NOW))


def in_window(f):
    return [r for r in DB.rows if f["since"] <= r["time"] < f["until"]]


class TestSummaryMetrics(unittest.TestCase):
    def test_kpis_match_the_events(self):
        d = get(range="8h")
        rows = in_window(dashboard.parse_filters(q(range="8h"), now=NOW))
        k = d["kpis"]
        self.assertGreater(k["events"], 100)
        self.assertEqual(k["events"], len(rows))
        self.assertEqual(k["high"], sum(r["severity"] == "high" for r in rows))
        self.assertEqual(k["multimodal"], sum(r["fusion"] == "MULTIMODAL" for r in rows))
        self.assertEqual(k["audio_only"] + k["vision_only"] + k["multimodal"], k["events"])
        self.assertEqual(k["with_audio"], k["audio_only"] + k["multimodal"])
        self.assertEqual(k["with_video"], k["vision_only"] + k["multimodal"])
        self.assertEqual(sum(s["events"] for s in d["severity"]), k["events"])
        self.assertEqual(sum(r["events"] for r in d["register"]), k["events"])

    def test_timeline_covers_every_bucket_and_sums_to_total(self):
        d = get(range="8h")
        self.assertEqual(d["filters"]["bucket_seconds"], 600)
        self.assertEqual(len(d["timeline"]["buckets"]), 48)            # 8 h / 10 min, empty ones included
        total = sum(sum(v) for v in d["timeline"]["series"].values())
        self.assertEqual(total, d["kpis"]["events"])
        self.assertEqual(set(d["timeline"]["series"]), set(dashboard.FUSIONS))

    def test_multimodal_events_are_distinguishable(self):
        d = get(range="8h")
        multi = [e for e in d["events"] if e["fusion"] == "MULTIMODAL"]
        self.assertTrue(multi)
        for e in multi:
            self.assertIsNotNone(e["audio_score"])
            self.assertIsNotNone(e["video_score"])
            self.assertIn("pairs them within", e["explanation"])
            self.assertTrue(e["control"])                               # from hazard_types
        for e in d["events"]:
            if e["fusion"] == "AUDIO_ONLY":
                self.assertIsNone(e["video_score"])
            if e["fusion"] == "VISION_ONLY":
                self.assertIsNone(e["audio_score"])

    def test_payload_is_json_and_events_newest_first(self):
        d = get(range="24h")
        json.dumps(d)
        times = [e["time"] for e in d["events"]]
        self.assertEqual(times, sorted(times, reverse=True))
        self.assertLessEqual(len(d["events"]), dashboard.RECENT_LIMIT)
        self.assertEqual(len({e["event_id"] for e in d["events"]}), len(d["events"]))

    def test_register_columns(self):
        reg = {r["hazard"]: r for r in get(range="8h")["register"]}
        self.assertEqual((reg["PERSON"]["severity"], reg["PERSON"]["source"]), ("low", "video"))
        self.assertEqual((reg["VEHICLE_HAZARD"]["severity"], reg["VEHICLE_HAZARD"]["source"]), ("high", "combined"))
        self.assertTrue(all(r["severity"] in dashboard.SEVERITIES for r in reg.values()))

    def test_pre_fusion_register_entry_hidden(self):
        hazards = {r["hazard"] for r in get(range="8h")["register"]}
        self.assertNotIn("COMBINED", hazards)
        self.assertTrue({"VEHICLE_HAZARD", "WORKER_DISTRESS"} <= hazards)


class TestFiltering(unittest.TestCase):
    def test_each_filter_narrows_to_matching_events(self):
        cases = {"fusion": "MULTIMODAL", "severity": "high", "hazard": "VEHICLE_HAZARD", "device": "demo-01"}
        for key, value in cases.items():
            d = get(range="8h", **{key: value})
            col = "device_id" if key == "device" else key
            self.assertTrue(d["events"], key)
            self.assertTrue(all(e[col] == value for e in d["events"]), key)
            self.assertEqual(d["filters"][key], value)

    def test_filtered_kpis_agree(self):
        all_ = get(range="8h")["kpis"]
        multi = get(range="8h", fusion="MULTIMODAL")["kpis"]
        self.assertEqual(multi["events"], all_["multimodal"])
        self.assertEqual(multi["audio_only"] + multi["vision_only"], 0)
        high = get(range="8h", severity="high")["kpis"]
        self.assertEqual(high["events"], all_["high"])

    def test_source_filter(self):
        heard = get(range="8h", source="audio")
        self.assertTrue(all(e["audio_label"] for e in heard["events"]))
        self.assertEqual(heard["kpis"]["events"], get(range="8h")["kpis"]["with_audio"])
        self.assertEqual(heard["kpis"]["vision_only"], 0)
        seen = get(range="8h", source="video")
        self.assertEqual(seen["kpis"]["audio_only"], 0)

    def test_filters_combine(self):
        d = get(range="8h", fusion="VISION_ONLY", hazard="PERSON")
        self.assertTrue(d["events"])
        self.assertTrue(all(e["fusion"] == "VISION_ONLY" and e["hazard"] == "PERSON" for e in d["events"]))

    def test_custom_time_window(self):
        since, until = NOW - timedelta(hours=3), NOW - timedelta(hours=2)
        d = get(range="custom", **{"from": since.isoformat(), "to": until.isoformat()})
        self.assertEqual(d["filters"]["range"], "custom")
        self.assertEqual(d["filters"]["bucket_seconds"], 60)
        self.assertTrue(all(since.isoformat() <= e["time"] < until.isoformat() for e in d["events"]))
        expected = sum(since <= r["time"] < until for r in DB.rows)
        self.assertEqual(d["kpis"]["events"], expected)

    def test_shorter_range_is_a_subset(self):
        self.assertLessEqual(get(range="1h")["kpis"]["events"], get(range="8h")["kpis"]["events"])


class TestFilterParsing(unittest.TestCase):
    def test_invalid_values_are_ignored(self):
        f = dashboard.parse_filters(q(range="5y", severity="extreme", fusion="BOTH", source="smell",
                                      hazard="X'; DROP TABLE detections; --"), now=NOW)
        self.assertEqual(f["range"], "8h")
        self.assertIsNone(f["severity"]); self.assertIsNone(f["fusion"])
        self.assertIsNone(f["source"]); self.assertIsNone(f["hazard"])

    def test_bad_custom_window_falls_back(self):
        f = dashboard.parse_filters(q(range="custom", **{"from": "yesterday", "to": "now"}), now=NOW)
        self.assertEqual(f["range"], "8h")
        f = dashboard.parse_filters(q(range="custom", **{"from": NOW.isoformat(),
                                                          "to": (NOW - timedelta(hours=1)).isoformat()}), now=NOW)
        self.assertEqual(f["range"], "8h")

    def test_custom_window_clamped(self):
        f = dashboard.parse_filters(q(range="custom", **{"from": "2020-01-01T00:00:00Z",
                                                          "to": (NOW + timedelta(days=5)).isoformat()}), now=NOW)
        self.assertEqual(f["until"], NOW)
        self.assertEqual((f["until"] - f["since"]).days, 90)
        self.assertEqual(f["bucket"], 86400)

    def test_values_are_bound_not_interpolated(self):
        f = dashboard.parse_filters(q(range="8h", device="hat'01", hazard="VEHICLE", severity="low",
                                      fusion="VISION_ONLY", source="video"), now=NOW)
        sql, params = dashboard.where(f)
        self.assertNotIn("hat'01", sql)
        self.assertEqual(params["device"], "hat'01")
        self.assertIn("video_group IS NOT NULL", sql)

    def test_bucket_starts_align_with_time_bucket(self):
        f = dashboard.parse_filters(q(range="24h"), now=NOW)
        starts = dashboard.bucket_starts(f)
        self.assertTrue(all(int(b.timestamp()) % 1800 == 0 for b in starts))
        self.assertLessEqual(starts[0], f["since"])
        self.assertLess(starts[-1], f["until"])


class TestHttp(unittest.TestCase):
    def test_server_serves_page_and_api(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(DB))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/") as r:
            self.assertIn(b"Event timeline", r.read())
        with urllib.request.urlopen(base + "/api/data?range=7d&fusion=MULTIMODAL") as r:
            d = json.load(r)
        self.assertEqual(d["filters"]["fusion"], "MULTIMODAL")
        self.assertTrue(all(e["fusion"] == "MULTIMODAL" for e in d["events"]))


if __name__ == "__main__":
    unittest.main()
