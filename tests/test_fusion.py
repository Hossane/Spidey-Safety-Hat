"""Tests for fusion.py with simulated detections (no models, camera or mic needed).

Run from the project root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fusion import AUDIO_ONLY, MULTIMODAL, VISION_ONLY, Detection, FusionEngine  # noqa: E402

with open(os.path.join(ROOT, "hat_config.json")) as f:
    CONFIG = json.load(f)
BASE = datetime(2026, 11, 21, 19, 30, tzinfo=timezone.utc)

HORN = ("audio", "HORN", "Vehicle horn, car horn, honking")
SIREN = ("audio", "SIREN", "Siren")
SHOUT = ("audio", "VOICE", "Shout")
ALARM = ("audio", "ALARM", "Fire alarm")
CAR = ("video", "VEHICLE", "car")
PERSON = ("video", "PERSON", "person")


def det(kind, t, score=0.9, position=None):
    source, group, label = kind
    if source == "video" and position is None:
        position = "left"
    return Detection(source, group, label, score, t, (BASE + timedelta(seconds=t)).isoformat(),
                     position, [10, 20, 200, 220] if source == "video" else None)


def engine(**overrides):
    fusion = dict(CONFIG["fusion"])
    fusion.update(overrides)
    return FusionEngine(fusion, CONFIG["alerts"]["severity"], "hat-test")


def run(detections, eng=None, tail=10.0, step=0.2):
    """Feed detections in time order with tick() every `step` s, like hat_detect's main
    loop, then keep ticking `tail` s so every episode closes. Returns all records."""
    eng = eng or engine()
    detections = sorted(detections, key=lambda d: d.t)
    out, t, i = [], 0.0, 0
    end = (detections[-1].t if detections else 0) + tail
    while t <= end:
        while i < len(detections) and detections[i].t <= t:
            out += eng.add(detections[i])
            i += 1
        out += eng.tick(t)
        t = round(t + step, 6)
    return out


def final(records):
    """Latest record per event_id (what the database ends up holding)."""
    last = {}
    for r in records:
        last[r["event_id"]] = r
    return list(last.values())


def notifications(records):
    return [r for r in records if r["notify"]]


def repeat(kind, start, end, every, score=0.9):
    n = int(round((end - start) / every)) + 1
    return [det(kind, round(start + i * every, 6), score) for i in range(n)]


class Test1AudioOnly(unittest.TestCase):
    def test_horn_without_vehicle(self):
        recs = run([det(HORN, 0.0, 0.94), det(HORN, 0.5, 0.80), det(HORN, 1.0, 0.85)])
        events = final(recs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["fusion"], AUDIO_ONLY)
        self.assertEqual(ev["hazard"], "HORN")
        self.assertEqual(ev["severity"], "medium")
        self.assertEqual(ev["audio"]["score"], 0.94)        # peak, not an average
        self.assertEqual(ev["audio"]["detections"], 3)
        self.assertIsNone(ev["video"])
        self.assertIsNone(ev["correlation"])
        self.assertEqual(ev["status"], "closed")
        self.assertTrue(ev["alerted"])
        self.assertEqual(len(notifications(recs)), 1)

    def test_high_severity_sound_is_not_delayed(self):
        eng = engine()
        out = eng.add(det(SIREN, 0.0))       # nothing can make a siren more severe
        self.assertEqual([r["status"] for r in out], ["open"])
        self.assertTrue(out[0]["notify"])

    def test_medium_sound_waits_for_confirmation(self):
        eng = engine()
        self.assertEqual(eng.add(det(HORN, 0.0)), [])      # held for confirm_wait_seconds
        self.assertEqual(eng.tick(0.8), [])
        out = eng.tick(1.2)
        self.assertEqual([(r["status"], r["fusion"]) for r in out], [("open", AUDIO_ONLY)])


class Test2VisionOnly(unittest.TestCase):
    def test_vehicle_without_sound(self):
        recs = run(repeat(CAR, 0.0, 2.0, 0.2, 0.91))
        events = final(recs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["fusion"], VISION_ONLY)
        self.assertEqual(ev["hazard"], "VEHICLE")          # a sighting, not VEHICLE_HAZARD
        self.assertEqual(ev["severity"], "medium")         # from config, not escalated
        self.assertIsNone(ev["correlation"])
        self.assertEqual(ev["video"]["detections"], 11)
        self.assertEqual(ev["video"]["position"], "left")
        self.assertEqual(len(notifications(recs)), 1)      # once per sighting, not every 3 s


class Test3Multimodal(unittest.TestCase):
    def test_horn_then_car(self):
        recs = run([det(HORN, 0.0, 0.94), det(CAR, 1.0, 0.91)])
        events = final(recs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["fusion"], MULTIMODAL)
        self.assertEqual(ev["hazard"], "VEHICLE_HAZARD")
        self.assertEqual(ev["severity"], "high")
        self.assertEqual(ev["audio"]["label"], "Vehicle horn, car horn, honking")
        self.assertEqual(ev["audio"]["score"], 0.94)
        self.assertEqual(ev["video"]["label"], "car")
        self.assertEqual(ev["video"]["score"], 0.91)
        self.assertEqual(ev["correlation"]["window_seconds"], 3.0)
        self.assertIn("heard 1.0 s before car (VEHICLE) was seen on the left",
                      ev["correlation"]["explanation"])
        self.assertEqual(ev["time"], (BASE).isoformat())   # starts at the horn
        n = notifications(recs)
        self.assertEqual(len(n), 1)
        self.assertEqual(n[0]["fusion"], MULTIMODAL)

    def test_car_seen_just_before_horn(self):
        recs = run(repeat(CAR, 0.0, 0.6, 0.2) + [det(HORN, 0.5)])
        events = final(recs)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["fusion"], MULTIMODAL)
        self.assertEqual(len(notifications(recs)), 1)

    def test_late_confirmation_upgrades_same_event(self):
        # Car announced on its own at ~1 s; horn at 2.5 s escalates that same event.
        recs = run(repeat(CAR, 0.0, 3.0, 0.2) + [det(HORN, 2.5)])
        events = final(recs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["fusion"], MULTIMODAL)
        self.assertEqual([r["status"] for r in recs if r["event_id"] == ev["event_id"]],
                         ["open", "update", "closed"])
        self.assertEqual([r["severity"] for r in notifications(recs)], ["medium", "high"])

    def test_late_confirmation_without_realert(self):
        recs = run(repeat(CAR, 0.0, 3.0, 0.2) + [det(HORN, 2.5)],
                   eng=engine(realert_on_escalation=False))
        self.assertEqual(len(final(recs)), 1)
        self.assertEqual(len(notifications(recs)), 1)

    def test_shout_with_person_is_worker_distress(self):
        recs = run([det(SHOUT, 0.0, 0.7), det(PERSON, 0.4, 0.8)])
        events = final(recs)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["hazard"], "WORKER_DISTRESS")


class Test4CorrelationTimeout(unittest.TestCase):
    def test_car_outside_window_stays_separate(self):
        recs = run([det(HORN, 0.0)] + repeat(CAR, 5.0, 6.0, 0.2))
        events = sorted(final(recs), key=lambda e: e["time"])
        self.assertEqual([e["fusion"] for e in events], [AUDIO_ONLY, VISION_ONLY])
        self.assertEqual([e["hazard"] for e in events], ["HORN", "VEHICLE"])
        self.assertNotIn(MULTIMODAL, {r["fusion"] for r in recs})
        self.assertEqual(len(notifications(recs)), 2)

    def test_unrelated_pair_is_not_combined(self):
        # Shouting while a car is in view: close in time, but no rule pairs VOICE + VEHICLE.
        recs = run([det(SHOUT, 0.0)] + repeat(CAR, 0.0, 1.0, 0.2))
        events = final(recs)
        self.assertEqual(sorted(e["fusion"] for e in events), [AUDIO_ONLY, VISION_ONLY])

    def test_window_is_configurable(self):
        rules = [dict(r, window_seconds=6.0) for r in CONFIG["fusion"]["rules"]]
        recs = run([det(HORN, 0.0)] + repeat(CAR, 5.0, 6.0, 0.2), eng=engine(rules=rules))
        self.assertEqual([e["fusion"] for e in final(recs)], [MULTIMODAL])


class Test5DuplicateSuppression(unittest.TestCase):
    def test_ongoing_horn_and_car(self):
        recs = run(repeat(HORN, 0.0, 10.0, 0.5) + repeat(CAR, 0.2, 10.0, 0.2))
        events = final(recs)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["fusion"], MULTIMODAL)
        self.assertEqual(events[0]["audio"]["detections"], 21)
        self.assertEqual(events[0]["video"]["detections"], 50)
        self.assertEqual(len(notifications(recs)), 1)

    def test_same_hazard_again_after_gap_is_new(self):
        recs = run([det(ALARM, 0.0), det(ALARM, 10.0)])
        self.assertEqual(len(final(recs)), 2)
        self.assertEqual(len(notifications(recs)), 2)

    def test_new_hazard_not_suppressed_by_another(self):
        recs = run(repeat(PERSON, 0.0, 6.0, 0.2) + [det(ALARM, 2.0)])
        hazards = sorted(e["hazard"] for e in final(recs))
        self.assertEqual(hazards, ["ALARM", "PERSON"])
        self.assertEqual(len(notifications(recs)), 2)

    def test_siren_joins_ongoing_vehicle_hazard(self):
        recs = run([det(HORN, 0.0)] + repeat(CAR, 0.2, 3.0, 0.2) + [det(SIREN, 2.0)])
        self.assertEqual(len(final(recs)), 1)
        self.assertEqual(len(notifications(recs)), 1)


class TestPolicyAndFormat(unittest.TestCase):
    def test_rule_without_severity_escalates_one_level(self):
        rules = [{"hazard": "VEHICLE_HAZARD", "audio": ["HORN"], "video": ["VEHICLE"]}]
        recs = run([det(HORN, 0.0), det(CAR, 0.5)], eng=engine(rules=rules))
        self.assertEqual(final(recs)[0]["severity"], "high")      # medium + confirmation

    def test_rule_without_severity_no_escalation(self):
        rules = [{"hazard": "VEHICLE_HAZARD", "audio": ["HORN"], "video": ["VEHICLE"]}]
        eng = engine(rules=rules, escalate_on_confirmation=False)
        recs = run([det(HORN, 0.0), det(CAR, 0.5)], eng=eng)
        self.assertEqual(final(recs)[0]["severity"], "medium")
        self.assertEqual(final(recs)[0]["fusion"], MULTIMODAL)
        # No hold (confirmation can't raise severity): horn announced at once, car joins
        # the same event silently.
        self.assertEqual([(r["status"], r["notify"]) for r in recs][:2], [("open", True), ("update", False)])
        self.assertEqual(len(notifications(recs)), 1)

    def test_records_are_json_and_ids_unique(self):
        recs = run([det(HORN, 0.0), det(CAR, 1.0), det(ALARM, 20.0)])
        for r in recs:
            json.dumps(r)
            self.assertTrue(r["time"].endswith("+00:00"))
        self.assertEqual(len({r["event_id"] for r in final(recs)}), 2)

    def test_time_is_stable_across_records(self):
        # Car announced alone (~1 s), then a horn heard earlier... and later: the merged
        # event keeps the announced event's time so its database key never changes.
        recs = run(repeat(CAR, 0.0, 3.0, 0.2) + [det(HORN, 2.5)])
        self.assertEqual(len({(r["event_id"], r["time"]) for r in recs}), 1)

    def test_flush_closes_open_events(self):
        eng = engine()
        eng.add(det(HORN, 0.0))
        out = eng.flush()
        self.assertEqual([r["status"] for r in out], ["open", "closed"])
        self.assertEqual(eng.tick(100.0), [])


if __name__ == "__main__":
    unittest.main()
