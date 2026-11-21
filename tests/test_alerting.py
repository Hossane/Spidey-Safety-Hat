"""Phase 3 tests: unified events through hat_detect's EventPipeline (Pi side) and the
laptop companion's phrases and Speaker. No models, GPIO, mic or speaker needed.

Run from the project root:  python3 -m unittest discover -s tests -v
"""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hat_detect  # noqa: E402
import laptop_companion as lc  # noqa: E402


class FakeBuzzer:
    def __init__(self):
        self.beeps = []

    def beep(self, severity):
        self.beeps.append(severity)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def pipeline(tmp):
    cfg = hat_detect.load_config(os.path.join(ROOT, "hat_config.json"))
    cfg["event_log"] = os.path.join(tmp, "events.jsonl")
    cfg["detection_log"] = os.path.join(tmp, "detections.jsonl")
    out, buzzer, clock = io.StringIO(), FakeBuzzer(), Clock()
    return hat_detect.EventPipeline(cfg, buzzer, out, clock=clock), out, buzzer, clock


def run_main_loop(pipe, clock, until):
    while clock.t < until:
        clock.t = round(clock.t + 0.2, 6)
        pipe.tick()


def event_lines(out):
    lines = out.getvalue().splitlines()
    assert all(line.startswith("EVENT ") for line in lines)
    return [json.loads(line[6:]) for line in lines]


class TestPiPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_horn_and_car_give_one_alert(self):
        pipe, out, buzzer, clock = pipeline(self.tmp.name)
        pipe.handle("audio", "HORN", "Vehicle horn, car horn, honking", 0.94)
        for i in range(10):                       # car in view for 2 s at 5 fps
            clock.t = 0.6 + i * 0.2
            pipe.handle("video", "VEHICLE", "car", 0.91, "left", [10, 20, 200, 220])
            pipe.tick()
        run_main_loop(pipe, clock, 10)
        pipe.close()

        recs = event_lines(out)
        self.assertEqual({r["event_id"] for r in recs}.__len__(), 1)
        self.assertEqual([r["status"] for r in recs], ["open", "closed"])
        self.assertEqual(recs[0]["fusion"], "MULTIMODAL")
        self.assertEqual(recs[0]["hazard"], "VEHICLE_HAZARD")
        self.assertEqual(buzzer.beeps, ["high"])               # buzzer once per event

        with open(os.path.join(self.tmp.name, "events.jsonl")) as f:
            self.assertEqual([json.loads(line) for line in f], recs)
        with open(os.path.join(self.tmp.name, "detections.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 11)           # every raw detection kept

    def test_close_flushes_open_events(self):
        pipe, out, buzzer, clock = pipeline(self.tmp.name)
        pipe.handle("audio", "ALARM", "Fire alarm", 0.8)
        pipe.close()
        self.assertEqual([r["status"] for r in event_lines(out)], ["open", "closed"])
        self.assertEqual(buzzer.beeps, ["high"])

    def test_broken_output_keeps_logging(self):
        pipe, out, buzzer, clock = pipeline(self.tmp.name)
        out.close()                                            # laptop went away
        pipe.handle("audio", "SIREN", "Siren", 0.9)
        self.assertTrue(pipe.out_broken)
        self.assertEqual(buzzer.beeps, ["high"])               # local alert still fires
        pipe.close()
        with open(os.path.join(self.tmp.name, "events.jsonl")) as f:
            self.assertEqual(len(f.readlines()), 2)


class TestPhrases(unittest.TestCase):
    def ev(self, fusion, hazard, audio=None, video=None):
        return {"fusion": fusion, "hazard": hazard, "audio": audio, "video": video}

    def test_multimodal(self):
        ev = self.ev("MULTIMODAL", "VEHICLE_HAZARD", {"label": "Siren"}, {"label": "truck", "position": "left"})
        self.assertEqual(lc.phrase(ev), "Warning! Vehicle hazard on your left")

    def test_unknown_multimodal_rule(self):
        self.assertEqual(lc.phrase(self.ev("MULTIMODAL", "FALLING_LOAD", {}, {"position": "centre"})),
                         "Warning! Falling load ahead")

    def test_audio_only(self):
        self.assertEqual(lc.phrase(self.ev("AUDIO_ONLY", "VOICE", {"label": "Shout"})), "Someone shouting")

    def test_vision_only(self):
        self.assertEqual(lc.phrase(self.ev("VISION_ONLY", "PERSON", None, {"label": "person", "position": "right"})),
                         "Person on your right")
        self.assertEqual(lc.phrase(self.ev("VISION_ONLY", "VEHICLE", None, {"label": "truck", "position": "centre"})),
                         "Truck ahead")


class FakeVoice:
    """Speaks until cancelled or released by the test."""

    def __init__(self):
        self.started, self.cancels = [], 0
        self.speaking = threading.Event()
        self.done = threading.Event()

    def start(self, text):
        self.started.append(text)
        self.done.clear()
        self.speaking.set()

    def wait(self):
        self.done.wait(5)
        self.speaking.clear()

    def cancel(self):
        self.cancels += 1
        self.done.set()

    def close(self):
        pass


class FakeTones:
    def __init__(self):
        self.played, self.stops = [], 0

    def play(self, severity):
        self.played.append(severity)

    def stop(self):
        self.stops += 1


def wait_for(cond, timeout=3):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.01)
    return cond()


class TestSpeaker(unittest.TestCase):
    def test_high_interrupts_low(self):
        voice, tones = FakeVoice(), FakeTones()
        sp = lc.Speaker(voice, tones)
        sp.say("low", "Person ahead")
        self.assertTrue(voice.speaking.wait(2))
        sp.say("high", "Warning! Vehicle hazard on your left")
        self.assertTrue(wait_for(lambda: len(voice.started) == 2))
        voice.done.set()
        self.assertTrue(wait_for(lambda: not sp.busy()))
        self.assertEqual(voice.cancels, 1)
        self.assertEqual(sp.played, [("low", "Person ahead", True),
                                     ("high", "Warning! Vehicle hazard on your left", False)])

    def test_equal_severity_does_not_interrupt(self):
        voice, tones = FakeVoice(), FakeTones()
        sp = lc.Speaker(voice, tones)
        sp.say("medium", "Horn")
        self.assertTrue(voice.speaking.wait(2))
        sp.say("medium", "Truck ahead")
        time.sleep(0.1)
        self.assertEqual(voice.cancels, 0)
        voice.done.set()
        self.assertTrue(wait_for(lambda: len(voice.started) == 2))
        voice.done.set()
        self.assertTrue(wait_for(lambda: not sp.busy()))
        self.assertEqual([p[1] for p in sp.played], ["Horn", "Truck ahead"])

    def test_queue_plays_most_severe_first(self):
        voice, tones = FakeVoice(), FakeTones()
        sp = lc.Speaker(voice, tones)
        sp.say("high", "Alarm")                     # occupies the speaker
        self.assertTrue(voice.speaking.wait(2))
        sp.say("low", "Person ahead")
        sp.say("medium", "Horn")
        voice.done.set()
        self.assertTrue(wait_for(lambda: len(voice.started) == 2))
        voice.done.set()
        self.assertTrue(wait_for(lambda: len(voice.started) == 3))
        voice.done.set()
        self.assertEqual(voice.started, ["Alarm", "Horn", "Person ahead"])

    def test_stale_alerts_are_dropped(self):
        clock = Clock()
        voice, tones = FakeVoice(), FakeTones()
        sp = lc.Speaker(voice, tones, clock=clock)
        sp.say("high", "Alarm")
        self.assertTrue(voice.speaking.wait(2))
        sp.say("medium", "Horn")
        clock.t = 10.0                              # Horn has now waited 10 s
        voice.done.set()
        self.assertTrue(wait_for(lambda: not sp.busy()))
        self.assertEqual(voice.started, ["Alarm"])


if __name__ == "__main__":
    unittest.main()
