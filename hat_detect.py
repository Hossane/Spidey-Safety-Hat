"""Hat Detect: the full on-device app. Runs on QNX (Raspberry Pi 5).

    microphone audio ──► YAMNet ─────────┐
                                         ├──► alert logic ──► GPIO buzz + EVENT lines
    QNX Camera unit ──► EfficientDet ────┘                     (laptop speaks them on the JBL)

- Audio and video run in separate threads; all inference is on the Pi (tflite_runtime).
- Audio comes in as raw s16le 16 kHz mono PCM on stdin (the laptop streams its mic,
  because QNX's USB audio driver rejects the webcam mic). Use --no-audio to skip it.
- Every detection becomes an event in the Tiger Data `detections` schema
  (time, device_id, source, label, score, position, bbox, alerted). Events are printed as
  `EVENT {json}` lines on stdout and appended to a JSONL log.
- An alert fires when an event's group is out of its cooldown (default 3 s). A horn or
  siren heard within 2 s of a vehicle seen becomes one COMBINED alert.
- People and vehicles are tracked (tracking.py): each one alerts once when first seen,
  not on every frame. Someone who steps out of view and comes back is recognised by
  their clothing colours and stays quiet. They alert again only if something happens:
  PERSON_DOWN (fell or collapsed), PERSON_CLOSE (right behind you, or closing in fast),
  or a vehicle closing in. Settings: video.tracking in hat_config.json.
- Tracks also carry motion (approaching / moving away / crossing / still) and an estimated
  time to reach the wearer. A person lying still is PERSON_DOWN even if the fall wasn't seen.
  motion.py flags sudden unlabelled movement (MOTION). Sound + sight together become
  stronger alerts (COMBINED, INJURY, FALLING_OBJECT). See docs/SAFETY_FEATURES.md.
- Alerts pulse a buzzer/LED/vibration motor on a GPIO pin (default BCM 18).

Run on the Pi:     python3 hat_detect.py --no-audio                (camera + GPIO only)
From the laptop:   python scripts/laptop_companion.py              (mic + speaker + this)
Settings:          hat_config.json (thresholds, classes, cooldowns, GPIO pin)
"""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_detect  # noqa: E402
import motion  # noqa: E402
import tracking  # noqa: E402
import vision_detect  # noqa: E402

SEVERITY = {
    "COMBINED": "high", "INJURY": "high", "FALLING_OBJECT": "high", "PERSON_DOWN": "high",
    "ALARM": "high", "SIREN": "high", "IMPACT": "high", "FIRE": "high", "GUNSHOT": "high", "SKID": "high",
    "HORN": "medium", "VOICE": "medium", "GLASS": "medium", "VEHICLE": "medium", "SHARP": "medium",
    "PERSON_CLOSE": "medium", "MOTION": "medium",
    "PERSON": "low", "MACHINERY": "low",
}
DEFAULTS = {
    "device_id": "hat-01",
    "audio": {"threshold": 0.3, "hop_seconds": 0.5,
              "alert_groups": ["ALARM", "SIREN", "HORN", "VOICE", "IMPACT", "FIRE", "GLASS", "GUNSHOT",
                               "SKID", "MACHINERY"]},
    "video": {"camera_unit": 5, "fps": 5, "threshold": 0.5,
              "alert_groups": ["PERSON", "VEHICLE", "SHARP", "MOTION"],
              "tracking": {"enabled": True, "groups": ["PERSON", "VEHICLE"]},
              "motion": {"enabled": True}},
    "alerts": {"cooldown_seconds": 3.0, "combine_window_seconds": 2.0,
               "cooldown_overrides": {"MACHINERY": 60.0},
               "severity": SEVERITY},
    "gpio": {"enabled": True, "pin_bcm": 18, "buzzer": "active", "tone_hz": 2700},
    "event_log": "~/hatdetect/events.jsonl",
}
# Beep patterns per severity: list of (on_seconds, off_seconds)
PATTERNS = {"high": [(0.12, 0.08)] * 3, "medium": [(0.15, 0.1)] * 2, "low": [(0.08, 0.0)]}
STATS = {"frames": 0, "windows": 0}   # for the periodic status line
STATUS_EVERY = 10.0
# Audio + video confirmation: (sounds, sights, resulting alert, window seconds, label filter
# for the sound). A matching pair within the window becomes one stronger alert.
COMBOS = [
    ({"HORN", "SIREN", "SKID"}, {"VEHICLE"}, "COMBINED", None, None),
    ({"ALARM"}, {"VEHICLE"}, "COMBINED", None, ("Reversing beeps", "Beep, bleep")),
    ({"IMPACT", "VOICE"}, {"PERSON_DOWN"}, "INJURY", 8.0, None),
    ({"IMPACT", "GLASS"}, {"MOTION"}, "FALLING_OBJECT", 3.0, None),
]


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Buzzer:
    """Pulses a GPIO pin in a background thread. 'active' buzzers, LEDs and vibration
    motors are switched on/off; 'passive' buzzers get a PWM tone. No-op if GPIO is off
    or unavailable (e.g. on the laptop)."""

    def __init__(self, cfg):
        self.ok, self.pwm = False, None
        self.pin, self.mode, self.hz = cfg["pin_bcm"], cfg["buzzer"], cfg["tone_hz"]
        self.q = queue.Queue(maxsize=1)
        if not cfg["enabled"]:
            return
        try:
            import rpi_gpio as gpio
            self.gpio = gpio
            gpio.setmode(gpio.BCM)
            gpio.setup(self.pin, gpio.OUT)
            gpio.output(self.pin, gpio.LOW)
            if self.mode == "passive":
                self.pwm = gpio.PWM(self.pin, self.hz)
            self.ok = True
        except Exception as e:  # noqa: BLE001 - any GPIO failure just disables the buzzer
            print(f"[gpio] disabled: {e}", file=sys.stderr)
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _on(self, on):
        if self.pwm is not None:
            self.pwm.start(50) if on else self.pwm.stop()
        else:
            self.gpio.output(self.pin, self.gpio.HIGH if on else self.gpio.LOW)

    def _run(self):
        while True:
            pattern = self.q.get()
            for on_s, off_s in pattern:
                self._on(True)
                time.sleep(on_s)
                self._on(False)
                time.sleep(off_s)

    def beep(self, severity):
        if self.ok:
            try:  # if a pattern is already playing, the newest alert replaces the queued one
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(PATTERNS.get(severity, PATTERNS["low"]))

    def close(self):
        if self.ok:
            try:
                self._on(False)
                self.gpio.cleanup()
            except Exception:  # noqa: BLE001
                pass


class AlertLogic:
    def __init__(self, cfg, buzzer, out):
        self.cfg, self.buzzer, self.out = cfg, buzzer, out
        self.cool = cfg["alerts"]["cooldown_seconds"]
        self.cool_for = cfg["alerts"].get("cooldown_overrides", {})
        self.window = cfg["alerts"]["combine_window_seconds"]
        self.sev = cfg["alerts"]["severity"]
        self.last_alert = {}            # group -> monotonic time of last alert
        self.recent = {}                # group -> (monotonic time, event) for combining
        self.lock = threading.Lock()
        log = os.path.expanduser(cfg["event_log"])
        os.makedirs(os.path.dirname(log), exist_ok=True)
        self.log = open(log, "a", buffering=1)
        self.out_broken = False

    def _emit(self, ev):
        line = json.dumps(ev, separators=(",", ":"))
        self.log.write(line + "\n")
        if self.out_broken:
            return
        try:
            self.out.write("EVENT " + line + "\n")
            self.out.flush()
        except (BrokenPipeError, OSError):   # laptop disconnected; main loop stops the app
            self.out_broken = True

    def _maybe_alert(self, group, ev, t):
        if t - self.last_alert.get(group, -1e9) >= self.cool_for.get(group, self.cool):
            self.last_alert[group] = t
            ev["alerted"] = True
            self.buzzer.beep(self.sev.get(group, "low"))

    def handle(self, source, group, label, score, position=None, bbox=None, alert=True, extra=None):
        """alert=False logs the event without alerting (e.g. a person who is already being
        tracked); alerts are still limited to one per group per cooldown."""
        t = time.monotonic()
        ev = {"time": now_iso(), "device_id": self.cfg["device_id"], "source": source,
              "label": label, "score": round(float(score), 3), "position": position,
              "bbox": bbox, "alerted": False, "group": group}
        if extra:
            ev.update(extra)
        with self.lock:
            if alert:
                self._maybe_alert(group, ev, t)
            self.recent[group] = (t, ev)
            self._emit(ev)
            self._combine(source, group, t)

    def _combine(self, source, group, t):
        """Turn a sound and a sight close together in time into one confirmed alert."""
        for sounds, sights, result, window, labels in COMBOS:
            if source == "audio" and group in sounds:
                others = sights
            elif source != "audio" and group in sights:
                others = sounds
            else:
                continue
            for g in others:
                seen = self.recent.get(g)
                if not seen or t - seen[0] > (window or self.window):
                    continue
                a, v = (self.recent[group][1], seen[1]) if source == "audio" else (seen[1], self.recent[group][1])
                if labels and a["label"] not in labels:
                    continue
                ev = {"time": now_iso(), "device_id": self.cfg["device_id"], "source": "combined",
                      "label": f"{a['label']} + {v['label']}", "score": round(min(a["score"], v["score"]), 3),
                      "position": v["position"], "bbox": v["bbox"], "alerted": False, "group": result}
                for k in ("track", "motion", "ttc"):
                    if k in v:
                        ev[k] = v[k]
                self._maybe_alert(result, ev, t)
                self._emit(ev)
                return


def position_of(box, width):
    cx = (box[0] + box[2]) / 2 / width
    return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"


def audio_worker(cfg, logic, stop, model_dir):
    model = audio_detect.Yamnet(model_dir)
    groups = set(cfg["audio"]["alert_groups"])
    hop = int(cfg["audio"]["hop_seconds"] * audio_detect.SAMPLE_RATE)
    for _, window in audio_detect.stream_windows(sys.stdin.buffer, hop):
        if stop.is_set():
            break
        STATS["windows"] += 1
        _, hits, _ = model.analyze(window, cfg["audio"]["threshold"])
        best = {}
        for g, name, score in hits:              # strongest class per group
            if g in groups and score > best.get(g, ("", 0))[1]:
                best[g] = (name, score)
        for g, (name, score) in best.items():
            logic.handle("audio", g, name, score)
    # Audio comes from the laptop over SSH; when it ends, the laptop has disconnected.
    print("[audio] input ended; stopping", file=sys.stderr)
    stop.set()


def video_worker(cfg, logic, stop, model_dir):
    det = vision_detect.Detector(model_dir)
    vc = cfg["video"]
    groups = set(vc["alert_groups"])
    tc = vc.get("tracking", {})
    tracked = set(tc.get("groups", ["PERSON", "VEHICLE"])) if tc.get("enabled", True) else set()
    tracker = tracking.Tracker.from_config(tc)
    mc = vc.get("motion", {})
    watch = motion.MotionWatch.from_config(mc) if mc.get("enabled", True) and "MOTION" in groups else None
    for _, img in vision_detect.camera_frames(vc["camera_unit"], vc["fps"], stop=stop):
        STATS["frames"] += 1
        t = time.monotonic()
        width = img.shape[1]
        all_dets = det.detect(img, vc["threshold"])
        dets = [d for d in all_dets if d[3] in groups]

        # Tracked groups (people, vehicles) alert only when someone new appears or something
        # happens to them (fall, right behind you, closing in fast), not on every frame.
        visible, incidents = tracker.update([d for d in dets if d[3] in tracked], img, t)
        STATS.update(tracker.counts())
        done = set()
        for kind, tr in incidents:
            print(f"[track] {kind:<8} {tr.name} #{tr.id} ({tracking.side_of(tr.box, width)})", file=sys.stderr)
            g = tracking.incident_group(kind, tr)
            if g:
                people = sum(1 for v in visible if v.group == "PERSON")
                logic.handle("video", g, tr.name, tr.score, position_of(tr.box, width), list(tr.box),
                             extra={"track": tr.id, "incident": kind, "motion": tr.motion,
                                    "ttc": tr.ttc, "people": people})
                done.add(tr.group)

        # Sudden movement the detector has no label for (falling load, swinging object, ...).
        if watch:
            m = watch.update(img, t, [d[2] for d in all_dets])
            if m:
                print(f"[motion] {m['label']} ({m['position']}, {m['score']:.2f})", file=sys.stderr)
                logic.handle("video", "MOTION", m["label"], m["score"], m["position"], m["bbox"])

        # Everything else is logged as before (one event per group per frame); tracked groups
        # never alert from here.
        best = {}
        for name, score, box, group in dets:
            if score > best.get(group, (None, 0))[1]:
                best[group] = (name, score, box)
        for g, (name, score, box) in best.items():
            if g not in done:
                logic.handle("video", g, name, score, position_of(box, width), list(box), alert=g not in tracked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "hat_config.json"))
    ap.add_argument("--models", help="model folder (default ~/models, then ./models)")
    ap.add_argument("--no-audio", action="store_true", help="don't read audio from stdin")
    ap.add_argument("--no-video", action="store_true", help="don't use the camera")
    ap.add_argument("--stop-on-stdin-eof", action="store_true",
                    help="with --no-audio: stop when stdin closes (used by the laptop companion)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = audio_detect.find_models(args.models) if not args.no_audio else vision_detect.find_models(args.models)
    stop = threading.Event()
    buzzer = Buzzer(cfg["gpio"])
    logic = AlertLogic(cfg, buzzer, sys.stdout)

    def on_signal(signum, frame):
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    workers = []
    if not args.no_audio:
        workers.append(threading.Thread(target=audio_worker, args=(cfg, logic, stop, model_dir), daemon=True))
    if not args.no_video:
        workers.append(threading.Thread(target=video_worker, args=(cfg, logic, stop, model_dir), daemon=True))
    if not workers:
        ap.error("nothing to do: both --no-audio and --no-video")
    if args.no_audio and args.stop_on_stdin_eof:
        def watch_stdin():
            while sys.stdin.buffer.read(4096):
                pass
            stop.set()
        threading.Thread(target=watch_stdin, daemon=True).start()
    for w in workers:
        w.start()
    print(f"[hat] running: audio={'on' if not args.no_audio else 'off'} "
          f"video={'on' if not args.no_video else 'off'} gpio={'BCM %d' % buzzer.pin if buzzer.ok else 'off'}",
          file=sys.stderr)
    try:
        next_status, last = time.monotonic() + STATUS_EVERY, dict(STATS)
        while not stop.is_set() and not logic.out_broken and any(w.is_alive() for w in workers):
            time.sleep(0.2)
            if time.monotonic() >= next_status:
                f, w = STATS["frames"] - last["frames"], STATS["windows"] - last["windows"]
                print(f"[hat] status: video {f / STATUS_EVERY:.1f} fps, audio {w / STATUS_EVERY:.1f} windows/s, "
                      f"tracking {STATS.get('tracked', 0)} in view ({STATS.get('people', 0)} people), "
                      f"{STATS.get('remembered', 0)} remembered", file=sys.stderr)
                next_status, last = time.monotonic() + STATUS_EVERY, dict(STATS)
    finally:
        stop.set()
        for w in workers:          # video worker stops its recorder and deletes the file
            w.join(timeout=8)
        buzzer.close()
        print("[hat] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
