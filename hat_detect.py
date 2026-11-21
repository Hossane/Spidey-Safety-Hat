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
import vision_detect  # noqa: E402

DEFAULTS = {
    "device_id": "hat-01",
    "audio": {"threshold": 0.3, "hop_seconds": 0.5,
              "alert_groups": ["ALARM", "SIREN", "HORN", "VOICE", "IMPACT", "FIRE"]},
    "video": {"camera_unit": 5, "fps": 5, "threshold": 0.5,
              "alert_groups": ["PERSON", "VEHICLE", "SHARP"]},
    "alerts": {"cooldown_seconds": 3.0, "combine_window_seconds": 2.0,
               "severity": {"COMBINED": "high", "ALARM": "high", "SIREN": "high", "IMPACT": "high",
                            "FIRE": "high", "HORN": "medium", "VOICE": "medium", "VEHICLE": "medium",
                            "SHARP": "medium", "PERSON": "low"}},
    "gpio": {"enabled": True, "pin_bcm": 18, "buzzer": "active", "tone_hz": 2700},
    "event_log": "~/hatdetect/events.jsonl",
}
# Beep patterns per severity: list of (on_seconds, off_seconds)
PATTERNS = {"high": [(0.12, 0.08)] * 3, "medium": [(0.15, 0.1)] * 2, "low": [(0.08, 0.0)]}
STATS = {"frames": 0, "windows": 0}   # for the periodic status line
STATUS_EVERY = 10.0
COMBINE_AUDIO = {"HORN", "SIREN"}
COMBINE_VIDEO = {"VEHICLE"}


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
        if t - self.last_alert.get(group, -1e9) >= self.cool:
            self.last_alert[group] = t
            ev["alerted"] = True
            self.buzzer.beep(self.sev.get(group, "low"))

    def handle(self, source, group, label, score, position=None, bbox=None):
        t = time.monotonic()
        ev = {"time": now_iso(), "device_id": self.cfg["device_id"], "source": source,
              "label": label, "score": round(float(score), 3), "position": position,
              "bbox": bbox, "alerted": False, "group": group}
        with self.lock:
            self._maybe_alert(group, ev, t)
            self.recent[group] = (t, ev)
            self._emit(ev)
            self._combine(source, group, t)

    def _combine(self, source, group, t):
        if source == "audio" and group in COMBINE_AUDIO:
            others = COMBINE_VIDEO
        elif source == "video" and group in COMBINE_VIDEO:
            others = COMBINE_AUDIO
        else:
            return
        for g in others:
            seen = self.recent.get(g)
            if seen and t - seen[0] <= self.window:
                a, v = (self.recent[group][1], seen[1]) if source == "audio" else (seen[1], self.recent[group][1])
                ev = {"time": now_iso(), "device_id": self.cfg["device_id"], "source": "combined",
                      "label": f"{a['label']} + {v['label']}", "score": round(min(a["score"], v["score"]), 3),
                      "position": v["position"], "bbox": v["bbox"], "alerted": False, "group": "COMBINED"}
                self._maybe_alert("COMBINED", ev, t)
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
    for _, img in vision_detect.camera_frames(vc["camera_unit"], vc["fps"], stop=stop):
        STATS["frames"] += 1
        best = {}
        for name, score, box, group in det.detect(img, vc["threshold"]):
            if group in groups and score > best.get(group, (None, 0))[1]:
                best[group] = (name, score, box)
        for g, (name, score, box) in best.items():
            logic.handle("video", g, name, score, position_of(box, img.shape[1]), list(box))


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
                print(f"[hat] status: video {f / STATUS_EVERY:.1f} fps, audio {w / STATUS_EVERY:.1f} windows/s",
                      file=sys.stderr)
                next_status, last = time.monotonic() + STATUS_EVERY, dict(STATS)
    finally:
        stop.set()
        for w in workers:          # video worker stops its recorder and deletes the file
            w.join(timeout=8)
        buzzer.close()
        print("[hat] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
