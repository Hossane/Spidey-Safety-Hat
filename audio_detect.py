"""YAMNet sound classifier for Hat Detect. Runs on QNX with tflite_runtime + numpy only.

Input: 16 kHz mono float32 in 0.975 s windows (15600 samples). Prints the top classes for
each window and flags safety sounds (alarms, sirens, shouting, crashes, glass breaking)
separately from normal construction-site noise.

Usage (on the Pi):
    python3 audio_detect.py samples/siren.wav            # one or more .wav files
    python3 audio_detect.py --stdin < raw_s16le_16k.pcm  # live: raw 16-bit mono PCM on stdin
Options: --models DIR (default ~/models, then ./models), --threshold, --top, --hop
"""
import argparse
import csv
import os
import sys
import time
import wave
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # tflite_runtime prints a deprecation warning on import
try:
    from tflite_runtime.interpreter import Interpreter  # QNX: python3-tflite-runtime (oss.qnx.com)
except ImportError:  # laptop fallback for testing
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

SAMPLE_RATE = 16000
WINDOW = 15600  # 0.975 s, YAMNet's fixed input size

# Names must match yamnet_class_map.csv exactly (checked at startup).
SAFETY = {
    "ALARM": ["Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Buzzer", "Car alarm",
              "Alarm clock", "Reversing beeps", "Beep, bleep"],
    "SIREN": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)"],
    "HORN": ["Vehicle horn, car horn, honking", "Air horn, truck horn", "Foghorn"],
    "VOICE": ["Shout", "Yell", "Screaming", "Children shouting", "Crying, sobbing", "Groan"],
    "IMPACT": ["Smash, crash", "Breaking", "Shatter", "Glass", "Explosion", "Bang",
               "Thump, thud", "Crack", "Gunshot, gunfire"],
    "FIRE": ["Fire", "Crackle"],
}
# Expected on a construction site: reported, but never raises an alert on its own.
NORMAL = ["Jackhammer", "Drill", "Power tool", "Hammer", "Sawing", "Chainsaw", "Tools",
          "Engine", "Heavy engine (low frequency)", "Medium engine (mid frequency)",
          "Engine starting", "Speech", "Silence"]


def find_models(arg):
    for d in ([arg] if arg else [os.path.expanduser("~/models"), "models"]):
        if d and os.path.exists(os.path.join(d, "yamnet.tflite")):
            return d
    sys.exit("yamnet.tflite not found (looked in ~/models and ./models; use --models DIR)")


def load_labels(path):
    with open(path, newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def read_wav(path):
    """Read a PCM .wav as float32 mono in [-1, 1], resampled to 16 kHz."""
    with wave.open(path, "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        sys.exit(f"{path}: unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:  # simple linear resample; fine for classification
        t_new = np.arange(int(len(x) * SAMPLE_RATE / rate)) / SAMPLE_RATE
        x = np.interp(t_new, np.arange(len(x)) / rate, x).astype(np.float32)
    return x


class Yamnet:
    def __init__(self, model_dir, threads=4):
        self.labels = load_labels(os.path.join(model_dir, "yamnet_class_map.csv"))
        self.it = Interpreter(model_path=os.path.join(model_dir, "yamnet.tflite"), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        idx = {n: i for i, n in enumerate(self.labels)}
        for n in [n for g in SAFETY.values() for n in g] + NORMAL:
            if n not in idx:
                print(f"WARNING: class '{n}' not in class map", file=sys.stderr)
        self.safety = [(g, n, idx[n]) for g, names in SAFETY.items() for n in names if n in idx]
        self.normal = [(n, idx[n]) for n in NORMAL if n in idx]

    def scores(self, window):
        self.it.set_tensor(self.inp, window.astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.out)[0]

    def analyze(self, window, threshold):
        s = self.scores(window)
        hits = sorted(((g, n, float(s[i])) for g, n, i in self.safety if s[i] >= threshold),
                      key=lambda h: -h[2])
        normal = sorted(((n, float(s[i])) for n, i in self.normal if s[i] >= threshold),
                        key=lambda h: -h[1])
        return s, hits, normal


def report(t, window, model, args):
    t0 = time.time()
    s, hits, normal = model.analyze(window, args.threshold)
    ms = (time.time() - t0) * 1000
    top = s.argsort()[::-1][:args.top]
    top_str = " | ".join(f"{model.labels[i]} {s[i]:.2f}" for i in top)
    level = float(np.abs(window).max())
    if hits:
        tag = "ALERT " + ",".join(sorted({g for g, _, _ in hits}))
    elif normal:
        tag = "normal"
    else:
        tag = "-"
    print(f"[{t:6.2f}s lvl {level:.2f} {ms:4.0f}ms] {tag:<18} {top_str}")
    for g, n, sc in hits:
        print(f"      !! {g:<6} {n} {sc:.2f}")
    return hits


def stream_windows(stream, hop):
    """Yield (t_seconds, window) from raw s16le 16 kHz mono PCM on a binary stream,
    one window every `hop` samples, each the latest WINDOW samples."""
    buf, t = np.zeros(0, np.float32), 0.0
    while True:
        chunk = stream.read(hop * 2)
        if not chunk:
            return
        samples = np.frombuffer(chunk[:len(chunk) // 2 * 2], "<i2").astype(np.float32) / 32768.0
        buf = np.concatenate([buf, samples])[-WINDOW:]
        if len(buf) >= WINDOW:
            yield t, buf
            t += hop / SAMPLE_RATE


def windows(x, hop):
    if len(x) < WINDOW:
        x = np.pad(x, (0, WINDOW - len(x)))
    for start in range(0, len(x) - WINDOW + 1, hop):
        yield start / SAMPLE_RATE, x[start:start + WINDOW]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="*", help=".wav files to classify")
    ap.add_argument("--stdin", action="store_true", help="read raw s16le 16 kHz mono PCM from stdin")
    ap.add_argument("--models", help="folder with yamnet.tflite + yamnet_class_map.csv")
    ap.add_argument("--threshold", type=float, default=0.25, help="score needed to flag a class")
    ap.add_argument("--top", type=int, default=3, help="top classes to print per window")
    ap.add_argument("--hop", type=float, default=0.5, help="hop between windows, in seconds")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if not args.wavs and not args.stdin:
        ap.error("give one or more .wav files, or --stdin")

    model = Yamnet(find_models(args.models), args.threads)
    hop = max(1, int(args.hop * SAMPLE_RATE))

    if args.stdin:
        try:
            for t, w in stream_windows(sys.stdin.buffer, hop):
                report(t, w, model, args)
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        return

    any_alert = False
    for path in args.wavs:
        x = read_wav(path)
        print(f"== {path}  ({len(x) / SAMPLE_RATE:.2f}s)")
        alerts = set()
        for t, w in windows(x, hop):
            for g, _, _ in report(t, w, model, args):
                alerts.add(g)
        print(f"   summary: {'ALERT ' + ', '.join(sorted(alerts)) if alerts else 'no safety sounds'}")
        any_alert |= bool(alerts)
    sys.exit(1 if any_alert else 0)


if __name__ == "__main__":
    main()
