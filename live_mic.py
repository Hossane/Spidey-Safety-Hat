import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # hide TF info spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import csv
import numpy as np
import sounddevice as sd

# TFLite runtime: oss.qnx.com package on the Pi, TensorFlow fallback on the laptop
try:
    from tflite_runtime.interpreter import Interpreter   # Pi: python3-tflite-runtime (oss.qnx.com)
    RUNTIME = "tflite_runtime (oss.qnx.com)"
except ImportError:
    import tensorflow as tf                               # laptop fallback for testing
    Interpreter = tf.lite.Interpreter
    RUNTIME = "tensorflow (laptop)"

SAMPLE_RATE = 16000
WINDOW = 15600                                 # ~1 second
SHOW_ABOVE = 0.10                              # print target labels scoring at least this

# Sounds the hat cares about, grouped by category.
# Names must match yamnet_class_map.csv exactly.
TARGETS = {
    "EMERGENCY": [
        "Siren", "Police car (siren)", "Ambulance (siren)",
        "Fire engine, fire truck (siren)", "Civil defense siren",
        "Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Buzzer",
    ],
    "VEHICLE": [
        "Vehicle horn, car horn, honking", "Air horn, truck horn", "Honk",
        "Reversing beeps", "Truck", "Air brake", "Car alarm",
        "Engine", "Heavy engine (low frequency)", "Engine starting",
    ],
    "TOOLS": [
        "Jackhammer", "Drill", "Power tool", "Chainsaw", "Sawing",
        "Hammer", "Sanding", "Tools",
    ],
    "IMPACT": [
        "Explosion", "Bang", "Smash, crash", "Breaking", "Shatter",
        "Glass", "Crack", "Thump, thud",
    ],
    "PEOPLE": [
        "Shout", "Yell", "Screaming", "Whistle",
    ],
}

with open("models/yamnet_class_map.csv", newline="") as f:
    labels = [row["display_name"] for row in csv.DictReader(f)]

# Look up each target's index once, and warn about any name that doesn't match
target_idx = {}
for group, names in TARGETS.items():
    for name in names:
        if name in labels:
            target_idx[name] = (group, labels.index(name))
        else:
            print(f"WARNING: '{name}' not found in class map")

interp = Interpreter(model_path="models/yamnet.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]["index"]
out = interp.get_output_details()[0]["index"]
print(f"Runtime: {RUNTIME}")

print("Listening... (Ctrl+C to stop)")
try:
    while True:
        audio = sd.rec(WINDOW, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        interp.set_tensor(inp, audio)
        interp.invoke()
        scores = interp.get_tensor(out)[0]

        # Top 3 overall
        top = scores.argsort()[::-1][:3]
        line = "  |  ".join(f"{labels[i]} {scores[i]:.2f}" for i in top)
        print(f"[level {np.abs(audio).max():.3f}]  {line}")

        # Target sounds above SHOW_ABOVE, even if not in the top 3
        hits = [(group, name, scores[i]) for name, (group, i) in target_idx.items()
                if scores[i] >= SHOW_ABOVE]
        for group, name, score in sorted(hits, key=lambda h: -h[2]):
            print(f"    {group:9s} -> {name} {score:.2f}")
except KeyboardInterrupt:
    print("\nStopped.")