import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # hide TF info spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf

SAMPLE_RATE = 16000
WINDOW = 15600                                 # ~1 second

# Sounds the hat cares about (exact names from yamnet_class_map.csv)
TARGETS = ["Siren", "Police car (siren)", "Ambulance (siren)",
           "Fire engine, fire truck (siren)", "Civil defense siren",
           "Vehicle horn, car horn, honking", "Car alarm",
           "Smoke detector, smoke alarm", "Fire alarm", "Alarm", "Bark"]

with open("models/yamnet_class_map.csv", newline="") as f:
    labels = [row["display_name"] for row in csv.DictReader(f)]

interp = tf.lite.Interpreter("models/yamnet.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]["index"]
out = interp.get_output_details()[0]["index"]

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
        print(line)

        # Any target sound scoring 0.10 or more, even if not in the top 3
        hits = [f"{t} {scores[labels.index(t)]:.2f}" for t in TARGETS
                if t in labels and scores[labels.index(t)] >= 0.10]
        if hits:
            print("    TARGETS ->", "  ".join(hits))
except KeyboardInterrupt:
    print("\nStopped.")