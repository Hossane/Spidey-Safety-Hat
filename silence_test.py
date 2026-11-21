import csv
import numpy as np
import tensorflow as tf

# Load labels (3rd column of the CSV)
with open("models/yamnet_class_map.csv", newline="") as f:
    labels = [row["display_name"] for row in csv.DictReader(f)]

# Load model
interp = tf.lite.Interpreter("models/yamnet.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]["index"]
out = interp.get_output_details()[0]["index"]

# 1 second of silence
audio = np.zeros(15600, dtype=np.float32)

interp.set_tensor(inp, audio)
interp.invoke()
scores = interp.get_tensor(out)[0]

# Top 3
for idx in scores.argsort()[::-1][:3]:
    print(f"{labels[idx]:30s} {scores[idx]:.3f}")