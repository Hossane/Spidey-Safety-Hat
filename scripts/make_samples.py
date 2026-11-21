"""Laptop-side: build test inputs in samples/ for audio_detect.py and vision_detect.py.

- Downloads a street photo (people + bus) and two YAMNet example clips.
- Synthesizes siren.wav (wailing sweep) and alarm_beeps.wav (3 kHz beeps).
- Converts every .jpg to .ppm, which the Pi can read with numpy alone.

Run from the project root:  python scripts/make_samples.py   (needs numpy + Pillow)
"""
import os
import urllib.request
import wave

import numpy as np
from PIL import Image

OUT = "samples"
SR = 16000
DOWNLOADS = {
    "street.jpg": "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
    "speech_whistling.wav": "https://storage.googleapis.com/audioset/speech_whistling2.wav",
    "cat_miaow.wav": "https://storage.googleapis.com/audioset/miaow_16k.wav",
}


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, url in DOWNLOADS.items():
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            print("download", name)
            urllib.request.urlretrieve(url, p)

    t = np.arange(4 * SR) / SR
    # Wailing siren: 600-1400 Hz sweep, 4 cycles over 4 s, with harmonics
    f = 1000 + 400 * np.sin(2 * np.pi * 1.0 * t)
    phase = 2 * np.pi * np.cumsum(f) / SR
    write_wav(os.path.join(OUT, "siren.wav"), 0.5 * np.sin(phase) + 0.15 * np.sin(2 * phase))
    # Alarm: 3 kHz beeps, 0.25 s on / 0.25 s off
    beeps = 0.5 * np.sin(2 * np.pi * 3000 * t) * ((t % 0.5) < 0.25)
    write_wav(os.path.join(OUT, "alarm_beeps.wav"), beeps)
    write_wav(os.path.join(OUT, "silence.wav"), np.zeros(2 * SR))

    for name in os.listdir(OUT):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(OUT, name)).convert("RGB")
            img.save(os.path.join(OUT, os.path.splitext(name)[0] + ".ppm"))
    print("samples:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
