"""Record what the tracker sees, for tuning it offline (runs on the Pi).

Writes one JSON line per camera frame: time, frame size, and each person/vehicle
detection with its box, score and colour signature. Replay it on the laptop with
scripts/replay_tracks.py to try tracker settings without the camera.

    python3 scripts/capture_tracks.py --seconds 60 > tracks.jsonl
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tracking  # noqa: E402
import vision_detect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--unit", type=int, default=5)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    det = vision_detect.Detector(vision_detect.find_models(None))
    stop = threading.Event()
    t0 = time.monotonic()
    for _, img in vision_detect.camera_frames(args.unit, args.fps, stop=stop):
        t = time.monotonic() - t0
        if t > args.seconds:
            stop.set()
            break
        dets = [d for d in det.detect(img, args.threshold) if d[3] in ("PERSON", "VEHICLE")]
        out = {"t": round(t, 3), "w": img.shape[1], "h": img.shape[0],
               "dets": [{"name": n, "score": round(s, 3), "box": list(b), "group": g,
                         "sig": [round(float(x), 4) for x in tracking.signature(img, b)]}
                        for n, s, b, g in dets]}
        print(json.dumps(out), flush=True)
        print(f"t={t:5.1f}s  {len(dets)} tracked-class detections", file=sys.stderr)


if __name__ == "__main__":
    main()
