"""Pi side of the live camera view (started over SSH by scripts/live_view.py).

Runs the same detector and tracker as hat_detect.py and writes, per frame, one JSON line
(boxes, track state, incidents) followed by the frame as raw RGB bytes, half size.
The camera can only be used by one program, so stop hat_detect.py first.
"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import hat_detect  # noqa: E402
import tracking  # noqa: E402
import vision_detect  # noqa: E402


def main():
    cfg = hat_detect.load_config(os.path.join(HERE, "hat_config.json"))
    vc = cfg["video"]
    tc = vc.get("tracking", {})
    tracked = set(tc.get("groups", ["PERSON", "VEHICLE"]))
    tracker = tracking.Tracker.from_config(tc)
    det = vision_detect.Detector(vision_detect.find_models(None))
    out = sys.stdout.buffer
    stop = threading.Event()
    last, fps = time.monotonic(), 0.0
    try:
        for _, img in vision_detect.camera_frames(vc["camera_unit"], vc["fps"], stop=stop):
            t = time.monotonic()
            fps = 0.8 * fps + 0.2 / max(t - last, 1e-3)
            last = t
            dets = det.detect(img, vc["threshold"])
            visible, incidents = tracker.update([d for d in dets if d[3] in tracked], img, t)
            tracks = []
            for tr in visible:
                lying = (t - tr.lying_since) if tr.lying_since else 0.0
                tracks.append({"id": tr.id, "name": tr.name, "box": list(tr.box), "score": round(tr.score, 2),
                               "motion": tr.motion, "ttc": tr.ttc,
                               "aspect": round(tracking.aspect(tr.box), 2),
                               "floor": tr.on_floor() if tr.name == "person" else False,
                               "down": tr.down_frames, "need_down": tracker.need_down,
                               "lying_s": round(lying, 1), "lying_need": tracker.lying_still_s})
            other = [{"name": n, "box": list(b), "score": round(s, 2)}
                     for n, s, b, g in dets if g not in tracked]
            head = {"w": img.shape[1] // 2, "h": img.shape[0] // 2, "fps": round(fps, 1),
                    "tracks": tracks, "other": other,
                    "incidents": [f"{k} {tr.name} #{tr.id}" for k, tr in incidents]}
            out.write((json.dumps(head) + "\n").encode())
            out.write(img[::2, ::2].tobytes())
            out.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
