"""EfficientDet-Lite0 object detector for Hat Detect. Runs on QNX with tflite_runtime + numpy.

Flags people and other hazards (vehicles, etc.) in camera frames.

Image input: .ppm/.pgm are decoded with numpy alone (no extra packages). .jpg/.png need
cv2 or PIL; neither is on the Pi by default (apk add python3-opencv adds cv2).
.ucv files are uncompressed recordings from the QNX Camera service (camera_example4_record).

Usage (laptop testing, needs opencv-python):
    python vision_detect.py --webcam 0

Usage (on the Pi):
    python3 vision_detect.py --camera 5                     # live: USB webcam via QNX Camera service
    python3 vision_detect.py samples/street.ppm             # one or more images
    python3 vision_detect.py /data/share/sensor/VID_x.ucv   # every frame of a recording
    python3 vision_detect.py --watch /tmp/frames            # process frames as a camera
                                                            # process drops them in a folder
Options: --models DIR (default ~/models, then ./models), --threshold, --fps (camera mode)

Tracking: in camera, --watch and .ucv modes, people and vehicles get a stable ID (#1, #2, ...)
across frames. Each track reports its side (left/centre/right), whether it is approaching
or moving away (box growing/shrinking), and events when it appears or leaves.
"""
import argparse
from collections import deque
import glob
import os
import signal
import struct
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
try:
    from tflite_runtime.interpreter import Interpreter  # QNX: python3-tflite-runtime (oss.qnx.com)
except ImportError:  # laptop fallback for testing
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# COCO names as written in coco_labels.txt.
HAZARDS = {
    "PERSON": ["person"],
    "VEHICLE": ["car", "truck", "bus", "motorcycle", "bicycle", "train"],
    "SHARP": ["knife", "scissors"],
}
IMAGE_EXT = (".ppm", ".pgm", ".jpg", ".jpeg", ".png")


def find_models(arg):
    for d in ([arg] if arg else [os.path.expanduser("~/models"), "models"]):
        if d and os.path.exists(os.path.join(d, "efficientdet_lite0.tflite")):
            return d
    sys.exit("efficientdet_lite0.tflite not found (looked in ~/models and ./models; use --models DIR)")


def read_pnm(path):
    """Decode binary PPM (P6) / PGM (P5) with numpy. Returns HxWx3 uint8 RGB."""
    with open(path, "rb") as f:
        data = f.read()
    fields, pos = [], 0
    while len(fields) < 4:  # magic, width, height, maxval; skip '#' comments
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    pos += 1  # single whitespace before pixel data
    magic, w, h, maxval = fields[0], int(fields[1]), int(fields[2]), int(fields[3])
    if magic not in (b"P6", b"P5") or maxval > 255:
        raise ValueError(f"{path}: only 8-bit binary P6/P5 supported")
    ch = 3 if magic == b"P6" else 1
    img = np.frombuffer(data, np.uint8, w * h * ch, pos).reshape(h, w, ch)
    return np.repeat(img, 3, axis=2) if ch == 1 else img


def read_image(path):
    if path.lower().endswith((".ppm", ".pgm")):
        return read_pnm(path)
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except ImportError:
        sys.exit(f"{path}: need cv2 or PIL for this format; convert it to .ppm instead")


def yuy2_to_rgb(buf, w, h):
    """YUY2 (Y0 U Y1 V, BT.601 limited range) -> HxWx3 uint8 RGB."""
    p = np.frombuffer(buf, np.uint8, w * h * 2).reshape(h, w // 2, 4).astype(np.float32)
    y = np.empty((h, w), np.float32)
    y[:, 0::2], y[:, 1::2] = p[..., 0], p[..., 2]
    u = np.repeat(p[..., 1], 2, axis=1) - 128.0
    v = np.repeat(p[..., 3], 2, axis=1) - 128.0
    y = (y - 16.0) * 1.164
    rgb = np.stack([y + 1.596 * v, y - 0.392 * u - 0.813 * v, y + 2.017 * u], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class UcvReader:
    """Reads QNX camera 'uncv_raw' recordings: a header (magic 'uncv_raw', header size,
    version, frame type, fourcc, fps, width, height, bpp, frame size), then per frame an
    8-byte timestamp in microseconds followed by the raw frame. Works on a growing file."""

    def __init__(self, path):
        self.f = open(path, "rb")
        hdr = self.f.read(48)
        if len(hdr) < 48 or hdr[:8] != b"uncv_raw":
            raise ValueError(f"{path}: not a uncv_raw recording (or header not written yet)")
        (self.header_size, _ver, _ftype, fourcc, self.fps, self.w, self.h, _bpp,
         self.frame_size) = struct.unpack_from("<III4sIIIIQ", hdr, 8)
        self.fourcc = fourcc.decode("ascii", "replace")
        if self.fourcc not in ("YUY2", "YUYV"):
            raise ValueError(f"{path}: frame format {self.fourcc} not supported (need YUY2)")
        self.record = 8 + self.frame_size
        self.path = path

    def count(self):
        return max(0, (os.path.getsize(self.path) - self.header_size) // self.record)

    def frame(self, i):
        self.f.seek(self.header_size + i * self.record)
        data = self.f.read(self.record)
        if len(data) < self.record:
            return None
        ts = struct.unpack_from("<Q", data)[0]
        return ts, yuy2_to_rgb(data[8:], self.w, self.h)

    def close(self):
        self.f.close()


def resize_nearest(img, size):
    h, w = img.shape[:2]
    ys = (np.arange(size) * h / size).astype(np.int32)
    xs = (np.arange(size) * w / size).astype(np.int32)
    return img[ys[:, None], xs[None, :]]


class Detector:
    def __init__(self, model_dir, threads=4):
        with open(os.path.join(model_dir, "coco_labels.txt")) as f:
            self.labels = [line.strip() for line in f]
        self.it = Interpreter(model_path=os.path.join(model_dir, "efficientdet_lite0.tflite"),
                              num_threads=threads)
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        self.inp, self.size, self.dtype = inp["index"], int(inp["shape"][1]), inp["dtype"]
        outs = self.it.get_output_details()
        if len(outs) != 4:
            sys.exit("expected the EfficientDet-Lite0 *detection* model (4 outputs: boxes, classes, "
                     "scores, count). See models/README.md.")
        # TFLite_Detection_PostProcess output order: boxes, classes, scores, count
        self.boxes_i, self.classes_i, self.scores_i, self.count_i = [o["index"] for o in outs]
        self.hazard_of = {n: g for g, names in HAZARDS.items() for n in names}

    def detect(self, img, threshold):
        h, w = img.shape[:2]
        x = resize_nearest(img, self.size)[None].astype(self.dtype)
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        boxes = self.it.get_tensor(self.boxes_i)[0]
        classes = self.it.get_tensor(self.classes_i)[0]
        scores = self.it.get_tensor(self.scores_i)[0]
        count = int(self.it.get_tensor(self.count_i)[0])
        dets = []
        for i in range(count):
            if scores[i] < threshold:
                continue
            c = int(classes[i])
            name = self.labels[c] if 0 <= c < len(self.labels) else f"class{c}"
            y0, x0, y1, x1 = np.clip(boxes[i], 0, 1)  # normalized ymin, xmin, ymax, xmax
            box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            dets.append((name, float(scores[i]), box, self.hazard_of.get(name)))
        return dets


def iou(a, b):
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def area(box):
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


class Tracker:
    """Gives each hazard detection (people, vehicles, ...) a stable ID across frames.

    Matching is greedy, same class only: first by box overlap (IoU), then by how close
    the box centres are, for objects that moved further than their own size between
    frames (low fps). A track that goes unseen for `max_missed` frames is dropped.

    Per track it reports:
      side      left / centre / right, from the box centre
      motion    approaching / moving away / steady, from how the box area changed
                over the last `history` frames (bigger box = closer)
    """

    def __init__(self, iou_min=0.25, max_dist=0.2, max_missed=8, history=6,
                 grow=1.3, shrink=0.75):
        self.iou_min, self.max_dist, self.max_missed = iou_min, max_dist, max_missed
        self.history, self.grow, self.shrink = history, grow, shrink
        self.tracks = {}   # id -> dict(name, hazard, box, score, areas, missed, frames)
        self.next_id = 1

    def update(self, dets, w, h):
        """dets: list of (name, score, box, hazard). Only hazard detections are tracked.
        Returns (visible_tracks, events)."""
        dets = [d for d in dets if d[3]]
        diag = (w * w + h * h) ** 0.5
        events = []

        # Score every (track, detection) pair of the same class.
        pairs = []
        for tid, t in self.tracks.items():
            for j, (name, _s, box, _hz) in enumerate(dets):
                if name != t["name"]:
                    continue
                o = iou(t["box"], box)
                if o >= self.iou_min:
                    pairs.append((1.0 + o, tid, j))          # overlap beats distance
                else:
                    (ax, ay), (bx, by) = centre(t["box"]), centre(box)
                    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diag
                    if d <= self.max_dist:
                        pairs.append((1.0 - d, tid, j))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _score, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            name, score, box, hz = dets[j]
            t = self.tracks[tid]
            t.update(box=box, score=score, missed=0, frames=t["frames"] + 1)
            t["areas"].append(area(box))

        # New tracks for unmatched detections.
        for j, (name, score, box, hz) in enumerate(dets):
            if j in used_d:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = dict(name=name, hazard=hz, box=box, score=score, missed=0,
                                    frames=1, areas=deque([area(box)], maxlen=self.history),
                                    motion="steady", warned=False)
            used_t.add(tid)
            events.append(f"NEW    {name} #{tid} ({self._side(box, w)})")

        # Age out tracks that weren't seen this frame.
        for tid in list(self.tracks):
            if tid in used_t:
                continue
            t = self.tracks[tid]
            t["missed"] += 1
            if t["missed"] > self.max_missed:
                events.append(f"LEFT   {t['name']} #{tid} (seen {t['frames']} frames)")
                del self.tracks[tid]

        visible = []
        for tid, t in self.tracks.items():
            if t["missed"]:
                continue
            t["side"] = self._side(t["box"], w)
            motion = self._motion(t["areas"])
            if motion == "approaching" and t["motion"] != "approaching":
                events.append(f"CLOSER {t['name']} #{tid} ({t['side']})")
            t["motion"] = motion
            visible.append((tid, t))
        return visible, events

    def _side(self, box, w):
        cx = centre(box)[0] / w
        return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"

    def _motion(self, areas):
        if len(areas) < 3:
            return "steady"
        ratio = areas[-1] / areas[0]
        if ratio >= self.grow:
            return "approaching"
        if ratio <= self.shrink:
            return "moving away"
        return "steady"

    def people(self):
        return sum(1 for t in self.tracks.values() if t["hazard"] == "PERSON" and not t["missed"])


def report_frame(label, img, model, args, tracker=None):
    t0 = time.time()
    dets = model.detect(img, args.threshold)
    ms = (time.time() - t0) * 1000
    hazards = sorted({d[3] for d in dets if d[3]})
    tag = "ALERT " + ",".join(hazards) if hazards else "clear"

    if tracker is None:                      # single images: no tracking
        print(f"== {label} ({img.shape[1]}x{img.shape[0]}, {ms:.0f}ms) {tag}")
        for name, score, (x0, y0, x1, y1), hz in dets:
            mark = f"!! {hz:<7}" if hz else "          "
            print(f"   {mark} {name:<14} {score:.2f}  box=({x0},{y0})-({x1},{y1})")
        sys.stdout.flush()
        return hazards

    visible, events = tracker.update(dets, img.shape[1], img.shape[0])
    print(f"== {label} ({img.shape[1]}x{img.shape[0]}, {ms:.0f}ms) {tag}  "
          f"people in view: {tracker.people()}")
    for ev in events:
        print(f"   >> {ev}")
    for tid, t in sorted(visible):
        x0, y0, x1, y1 = t["box"]
        print(f"   !! {t['hazard']:<7} {t['name']:<10} #{tid:<3} {t['score']:.2f}  "
              f"{t['side']:<6} {t['motion']:<11} box=({x0},{y0})-({x1},{y1})")
    for name, score, (x0, y0, x1, y1), hz in dets:  # untracked, non-hazard objects
        if not hz:
            print(f"              {name:<14} {score:.2f}  box=({x0},{y0})-({x1},{y1})")
    sys.stdout.flush()
    return hazards


def report(path, model, args, tracker=None):
    if path.lower().endswith(".ucv"):
        r = UcvReader(path)
        hazards = set()
        tracker = Tracker()                  # frames of one recording are a sequence
        for i in range(r.count()):
            ts, img = r.frame(i)
            hazards.update(report_frame(f"{os.path.basename(path)} #{i} t={ts / 1e6:.2f}s",
                                        img, model, args, tracker))
        r.close()
        return sorted(hazards)
    return report_frame(path, read_image(path), model, args, tracker)


SENSOR_DIR = "/data/share/sensor"


class CameraRecorder:
    """Drives the stock QNX `camera_example4_record` to write an uncompressed .ucv stream,
    which we read while it grows. Each recording is rotated every `chunk` seconds and the
    finished file is deleted, so the SD card doesn't fill up."""

    def __init__(self, unit, fps, chunk):
        self.unit, self.fps, self.chunk = unit, fps, chunk
        self.proc, self.path, self.started = None, None, 0.0

    def start(self):
        before = set(glob.glob(os.path.join(SENSOR_DIR, "*.ucv")))
        self.proc = subprocess.Popen(
            ["camera_example4_record", "-u", str(self.unit), "-f", "ucv", "-c", "none", "-r", str(self.fps)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.5)                      # viewfinder setup
        self._key()                          # "press any key to start recording"
        deadline = time.time() + 5
        while time.time() < deadline:
            new = set(glob.glob(os.path.join(SENSOR_DIR, "*.ucv"))) - before
            if new:
                self.path = max(new, key=os.path.getmtime)
                break
            time.sleep(0.1)
        else:
            self.stop()
            raise RuntimeError(f"camera unit {self.unit}: recorder produced no file")
        self.started = time.time()
        return self.path

    def _key(self):
        try:
            self.proc.stdin.write(b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def expired(self):
        return time.time() - self.started > self.chunk

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self._key()                      # "press any key to stop"
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        self.proc, self.path = None, None


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def camera_frames(unit, fps=5, chunk=30, stop=None):
    """Yield (timestamp_s, RGB frame) live from a QNX camera unit, newest frame only.
    Stops when `stop` (a threading.Event) is set; always stops the recorder and deletes
    its file on exit."""
    cam = CameraRecorder(unit, fps, chunk)
    try:
        while not (stop and stop.is_set()):
            path = cam.start()
            reader, last = None, -1
            while not cam.expired() and not (stop and stop.is_set()):
                if cam.proc.poll() is not None:
                    print("   recorder exited; restarting", file=sys.stderr)
                    break
                if reader is None:
                    try:
                        reader = UcvReader(path)
                    except ValueError:      # header not on disk yet
                        time.sleep(0.1)
                        continue
                n = reader.count()
                if n - 1 > last:
                    last = n - 1             # newest complete frame only; skip backlog
                    got = reader.frame(last)
                    if got:
                        yield got[0] / 1e6, got[1]
                else:
                    time.sleep(0.02)
            if reader:
                reader.close()
            cam.stop()
    finally:
        cam.stop()


def run_camera(model, args):
    # Clean shutdown (stop recorder, delete its file) on Ctrl+C or kill, even when started
    # in the background, where the shell leaves SIGINT ignored.
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    print(f"Camera unit {args.camera} at {args.fps} fps (Ctrl+C to stop)")
    tracker = Tracker()
    try:
        for ts, img in camera_frames(args.camera, args.fps, args.chunk):
            report_frame(f"cam{args.camera} t={ts:.2f}s", img, model, args, tracker)
    except KeyboardInterrupt:
        pass


def run_webcam(model, args):
    """Laptop testing: read an ordinary webcam with OpenCV, track, and show boxes in a window."""
    try:
        import cv2
    except ImportError:
        sys.exit("--webcam needs OpenCV: pip install opencv-python")
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        sys.exit(f"cannot open webcam {args.webcam} (try --webcam 1)")
    tracker = Tracker()
    print(f"Webcam {args.webcam} (press q in the window or Ctrl+C to stop)")
    t_start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("no frame from webcam")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            report_frame(f"webcam t={time.time() - t_start:.2f}s", rgb, model, args, tracker)
            for tid, t in tracker.tracks.items():
                if t["missed"]:
                    continue
                x0, y0, x1, y1 = t["box"]
                color = (0, 0, 255) if t.get("motion") == "approaching" else (0, 200, 255)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{t['name']} #{tid} {t.get('motion', '')}", (x0, max(15, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, f"people: {tracker.people()}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("hat camera (laptop test)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files (.ppm/.pgm, or .jpg/.png with cv2/PIL)")
    ap.add_argument("--camera", type=int, metavar="UNIT",
                    help="live: QNX Camera service unit (USB webcam is 5; list: camera_example1_callback)")
    ap.add_argument("--webcam", type=int, metavar="N",
                    help="laptop testing: OpenCV webcam index (usually 0); shows a window")
    ap.add_argument("--fps", type=int, default=5, help="camera mode frame rate (divisor of 30)")
    ap.add_argument("--chunk", type=int, default=30, help="camera mode: seconds per recording file")
    ap.add_argument("--watch", metavar="DIR", help="process new frames written to DIR, then delete them")
    ap.add_argument("--models", help="folder with efficientdet_lite0.tflite + coco_labels.txt")
    ap.add_argument("--threshold", type=float, default=0.4, help="minimum detection score")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if not args.images and not args.watch and args.camera is None and args.webcam is None:
        ap.error("give one or more images, --camera UNIT (Pi), --webcam N (laptop), or --watch DIR")

    model = Detector(find_models(args.models), args.threads)

    if args.webcam is not None:
        run_webcam(model, args)
        return

    if args.camera is not None:
        run_camera(model, args)
        return

    if args.watch:
        print(f"Watching {args.watch} for frames ... (Ctrl+C to stop)")
        tracker = Tracker()                  # frames arrive in order, so track across them
        try:
            while True:
                names = sorted(n for n in os.listdir(args.watch) if n.lower().endswith(IMAGE_EXT))
                for n in names:
                    p = os.path.join(args.watch, n)
                    try:
                        report(p, model, args, tracker)
                    except (ValueError, OSError) as e:  # frame still being written
                        print(f"   skip {n}: {e}")
                        continue
                    os.remove(p)
                sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            return

    any_alert = False
    for path in args.images:
        any_alert |= bool(report(path, model, args))
    sys.exit(1 if any_alert else 0)


if __name__ == "__main__":
    main()