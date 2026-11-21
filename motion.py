"""Sudden-movement detector for Hat Detect: catches things the object model has no label for
(a falling load, a swinging hook, a door or panel slamming, debris).

Compares each camera frame with the previous one at low resolution (80x60 grey). Movement
inside boxes of people and vehicles is ignored, since the tracker already covers those. It
alerts when a burst of unexplained movement follows a quiet scene. If most of the frame
changes at once, it's treated as the camera itself moving (the wearer turning their head)
and ignored.

Pure numpy, so it runs on QNX as is.
"""
import numpy as np


class MotionWatch:
    def __init__(self, step=8, pixel_delta=30, burst=0.06, quiet=0.02, quiet_s=1.0,
                 ego=0.45, min_blob=0.03):
        self.step, self.pixel_delta = step, pixel_delta
        self.burst, self.quiet, self.quiet_s = burst, quiet, quiet_s
        self.ego, self.min_blob = ego, min_blob
        self.prev = None
        self.quiet_since = None
        self.last = 0.0          # unexplained changed fraction in the latest frame

    @classmethod
    def from_config(cls, cfg):
        keys = ("step", "pixel_delta", "burst", "quiet", "quiet_s", "ego", "min_blob")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def update(self, img, t, boxes=()):
        """img: HxWx3 uint8. boxes: (x0, y0, x1, y1) of tracked objects, in pixels.
        Returns None, or a dict (label, score, position, bbox) for a sudden movement."""
        h, w = img.shape[:2]
        s = self.step
        g = img[::s, ::s].astype(np.int16).mean(axis=2)
        prev, self.prev = self.prev, g
        if prev is None or prev.shape != g.shape:
            return None
        changed = np.abs(g - prev) > self.pixel_delta
        if changed.mean() > self.ego:              # the whole view moved: the camera, not the scene
            self.quiet_since = None
            self.last = 0.0
            return None
        for x0, y0, x1, y1 in boxes:               # movement of tracked people / vehicles
            changed[max(0, y0 // s - 1):y1 // s + 2, max(0, x0 // s - 1):x1 // s + 2] = False
        frac = float(changed.mean())
        self.last = frac

        if frac < self.quiet:
            self.quiet_since = self.quiet_since or t
            return None
        was_quiet = self.quiet_since is not None and t - self.quiet_since >= self.quiet_s
        self.quiet_since = None
        if not was_quiet or frac < self.burst:
            return None

        ys, xs = np.nonzero(changed)
        if len(xs) < self.min_blob * changed.size:
            return None
        x0, x1 = int(np.percentile(xs, 5)), int(np.percentile(xs, 95))
        y0, y1 = int(np.percentile(ys, 5)), int(np.percentile(ys, 95))
        cx, cy = (x0 + x1) / 2 / changed.shape[1], (y0 + y1) / 2 / changed.shape[0]
        where = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
        label = "movement overhead" if cy < 0.35 else "sudden movement"
        return {"label": label, "score": round(min(1.0, frac / 0.3), 3), "position": where,
                "bbox": [x0 * s, y0 * s, min(w, (x1 + 1) * s), min(h, (y1 + 1) * s)]}
