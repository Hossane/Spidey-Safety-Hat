"""Person/vehicle tracking for Hat Detect, so a familiar person doesn't keep setting off alerts.

Based on the IoU/centre-distance Tracker from the cv-tracking branch (live_cam.py), with:
- Re-identification: when someone leaves the camera's view, their track is kept for
  `memory_s` seconds with a colour signature of their clothing. If they come back, they
  get their old ID and don't trigger a new alert.
- Incidents, which alert even for someone already known:
    DOWN      a person whose box was upright (tall) becomes wide: they fell or collapsed
    CLOSE     a person or vehicle fills a large part of the frame (right behind you)
    APPROACH  a box grows quickly (closing in fast)
    LYING     a person lying still (wide shape, not moving) for a few seconds, even if
              the fall itself wasn't seen
  Each incident fires once, then re-arms when the condition clears.
- Motion for every track: approaching / moving away / crossing left|right / still, a
  speed, and for anything approaching an estimated time to reach the wearer (from how
  fast its box grows). Vehicles moving away are never announced.

Pure numpy, so it runs on QNX with nothing extra installed.
"""
from collections import deque

import numpy as np


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


def aspect(box):
    """Height / width: well above 1 for someone standing, below 1 for someone lying down."""
    return (box[3] - box[1]) / max(1, box[2] - box[0])


def signature(img, box, bins=4):
    """Normalised RGB colour histogram of the middle of the box (torso / clothing)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx0, cx1 = x0 + int(w * 0.2), x1 - int(w * 0.2)
    cy0, cy1 = y0 + int(h * 0.15), y0 + int(h * 0.7)
    crop = img[max(0, cy0):max(cy0 + 1, cy1), max(0, cx0):max(cx0 + 1, cx1)]
    if crop.size == 0:
        return None
    step = max(1, max(crop.shape[0], crop.shape[1]) // 32)    # ~32x32 samples is plenty
    px = crop[::step, ::step].reshape(-1, 3).astype(np.int32) * bins // 256
    idx = px[:, 0] * bins * bins + px[:, 1] * bins + px[:, 2]
    hist = np.bincount(idx, minlength=bins ** 3).astype(np.float32)
    return hist / max(1.0, hist.sum())


def similarity(a, b):
    """Histogram intersection, 0 (nothing alike) .. 1 (identical colours)."""
    if a is None or b is None:
        return 0.0
    return float(np.minimum(a, b).sum())


def side_of(box, w):
    cx = centre(box)[0] / w
    return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"


def incident_group(kind, tr):
    """Which hazard group a tracker incident alerts as (None: log only, no alert)."""
    if kind == "NEW":
        return tr.group
    if kind in ("DOWN", "LYING"):
        return "PERSON_DOWN"
    if kind in ("CLOSE", "APPROACH"):
        return "PERSON_CLOSE" if tr.group == "PERSON" else tr.group
    return None


class Track:
    frame_size = (640, 480)

    def __init__(self, tid, name, group, box, score, sig, t):
        self.id, self.name, self.group = tid, name, group
        self.box, self.score, self.sig = box, score, sig
        self.gallery = deque([sig] if sig is not None else [], maxlen=12)   # varied looks, for re-id
        self.first_seen = self.last_seen = self.entered = t
        self.frames = 1
        self.areas = deque(maxlen=30)                       # (time, box area, cut off by edge)
        self.path = deque(maxlen=40)                        # (time, centre x, centre y, size), 0..1
        self.motion, self.ttc, self.speed = "steady", None, 0.0
        self.lying_since = None
        self.announced = False
        self.below_since = None
        self.aspects = deque([aspect(box)] if self.shape_ok else [], maxlen=30)   # usable views only
        self.down_frames = 0
        self.armed = {"DOWN": True, "CLOSE": True, "APPROACH": True}
        self.visible = True
        self.areas.append((t, area(box), self.clipped))
        self._add_path(t)

    def _add_path(self, t):
        w, h = Track.frame_size
        cx, cy = centre(self.box)
        self.path.append((t, cx / w, cy / h, (area(self.box) / float(w * h)) ** 0.5))

    def seen(self, box, score, sig, t):
        self.box, self.score, self.last_seen = box, score, t
        self.frames += 1
        self.areas.append((t, area(box), self.clipped))
        self._add_path(t)
        if self.shape_ok:
            self.aspects.append(aspect(box))
        if sig is not None:
            self.sig = sig if self.sig is None else 0.8 * self.sig + 0.2 * sig
            if not self.gallery or self.likeness(sig) < 0.85:    # keep looks that differ
                self.gallery.append(sig)

    @property
    def clipped(self):
        """Box touches two or more frame edges: part of the body is out of view."""
        w, h = Track.frame_size
        x0, y0, x1, y1 = self.box
        return (x0 <= 4) + (y0 <= 4) + (x1 >= w - 4) + (y1 >= h - 4) >= 2

    @property
    def full(self):
        """Whole body in view: the box keeps clear of every frame edge (a box cut off by an
        edge is a partial body, and its shape says nothing about standing or lying)."""
        w, h = Track.frame_size
        x0, y0, x1, y1 = self.box
        m = 8
        return x0 > m and y0 > m and x1 < w - m and y1 < h - m

    def edges(self, m=8):
        """Which frame edges the box touches."""
        w, h = Track.frame_size
        x0, y0, x1, y1 = self.box
        return {"left": x0 <= m, "top": y0 <= m, "right": x1 >= w - m, "bottom": y1 >= h - m}

    @property
    def shape_ok(self):
        """The box's shape says something about standing vs lying. The camera is close, so a
        body often runs off the top, the bottom or one side; that's fine. A box touching both
        sides is a torso filling the view, and its shape means nothing."""
        e = self.edges()
        return not (e["left"] and e["right"])

    def on_floor(self, min_frac=0.02):
        """Could be someone lying on the ground: low in the image (the camera is at head
        height) and not a tiny far-away box. A lying body may run off one side or the bottom
        of the view. Rejected: a box touching the top (a head too close to judge), a box
        touching both sides, and a thin strip along the bottom edge (head and shoulders)."""
        w, h = Track.frame_size
        x0, y0, x1, y1 = self.box
        e = self.edges()
        if e["top"] or not self.shape_ok or (e["bottom"] and y0 > 0.6 * h):
            return False
        return (y0 + y1) / 2 >= 0.45 * h and (x1 - x0) * (y1 - y0) >= min_frac * w * h

    def likeness(self, sig):
        """Best colour match against any of this track's remembered looks."""
        return max((similarity(g, sig) for g in self.gallery), default=similarity(self.sig, sig))


class Tracker:
    """update(dets, img, t) -> (visible tracks, incidents).

    dets: (name, score, box, group) for the groups to track. Incidents are
    (kind, track) with kind NEW, BACK, DOWN, CLOSE or APPROACH. NEW is a person or
    vehicle seen for the first time; BACK is a known one re-entering the view (no alert).
    """

    def __init__(self, iou_min=0.25, max_dist=0.2, lost_after_s=1.0, memory_s=120.0,
                 reid_min=0.6, close_area=0.35, approach_ratio=1.8, approach_window_s=1.5,
                 stand_aspect=1.3, down_aspect=0.9, down_frames=8, settle_s=1.0, rearm_s=3.0,
                 lying_still_s=5.0, ttc_alert_s=3.0, motion_window_s=1.5):
        self.iou_min, self.max_dist = iou_min, max_dist
        self.lost_after_s, self.memory_s, self.reid_min = lost_after_s, memory_s, reid_min
        self.close_area, self.approach_ratio, self.approach_window_s = close_area, approach_ratio, approach_window_s
        self.stand_aspect, self.down_aspect, self.need_down = stand_aspect, down_aspect, down_frames
        self.settle_s, self.rearm_s = settle_s, rearm_s
        self.lying_still_s, self.ttc_alert_s, self.motion_window_s = lying_still_s, ttc_alert_s, motion_window_s
        self.tracks = {}
        self.next_id = 1

    @classmethod
    def from_config(cls, cfg):
        keys = ("iou_min", "max_dist", "lost_after_s", "memory_s", "reid_min", "close_area",
                "approach_ratio", "approach_window_s", "stand_aspect", "down_aspect", "down_frames",
                "settle_s", "rearm_s", "lying_still_s", "ttc_alert_s", "motion_window_s")
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def update(self, dets, img, t, sigs=None, size=None):
        """img gives the colour signatures; offline replays can pass sigs and size=(w, h)."""
        h, w = img.shape[:2] if img is not None else (size[1], size[0])
        Track.frame_size = (w, h)
        diag = (w * w + h * h) ** 0.5
        incidents = []
        if sigs is None:
            sigs = [signature(img, d[2]) for d in dets]

        # 1. Continuity: tracks seen in the last `lost_after_s`, matched by overlap, then distance.
        pairs = []
        for tid, tr in self.tracks.items():
            if t - tr.last_seen > self.lost_after_s:
                continue
            for j, (name, _s, box, _g) in enumerate(dets):
                if name != tr.name:
                    continue
                o = iou(tr.box, box)
                if o >= self.iou_min:
                    pairs.append((2.0 + o, tid, j))
                else:
                    (ax, ay), (bx, by) = centre(tr.box), centre(box)
                    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diag
                    if d <= self.max_dist:
                        pairs.append((1.0 - d + tr.likeness(sigs[j]), tid, j))
        matched = self._assign(pairs)

        # 2. Re-identification: unmatched detections vs. tracks out of view, by clothing colour
        #    and a plausible size.
        pairs = []
        for tid, tr in self.tracks.items():
            if tid in matched.values():
                continue
            for j, (name, _s, box, _g) in enumerate(dets):
                if j in matched or name != tr.name:
                    continue
                ratio = area(box) / area(tr.box)
                s = tr.likeness(sigs[j])
                if s >= self.reid_min and 1 / 6 <= ratio <= 6:   # they may come back nearer or further
                    pairs.append((s, tid, j))
        back = self._assign(pairs, taken=matched)
        for j, tid in back.items():
            matched[j] = tid
            tr = self.tracks[tid]
            if t - tr.last_seen > self.lost_after_s:   # was out of view, not just a jump
                tr.areas.clear()                       # don't read the gap as "approaching"
                tr.path.clear()
                tr.down_frames = 0
                tr.lying_since = None
                tr.entered = t
                incidents.append(("BACK", tr))

        for j, tid in matched.items():
            name, score, box, _g = dets[j]
            self.tracks[tid].seen(box, score, sigs[j], t)

        # 3. New tracks.
        for j, (name, score, box, group) in enumerate(dets):
            if j in matched:
                continue
            tr = Track(self.next_id, name, group, box, score, sigs[j], t)
            self.tracks[tr.id] = tr
            self.next_id += 1
            matched[j] = tr.id
            if group == "PERSON":                      # people: announce at once
                tr.announced = True
                incidents.append(("NEW", tr))

        # 4. Forget tracks gone too long; check incidents on the visible ones.
        visible = []
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            tr.visible = tr.last_seen == t
            if not tr.visible:
                if t - tr.last_seen > self.memory_s:
                    del self.tracks[tid]
                continue
            visible.append(tr)
            self._kinematics(tr, t)
            if not tr.announced and t - tr.first_seen >= self.settle_s:
                tr.announced = True                    # vehicles: announce unless moving away
                if tr.motion != "moving away":
                    incidents.append(("NEW", tr))
            for kind in self._check(tr, w, h, t):
                incidents.append((kind, tr))
        return visible, incidents

    @staticmethod
    def _assign(pairs, taken=None):
        """Greedy best-first matching. Returns {detection index: track id}."""
        out = {}
        used_t = set(taken.values()) if taken else set()
        used_d = set(taken) if taken else set()
        for _score, tid, j in sorted(pairs, reverse=True):
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            out[j] = tid
        return out

    def _kinematics(self, tr, t):
        """Fit the last motion_window_s of the track: size growth (approach), sideways speed."""
        pts = [p for p in tr.path if t - p[0] <= self.motion_window_s]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.4:
            tr.motion, tr.ttc, tr.speed = "steady", None, 0.0
            return
        ts = np.array([p[0] for p in pts]) - pts[0][0]
        xs, size = np.array([p[1] for p in pts]), np.array([p[3] for p in pts])
        vx = float(np.polyfit(ts, xs, 1)[0])                 # frame widths per second
        growth = float(np.polyfit(ts, size, 1)[0]) / max(1e-3, float(size[-1]))   # per second
        tr.speed = round(max(abs(vx), abs(growth)), 2)
        tr.ttc = None
        if growth > 0.15:
            tr.motion = "approaching"
            tr.ttc = round(1.0 / growth, 1)                  # looming: size / d(size)/dt
        elif growth < -0.15:
            tr.motion = "moving away"
        elif abs(vx) > 0.15:
            tr.motion = "crossing left" if vx < 0 else "crossing right"
        elif max(float(abs(xs - xs.mean()).max()),
                 float(abs(size - size.mean()).max()) / max(1e-3, float(size.mean()))) < 0.05:
            tr.motion = "still"
        else:
            tr.motion = "steady"

    def _still_for(self, tr, t, secs):
        pts = [p for p in tr.path if t - p[0] <= secs]
        if not pts or t - pts[0][0] < secs * 0.8:
            return False
        xs, ys, size = (np.array([p[i] for p in pts]) for i in (1, 2, 3))
        return (xs.max() - xs.min() < 0.08 and ys.max() - ys.min() < 0.08
                and size.max() / max(1e-3, size.min()) < 1.3)

    def _check(self, tr, w, h, t):
        found = []
        frac = area(tr.box) / float(w * h)

        settling = t - tr.entered < self.settle_s    # just appeared: NEW/BACK already covers it

        # CLOSE: fills a big part of the frame. Someone who arrives already close doesn't
        # count (the NEW alert covers them). Re-arms after they've stayed further back
        # for rearm_s.
        if frac >= self.close_area:
            tr.below_since = None
            if tr.armed["CLOSE"] and not settling:
                found.append("CLOSE")
            tr.armed["CLOSE"] = False
        elif frac < self.close_area * 0.6:
            tr.below_since = tr.below_since or t
            if t - tr.below_since >= self.rearm_s:
                tr.armed["CLOSE"] = True

        # APPROACH: box grew by approach_ratio within approach_window_s, measured from boxes
        # that weren't cut off by the frame edge (a box also grows as someone walks into view).
        if not settling:
            base = [a for (ts, a, clipped) in tr.areas
                    if t - ts <= self.approach_window_s and not clipped][:-1]
            if len(base) >= 2:
                ratio = tr.areas[-1][1] / max(1, min(base))
                fast = tr.motion == "approaching" and tr.ttc is not None and tr.ttc <= self.ttc_alert_s
                if ratio >= self.approach_ratio or fast:
                    if tr.armed["APPROACH"]:
                        found.append("APPROACH")
                    tr.armed["APPROACH"] = False
                elif ratio < 1.2 and tr.motion != "approaching":
                    tr.armed["APPROACH"] = True

        # DOWN (people only): their usual upright shape turns wide for a few frames.
        if tr.name == "person":
            a = aspect(tr.box)
            floor = tr.on_floor()
            seen_upright = list(tr.aspects)[:-1] if tr.shape_ok else list(tr.aspects)
            usual = float(np.median(seen_upright)) if len(seen_upright) >= 3 else 0.0
            if floor and usual >= self.stand_aspect and a <= self.down_aspect:
                tr.down_frames += 1
            elif tr.shape_ok and a >= self.stand_aspect:
                tr.down_frames = 0
                tr.armed["DOWN"] = True
            elif not floor:
                tr.down_frames = 0
            if tr.down_frames >= self.need_down and tr.armed["DOWN"]:
                tr.armed["DOWN"] = False
                found.append("DOWN")

            # LYING: wide and not moving for lying_still_s, even if we never saw them upright.
            if floor and a <= self.down_aspect:
                tr.lying_since = tr.lying_since or t
                if (t - tr.lying_since >= self.lying_still_s and tr.armed["DOWN"]
                        and self._still_for(tr, t, self.lying_still_s)):
                    tr.armed["DOWN"] = False
                    found.append("LYING")
            else:
                tr.lying_since = None
        return found

    def counts(self):
        vis = [tr for tr in self.tracks.values() if tr.visible]
        return {"people": sum(tr.name == "person" for tr in vis), "tracked": len(vis),
                "remembered": len(self.tracks) - len(vis)}
