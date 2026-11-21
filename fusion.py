"""Multimodal event fusion for Hat Detect: raw model detections in, unified hazard events out.

Pure Python (stdlib only), so it runs on QNX next to the models and can be tested on a
laptop without numpy or TFLite.

A *detection* is one model prediction: YAMNet heard "Siren" at 0.91 in one 0.975 s window,
or EfficientDet saw "car" at 0.88 in one frame. A horn lasting two seconds is ~4 audio
detections; a car in view for ten seconds is ~50 video detections.

A *hazard event* is the interpretation of one or more related detections:

- AUDIO_ONLY   sound with no related object in view      (alarm, shouting, crash)
- VISION_ONLY  object with no related sound              (person, vehicle, knife)
- MULTIMODAL   a sound and an object that a fusion rule says belong together
               (horn or siren + vehicle -> VEHICLE_HAZARD)

How detections become events:

1. Episodes. Detections of the same hazard extend one open episode until none arrive for
   `episode_gap_seconds` (or a fusion rule's longer `window_seconds`, so a partner seen
   late can still join). One episode = one event = one alert, however long it lasts.
   Unrelated hazards each get their own episode, so a new hazard is never suppressed by
   another one.
2. Fusion rules. A detection is combined with the other modality only when a rule in the
   config pairs their groups, and only if the two were seen within the rule's
   `window_seconds`. Things that merely happen at the same time stay separate.
3. Confirmation wait. If a single-modality detection could become a more severe
   multimodal event (a horn, medium, could become VEHICLE_HAZARD, high), it waits up to
   `confirm_wait_seconds` for the other modality before being announced, so horn + car
   gives one warning instead of two. Detections that are already as severe as any
   combination they could join (siren, impact) are announced immediately.
4. Late confirmation. If the partner arrives after the event was announced, the same
   event (same event_id) is upgraded to MULTIMODAL. It is announced again only if its
   severity went up and `realert_on_escalation` is on.

Severity comes from the configured hazard rules, never from model confidence. Each
modality's original peak confidence is kept as is (no averaging).

Every emitted record is a full snapshot of the event, keyed by `event_id`:
  status "open"    first announcement (notify = true)
  status "update"  upgraded to MULTIMODAL (notify = true only if severity went up)
  status "closed"  episode ended; final counts and peak scores (notify = false)
Consumers keep the latest record per event_id (the database upserts on it). `time` is
when the event was first detected and never changes between its records.
"""
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

AUDIO_ONLY, VISION_ONLY, MULTIMODAL = "AUDIO_ONLY", "VISION_ONLY", "MULTIMODAL"
SEVERITIES = ["low", "medium", "high"]
RANK = {s: i for i, s in enumerate(SEVERITIES)}
SIDE = {"audio": "audio", "video": "video", "vision": "video"}
OTHER = {"audio": "video", "video": "audio"}

DEFAULT_FUSION = {
    "confirm_wait_seconds": 1.0,
    "episode_gap_seconds": 3.0,
    "escalate_on_confirmation": True,   # rule without a severity: one level above its parts
    "realert_on_escalation": True,
    "rules": [
        {"hazard": "VEHICLE_HAZARD", "audio": ["HORN", "SIREN"], "video": ["VEHICLE"],
         "window_seconds": 3.0, "severity": "high"},
        {"hazard": "WORKER_DISTRESS", "audio": ["VOICE", "IMPACT"], "video": ["PERSON"],
         "window_seconds": 3.0, "severity": "high"},
    ],
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Detection:
    """One raw model prediction."""
    source: str                 # "audio" | "video"
    group: str                  # ALARM, HORN, VEHICLE, PERSON, ...
    label: str                  # model class, e.g. "Vehicle horn, car horn, honking", "car"
    score: float                # model confidence 0..1
    t: float                    # monotonic seconds (for windows and gaps)
    time: str                   # UTC ISO 8601 (for the record)
    position: str = None        # left | centre | right (video)
    bbox: list = None           # [x0, y0, x1, y1] px (video)


@dataclass
class Rule:
    hazard: str
    audio: frozenset
    video: frozenset
    window: float
    severity: str = None

    @classmethod
    def from_cfg(cls, r):
        sev = r.get("severity")
        if sev is not None and sev not in RANK:
            raise ValueError(f"fusion rule {r.get('hazard')}: unknown severity {sev!r}")
        return cls(r["hazard"], frozenset(r["audio"]), frozenset(r["video"]),
                   float(r.get("window_seconds", 3.0)), sev)

    def groups(self, side):
        return self.audio if side == "audio" else self.video


class Part:
    """What one modality contributed to an event: its strongest detection and a count."""

    def __init__(self, d):
        self.first_t, self.first_time = d.t, d.time
        self.count = 0
        self.score = -1.0
        self.update(d)

    def update(self, d):
        self.count += 1
        self.last_t, self.last_time = d.t, d.time
        if d.score > self.score:
            self.group, self.label, self.score = d.group, d.label, float(d.score)
            self.position, self.bbox = d.position, d.bbox

    def to_dict(self, side):
        out = {"group": self.group, "label": self.label, "score": round(self.score, 3),
               "detections": self.count}
        if side == "video":
            out["position"], out["bbox"] = self.position, self.bbox
        return out


class Episode:
    def __init__(self, d, severity, hold_until):
        self.id = str(uuid.uuid4())
        self.parts = {"audio": None, "video": None}
        self.parts[SIDE[d.source]] = Part(d)
        self.start_time = d.time  # fixed for the event's life: (event_id, time) is its DB key
        self.rule = None
        self.hazard = d.group
        self.severity = severity
        self.hold_until = hold_until
        self.emitted = False
        self.alerted = False
        self.explanation = None
        self.keep_open = 0.0      # seconds without detections before the episode closes

    def present(self):
        return [p for p in self.parts.values() if p]

    @property
    def first_t(self):
        return min(p.first_t for p in self.present())

    @property
    def last_t(self):
        return max(p.last_t for p in self.present())

    def single_side(self):
        return None if self.rule else next(s for s, p in self.parts.items() if p)


class FusionEngine:
    """Thread-safe. Call add() for every raw detection and tick() a few times a second
    (it releases held events and closes finished ones). Both return the event records to
    emit. Pass `t` explicitly in tests; on the hat it is time.monotonic()."""

    def __init__(self, fusion_cfg, severity_map, device_id):
        cfg = dict(DEFAULT_FUSION)
        cfg.update(fusion_cfg or {})
        self.rules = [Rule.from_cfg(r) for r in cfg["rules"]]
        self.confirm_wait = float(cfg["confirm_wait_seconds"])
        self.gap = float(cfg["episode_gap_seconds"])
        self.escalate = bool(cfg["escalate_on_confirmation"])
        self.realert = bool(cfg["realert_on_escalation"])
        self.sev = dict(severity_map)
        self.device_id = device_id
        self.open = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ public API

    def add(self, d):
        side = SIDE.get(d.source)
        if side is None:
            raise ValueError(f"unknown detection source {d.source!r}")
        with self.lock:
            out = []
            self._advance(d.t, out)
            ep = self._find(side, d.group)
            if ep:                              # _find only returns episodes that have this side
                ep.parts[side].update(d)
            else:
                hold = self._hold(side, d.group)
                ep = Episode(d, self._severity(d.group), d.t + hold if hold > 0 else -1.0)
                ep.keep_open = max([self.gap] + [r.window for r in self._rules_for(side, d.group)])
                self.open.append(ep)
            if ep.rule is None:
                self._try_merge(ep, d.t, out)
            self._release(d.t, out)
            return out

    def tick(self, t):
        with self.lock:
            out = []
            self._advance(t, out)
            return out

    def flush(self):
        """Emit and close every open event (on shutdown)."""
        with self.lock:
            out = []
            for ep in list(self.open):
                if not ep.emitted:
                    self._announce(ep, out)
                out.append(self._record(ep, "closed", False))
            self.open.clear()
            return out

    # ------------------------------------------------------------------ internals

    def _severity(self, group):
        return self.sev.get(group, "low")

    def _rules_for(self, side, group):
        return [r for r in self.rules if group in r.groups(side)]

    def _combined_severity(self, rule, parts_severities):
        if rule.severity:
            return rule.severity
        top = max(RANK[s] for s in parts_severities)
        return SEVERITIES[min(top + 1, RANK["high"])] if self.escalate else SEVERITIES[top]

    def _hold(self, side, group):
        """Wait for confirmation only when the other modality could raise the severity."""
        own = self._severity(group)
        for r in self._rules_for(side, group):
            if r.severity:
                if RANK[r.severity] > RANK[own]:
                    return self.confirm_wait
            elif self.escalate and RANK[own] < RANK["high"]:
                return self.confirm_wait
        return 0.0

    def _find(self, side, group):
        for ep in self.open:                    # an ongoing multimodal hazard this belongs to
            if ep.rule and group in ep.rule.groups(side):
                return ep
        for ep in self.open:                    # an ongoing sighting of the same thing
            p = ep.parts[side]
            if ep.rule is None and p and p.group == group:
                return ep
        return None

    def _try_merge(self, ep, t, out):
        side = ep.single_side()
        other = OTHER[side]
        mine = ep.parts[side]
        for rule in self._rules_for(side, mine.group):
            for p in self.open:
                theirs = p.parts[other]
                if p is ep or p.rule or theirs is None or theirs.group not in rule.groups(other):
                    continue
                # gap between the two sightings (0 if they overlap)
                gap = max(0.0, mine.first_t - theirs.last_t, theirs.first_t - mine.last_t)
                if gap <= rule.window:
                    self._merge(ep, p, rule, out)
                    return

    def _merge(self, a, b, rule, out):
        # Keep the event the wearer already heard about; otherwise the earlier one.
        if a.emitted != b.emitted:
            keep, drop = (a, b) if a.emitted else (b, a)
        else:
            keep, drop = (a, b) if a.first_t <= b.first_t else (b, a)
        other = OTHER[keep.single_side()]
        before = keep.severity
        keep.parts[other] = drop.parts[other]
        keep.rule, keep.hazard = rule, rule.hazard
        keep.keep_open = max(self.gap, rule.window)
        keep.severity = self._combined_severity(rule, [self._severity(p.group) for p in keep.present()])
        keep.explanation = self._explain(keep)
        self.open.remove(drop)
        if drop.emitted:                        # both were already announced separately
            rec = self._record(drop, "closed", False)
            rec["merged_into"] = keep.id
            out.append(rec)
        if keep.emitted:
            notify = self.realert and RANK[keep.severity] > RANK[before]
            keep.alerted = keep.alerted or notify
            out.append(self._record(keep, "update", notify))
        else:
            keep.hold_until = -1.0              # confirmed: announce now

    def _explain(self, ep):
        a, v = ep.parts["audio"], ep.parts["video"]
        dt = a.first_t - v.first_t
        where = {"left": " on the left", "right": " on the right", "centre": " ahead"}.get(v.position, "")
        order = "after" if dt >= 0 else "before"
        return (f"{a.label} ({a.group}) heard {abs(dt):.1f} s {order} {v.label} ({v.group}) "
                f"was seen{where}; rule {ep.rule.hazard} pairs them within {ep.rule.window:g} s")

    def _advance(self, t, out):
        self._release(t, out)
        for ep in list(self.open):
            if t - ep.last_t > ep.keep_open:
                if not ep.emitted:
                    self._announce(ep, out)
                out.append(self._record(ep, "closed", False))
                self.open.remove(ep)

    def _release(self, t, out):
        for ep in self.open:
            if not ep.emitted and t > ep.hold_until:
                self._announce(ep, out)

    def _announce(self, ep, out):
        ep.emitted = ep.alerted = True
        out.append(self._record(ep, "open", True))

    def _record(self, ep, status, notify):
        a, v = ep.parts["audio"], ep.parts["video"]
        fusion = MULTIMODAL if ep.rule else AUDIO_ONLY if a else VISION_ONLY
        last = max(ep.present(), key=lambda p: p.last_t)
        return {
            "event_id": ep.id,
            "time": ep.start_time,
            "last_time": last.last_time,
            "device_id": self.device_id,
            "hazard": ep.hazard,
            "severity": ep.severity,
            "fusion": fusion,
            "label": " + ".join(p.label for p in (a, v) if p),
            "audio": a.to_dict("audio") if a else None,
            "video": v.to_dict("video") if v else None,
            "correlation": ({"window_seconds": ep.rule.window, "explanation": ep.explanation}
                            if ep.rule else None),
            "alerted": ep.alerted,
            "notify": notify,
            "status": status,
        }
