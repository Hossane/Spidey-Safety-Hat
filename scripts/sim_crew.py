"""Simulated crew for the dashboard's site overview (proof of concept for scale).

Adds made-up workers (device_id sim-01, sim-02, ...) and hazard detections in Tiger
Cloud, so the overview can show a whole site. They are marked simulated and only show
on the dashboard when its "Simulated crew" switch is on; real hats (hat-01, ...) are
never touched.

    python scripts/sim_crew.py add                 # 12 workers, last 8 hours
    python scripts/sim_crew.py add --workers 40 --hours 24
    python scripts/sim_crew.py add --live          # ...then keep adding detections live (Ctrl+C stops)
    python scripts/sim_crew.py remove              # delete all simulated workers and their data
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiger_bridge  # noqa: E402

PREFIX = "sim-"
FIRST = ["Alex", "Priya", "Jordan", "Mei", "Sam", "Omar", "Lucia", "Ben", "Aisha", "Tom", "Nadia", "Raj",
         "Chloe", "Diego", "Hana", "Kwame", "Sofia", "Liam", "Yusuf", "Grace"]
LAST = ["Chen", "Patel", "Nguyen", "Okafor", "Smith", "Garcia", "Kim", "Martin", "Haddad", "Silva"]

# (role, zone, incidents per hour, hazard mix: (source, hazard, label, weight))
ROLES = [
    ("Forklift operator", "Loading dock", 70, [
        ("video", "PERSON", "person", 30), ("video", "VEHICLE", "truck", 25), ("audio", "ALARM", "Reversing beeps", 15),
        ("audio", "HORN", "Vehicle horn, car horn, honking", 12), ("audio", "IMPACT", "Smash, crash", 2)]),
    ("Yard marshal", "Truck yard", 90, [
        ("video", "VEHICLE", "truck", 35), ("video", "VEHICLE", "car", 15), ("audio", "HORN", "Air horn, truck horn", 15),
        ("audio", "ALARM", "Reversing beeps", 15), ("video", "PERSON", "person", 15), ("audio", "SIREN", "Siren", 1)]),
    ("Welder", "Fabrication shop", 40, [
        ("video", "PERSON", "person", 30), ("audio", "IMPACT", "Bang", 6), ("audio", "FIRE", "Fire", 2),
        ("audio", "ALARM", "Smoke detector, smoke alarm", 2), ("audio", "VOICE", "Shout", 4)]),
    ("Picker", "Warehouse aisles", 50, [
        ("video", "PERSON", "person", 40), ("video", "VEHICLE", "truck", 6), ("audio", "ALARM", "Reversing beeps", 6),
        ("video", "SHARP", "knife", 3), ("audio", "VOICE", "Shout", 2)]),
    ("Maintenance tech", "Plant room", 25, [
        ("video", "PERSON", "person", 20), ("audio", "ALARM", "Buzzer", 4), ("video", "SHARP", "scissors", 3),
        ("audio", "IMPACT", "Smash, crash", 2)]),
    ("Supervisor", "Site office", 15, [
        ("video", "PERSON", "person", 40), ("audio", "VOICE", "Shout", 2)]),
]


def make_crew(n, seed=7):
    rng = random.Random(seed)
    crew = []
    for i in range(n):
        role, zone, rate, mix = ROLES[i % len(ROLES)]
        crew.append({"device_id": f"{PREFIX}{i + 1:02d}",
                     "name": f"{FIRST[i % len(FIRST)]} {LAST[(i * 3 + i // len(FIRST)) % len(LAST)]}",
                     "role": role, "zone": zone, "mix": mix,
                     # some workers have a rougher shift than others
                     "rate": rate * rng.uniform(0.5, 1.6)})
    return crew


class Hat:
    """Generates detections like the real hat: each incident is a burst (video ~5 fps,
    audio every 0.5 s) and only the first per hazard in a 3 s cooldown alerts."""

    def __init__(self, worker, rng):
        self.w, self.rng, self.last_alert = worker, rng, {}

    def _event(self, t, src, hazard, label, pos, score):
        prev = self.last_alert.get(hazard)
        alerted = prev is None or (t - prev).total_seconds() >= 3
        if alerted:
            self.last_alert[hazard] = t
        r = self.rng
        return {"time": t.isoformat(), "device_id": self.w["device_id"], "source": src, "group": hazard,
                "label": label, "score": round(score, 3), "position": pos, "alerted": alerted,
                "bbox": [r.randint(0, 300), r.randint(0, 200), r.randint(320, 640), r.randint(240, 480)] if pos else None}

    def incident(self, t):
        r, mix = self.rng, self.w["mix"]
        src, hazard, label, _ = r.choices(mix, [m[3] for m in mix])[0]
        pos = r.choice(["left", "centre", "right"]) if src == "video" else None
        n, step = (r.randint(3, 20), 0.2) if src == "video" else (r.randint(1, 4), 0.5)
        base = r.uniform(0.5, 0.9)
        out = [self._event(t + timedelta(seconds=i * step), src, hazard, label, pos,
                           min(0.99, max(0.4, base + r.uniform(-0.08, 0.08)))) for i in range(n)]
        if hazard == "HORN" and r.random() < 0.35:   # horn + vehicle in view -> combined alert
            side = r.choice(["left", "right"])
            out.append(self._event(t + timedelta(seconds=0.4), "video", "VEHICLE", "truck", side, r.uniform(0.55, 0.85)))
            out.append(self._event(t + timedelta(seconds=0.5), "combined", "COMBINED", f"{label} + truck", side, 0.6))
        return out


def register(conn, crew):
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO workers (device_id, name, role, zone, simulated) VALUES (%s, %s, %s, %s, true)
                           ON CONFLICT (device_id) DO UPDATE SET name = EXCLUDED.name, role = EXCLUDED.role,
                           zone = EXCLUDED.zone, simulated = true""",
                        [(w["device_id"], w["name"], w["role"], w["zone"]) for w in crew])
    conn.commit()


def cmd_add(url, args):
    crew = make_crew(args.workers)
    rng = random.Random(42)
    hats = [Hat(w, rng) for w in crew]
    now = datetime.now(timezone.utc).replace(microsecond=0)   # whole seconds: re-runs dedup
    events = []
    for h in hats:
        t = now - timedelta(hours=args.hours)
        while True:
            t += timedelta(seconds=rng.expovariate(h.w["rate"] / 3600))
            if t >= now:
                break
            events.extend(h.incident(t))
    with tiger_bridge.connect(url) as conn:
        register(conn, crew)
        for i in range(0, len(events), 5000):
            tiger_bridge.insert_events(conn, events[i:i + 5000])
            print(f"  {min(i + 5000, len(events)):,} / {len(events):,} detections", flush=True)
    print(f"Added {len(crew)} simulated workers and {len(events):,} detections over the last {args.hours:g} h.")
    if args.live:
        live(url, hats, rng)


def live(url, hats, rng):
    print("Adding live detections (Ctrl+C to stop)...", flush=True)
    writer = tiger_bridge.TigerWriter(url, log=lambda m: print(m, flush=True))
    try:
        while True:
            time.sleep(1)
            t = datetime.now(timezone.utc).replace(microsecond=0)
            for h in hats:
                if rng.random() < h.w["rate"] / 3600:
                    for ev in h.incident(t):
                        writer.add(ev)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()


def cmd_remove(url, args):
    with tiger_bridge.connect(url) as conn:
        n = conn.execute("DELETE FROM detections WHERE device_id LIKE %s", (PREFIX + "%",)).rowcount
        w = conn.execute("DELETE FROM workers WHERE simulated OR device_id LIKE %s", (PREFIX + "%",)).rowcount
        conn.commit()
    with tiger_bridge.connect(url, autocommit=True) as conn:   # drop their rolled-up minutes too
        conn.execute("CALL refresh_continuous_aggregate('detections_per_minute', NULL, NULL)")
    print(f"Removed {w} simulated workers and {n:,} detections.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--live", action="store_true", help="keep adding detections in real time")
    sub.add_parser("remove")
    args = ap.parse_args()
    url = tiger_bridge.database_url()
    if not url:
        sys.exit("TIGER_DATABASE_URL is not set (see docs/TIGER_DATA.md).")
    {"add": cmd_add, "remove": cmd_remove}[args.cmd](url, args)


if __name__ == "__main__":
    main()
