"""Replay a capture from scripts/capture_tracks.py through the tracker (laptop, no camera).

    python scripts/replay_tracks.py tracks.jsonl
    python scripts/replay_tracks.py tracks.jsonl --set reid_min=0.5 --set close_area=0.5

Prints every tracker incident and which ones would alert.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tracking  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="tracker setting")
    args = ap.parse_args()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hat_config.json")) as f:
        cfg = dict(json.load(f)["video"].get("tracking", {}))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = float(v)
    tr = tracking.Tracker.from_config(cfg)
    alerts = 0
    with open(args.capture) as f:
        for line in f:
            r = json.loads(line)
            dets = [(d["name"], d["score"], tuple(d["box"]), d["group"]) for d in r["dets"]]
            sigs = [np.array(d["sig"], np.float32) for d in r["dets"]]
            _, incidents = tr.update(dets, None, r["t"], sigs=sigs, size=(r["w"], r["h"]))
            for kind, t in incidents:
                g = tracking.incident_group(kind, t)
                alerts += bool(g)
                print(f"{r['t']:6.1f}s  {kind:<8} {t.name} #{t.id:<3} {'ALERT ' + g if g else '(no alert)'}")
    print(f"{alerts} alerts, {tr.next_id - 1} tracks created")


if __name__ == "__main__":
    main()
