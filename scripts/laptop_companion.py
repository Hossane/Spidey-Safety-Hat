"""Laptop companion for Hat Detect: mic in, JBL speaker out. All detection stays on the Pi.

The Pi's QNX image has no Bluetooth stack and its USB audio driver rejects the webcam mic,
so the laptop lends both:
  - streams its microphone (16 kHz mono PCM) over SSH into `hat_detect.py` on the Pi
  - reads the `EVENT {json}` lines that come back, and for each alert plays a warning tone
    and speaks it ("Siren", "Car on your left") on the default speaker: pair the JBL with
    the laptop over Bluetooth and make it the default output device.
  - stores every event in Tiger Cloud (scripts/tiger_bridge.py) when TIGER_DATABASE_URL is
    set in .env; events are buffered locally if the connection drops.
  - listens for spoken commands (scripts/voice_commands.py): "I need help" / "mayday" sends an
    SOS; "call nine one one" then "confirm" places a SIMULATED 911 call (nothing is dialled).

Run from the project root on the laptop (needs: pip install sounddevice numpy):
    python scripts/laptop_companion.py               # mic + camera + speaker
    python scripts/laptop_companion.py --no-mic      # camera only (no audio streamed)
    python scripts/laptop_companion.py --test-speaker  # just play a sample alert
Options: --camera-facing back|front (default back: the camera looks behind the wearer),
         --no-voice (no spoken commands), --no-db, --quiet-events
Ctrl+C stops everything (the Pi stops its camera recorder and cleans up).
"""
import argparse
import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd

RATE = 16000
BLOCK = 1600
STALE_SECONDS = 4.0  # don't announce alerts that waited longer than this in the queue

AUDIO_PHRASES = {
    "ALARM": "Alarm", "SIREN": "Siren", "HORN": "Horn", "VOICE": "Someone shouting",
    "IMPACT": "Crash", "FIRE": "Fire", "GLASS": "Glass breaking", "GUNSHOT": "Possible gunshot, take cover",
    "SKID": "Tires screeching", "MACHINERY": "Heavy machinery nearby",
}
TONES = {  # severity -> (frequency Hz, beeps)
    "high": (1400, 3), "medium": (1000, 2), "low": (800, 1),
}
SEVERITY = {
    "COMBINED": "high", "INJURY": "high", "FALLING_OBJECT": "high", "PERSON_DOWN": "high", "SOS": "high",
    "EMERGENCY_CALL": "high", "ALARM": "high", "SIREN": "high", "IMPACT": "high", "FIRE": "high",
    "GUNSHOT": "high", "SKID": "high", "HORN": "medium", "VOICE": "medium", "GLASS": "medium",
    "VEHICLE": "medium", "SHARP": "medium", "PERSON_CLOSE": "medium", "MOTION": "medium",
    "PERSON": "low", "MACHINERY": "low",
}
NUMBERS = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def where(pos, facing="back"):
    """Image position -> words for the wearer. A rear camera sees the wearer's left on
    the right of its image."""
    if facing == "back":
        side = {"left": "right", "right": "left"}.get(pos, pos)
        return {"left": "behind you on your left", "right": "behind you on your right",
                "centre": "right behind you"}.get(side, "behind you")
    return {"left": "on your left", "right": "on your right", "centre": "ahead"}.get(pos, "")


def movement(ev):
    m, ttc = ev.get("motion"), ev.get("ttc")
    if m == "approaching":
        return f", closing in, about {max(1, round(ttc))} seconds away" if ttc and ttc <= 6 else ", closing in"
    if m and m.startswith("crossing"):
        return ", crossing"
    if m == "moving away":
        return ", moving away"
    return ""


def phrase(ev, facing="back"):
    g, label, inc = ev.get("group"), ev.get("label", ""), ev.get("incident")
    at = where(ev.get("position"), facing)
    thing = label.capitalize()
    if g == "COMBINED":
        heard, seen = (label.split(" + ") + [""])[:2]
        h = heard.lower()
        heard = ("Siren" if "siren" in h else "Horn" if "horn" in h else "Reversing beeps" if "beep" in h
                 else "Tires screeching" if "skid" in h or "squeal" in h else heard)
        return f"Warning! {heard} and {seen} {at}{movement(ev)}"
    if g == "INJURY":
        return f"Warning! Possible injury {at}. Someone is down"
    if g == "FALLING_OBJECT":
        return f"Watch out! Something falling {at}"
    if g == "MOTION":
        return f"Movement overhead {at}" if "overhead" in label else f"Sudden movement {at}"
    if ev.get("source") == "audio":
        return AUDIO_PHRASES.get(g, label)
    if g == "PERSON":
        n = ev.get("people") or 1
        who = f"{NUMBERS.get(n, n)} people" if n > 1 else "Person"
        return f"{who} {at}{movement(ev)}"
    if g == "PERSON_DOWN":
        return (f"Warning! Person lying still {at}" if inc == "LYING" else f"Warning! Person down {at}")
    if g == "PERSON_CLOSE":
        if inc == "CLOSE":
            return f"Person very close, {at}"
        return f"Person approaching fast {at}{movement(ev).replace(', closing in', '')}"
    if g == "VEHICLE":
        if inc in ("CLOSE", "APPROACH"):
            return f"Warning! {thing} closing in {at}{movement(ev).replace(', closing in', '')}"
        return f"{thing} {at}{movement(ev)}"
    return f"{thing} {at}"


class Speaker:
    """Plays a tone + spoken phrase per alert, one at a time, on the default output device.
    Speech uses Windows' built-in voice through one long-running PowerShell process."""

    def __init__(self):
        self.q = queue.Queue()
        self.busy_until = 0.0
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; "
              "while (($l = [Console]::In.ReadLine()) -ne $null) { $s.Speak($l); [Console]::Out.WriteLine('done') }")
        self.tts = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, severity, text):
        self.q.put((time.monotonic(), severity, text))

    def muted(self):
        """True while speaking (and briefly after): the mic would hear our own voice."""
        return not self.q.empty() or time.monotonic() < self.busy_until

    def _tone(self, severity):
        hz, n = TONES.get(severity, TONES["low"])
        t = np.arange(int(0.12 * RATE)) / RATE
        beep = 0.4 * np.sin(2 * np.pi * hz * t) * np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        gap = np.zeros(int(0.08 * RATE))
        sd.play(np.concatenate([np.concatenate([beep, gap]) for _ in range(n)]).astype(np.float32), RATE)
        sd.wait()

    def _run(self):
        while True:
            t, severity, text = self.q.get()
            if time.monotonic() - t > STALE_SECONDS:
                continue
            self.busy_until = time.monotonic() + 30
            try:
                self._tone(severity)
                self.tts.stdin.write(text.replace("\n", " ") + "\n")
                self.tts.stdin.flush()
                self.tts.stdout.readline()        # wait until spoken
            except Exception as e:  # noqa: BLE001
                print(f"[speaker] {e}", file=sys.stderr)
            self.busy_until = time.monotonic() + 1.5

    def drain(self, timeout=10):
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.1)

    def close(self):
        try:
            self.tts.stdin.close()
            self.tts.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self.tts.kill()


class Emergency:
    """Handles spoken commands: SOS events and the SIMULATED 911 call (proof of concept:
    nothing is dialled; it only logs an EMERGENCY_CALL event and announces it)."""

    def __init__(self, speaker, tiger, device_id, worker):
        self.speaker, self.tiger, self.device_id, self.worker = speaker, tiger, device_id, worker
        self.recent = collections.deque(maxlen=20)        # (monotonic time, phrase) of alerts

    def note_alert(self, text):
        self.recent.append((time.monotonic(), text))

    def _event(self, group, label, score):
        from datetime import datetime, timezone
        ev = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
              "device_id": self.device_id, "source": "audio", "group": group, "label": label,
              "score": round(score, 3), "position": None, "bbox": None, "alerted": True}
        if self.tiger:
            self.tiger.add(ev)
        return ev

    def summary(self):
        now = time.monotonic()
        recent = [f"{text} ({int(now - t)} s ago)" for t, text in self.recent if now - t <= 120][-3:]
        name, zone = self.worker
        return (f"{name} ({self.device_id}) at {zone} needs emergency help"
                + (". Recent alerts: " + "; ".join(recent) if recent else ""))

    def __call__(self, kind, text, conf):
        print(f"[voice] heard {text!r} ({conf:.2f}) -> {kind}", file=sys.stderr)
        if kind == "SOS":
            self._event("SOS", f"voice: {text}", conf)
            print(f"!! SOS  {self.worker[0]} asked for help", flush=True)
            self.speaker.say("high", "Help requested. The site dashboard has been alerted.")
        elif kind == "CALL_REQUESTED":
            self.speaker.say("high", "Emergency call requested. Say confirm to call nine one one, or say cancel.")
        elif kind == "CALL_CANCELLED":
            self.speaker.say("low", "Call cancelled.")
        elif kind == "CALL_CONFIRMED":
            msg = self.summary()
            self._event("EMERGENCY_CALL", f"SIMULATED 911 call: {msg}", conf)
            bar = "=" * 72
            print(f"\n{bar}\n  SIMULATED 911 CALL (proof of concept: nothing was dialled)\n"
                  f"  Message to dispatch: {msg}\n{bar}\n", flush=True)
            self.speaker.say("high", "Simulated nine one one call placed. This is a demo; no real call was made.")


def worker_info(device_id):
    """(name, zone) for this hat from the Tiger workers table, if reachable."""
    try:
        import tiger_bridge
        url = tiger_bridge.database_url()
        if url:
            with tiger_bridge.connect(url) as conn:
                row = conn.execute("SELECT name, zone FROM workers WHERE device_id = %s", (device_id,)).fetchone()
                if row:
                    return row[0], row[1]
    except Exception:  # noqa: BLE001
        pass
    return device_id, "unknown location"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="qnxpi", help="SSH host (default: qnxpi from ~/.ssh/config)")
    ap.add_argument("--remote-dir", default="~/hatdetect")
    ap.add_argument("--no-mic", action="store_true", help="don't stream the laptop mic")
    ap.add_argument("--mic", type=int, help="input device index (see python -m sounddevice)")
    ap.add_argument("--quiet-events", action="store_true", help="print alerts only, not every event")
    ap.add_argument("--test-speaker", action="store_true", help="play one sample alert and exit")
    ap.add_argument("--no-db", action="store_true", help="don't store events in Tiger Cloud")
    ap.add_argument("--no-voice", action="store_true", help="don't listen for spoken commands")
    ap.add_argument("--camera-facing", choices=["back", "front"], default="back",
                    help="where the hat camera points (default back: it watches behind the wearer)")
    args = ap.parse_args()

    out = sd.query_devices(kind="output")
    print(f"[speaker] output device: {out['name']}  (pair the JBL and make it the default output)",
          file=sys.stderr)
    speaker = Speaker()
    if args.test_speaker:
        speaker.say("high", "Warning! Vehicle horn and car on your left")
        time.sleep(0.5)
        speaker.drain()
        time.sleep(4)
        speaker.close()
        return

    tiger = None
    if not args.no_db:
        import tiger_bridge
        url = tiger_bridge.database_url()
        if url:
            tiger = tiger_bridge.TigerWriter(url, log=lambda m: print(m, file=sys.stderr))
        else:
            print("[tiger] TIGER_DATABASE_URL not set in .env; events won't be stored", file=sys.stderr)

    device_id = os.environ.get("DEVICE_ID", "hat-01")
    emergency = Emergency(speaker, tiger, device_id, worker_info(device_id))
    voice = None
    if not args.no_voice:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import voice_commands
        voice = voice_commands.VoiceCommands(emergency, muted=speaker.muted,
                                             log=lambda m: print(m, file=sys.stderr))

    remote = f"cd {args.remote_dir} && python3 -u hat_detect.py" + (" --no-audio --stop-on-stdin-eof" if args.no_mic else "")
    ssh = subprocess.Popen(["ssh", "-o", "BatchMode=yes", args.host, remote],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def read_events():
        for raw in iter(ssh.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line.startswith("EVENT "):
                print(line)
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if tiger:
                tiger.add(ev)
            if ev.get("alerted"):
                text = phrase(ev, args.camera_facing)
                emergency.note_alert(text)
                print(f"!! ALERT {ev['source']:<8} {ev['label']} {ev['score']:.2f} "
                      f"{ev.get('position') or ''} -> \"{text}\"")
                speaker.say(SEVERITY.get(ev.get("group"), "low"), text)
            elif not args.quiet_events:
                print(f"   event {ev['source']:<8} {ev['label']} {ev['score']:.2f} {ev.get('position') or ''}")
    reader = threading.Thread(target=read_events, daemon=True)
    reader.start()

    chunks = queue.Queue(maxsize=50)

    def on_audio(indata, frames, time_info, status):
        try:
            chunks.put_nowait(bytes(indata))
        except queue.Full:
            pass

    try:
        if args.no_mic:
            while ssh.poll() is None:
                time.sleep(0.2)
        else:
            mic = sd.query_devices(args.mic, kind="input")
            print(f"[mic] {mic['name']} -> {args.host} (Ctrl+C to stop)", file=sys.stderr)
            with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK,
                                   device=args.mic, callback=on_audio):
                while ssh.poll() is None:
                    try:
                        ssh.stdin.write(chunks.get(timeout=1))
                    except queue.Empty:
                        continue
    except KeyboardInterrupt:
        print("\n[laptop] stopping...", file=sys.stderr)
    except (BrokenPipeError, OSError) as e:
        print(f"[laptop] connection to Pi lost: {e}", file=sys.stderr)
    finally:
        try:
            ssh.stdin.close()       # Pi sees end of audio and shuts down cleanly
        except OSError:
            pass
        try:
            ssh.wait(timeout=12)
        except subprocess.TimeoutExpired:
            ssh.terminate()
        reader.join(timeout=2)
        if voice:
            voice.close()
        speaker.close()
        if tiger:
            tiger.close()


if __name__ == "__main__":
    main()
