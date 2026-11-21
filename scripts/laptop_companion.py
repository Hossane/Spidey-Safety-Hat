"""Laptop companion for Hat Detect: mic in, JBL speaker out. All detection stays on the Pi.

The Pi's QNX image has no Bluetooth stack and its USB audio driver rejects the webcam mic,
so the laptop lends both:
  - streams its microphone (16 kHz mono PCM) over SSH into `hat_detect.py` on the Pi
  - reads the `EVENT {json}` lines that come back, and for each alert plays a warning tone
    and speaks it ("Siren", "Car on your left") on the default speaker: pair the JBL with
    the laptop over Bluetooth and make it the default output device.

Run from the project root on the laptop (needs: pip install sounddevice numpy):
    python scripts/laptop_companion.py               # mic + camera + speaker
    python scripts/laptop_companion.py --no-mic      # camera only (no audio streamed)
    python scripts/laptop_companion.py --test-speaker  # just play a sample alert
Ctrl+C stops everything (the Pi stops its camera recorder and cleans up).
"""
import argparse
import json
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
    "IMPACT": "Crash", "FIRE": "Fire",
}
TONES = {  # severity -> (frequency Hz, beeps)
    "high": (1400, 3), "medium": (1000, 2), "low": (800, 1),
}
SEVERITY = {"COMBINED": "high", "ALARM": "high", "SIREN": "high", "IMPACT": "high", "FIRE": "high",
            "HORN": "medium", "VOICE": "medium", "VEHICLE": "medium", "SHARP": "medium", "PERSON": "low"}


def phrase(ev):
    g, label, pos = ev.get("group"), ev.get("label", ""), ev.get("position")
    side = {"left": "on your left", "right": "on your right", "centre": "ahead"}.get(pos, "")
    if g == "COMBINED":
        heard, seen = (label.split(" + ") + [""])[:2]
        heard = "Siren" if "siren" in heard.lower() else "Horn" if "horn" in heard.lower() else heard
        return f"Warning! {heard} and {seen} {side}".strip()
    if ev.get("source") == "audio":
        return AUDIO_PHRASES.get(g, label)
    if g == "PERSON":
        return f"Person {side}".strip()
    return f"{label.capitalize()} {side}".strip()


class Speaker:
    """Plays a tone + spoken phrase per alert, one at a time, on the default output device.
    Speech uses Windows' built-in voice through one long-running PowerShell process."""

    def __init__(self):
        self.q = queue.Queue()
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; "
              "while (($l = [Console]::In.ReadLine()) -ne $null) { $s.Speak($l); [Console]::Out.WriteLine('done') }")
        self.tts = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, severity, text):
        self.q.put((time.monotonic(), severity, text))

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
            try:
                self._tone(severity)
                self.tts.stdin.write(text.replace("\n", " ") + "\n")
                self.tts.stdin.flush()
                self.tts.stdout.readline()        # wait until spoken
            except Exception as e:  # noqa: BLE001
                print(f"[speaker] {e}", file=sys.stderr)

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="qnxpi", help="SSH host (default: qnxpi from ~/.ssh/config)")
    ap.add_argument("--remote-dir", default="~/hatdetect")
    ap.add_argument("--no-mic", action="store_true", help="don't stream the laptop mic")
    ap.add_argument("--mic", type=int, help="input device index (see python -m sounddevice)")
    ap.add_argument("--quiet-events", action="store_true", help="print alerts only, not every event")
    ap.add_argument("--test-speaker", action="store_true", help="play one sample alert and exit")
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
            if ev.get("alerted"):
                text = phrase(ev)
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
        speaker.close()


if __name__ == "__main__":
    main()
