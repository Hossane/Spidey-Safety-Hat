"""Laptop companion for Hat Detect: mic in, JBL speaker out. All detection stays on the Pi.

The Pi's QNX image has no Bluetooth stack and its USB audio driver rejects the webcam mic,
so the laptop lends both:
  - streams its microphone (16 kHz mono PCM) over SSH into `hat_detect.py` on the Pi
  - reads the unified hazard events that come back as `EVENT {json}` lines. The Pi's
    fusion engine has already decided which ones to announce ("notify": true); for those
    the laptop plays a warning tone and speaks a phrase ("Warning! Vehicle hazard on your
    left") on the default speaker. Pair the JBL with the laptop over Bluetooth and make it
    the default output device.
  - stores every event in Tiger Cloud (scripts/tiger_bridge.py) when TIGER_DATABASE_URL is
    set in .env; events are buffered locally if the connection drops.

Alerts play one at a time, most severe first. A more severe alert cuts off a less severe
one that is playing; alerts that waited more than STALE_SECONDS are dropped.
Speech: macOS `say`, Windows System.Speech (PowerShell), Linux espeak-ng/espeak; if none
is available the phrase is printed instead.

Run from the project root on the laptop (needs: pip install sounddevice numpy):
    python scripts/laptop_companion.py               # mic + camera + speaker
    python scripts/laptop_companion.py --no-mic      # camera only (no audio streamed)
    python scripts/laptop_companion.py --test-speaker  # just play a sample alert
Ctrl+C stops everything (the Pi stops its camera recorder and cleans up).
"""
import argparse
import itertools
import json
import queue
import shutil
import subprocess
import sys
import threading
import time

RATE = 16000
BLOCK = 1600
STALE_SECONDS = 4.0  # don't announce alerts that waited longer than this in the queue
RANK = {"low": 0, "medium": 1, "high": 2}

AUDIO_PHRASES = {
    "ALARM": "Alarm", "SIREN": "Siren", "HORN": "Horn", "VOICE": "Someone shouting",
    "IMPACT": "Crash", "FIRE": "Fire",
}
HAZARD_PHRASES = {  # multimodal hazards (fusion rules in hat_config.json)
    "VEHICLE_HAZARD": "Warning! Vehicle hazard",
    "WORKER_DISTRESS": "Warning! Worker in distress",
}
SIDES = {"left": "on your left", "right": "on your right", "centre": "ahead"}
TONES = {  # severity -> (frequency Hz, beeps)
    "high": (1400, 3), "medium": (1000, 2), "low": (800, 1),
}


def phrase(ev):
    """What to say for one unified hazard event."""
    hazard = ev.get("hazard") or ""
    video = ev.get("video") or {}
    side = SIDES.get(video.get("position"), "")
    fusion = ev.get("fusion")
    if fusion == "MULTIMODAL":
        text = HAZARD_PHRASES.get(hazard) or "Warning! " + hazard.replace("_", " ").capitalize()
    elif fusion == "AUDIO_ONLY":
        text = AUDIO_PHRASES.get(hazard) or (ev.get("audio") or {}).get("label") or hazard.capitalize()
        side = ""                                   # no direction from a single mic
    elif hazard == "PERSON":
        text = "Person"
    else:
        text = (video.get("label") or hazard or "Hazard").capitalize()
    return f"{text} {side}".strip()


# ---------------------------------------------------------------- speech backends
# Each has start(text) (begin speaking, return quickly), wait() (block until finished or
# cancelled), cancel() (stop now; safe from another thread) and close().

class CommandVoice:
    """One process per phrase: macOS `say`, or espeak on Linux. Cancel = terminate it."""

    def __init__(self, cmd):
        self.cmd, self.proc = cmd, None

    def start(self, text):
        self.proc = subprocess.Popen(self.cmd + [text], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def wait(self):
        if self.proc:
            self.proc.wait()

    def cancel(self):
        p = self.proc
        if p and p.poll() is None:
            p.terminate()

    def close(self):
        self.cancel()


class PowerShellVoice:
    """Windows' built-in voice through one long-running PowerShell process (no start-up
    delay per phrase). Cancel kills it; the next phrase starts a fresh one."""

    SCRIPT = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; "
              "while (($l = [Console]::In.ReadLine()) -ne $null) { $s.Speak($l); [Console]::Out.WriteLine('done') }")

    def __init__(self):
        self.proc = None
        self._spawn()

    def _spawn(self):
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", self.SCRIPT],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def start(self, text):
        if self.proc.poll() is not None:
            self._spawn()
        self.proc.stdin.write(text.replace("\n", " ") + "\n")
        self.proc.stdin.flush()

    def wait(self):
        self.proc.stdout.readline()                 # "done", or "" if cancelled

    def cancel(self):
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class PrintVoice:
    """No speech engine found: print the phrase."""

    def start(self, text):
        print(f"[voice] {text}", file=sys.stderr)

    def wait(self):
        pass

    def cancel(self):
        pass

    def close(self):
        pass


def make_voice():
    if sys.platform == "darwin" and shutil.which("say"):
        return CommandVoice(["say"])
    if sys.platform == "win32":
        return PowerShellVoice()
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return CommandVoice([exe])
    print("[speaker] no speech engine found; alerts will be printed", file=sys.stderr)
    return PrintVoice()


class Tones:
    """Warning beeps on the default output device (sounddevice)."""

    def __init__(self):
        import numpy as np
        import sounddevice as sd
        self.np, self.sd = np, sd

    def play(self, severity):
        np = self.np
        hz, n = TONES.get(severity, TONES["low"])
        t = np.arange(int(0.12 * RATE)) / RATE
        beep = 0.4 * np.sin(2 * np.pi * hz * t) * np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        gap = np.zeros(int(0.08 * RATE))
        self.sd.play(np.concatenate([np.concatenate([beep, gap]) for _ in range(n)]).astype(np.float32), RATE)
        self.sd.wait()

    def stop(self):
        self.sd.stop()


class Speaker:
    """Plays a tone + spoken phrase per alert, one at a time, most severe first. A more
    severe alert interrupts a less severe one that is playing (the interrupted one is
    dropped, not replayed)."""

    def __init__(self, voice, tones, stale=STALE_SECONDS, clock=time.monotonic):
        self.voice, self.tones, self.stale, self.clock = voice, tones, stale, clock
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()
        self.lock = threading.Lock()
        self.current = None          # severity being played, or None when idle
        self.cut = False             # current alert was interrupted
        self.played = []             # (severity, text, interrupted) history, for tests/logs
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, severity, text):
        with self.lock:
            self.q.put((-RANK.get(severity, 0), next(self.seq), self.clock(), severity, text))
            if self.current is not None and RANK.get(severity, 0) > RANK.get(self.current, 0):
                self.cut = True
                try:
                    self.tones.stop()
                    self.voice.cancel()
                except Exception as e:  # noqa: BLE001
                    print(f"[speaker] interrupt failed: {e}", file=sys.stderr)

    def _run(self):
        while True:
            _, _, t, severity, text = self.q.get()
            if self.clock() - t > self.stale:
                continue
            with self.lock:
                self.current, self.cut = severity, False
            try:
                self.tones.play(severity)
                with self.lock:              # start speech under the lock so say() can't miss it
                    started = not self.cut
                    if started:
                        self.voice.start(text)
                if started:
                    self.voice.wait()
            except Exception as e:  # noqa: BLE001
                print(f"[speaker] {e}", file=sys.stderr)
            finally:
                with self.lock:
                    self.played.append((severity, text, self.cut))
                    self.current = None

    def busy(self):
        with self.lock:
            return self.current is not None or not self.q.empty()

    def drain(self, timeout=10):
        end = time.time() + timeout
        while self.busy() and time.time() < end:
            time.sleep(0.05)

    def close(self):
        self.voice.close()


def describe(ev):
    parts = []
    for side in ("audio", "video"):
        p = ev.get(side)
        if p:
            parts.append(f"{p['label']} {p['score']:.2f}" + (f" {p['position']}" if p.get("position") else ""))
    return " + ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="qnxpi", help="SSH host (default: qnxpi from ~/.ssh/config)")
    ap.add_argument("--remote-dir", default="~/hatdetect")
    ap.add_argument("--no-mic", action="store_true", help="don't stream the laptop mic")
    ap.add_argument("--mic", type=int, help="input device index (see python -m sounddevice)")
    ap.add_argument("--quiet-events", action="store_true", help="print alerts only, not every event")
    ap.add_argument("--test-speaker", action="store_true", help="play one sample alert and exit")
    ap.add_argument("--no-db", action="store_true", help="don't store events in Tiger Cloud")
    args = ap.parse_args()

    import sounddevice as sd
    out = sd.query_devices(kind="output")
    print(f"[speaker] output device: {out['name']}  (pair the JBL and make it the default output)",
          file=sys.stderr)
    speaker = Speaker(make_voice(), Tones())
    if args.test_speaker:
        speaker.say("high", "Warning! Vehicle hazard on your left")
        time.sleep(0.5)
        speaker.drain()
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
            if ev.get("notify"):
                text = phrase(ev)
                print(f"!! ALERT {ev['severity']:<6} {ev['fusion']:<11} {ev['hazard']:<15} "
                      f"{describe(ev)} -> \"{text}\"")
                if ev.get("correlation"):
                    print(f"         {ev['correlation']['explanation']}")
                speaker.say(ev["severity"], text)
            elif not args.quiet_events:
                print(f"   {ev['status']:<6} {ev['fusion']:<11} {ev['hazard']:<15} {describe(ev)}")
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
        if tiger:
            tiger.close()


if __name__ == "__main__":
    main()
