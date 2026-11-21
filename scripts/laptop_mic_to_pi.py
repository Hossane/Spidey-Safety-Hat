"""Fallback mic: stream the laptop microphone to the Pi, where YAMNet runs on QNX.

The Pi's USB audio driver rejects the C270 webcam mic, so the laptop records instead:
16 kHz mono 16-bit PCM goes over SSH into `audio_detect.py --stdin` on the Pi. Inference
and alert logic stay on QNX; the laptop only captures sound. Nothing to open on the
network: SSH (the `qnxpi` entry in ~/.ssh/config) carries the audio.

Run from the project root on the laptop (needs: pip install sounddevice):
    python scripts/laptop_mic_to_pi.py                    # default mic, default options
    python scripts/laptop_mic_to_pi.py --threshold 0.3    # extra args go to audio_detect.py
    python scripts/laptop_mic_to_pi.py --list             # show input devices
    python scripts/laptop_mic_to_pi.py --device 2         # pick a mic by index
Ctrl+C stops both ends.
"""
import argparse
import queue
import shlex
import subprocess
import sys

import sounddevice as sd

RATE = 16000
BLOCK = 1600  # 0.1 s of audio per chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="qnxpi", help="SSH host (default: qnxpi from ~/.ssh/config)")
    ap.add_argument("--remote-dir", default="~/hatdetect", help="folder with audio_detect.py on the Pi")
    ap.add_argument("--device", type=int, help="input device index (see --list)")
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    args, detect_args = ap.parse_known_args()

    if args.list:
        print(sd.query_devices())
        return

    remote = f"cd {args.remote_dir} && python3 -u audio_detect.py --stdin {shlex.join(detect_args)}"
    ssh = subprocess.Popen(["ssh", "-o", "BatchMode=yes", args.host, remote], stdin=subprocess.PIPE)

    chunks = queue.Queue(maxsize=50)

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        try:
            chunks.put_nowait(bytes(indata))
        except queue.Full:  # network stalled: drop audio rather than lag behind
            pass

    dev = sd.query_devices(args.device, kind="input")
    print(f"[mic] {dev['name']} -> {args.host}:{args.remote_dir}/audio_detect.py (Ctrl+C to stop)",
          file=sys.stderr)
    try:
        with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK,
                               device=args.device, callback=on_audio):
            while ssh.poll() is None:
                try:
                    ssh.stdin.write(chunks.get(timeout=1))
                    ssh.stdin.flush()
                except queue.Empty:
                    continue
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, OSError) as e:
        print(f"[mic] connection to Pi lost: {e}", file=sys.stderr)
    finally:
        try:
            ssh.stdin.close()
        except OSError:
            pass
        try:
            ssh.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ssh.terminate()
    sys.exit(ssh.returncode or 0)


if __name__ == "__main__":
    main()
