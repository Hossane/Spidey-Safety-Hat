"""Spoken commands for Hat Detect, recognised offline on the laptop (Windows' built-in
speech recogniser, a fixed phrase list, no internet).

    "help help" / "I need help"                           -> SOS alert to the site dashboard
    "call nine one one"                                  -> asks for confirmation
        "confirm" (within 10 s)                          -> SIMULATED 911 call (proof of concept)
        "cancel"                                         -> cancelled

The 911 call is a SIMULATION: nothing is dialled and nothing leaves the laptop except the
EMERGENCY_CALL event stored in Tiger. It shows what the hat would send to dispatch.

Test without a mic (synthesises the phrase, then recognises it):
    python scripts/voice_commands.py --selftest
Listen on the default mic and print what it hears:
    python scripts/voice_commands.py
"""
import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time

PHRASES = {
    "SOS": ["help help", "i need help"],   # plain "help" / "help me" sound too much like "hello"
    "CALL": ["call nine one one", "call 911"],
    "CONFIRM": ["confirm", "confirm call"],
    "CANCEL": ["cancel", "cancel call"],
    # Decoys: words that sound like a command. The recogniser can only pick from this list,
    # so without them "hello" or "yellow" comes back as "help". They match nothing.
    "NOISE": ["hello", "hello there", "yellow", "kelp", "hell", "held", "helmet", "helpful", "well", "hey",
              "hold on", "health", "self", "shelf", "elf", "yell", "help out", "help me out",
              "can sell", "cancer", "council", "counsel", "candle", "camp", "concern", "conference",
              "confirmed", "comfort", "conform", "information", "call me", "call you", "call back",
              "nine", "one", "okay", "yes", "no", "thanks", "what", "the", "and", "I",
              "help", "help me", "hello hello", "hey hello", "somebody", "somebody hold this", "i need",
              "i need that", "need a hand", "some help", "save that", "safe", "safety"],
}
MIN_CONFIDENCE = {"NOISE": 2.0, "SOS": 0.85, "CALL": 0.8, "CONFIRM": 0.7, "CANCEL": 0.7}
CONFIRM_WINDOW_S = 10.0

# PowerShell host for System.Speech: prints "REC|phrase|confidence" per recognition.
PS = r"""
Add-Type -AssemblyName System.Speech
$r = New-Object System.Speech.Recognition.SpeechRecognitionEngine([Globalization.CultureInfo]'en-US')
$c = New-Object System.Speech.Recognition.Choices
foreach ($p in @(PHRASES)) { $c.Add($p) }
$gb = New-Object System.Speech.Recognition.GrammarBuilder($c)
$gb.Culture = [Globalization.CultureInfo]'en-US'
$r.LoadGrammar((New-Object System.Speech.Recognition.Grammar($gb)))
$r.BabbleTimeout = [TimeSpan]::FromSeconds(0)
if ('WAVFILE' -ne '') { $r.SetInputToWaveFile('WAVFILE') } else { $r.SetInputToDefaultAudioDevice() }
[Console]::Out.WriteLine('READY'); [Console]::Out.Flush()
while ($true) {
  try { $res = $r.Recognize([TimeSpan]::FromSeconds(5)) } catch { break }
  if ($res -ne $null) { [Console]::Out.WriteLine('REC|' + $res.Text + '|' + $res.Confidence); [Console]::Out.Flush() }
  elseif ('WAVFILE' -ne '') { break }
}
"""


def kind_of(text):
    t = text.lower()
    for kind, phrases in PHRASES.items():
        if t in phrases:
            return kind
    return None


def start_recogniser(wav=None):
    phrases = ",".join(f"'{p}'" for ps in PHRASES.values() for p in ps)
    script = PS.replace("PHRASES", phrases).replace("WAVFILE", wav or "")
    return subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)


class VoiceCommands:
    """Listens on the default mic and calls on_command(kind, text, confidence) for SOS,
    CALL_REQUESTED, CALL_CONFIRMED (the simulated 911 call) and CALL_CANCELLED.
    `muted()` returns True while the laptop is speaking, so the hat can't hear its own
    prompts ("say confirm ...") and confirm itself."""

    def __init__(self, on_command, muted=lambda: False, log=print):
        self.on_command, self.muted, self.log = on_command, muted, log
        self.pending_call = None          # time a 911 call was requested
        self.proc = start_recogniser()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line == "READY":
                self.log("[voice] listening for: help help, I need help, call nine one one (then confirm)")
                continue
            if not line.startswith("REC|"):
                continue
            _, text, conf = line.split("|", 2)
            conf = float(conf.replace(",", "."))
            kind = kind_of(text)
            if not kind or conf < MIN_CONFIDENCE[kind]:
                continue
            if self.muted():
                continue
            self.handle(kind, text, conf)

    def handle(self, kind, text, conf):
        now = time.monotonic()
        if kind == "SOS":
            self.on_command("SOS", text, conf)
        elif kind == "CALL":
            self.pending_call = now
            self.on_command("CALL_REQUESTED", text, conf)
        elif kind == "CONFIRM":
            if self.pending_call and now - self.pending_call <= CONFIRM_WINDOW_S:
                self.pending_call = None
                self.on_command("CALL_CONFIRMED", text, conf)
        elif kind == "CANCEL":
            if self.pending_call:
                self.pending_call = None
                self.on_command("CALL_CANCELLED", text, conf)

    def close(self):
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass


def recognise_text(text):
    """Speak `text` into a wav file with Windows TTS, then run the recogniser on it."""
    wav = os.path.join(tempfile.mkdtemp(), "phrase.wav")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.SetOutputToWaveFile('{wav}'); $s.Speak('{text.replace(chr(39), chr(39) * 2)}'); $s.Dispose()"], check=True)
    out = start_recogniser(wav).communicate(timeout=60)[0]
    os.remove(wav)
    return [(l.split("|")[1], float(l.split("|")[2].replace(",", "."))) for l in out.splitlines() if l.startswith("REC|")]


def selftest():
    events = []
    vc = VoiceCommands.__new__(VoiceCommands)          # no live mic: drive handle() directly
    vc.on_command, vc.muted, vc.pending_call = (lambda k, t, c: events.append(k)), (lambda: False), None
    ok = True
    for said, expect in [("I need help", "SOS"), ("help help", "SOS"),
                         ("call nine one one", "CALL"), ("confirm", "CONFIRM"), ("cancel", "CANCEL")]:
        got = recognise_text(said)
        kind = kind_of(got[0][0]) if got else None
        good = kind == expect and got[0][1] >= MIN_CONFIDENCE[kind]
        ok &= good
        print(f"  said {said!r:<22} heard {got[0] if got else None}  -> {kind}  {'ok' if good else 'FAIL'}")
    for k in ["CALL", "CONFIRM", "CONFIRM", "CALL", "CANCEL", "CONFIRM", "SOS"]:
        vc.handle(k, k.lower(), 0.9)
    expect = ["CALL_REQUESTED", "CALL_CONFIRMED", "CALL_REQUESTED", "CALL_CANCELLED", "SOS"]
    print(f"  confirm flow: {events}  {'ok' if events == expect else 'FAIL'}")
    return ok and events == expect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    vc = VoiceCommands(lambda k, t, c: print(f"COMMAND {k}: {t!r} ({c:.2f})"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        vc.close()


if __name__ == "__main__":
    main()
