# Hat Detect ("safehat")

Hack Western 13 · QNX track

A hard hat that watches the area behind the wearer and listens for danger. A camera and
two TensorFlow Lite models run on a **Raspberry Pi 5 running QNX 8.0**. When something
needs attention, the hat warns the wearer out loud: a person or vehicle behind them, a
co-worker who has collapsed, an alarm, a horn, breaking glass. Every detection is stored in
**Tiger Data** (Tiger Cloud / TimescaleDB), and a site dashboard shows what every hat has
seen.

```
                       Raspberry Pi 5 · QNX 8.0 (all detection happens here)
  webcam ─► QNX Camera ─► EfficientDet-Lite0 ─► tracking.py / motion.py ─┐
                                                                          ├─► alert logic ─► GPIO buzzer
  laptop mic ─(SSH)──────────────────────────► YAMNet (audio_detect.py) ─┘   (hat_detect.py)
                                                                               │ EVENT lines over SSH
                                                                               ▼
                       Laptop: scripts/laptop_companion.py
                         speaks alerts on the speaker (JBL) · voice commands (SOS, simulated 911)
                         stores every event in Tiger Cloud ─────► scripts/dashboard.py (localhost:8050)
```

## What it does

**Sees** (camera on the Pi, EfficientDet-Lite0 + `tracking.py` + `motion.py`)
- **People and vehicles behind you**, announced once each. Someone who stays behind you, or
  steps out of view and comes back, is recognised by their clothing colours and doesn't
  keep beeping.
- **Speed and direction** for every track: approaching, moving away, crossing or still.
  For anything approaching, it estimates how many seconds until it reaches you.
- **Person down:** a co-worker who falls, or is lying still on the floor.
- **Sudden movement** the object model has no label for, such as a falling load or a
  swinging hook.
- **Sharp objects:** knives and scissors.

**Hears** (YAMNet on the Pi, fed by the laptop mic)
- Alarms and reversing beeps, sirens, horns, shouting, crashes and bangs, fire, breaking
  glass, gunshots, tires screeching and heavy machinery.

**Combines** the two: a horn heard while a truck is in view becomes one high-priority
alert. A crash followed by a person down becomes a possible injury. A crash together with
sudden movement becomes a falling object.

**Responds to the wearer's voice** (laptop, offline): "help help" or "I need help" sends an
SOS to the site dashboard. "Call nine one one", then "confirm", places a **simulated** 911
call: nothing is dialled.

**Reports to the site:** the dashboard has a site overview of every worker ranked by risk,
a per-worker view with an incident timeline, emergency banners, "What the data suggests"
insight cards and a supervisor activity log. A simulated crew shows how it scales past a
single hat.

All detection runs on the Pi, so the hat keeps working with no network. The laptop only
lends the mic and speaker (the QNX image has no Bluetooth, and its USB audio driver
rejects the webcam mic) and forwards events to the database.

## Running it

You need the Pi (set up as in [models/README.md](models/README.md)), a laptop with Python 3,
and optionally a Bluetooth speaker paired with the laptop and set as its default output.

**1. Connect to the Pi.** Add entries like these to `~/.ssh/config` on the laptop.
`Ciphers` and `MACs` are needed: without them, Windows OpenSSH fails with "Corrupted MAC
on input".

```
Host qnxpi                       # over Wi-Fi
    HostName <pi-wifi-address>
    User qnxuser
    IdentityFile ~/.ssh/id_ed25519_qnxpi
    Ciphers aes128-ctr
    MACs hmac-sha2-256

Host qnxpi-eth                   # over an Ethernet cable to the laptop
    HostName <pi-hostname>.local  # the Pi announces its name, so this follows address changes
    User qnxuser
    IdentityFile ~/.ssh/id_ed25519_qnxpi
    Ciphers aes128-ctr
    MACs hmac-sha2-256
```

For the cable, turn on Windows **Internet Connection Sharing** for the Ethernet adapter, so
the Pi gets an address from the laptop. The cable is more reliable than event Wi-Fi.

**2. Copy the code to the Pi** (the hat runs from `~/hatdetect`):

```powershell
scp hat_detect.py audio_detect.py vision_detect.py tracking.py motion.py hat_config.json qnxpi-eth:hatdetect/
```

**3. Start the hat** from the laptop:

```powershell
pip install sounddevice numpy "psycopg[binary]" python-dotenv
python scripts/laptop_companion.py --host qnxpi-eth --quiet-events
```

This starts `hat_detect.py` on the Pi over SSH, streams the laptop mic to it, speaks every
alert and stores every event in Tiger Cloud. Useful options:

| Option | What it does |
|---|---|
| `--test-speaker` | Plays one sample alert and exits (checks the speaker) |
| `--no-mic` | Camera only, no audio streamed |
| `--no-voice` | Don't listen for spoken commands |
| `--no-db` | Don't store events in Tiger Cloud |
| `--camera-facing front` | The camera looks forward (default `back`: left and right are flipped for the wearer) |

On the Pi alone, without the laptop: `python3 hat_detect.py --no-audio` (camera and GPIO
buzzer only).

`requirements.txt` has the extra packages (TensorFlow, OpenCV) for testing the models on
the laptop. The companion doesn't need them.

**4. Open the dashboard.** First set up Tiger Cloud ([docs/TIGER_DATA.md](docs/TIGER_DATA.md)), then:

```powershell
python scripts/dashboard.py      # http://localhost:8050
```

## Repo layout

```
hat_detect.py        the on-device app: audio and video threads, alert logic, GPIO, events
audio_detect.py      YAMNet sound classifier and the safety sound groups
vision_detect.py     EfficientDet-Lite0 object detector and the QNX Camera reader
tracking.py          tracks people and vehicles: re-identification, motion, person down
motion.py            sudden-movement detector (frame differencing at 80x60)
hat_config.json      thresholds, sound and object groups, cooldowns, severities, GPIO pin
scripts/
  laptop_companion.py  mic in, speaker out, voice commands, events into Tiger Cloud
  voice_commands.py    offline Windows speech recogniser: SOS and simulated 911
  tiger_bridge.py      Tiger Cloud: create tables, check, load the Pi's event log
  dashboard.py/.html   site dashboard (local web server)
  sim_crew.py          simulated crew for the dashboard
  capture_tracks.py, replay_tracks.py   record tracker input on the Pi, replay it on the laptop
  laptop_mic_to_pi.py, make_samples.py, pi_*.sh, install_pi_key.cmd   setup and test helpers
db/
  schema.sql           detections hypertable, rollup, hazard register, workers
  activity_log.sql     supervisor activity log tables
models/                .tflite models and labels (see models/README.md)
samples/               test sounds and images
docs/                  detailed docs (below)
```

## Docs

- [Safety features](docs/SAFETY_FEATURES.md): every alert, how it's detected, how to test it, and its settings.
- [Tiger Data](docs/TIGER_DATA.md): database setup, schema, dashboard and simulated crew.
- [Models](models/README.md): downloading the models and setting up the Pi's runtime.
- [Requirements](docs/REQUIREMENTS.md): the original plan and how it turned out.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ssh qnxpi` times out | The Pi's Wi-Fi chip sometimes fails to start at boot ("bus is not ready" in its log), and then there's no Wi-Fi until the next power cycle. Use the Ethernet cable and `qnxpi-eth`. |
| The companion starts but no alerts come through | It's connected to the wrong host. Add `--host qnxpi-eth` when you're on the cable. |
| Events don't show on the dashboard | Check the Pi's clock with `ssh qnxpi-eth date`. Without internet, the Pi starts at 1 Jan 1970 after a reboot. From Git Bash, set it from the laptop: `ssh qnxpi-eth "sudo date -u -s @$(date +%s)"`. |
| The camera won't start | Only one program can use it. Stop any other `hat_detect.py` on the Pi first. |
| Camera runs at about 2 fps instead of 5 | Check the Pi's power supply. A Pi 5 needs the official 27 W (5 V, 5 A) supply, and a weaker one slows it down. |
