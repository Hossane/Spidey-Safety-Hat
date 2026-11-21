# Hat Detect

**An AI safety helmet that listens and watches for hazards on a construction site, and
warns the wearer.** Hack Western 2026 · QNX track.

A Raspberry Pi 5 running **QNX 8.0** runs two TensorFlow Lite models on the helmet:

- **YAMNet** hears alarms, sirens, horns, shouting and crashes.
- **EfficientDet-Lite0** sees people, vehicles and sharp objects.

A **fusion engine** combines what the hat hears and sees into one hazard event. For
example, a horn heard while a truck is in view becomes one *vehicle hazard* with one
warning, not two separate alerts. The wearer hears the warning through a speaker. Every
event is stored in **Tiger Data** (Tiger Cloud / TimescaleDB), and a live dashboard
shows what happened on site.

All inference and alerting happen on the Pi, and they keep working without a network.
The laptop companion only lends a microphone, a speaker and the database connection.

---

## Contents

1. [The problem](#the-problem)
2. [Features](#features)
3. [Hardware architecture](#hardware-architecture)
4. [Software architecture](#software-architecture)
5. [Audio detection pipeline](#audio-detection-pipeline)
6. [Vision detection pipeline](#vision-detection-pipeline)
7. [Multimodal event fusion](#multimodal-event-fusion)
8. [Unified event schema](#unified-event-schema)
9. [Tiger Data integration](#tiger-data-integration)
10. [Dashboard](#dashboard)
11. [Repository structure](#repository-structure)
12. [Installation and setup](#installation-and-setup)
13. [Environment variables](#environment-variables)
14. [Running the individual components](#running-the-individual-components)
15. [Running the complete system](#running-the-complete-system)
16. [Running the dashboard](#running-the-dashboard)
17. [Running the tests](#running-the-tests)
18. [QNX-specific setup](#qnx-specific-setup)
19. [Limitations and future work](#limitations-and-future-work)

---

## The problem

Construction sites are loud. Workers wear hearing protection and concentrate on their
task, so they can miss a reversing truck's horn, an alarm, or a colleague shouting for
help. A helmet that is an extra pair of eyes and ears has to meet three conditions:

- **It must work on the helmet itself.** A warning that waits for the cloud arrives too
  late, or never arrives on a site with poor coverage.
- **It must not cry wolf.** A raw detector fires dozens of times a second. Workers learn
  to ignore a helmet that says "person, person, car, horn, car" all day.
- **It must leave a record.** Site managers need to see which hazards happen, where and
  when, so they can put controls in place.

Hat Detect runs both models on the QNX Pi, merges related detections into one meaningful
event, and logs those events to a time-series database for a site-safety dashboard.

## Features

**Implemented:**

- **Audio classification on the Pi:** YAMNet runs on 0.975 s windows every 0.5 s, and
  maps 33 YAMNet classes to six safety groups.
- **Object detection on the Pi:** EfficientDet-Lite0 reads frames from the QNX Camera
  service at 5 fps. It reports whether an object is left, centre or right of the wearer.
- **Both detectors run at the same time,** in separate threads, entirely on-device with
  `tflite_runtime` from oss.qnx.com.
- **A multimodal fusion engine** (`fusion.py`):
  - Produces AUDIO_ONLY, VISION_ONLY and MULTIMODAL events.
  - Combines detections only when a configurable rule pairs them, within a time window.
  - Suppresses duplicates, so an ongoing hazard is one event.
  - Waits briefly for confirmation from the other sensor before announcing.
  - Upgrades an already-announced event when that confirmation arrives late.
- **Exactly one alert per event:**
  - A GPIO buzzer pulse on the Pi (optional).
  - A warning tone plus a spoken phrase on the laptop's speaker (e.g. a paired JBL). A
    more severe alert interrupts a less severe one.
  - Speech uses `say` on macOS, PowerShell System.Speech on Windows, and espeak on
    Linux.
- **Tiger Cloud storage:**
  - One row per event, upserted on `event_id`, so it's idempotent.
  - When the database is offline, events are buffered locally and replayed when it's
    back. Invalid records are set aside without blocking the rest.
  - The Pi never holds database credentials.
- **Live dashboard:**
  - Headline numbers, events over time by type, and breakdowns.
  - A clickable event timeline with full details for each event: both confidences, why
    the detections were combined, and the recommended safety action.
  - Filters for time, hat, hazard, severity, event type and source. It refreshes every
    5 s.
- **71 automated tests,** including integration tests against a real TimescaleDB.

## Hardware architecture

| Part | Role |
|---|---|
| Raspberry Pi 5 running QNX 8.0 | Runs both models, the fusion engine, the GPIO buzzer and the event log |
| USB webcam (Logitech C270, QNX Camera unit 5) | Camera input for EfficientDet |
| Laptop microphone | Audio input. It is streamed to the Pi over SSH, because QNX's USB audio driver rejects the webcam's mic |
| JBL Bluetooth speaker, paired with the laptop | Spoken warnings. The QNX image has no Bluetooth stack |
| GPIO buzzer, LED or vibration motor on BCM 18 (optional) | Pulse on the helmet itself |
| Construction helmet | Holds the Pi and camera |

```mermaid
flowchart LR
  subgraph Helmet
    CAM[USB webcam]
    PI[Raspberry Pi 5<br/>QNX 8.0]
    BUZ[GPIO buzzer<br/>optional]
    CAM -- USB --> PI
    PI -- BCM 18 --> BUZ
  end
  subgraph Laptop
    MIC[Microphone]
    COMP[laptop_companion.py]
    SPK[JBL speaker<br/>Bluetooth]
    MIC --> COMP --> SPK
  end
  COMP <-->|"SSH: mic audio in, EVENT lines out"| PI
  COMP -- "TLS / SQL" --> TIGER[(Tiger Cloud)]
```

## Software architecture

```mermaid
flowchart TD
  subgraph Pi["Pi · QNX · hat_detect.py"]
    A[audio_detect.py<br/>YAMNet] -- "raw detections" --> F
    V[vision_detect.py<br/>EfficientDet-Lite0] -- "raw detections" --> F
    F["fusion.py<br/>FusionEngine"] -- "unified events" --> P[EventPipeline]
    P --> BZ[GPIO buzzer]
    P --> LOG["~/hatdetect/events.jsonl"]
    A & V -.-> RAW["~/hatdetect/detections.jsonl<br/>(raw, for debugging)"]
  end
  subgraph Laptop
    C[laptop_companion.py] --> S["Speaker<br/>tone + speech, priority"]
    C --> W["tiger_bridge.TigerWriter<br/>batch · upsert · spool"]
    D[dashboard.py :8050]
  end
  MIC[Laptop mic] -- "16 kHz PCM over SSH stdin" --> A
  P -- "EVENT {json} over SSH stdout" --> C
  W --> DB[("Tiger Cloud<br/>detections hypertable<br/>hazard_types · hazard_events")]
  W -. offline .-> SP[.tiger_spool.jsonl]
  SP -. replay .-> W
  DB --> D
  LOG -. "tiger_bridge.py ingest-pi" .-> DB
```

Inference runs on the Pi, and the database connection runs on the laptop. The Pi
decides what to announce: each event record carries a `notify` flag, and the laptop
speaks only those records. Nothing is announced by two devices independently.

## Audio detection pipeline

`audio_detect.py`, driven by `hat_detect.audio_worker`:

1. The laptop streams 16 kHz mono 16-bit PCM over SSH into `hat_detect.py`'s stdin.
2. Every `hop_seconds` (0.5 s), the latest 15,600 samples (0.975 s, YAMNet's fixed
   input) are scored. That's 521 AudioSet classes, in about 6 ms on the Pi 5.
3. Classes are mapped to safety groups. A class scoring at least `audio.threshold`
   (0.3) counts as a hit:

   | Group | YAMNet classes (examples) | Severity |
   |---|---|---|
   | ALARM | Alarm, Fire alarm, Smoke detector, Reversing beeps, Buzzer | high |
   | SIREN | Siren, Police car / Ambulance / Fire engine (siren) | high |
   | IMPACT | Smash, crash, Breaking, Shatter, Explosion, Bang | high |
   | FIRE | Fire, Crackle | high |
   | HORN | Vehicle horn, Air horn, Foghorn | medium |
   | VOICE | Shout, Yell, Screaming | medium |

   Normal site noise (jackhammer, drill, engine, speech) is recognised but never alerts.
4. The strongest class in each group becomes a raw detection for the fusion engine.

## Vision detection pipeline

`vision_detect.py`, driven by `hat_detect.video_worker`:

1. `camera_example4_record` (QNX Camera service, unit 5) records uncompressed YUY2
   `.ucv` chunks. Frames are read at `video.fps` (5) and converted to RGB with numpy.
2. EfficientDet-Lite0 (the TF Hub detection build, with built-in NMS) runs on a 320×320
   frame. That takes about 55 ms on the Pi 5, or about 85 ms including the resize.
3. COCO classes are mapped to groups. A detection scoring at least `video.threshold`
   (0.5) counts as a hit:

   | Group | COCO classes | Severity |
   |---|---|---|
   | VEHICLE | car, truck, bus, motorcycle, bicycle, train | medium |
   | SHARP | knife, scissors | medium |
   | PERSON | person | low |

4. The strongest box in each group becomes a raw detection, with `position` (left,
   centre or right, from the box centre) and `bbox`.

## Multimodal event fusion

`fusion.py` is plain Python with no numpy or TFLite, so it runs on QNX and is tested on
a laptop. It is configured by the `fusion` section of `hat_config.json`:

```json
"fusion": {
  "confirm_wait_seconds": 1.0,
  "episode_gap_seconds": 3.0,
  "escalate_on_confirmation": true,
  "realert_on_escalation": true,
  "rules": [
    {"hazard": "VEHICLE_HAZARD",  "audio": ["HORN", "SIREN"],  "video": ["VEHICLE"], "window_seconds": 3.0, "severity": "high"},
    {"hazard": "WORKER_DISTRESS", "audio": ["VOICE", "IMPACT"], "video": ["PERSON"],  "window_seconds": 3.0, "severity": "high"}
  ]
}
```

| Mechanism | What it does |
|---|---|
| **Episodes** (duplicate suppression) | Detections of the same group extend one open episode until none arrive for `episode_gap_seconds`, or for a rule's longer window. One episode is one event and one alert, however many frames or windows it spans. A different hazard always gets its own episode, so it is never suppressed |
| **Rules** (correlation) | A sound and an object combine **only** when a rule pairs their groups **and** their sightings are within `window_seconds` of each other. A shout while a car is in view stays two events |
| **Confirmation wait** | A detection that the other sensor could make more severe (a medium horn could become a high VEHICLE_HAZARD) waits up to `confirm_wait_seconds` before it is announced, so horn plus truck gives one warning. A detection that is already high (siren, alarm, impact, fire) is announced immediately |
| **Late confirmation** | If the partner arrives after the event was announced, the same event (same `event_id`) is upgraded to MULTIMODAL. It is announced again only if its severity went up |
| **Severity policy** | Severity comes from configured rules, never from model confidence. A rule without a `severity` raises the combined event one level above its parts (`escalate_on_confirmation`) |
| **Confidence** | Each modality keeps its own peak confidence. Scores are never averaged |

Example, with a horn at t = 0 s (0.94) and a car seen at t = 1 s (0.91):

| Raw detections | Without fusion | With fusion |
|---|---|---|
| ~2 audio windows + ~10 frames | "Horn" … "Car on your left" … | **One** MULTIMODAL `VEHICLE_HAZARD` (high), audio 0.94 and vision 0.91 kept, one buzz, one phrase: *"Warning! Vehicle hazard on your left"* |

Spoken phrases, from `scripts/laptop_companion.py`:

| Event | Phrase |
|---|---|
| VEHICLE_HAZARD / WORKER_DISTRESS | "Warning! Vehicle hazard on your left" / "Warning! Worker in distress ahead" |
| Audio only | "Alarm", "Siren", "Horn", "Someone shouting", "Crash", "Fire" |
| Vision only | "Person on your right", "Truck ahead" |

## Unified event schema

The Pi prints every record as `EVENT <json>` on stdout and appends it to
`~/hatdetect/events.jsonl`:

```json
{"event_id": "3f6c2a9e-…", "time": "2026-11-21T19:30:00.000+00:00", "last_time": "2026-11-21T19:30:02.400+00:00",
 "device_id": "hat-01", "hazard": "VEHICLE_HAZARD", "severity": "high", "fusion": "MULTIMODAL",
 "label": "Vehicle horn, car horn, honking + car",
 "audio": {"group": "HORN", "label": "Vehicle horn, car horn, honking", "score": 0.94, "detections": 3},
 "video": {"group": "VEHICLE", "label": "car", "score": 0.91, "detections": 12, "position": "left", "bbox": [10, 20, 200, 220]},
 "correlation": {"window_seconds": 3.0, "explanation": "Vehicle horn, car horn, honking (HORN) heard 1.0 s before car (VEHICLE) was seen on the left; rule VEHICLE_HAZARD pairs them within 3 s"},
 "alerted": true, "notify": true, "status": "open"}
```

| Field | Meaning |
|---|---|
| `event_id` | UUID. Stays the same across the event's `open`, `update` and `closed` records |
| `time` / `last_time` | First and last detection (UTC ISO 8601). `time` never changes |
| `fusion` | `AUDIO_ONLY`, `VISION_ONLY` or `MULTIMODAL` |
| `hazard`, `severity` | The group (e.g. `HORN`) or the rule's hazard (`VEHICLE_HAZARD`), and its configured severity |
| `audio`, `video` | Each modality's peak detection, its confidence and the number of raw detections merged. `null` if absent |
| `correlation` | Multimodal only: the rule window and a readable explanation |
| `alerted` / `notify` | The wearer has been alerted / this record is the one to announce |
| `status` | `open` on first announcement, `update` when upgraded, `closed` when the episode ends |

There is no recommended action in the record. It comes from the `hazard_types` table, so
it lives in one place.

## Tiger Data integration

Details are in [`docs/TIGER_DATA.md`](docs/TIGER_DATA.md).

- **`detections`** (hypertable): one row per event, with a unique key on
  `(event_id, time)`. Columns hold the fusion type, status, and the audio and video
  parts (group, label, peak score, detection count). Video events also store position
  and bbox, and multimodal events store the correlation window and explanation.
- **`hazard_types`:** the hazard register. Each hazard has a severity, a description and
  a recommended control.
- **`hazard_events`** (view): events joined with their control. The dashboard reads this.
- **`detections_per_minute`** (continuous aggregate), **`safety_alerts`** (view), and a
  **90-day retention** policy.
- **Idempotent ingestion:** an event's records upsert the same row, and an older status
  never overwrites a newer one. Re-sending a whole log changes nothing.
- **Reliability:**
  - `TigerWriter.add()` never blocks the alert path.
  - If Tiger Cloud is unreachable, records go to `.tiger_spool.jsonl` and are retried
    every 10 s, even while no new events arrive.
  - Records the database rejects as invalid go to `.tiger_rejected.jsonl`.
- **Credentials stay on the laptop.** QNX has no Postgres client, and the Pi's events
  reach the laptop over SSH.
- **Migration:** `tiger_bridge.py init` upgrades a database created by the earlier
  raw-detection schema in place. That includes compressed chunks. Old rows are kept with
  `event_id` NULL, and the dashboard ignores them.

## Dashboard

`scripts/dashboard.py` is a stdlib HTTP server on port 8050, serving one page
(`dashboard.html`, using Chart.js). It reads the `hazard_events` view, and the page polls
every 5 s.

- **Summary:** hazard events, high severity, multimodal, audio only and vision only, each
  with its share of all events.
- **Charts:** events over time stacked by type, plus breakdowns by severity, by source
  (heard or seen) and multimodal versus single-modality.
- **Event timeline:** the latest 100 events with hazard, severity, type, audio and vision
  confidence, and the recommended action. Newly arrived rows are highlighted.
- **Event details:** click a row, or use the arrow keys and Enter. It shows the event ID,
  the audio and vision classifications with their confidences, the fusion explanation,
  the recommended action, the alert status and every other field.
- **More panels:** events by hazard, where events were around the wearer (left, centre or
  right), and the hazard register.
- **Filters:** time range (1 h, 8 h shift, 24 h, 7 days, custom), hat, hazard, severity,
  event type and source. Clicking a tile or bar applies its filter.

## Repository structure

```
hack-western/
├── hat_detect.py            # Pi app: both detectors in threads → FusionEngine → buzzer + EVENT lines
├── fusion.py                # multimodal fusion engine and unified event records (stdlib only)
├── audio_detect.py          # YAMNet: safety-sound groups; .wav files or live PCM on stdin
├── vision_detect.py         # EfficientDet-Lite0: hazard groups; images, .ucv, or live QNX camera
├── hat_config.json          # thresholds, groups, severities, fusion rules, GPIO, log paths
├── scripts/
│   ├── laptop_companion.py  # laptop: mic → Pi over SSH, events → speaker + Tiger Cloud
│   ├── tiger_bridge.py      # Tiger Cloud writer + CLI: init, check, ingest, ingest-pi, demo, purge-demo
│   ├── dashboard.py         # dashboard server (localhost:8050)
│   ├── dashboard.html       # dashboard page
│   ├── laptop_mic_to_pi.py  # stream the laptop mic into audio_detect.py alone (audio-only testing)
│   ├── make_samples.py      # build test images and sounds in samples/
│   ├── install_pi_key.cmd   # Windows: install your SSH key on the Pi
│   ├── pi_audio_diag.sh     # QNX: read-only audio diagnostics
│   └── pi_mic_experiment.sh # QNX: try a USB mic with a temporary io-snd
├── db/schema.sql            # Tiger Cloud schema and migration (applied by tiger_bridge.py init)
├── tests/                   # unit tests, fakes, and real-database integration tests
├── docs/
│   ├── ARCHITECTURE.md      # design notes for each phase of this work
│   ├── TIGER_DATA.md        # database setup and the dashboard
│   └── REQUIREMENTS.md      # the original pre-hackathon requirements
├── models/README.md         # which .tflite files to download (models are git-ignored)
├── check_model.py, live_mic.py, silence_test.py   # early laptop model smoke tests (need TensorFlow)
├── requirements.txt         # laptop Python packages
└── .env.example             # template for the database connection string
```

## Installation and setup

### Laptop

Python 3.10+ is needed. It has been tested with Python 3.14 on macOS.

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install "psycopg[binary]" python-dotenv sounddevice numpy   # companion, bridge, dashboard
# optional, for testing the models on the laptop: pip install -r requirements.txt  (TensorFlow)
```

Then:

1. **Download the models** into `models/`, following [`models/README.md`](models/README.md).
2. **Set up SSH.** Add an SSH config entry named `qnxpi` for the Pi, and install your key
   (`scripts/install_pi_key.cmd` on Windows). See `models/README.md` for the cipher
   settings.
3. **Set up Tiger Cloud.** Create a service, then copy `.env.example` to `.env` and paste
   the connection string (see [Environment variables](#environment-variables)).
4. **Create the tables** (this is safe to run again, and it migrates an older database):

   ```bash
   python scripts/tiger_bridge.py init
   python scripts/tiger_bridge.py check
   ```

### Pi (QNX)

See [QNX-specific setup](#qnx-specific-setup). In short: install `python3-tflite-runtime`
and `python3-numpy` with `apk`, copy the models to `~/models`, and copy the app:

```bash
ssh qnxpi "mkdir -p ~/hatdetect ~/models"
scp audio_detect.py vision_detect.py hat_detect.py fusion.py hat_config.json qnxpi:hatdetect/
scp models/*.tflite models/*.csv models/*.txt qnxpi:models/
```

## Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `TIGER_DATABASE_URL` | `.env` on the laptop (git-ignored) | Tiger Cloud connection string, `postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require`. Used by `tiger_bridge.py`, `laptop_companion.py` (if it's not set, events aren't stored) and `dashboard.py` |
| `HATDETECT_TEST_DATABASE_URL` | your shell, tests only | A **local** Postgres + TimescaleDB admin URL for `tests/test_integration_db.py`. The tests refuse non-local hosts |

The hat's ID (`device_id`) and every threshold live in `hat_config.json`, not in the
environment. Never commit `.env`. The Pi never needs any of these variables.

## Running the individual components

On the Pi, from `~/hatdetect`:

```bash
python3 audio_detect.py samples/siren.wav              # classify .wav files
python3 audio_detect.py --stdin < audio.pcm            # raw s16le 16 kHz mono PCM
python3 vision_detect.py samples/street.ppm            # images (.ppm/.pgm with numpy alone)
python3 vision_detect.py --camera 5                    # live USB webcam via the QNX Camera service
python3 hat_detect.py --no-audio                       # camera + fusion + buzzer, events on stdout
```

On the laptop:

```bash
python scripts/laptop_companion.py --test-speaker      # play one sample alert on the default speaker
python scripts/laptop_mic_to_pi.py                     # laptop mic → audio_detect.py on the Pi
python scripts/make_samples.py                         # build samples/ (needs Pillow)
python scripts/tiger_bridge.py ingest-pi               # load the Pi's events.jsonl into Tiger Cloud
python scripts/tiger_bridge.py ingest FILE             # load a local events.jsonl
```

## Running the complete system

1. Pair the JBL speaker with the laptop over Bluetooth and make it the default output.
2. Plug the webcam into the Pi (a blue USB 3.0 port), and power the Pi.
3. From the project root on the laptop:

```bash
python scripts/laptop_companion.py              # mic + camera + speaker + Tiger Cloud
python scripts/laptop_companion.py --no-mic     # camera only
python scripts/laptop_companion.py --no-db      # don't store events
```

The companion starts `hat_detect.py` on the Pi over SSH, streams the mic, prints each
event, speaks the alerts and stores every record. Ctrl+C stops both ends. The Pi stops
its camera recorder and closes any events still open, writing their final records to its
event log. Use `tiger_bridge.py ingest-pi` to backfill anything the laptop missed.

## Running the dashboard

```bash
python scripts/dashboard.py              # http://localhost:8050
python scripts/dashboard.py --port 9000
```

To see it before the hat has run, `python scripts/tiger_bridge.py demo` fills the
database with 8 hours of sample events for device `demo-01`. It makes them by running
simulated detections through the real fusion engine, and it replaces earlier demo data.
`purge-demo` removes the sample events.

## Running the tests

```bash
python3 -m unittest discover -s tests -v
```

**62 unit tests** need no hardware, models or database. They use simulated detections,
fakes for the buzzer and voice, and an in-memory database:

| File | Covers |
|---|---|
| `test_fusion.py` | Audio-only, vision-only and multimodal events, correlation timeout, unrelated pairs, duplicate suppression, the severity policy, the event format |
| `test_alerting.py` | Pi pipeline (one buzz and one record stream per event, logging when the laptop disconnects), phrases, speaker priority and interruption |
| `test_tiger_bridge.py` | One row per event, upgrades, idempotent re-sends, offline spool and replay, invalid-record isolation, demo data |
| `test_dashboard.py` | Summary metrics match the events, the timeline sums, every filter and combination, input validation, HTTP endpoints |

**9 integration tests** run against a real PostgreSQL + TimescaleDB. They are skipped
unless `HATDETECT_TEST_DATABASE_URL` is set. They create and drop their own
`hatdetect_it` database. They cover:

- Migrating the old schema, with compressed chunks.
- Upserts, including into a compressed chunk.
- Going offline and reconnecting with the real driver.
- `ingest` and `check`.
- That the dashboard's SQL returns exactly what the reference computation returns, across
  9 filter combinations.

```bash
docker run -d --name hatdetect-tsdb -p 127.0.0.1:55432:5432 \
  -e POSTGRES_PASSWORD=localtest timescale/timescaledb:latest-pg17
HATDETECT_TEST_DATABASE_URL=postgres://postgres:localtest@127.0.0.1:55432/postgres \
  .venv/bin/python -m unittest discover -s tests -v
docker rm -f hatdetect-tsdb
```

All 71 pass on macOS with Python 3.14 and TimescaleDB 2.32.1 (Postgres 17).

## QNX-specific setup

- **OS:** Raspberry Pi 5 with QNX 8.0 (the loaner Pis come pre-loaded).
- **TensorFlow Lite** comes from QNX's open-source apk repo (`repo.oss.qnx.com`), not pip:

  ```bash
  sudo apk update
  sudo apk add python3-tflite-runtime python3-numpy    # tflite_runtime 2.21.0, numpy 2.4.1, Python 3.14
  ```

  `audio_detect.py` and `vision_detect.py` import `tflite_runtime` first. They fall back
  to TensorFlow only on a laptop.
- **Camera:** the QNX Camera service. The USB webcam is unit 5 (list the units with
  `camera_example1_callback`). Live frames come from `camera_example4_record` writing
  `.ucv` (YUY2), which is decoded with numpy. No OpenCV is needed on the Pi.
- **Audio:** QNX's USB audio driver rejects the C270 webcam microphone, so the laptop
  streams its mic over SSH. `scripts/pi_audio_diag.sh` and `scripts/pi_mic_experiment.sh`
  are the read-only and reversible diagnostics used to find this.
- **No Bluetooth stack** on the QNX image, so speech plays on the laptop.
- **GPIO:** the `rpi_gpio` module, BCM 18 by default (`gpio` in `hat_config.json`). If
  it isn't available, the buzzer is disabled and everything else runs.
- **SSH from Windows:** use `Ciphers aes128-ctr` and `MACs hmac-sha2-256` in the `qnxpi`
  config entry, or OpenSSH fails with "Corrupted MAC on input".
- **No Postgres client on QNX.** This is why the database connection lives on the laptop.

## Limitations and future work

**Known limitations of the current build:**

- **Not yet tested on the Pi:** the fusion engine, the new event format and the updated
  `hat_detect.py`. Everything up to the earlier per-detection alerts ran on QNX, and the
  new code was tested with simulated detections and a real TimescaleDB.
  - The macOS speech backend was checked, and the Windows one was not.
  - The full companion loop has not run end to end with the hardware.
- **No direction for sounds.** There is one microphone, so audio-only warnings have no
  direction. Position comes only from the camera.
- **Vision-only sightings are announced once per sighting.** On a busy site, people and
  vehicles can still produce frequent low and medium warnings. `alert_groups` in
  `hat_config.json` can remove a group.
- **No approach detection.** A vehicle is a hazard only when a horn or siren confirms it.
  The hat doesn't yet estimate whether an object is getting closer.
- **Only two fusion rules** (`VEHICLE_HAZARD`, `WORKER_DISTRESS`). More are a config
  change, but each needs a matching `hazard_types` row and a phrase.
- **Interrupting Windows speech** restarts the PowerShell voice process, which adds a
  short delay before the next phrase.
- **The per-minute rollup** (`detections_per_minute`) counts old raw rows and new event
  rows together. The dashboard doesn't use it.
- **No access control on the dashboard.** It binds to `127.0.0.1` by default; keep it
  that way.

**Future improvements (not implemented):**

- Approach detection from bounding-box growth across frames, so a vehicle getting closer
  is a hazard even without a horn.
- More fusion rules: impact plus falling object, alarm plus smoke or fire.
- A second microphone or the Pi's own audio input, for direction and to drop the laptop
  mic.
- Speech on the helmet itself, once a QNX audio output path is available.
- A C++ port using the QNX audio and camera APIs directly.
- Multi-hat site maps and shift reports in the dashboard.
