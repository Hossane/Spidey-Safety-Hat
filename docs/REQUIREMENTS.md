# Hat Detect — Requirements

Hack Western 2026 · QNX track

A wearable hat with a microphone and a camera that detects important sounds
and objects around the wearer and alerts them. Detection runs entirely
on-device: a Raspberry Pi running QNX, with TensorFlow Lite models. Every
detection event is stored in **Tiger Data** (Tiger Cloud / TimescaleDB) for
history, analytics and a live dashboard.

> **Scope:** video and audio detection only. Anything else from the original
> hat idea is out of scope for this build.

---

## 1. Functional requirements

### Audio detection
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-A1 | Capture mono audio from a USB microphone at 16 kHz | Must |
| FR-A2 | Classify each ~1 s window of audio with YAMNet | Must |
| FR-A3 | Report a sound event when a target class scores above a configurable threshold (default 0.5) | Must |
| FR-A4 | Support a configurable list of target sounds (e.g. siren, car horn, alarm, speech, dog bark) | Must |
| FR-A5 | Suppress repeated alerts for the same sound within a cooldown window (default 3 s) | Should |

### Video detection
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-V1 | Capture frames from the camera (USB webcam or Pi Camera Module 3) | Must |
| FR-V2 | Run object detection on frames with EfficientDet-Lite0 (or SSD MobileNet) | Must |
| FR-V3 | Report an object event when a target class scores above a configurable threshold (default 0.5) | Must |
| FR-V4 | Support a configurable list of target objects (e.g. person, car, bicycle, stop sign) | Must |
| FR-V5 | Report the approximate position of a detected object (left / centre / right) | Should |
| FR-V6 | Run video detection on demand, when an audio event fires, to save CPU | Could |

### Alerts and combined logic
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-L1 | Run audio and video detection at the same time (separate threads) | Must |
| FR-L2 | Alert the wearer when an audio or video event fires | Must |
| FR-L3 | Combine audio and video events that happen within a short time window into one alert (e.g. "horn heard + car seen") | Should |
| FR-L4 | Log every event with a timestamp, label and score to the console or a file for the demo | Should |
| FR-L5 | Choose the alert output (vibration motor / buzzer / speaker / screen) — **team decision, see §7** | Must |

### Event storage and dashboard (Tiger Data)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-D1 | Store every detection event (sound, object, combined alert) in a Tiger Cloud service | Must |
| FR-D2 | Store events in a TimescaleDB **hypertable** partitioned on event time | Must |
| FR-D3 | Send events from the Pi to a **bridge service** on the laptop (JSON over HTTP or UDP), which inserts them into Tiger Cloud — see §6 for why | Must |
| FR-D4 | Buffer events on the Pi when the network is down and resend them when it's back | Should |
| FR-D5 | Live dashboard showing recent events and counts per label over time (e.g. Grafana or a small web page reading Tiger Cloud) | Should |
| FR-D6 | Use a **continuous aggregate** for per-minute counts by label so the dashboard stays fast | Should |
| FR-D7 | Analytics queries for the demo: busiest times, most common sounds, how often audio and video agree | Could |

## 2. Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | All inference and alerts run on-device and work with no internet; only event upload to Tiger Data needs a network |
| NFR-2 | Audio detection latency ≤ 1.5 s from sound to alert |
| NFR-3 | Video detection at ≥ 2 frames per second while audio is also running |
| NFR-4 | Runs on battery power for at least the length of the demo (~30 min) |
| NFR-5 | Wearable: board, mic and camera mounted on or near the hat, with a power bank in a pocket |
| NFR-6 | Thresholds, target classes and cooldowns set in a config file, not hard-coded |
| NFR-7 | Written in C++17, cross-compiled with the QNX SDP 8.0 toolchain |
| NFR-8 | Database credentials kept in a `.env` file on the laptop, never committed and never stored on the Pi |
| NFR-9 | An event reaches Tiger Cloud within 2 s of detection while the network is up |

## 3. Hardware

| Item | Notes |
|------|-------|
| Raspberry Pi 4 or 5 | Running the QNX 8.0 quick-start image |
| microSD card (32 GB+) | For the QNX image |
| USB microphone | Plug in before `io-audio` starts |
| Camera | USB webcam (use a blue USB 3.0 port) **or** Raspberry Pi Camera Module 3 |
| USB-C power bank | 5 V / 3 A+ (Pi 5 needs 5 A for full USB power) |
| Alert output | Vibration motor / buzzer + transistor, or small speaker (§6) |
| Hat + mounting | Clips, velcro or a 3D-printed case for the Pi and camera |
| Heatsink / fan | Running both models keeps the CPU busy and the Pi throttles when hot |
| Dev laptop | Linux (or a Linux VM) with the QNX SDP installed |

## 4. Software

### On the dev laptop
- QNX Software Development Platform 8.0 (free non-commercial licence via QNX Everywhere)
- CMake ≥ 3.16, git, a native GCC/Clang toolchain (needed to build `flatc`)
- TensorFlow Lite cross-compiled for QNX aarch64le, from
  [qnx-ports/build-files](https://github.com/qnx-ports/build-files) (`ports/tensorflow`)
- Python 3.10+ with `requirements.txt`, for testing models and running the Tiger Data bridge
- A Tiger Data account with one Tiger Cloud service (free plan, or the MLH hackathon credits)
- `psql` (PostgreSQL client), for creating tables and checking data

### On the Pi (QNX)
- `io-audio` running with the USB mic (QNX Sound Architecture / `libasound`)
- The camera service running (QNX Camera library / `libcamapi`) — check with `camera_example3_viewfinder`
- `libtensorflow-lite.so` copied over from the cross-compile

## 5. Models

Both models are pretrained TensorFlow Lite files. No training is needed for the
default classes. Download them into `models/` (they are git-ignored); see
[`models/README.md`](../models/README.md).

| Purpose | Model | Input | Output |
|---------|-------|-------|--------|
| Sound | YAMNet (`yamnet.tflite`) + `yamnet_class_map.csv` | 15,600 float samples (0.975 s @ 16 kHz, range −1..1) | 521 class scores (AudioSet) |
| Objects | EfficientDet-Lite0 (`efficientdet_lite0.tflite`) + COCO labels | 320×320 RGB uint8 | Boxes, classes, scores (80 COCO classes) |

**Custom classes (only if needed):** if a target sound or object isn't in
YAMNet / COCO, train a small model in Edge Impulse or Teachable Machine and
export it as `.tflite`. Budget ~50–100 samples per class.

## 6. Tiger Data

### Why a bridge service
There is no official PostgreSQL client library (`libpq`) for QNX, so the Pi
does not talk to the database directly. The Pi sends small JSON events over the
local network to a bridge on the laptop. The bridge uses a normal Postgres
driver (`psycopg`) to insert them into Tiger Cloud. This also keeps database
credentials off the hat.

```
Pi (QNX) ──JSON over Wi-Fi──► bridge (laptop, Python) ──SQL/TLS──► Tiger Cloud
                                                                     │
                                                    dashboard ◄──────┘
```

### Data to store (`detections` hypertable)
| Column | Type | Notes |
|--------|------|-------|
| `time` | timestamptz | When the event was detected (hypertable time column) |
| `device_id` | text | Which hat, e.g. `hat-01` |
| `source` | text | `audio`, `video` or `combined` |
| `label` | text | e.g. `Siren`, `car`, `person` |
| `score` | real | Model confidence, 0–1 |
| `position` | text, nullable | `left` / `centre` / `right` for video events |
| `bbox` | jsonb, nullable | Box coordinates for video events |
| `alerted` | boolean | Whether the wearer was alerted (vs. below cooldown) |

A continuous aggregate (`detections_per_minute`) holds counts per minute by
`source` and `label` for the dashboard.

## 7. Open decisions (team)

- [ ] What exactly does the hat do, and who is it for? (accessibility for deaf / blind users, safety, other)
- [ ] Bridge transport: HTTP POST or UDP
- [ ] Dashboard: Grafana (connects to Postgres directly) or a custom web page
- [ ] Final list of target sounds
- [ ] Final list of target objects
- [ ] Alert output: vibration, buzzer, spoken alert, or screen
- [ ] Camera: USB webcam or Pi Camera Module 3
- [ ] Pi 4 or Pi 5

## 8. Before the hackathon (setup checklist)

- [ ] QNX image flashed and the Pi boots; SSH works
- [ ] QNX SDP installed on at least one laptop; `source ~/qnx800/qnxsdp-env.sh` works
- [ ] TensorFlow Lite cross-compiled for QNX and a test binary loads a `.tflite` on the Pi
- [ ] USB mic shows up under `/dev/snd` and a test recording plays back
- [ ] Camera shows frames in `camera_example3_viewfinder`
- [ ] Both models downloaded and tested on a laptop with `requirements.txt`
- [ ] Tiger Data account created and one Tiger Cloud service running; connection string saved in `.env`
- [ ] `psql` connects to the service from the laptop
- [ ] Pi and laptop are on the same network and can reach each other (ping both ways)
