# Hat Detect — Requirements

Hack Western 2026 · QNX track

A wearable hat with a microphone and a camera that detects important sounds
and objects around the wearer and alerts them. Detection runs entirely
on-device: a Raspberry Pi running QNX, with TensorFlow Lite models.

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
| FR-L5 | Choose the alert output (vibration motor / buzzer / speaker / screen) — **team decision, see §6** | Must |

## 2. Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | All inference runs on-device; no internet connection required |
| NFR-2 | Audio detection latency ≤ 1.5 s from sound to alert |
| NFR-3 | Video detection at ≥ 2 frames per second while audio is also running |
| NFR-4 | Runs on battery power for at least the length of the demo (~30 min) |
| NFR-5 | Wearable: board, mic and camera mounted on or near the hat, with a power bank in a pocket |
| NFR-6 | Thresholds, target classes and cooldowns set in a config file, not hard-coded |
| NFR-7 | Written in C++17, cross-compiled with the QNX SDP 8.0 toolchain |

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
- Python 3.10+ with `requirements.txt`, only for testing models on the laptop

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

## 6. Open decisions (team)

- [ ] What exactly does the hat do, and who is it for? (accessibility for deaf / blind users, safety, other)
- [ ] Final list of target sounds
- [ ] Final list of target objects
- [ ] Alert output: vibration, buzzer, spoken alert, or screen
- [ ] Camera: USB webcam or Pi Camera Module 3
- [ ] Pi 4 or Pi 5

## 7. Before the hackathon (setup checklist)

- [ ] QNX image flashed and the Pi boots; SSH works
- [ ] QNX SDP installed on at least one laptop; `source ~/qnx800/qnxsdp-env.sh` works
- [ ] TensorFlow Lite cross-compiled for QNX and a test binary loads a `.tflite` on the Pi
- [ ] USB mic shows up under `/dev/snd` and a test recording plays back
- [ ] Camera shows frames in `camera_example3_viewfinder`
- [ ] Both models downloaded and tested on a laptop with `requirements.txt`
