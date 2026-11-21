# Hat Detect

Hack Western 2026 · QNX track

A wearable hat that **hears** important sounds (YAMNet) and **sees** important
objects (EfficientDet-Lite0) and alerts the wearer. All detection and alerts run
on a Raspberry Pi 5 running QNX, using TensorFlow Lite (`tflite-runtime` from
oss.qnx.com).

```
USB mic ──► io-audio ──► YAMNet ─────────┐
                                         ├──► alert logic ──► buzz / beep / voice
Camera  ──► QNX Camera ──► EfficientDet ─┘
```

## Status

Requirements only, no code yet.

## Docs

- [Requirements](docs/REQUIREMENTS.md): features, hardware, software, models, open decisions, setup checklist
- [Models](models/README.md): which `.tflite` files to download and where they go

## Repo layout

```
hat-detect/
├── docs/
│   └── REQUIREMENTS.md
├── models/            # .tflite files go here (git-ignored)
│   └── README.md
├── requirements.txt   # laptop-side Python tooling for testing models
├── .gitignore
└── README.md
```

## Quick start (laptop testing)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# then download the models listed in models/README.md
```
