# Hat Detect: current architecture (Phase 0 audit)

How the code works as of commit `f9548d0`, before the multimodal fusion work. Later
phases update this file as they change the design.

```
Laptop mic ──s16le 16 kHz over SSH stdin──► hat_detect.py (Pi, QNX)
                                              ├─ audio_worker ─► YAMNet ─┐
USB webcam ──QNX Camera (unit 5)───────────►  ├─ video_worker ─► EffDet ─┤
                                              │                          ▼
                                              │                 AlertLogic.handle()
                                              │                  ├─ GPIO buzzer (optional)
                                              │                  ├─ ~/hatdetect/events.jsonl
                                              └──────────────────┴─ stdout: EVENT {json}
                                                                          │ (SSH stdout)
                                                                          ▼
                                     scripts/laptop_companion.py ──► Speaker (tone + TTS on JBL)
                                                                └──► TigerWriter ──► Tiger Cloud
                                                                     (tiger_bridge.py)   │
                                                         scripts/dashboard.py ◄──────────┘
                                                         (polls /api/data every 5 s)
```

## 1. Audio detections

- `audio_detect.Yamnet.analyze()` scores one 0.975 s window and returns
  `(group, yamnet_class, score)` hits above the threshold. Groups come from `SAFETY` in
  `audio_detect.py`: ALARM, SIREN, HORN, VOICE, IMPACT and FIRE.
- `hat_detect.audio_worker` runs a window every `hop_seconds` (0.5 s). It keeps the
  strongest class in each group and calls `logic.handle("audio", group, class, score)`.
- A horn lasting 2 s therefore produces about 4 raw detections.

## 2. Visual detections

- `vision_detect.Detector.detect()` returns `(coco_label, score, box, group)`. Groups
  come from `HAZARDS`: PERSON, VEHICLE (car, truck, bus, motorcycle, bicycle, train) and
  SHARP (knife, scissors).
- `hat_detect.video_worker` runs at `fps` (5). It keeps the strongest detection in each
  group per frame and calls `logic.handle("video", group, label, score, position, bbox)`.
  `position` is left, centre or right, taken from the box centre.
- A car in view for 10 s produces about 50 raw detections.

## 3. Correlation (`hat_detect.AlertLogic`)

- **Cooldown:** each group alerts at most once per `cooldown_seconds` (3 s). Every raw
  detection is still emitted, with `alerted` set to true or false.
- **Combining:** `_combine()` pairs HORN or SIREN (audio) with VEHICLE (video) when both
  were seen within `combine_window_seconds` (2 s). It emits an extra event with
  `source="combined"`, `group="COMBINED"`, `label="<audio> + <video>"` and
  `score=min(a, v)`. The pairing sets `COMBINE_AUDIO` and `COMBINE_VIDEO` are hard-coded.

Limits this work needs to address:

- The raw detections still alert on their own. A horn with a car in view produces a HORN
  alert, a VEHICLE alert and a COMBINED alert, which means three spoken warnings.
- A COMBINED event is emitted on every raw detection inside the window, not once per
  incident.
- `score=min(a, v)` throws away both original confidences.
- PERSON alerts every 3 s whenever someone is in view, and so does VEHICLE whenever a
  parked car is.
- There is no event ID, and `severity` is not included in the event.

## 4. EVENT JSON line

```json
{"time":"2026-11-21T19:30:00.123+00:00","device_id":"hat-01","source":"audio|video|combined",
 "label":"Siren","score":0.91,"position":null,"bbox":null,"alerted":true,"group":"SIREN"}
```

Each line is printed as `EVENT <json>` on stdout and appended to `~/hatdetect/events.jsonl`.

## 5. Pi to laptop

- `laptop_companion.py` runs `ssh qnxpi "cd ~/hatdetect && python3 -u hat_detect.py"`.
  It writes mic PCM to the SSH stdin and reads stdout line by line.
- For each event with `alerted: true`, it builds a phrase (`phrase()`) and queues a tone
  plus speech. Speech is dropped if it has waited more than 4 s, and there is no priority
  or interruption.
- Only the laptop speaks. The Pi only buzzes, so no event is announced twice across
  devices.
- Speech uses Windows `System.Speech` through PowerShell, which does not work on macOS.

## 6. Tiger Cloud ingestion (`scripts/tiger_bridge.py`)

- `TigerWriter.add(ev)` never blocks. A background thread inserts batches of up to 1 s
  or 200 events.
- On failure, batches are written to `.tiger_spool.jsonl` and retried every 10 s. The
  spool is replayed on reconnect.
- Duplicates are prevented by the unique index `(device_id, source, label, time)` with
  `ON CONFLICT DO NOTHING`.
- `ingest` and `ingest-pi` load JSONL logs through the same `insert_events()`.
- Severity is filled in from a copy of the severity map when the event has none.

## 7. Dashboard (`scripts/dashboard.py` + `dashboard.html`)

- A stdlib `ThreadingHTTPServer` serves `/api/data?range=&device=`, and the page polls
  it every 5 s.
- Queries use `detections` for the headline numbers, positions and register stats,
  `detections_per_minute` for the timeline and per-hazard totals, and `safety_alerts`
  for recent alerts with their control.
- All counts are raw model detections. "Alerts" means rows with `alerted = true`.

## Duplicated configuration

The hazard severity map is defined four times: in `hat_config.json`, in
`hat_detect.DEFAULTS`, as `laptop_companion.SEVERITY` and as `tiger_bridge.SEVERITY`.
`db/schema.sql`'s `hazard_types.severity` is a fifth copy.

## Environment notes

- There are no automated tests in the repo.
- Local Python is 3.14 with no numpy or psycopg installed. Importing `hat_detect` pulls
  in numpy and tflite through `audio_detect` and `vision_detect`, so fusion logic that
  needs testing on the laptop should live in its own module with no numpy dependency.
- There is no `.env` locally, so Tiger Cloud is not reachable from this checkout yet.
- Hypertable unique indexes must include the `time` column. A future `event_id`
  uniqueness rule therefore has to be `UNIQUE (event_id, time)`.

---

# Phases 1 and 2: unified events and the fusion engine

New module `fusion.py`, written in plain Python with no numpy or TFLite so it runs on
QNX and on the laptop. It is configured by the `fusion` section of `hat_config.json`.
Until Phase 3 it is not wired into `hat_detect.py`.

## Raw detection vs hazard event

- `fusion.Detection` is one model prediction:
  `source, group, label, score, t (monotonic), time (UTC ISO), position, bbox`.
- `FusionEngine.add(detection)` and `FusionEngine.tick(t)` return unified **event
  records**:

```json
{"event_id": "3f6c…", "time": "2026-11-21T19:30:00+00:00", "last_time": "…",
 "device_id": "hat-01", "hazard": "VEHICLE_HAZARD", "severity": "high",
 "fusion": "MULTIMODAL",
 "label": "Vehicle horn, car horn, honking + car",
 "audio": {"group": "HORN", "label": "Vehicle horn, car horn, honking", "score": 0.94, "detections": 3},
 "video": {"group": "VEHICLE", "label": "car", "score": 0.91, "detections": 12,
           "position": "left", "bbox": [10, 20, 200, 220]},
 "correlation": {"window_seconds": 3.0,
                 "explanation": "Vehicle horn… (HORN) heard 1.0 s before car (VEHICLE) was seen on the left; …"},
 "alerted": true, "notify": true, "status": "open"}
```

| Field | Meaning |
|---|---|
| `event_id` | UUID4. Stays the same across the event's open, update and closed records |
| `fusion` | `AUDIO_ONLY`, `VISION_ONLY` or `MULTIMODAL` |
| `hazard` | The group (`HORN`, `VEHICLE`, …), or the rule's hazard (`VEHICLE_HAZARD`, `WORKER_DISTRESS`) |
| `severity` | From the config rules, never from confidence |
| `audio` / `video` | Each modality's peak detection, its original score and the number of raw detections merged. `null` if absent |
| `correlation` | Only for multimodal events: the rule window and a readable explanation |
| `alerted` | The wearer has been alerted for this event |
| `notify` | This record should be announced. Only the `open` record, or an `update` that raised the severity |
| `status` | `open` on first announcement, `update` when upgraded to multimodal, `closed` when the episode ends |

There is no recommended action in the record. It comes from `hazard_types.control` in
Tiger Cloud, which stays the single source for it.

## Fusion behaviour

- **Episodes** (duplicate suppression). Detections of the same group extend one open
  episode. The episode closes after `episode_gap_seconds` (3 s) without detections, or
  after the longest window of a rule it could join, whichever is longer. A different
  hazard always starts its own episode.
- **Rules.** Detections combine only when a rule pairs their groups:
  - HORN or SIREN with VEHICLE gives `VEHICLE_HAZARD` (high).
  - VOICE or IMPACT with PERSON gives `WORKER_DISTRESS` (high).
  - Their two sightings must fall within `window_seconds` (3 s) of each other.
- **Confirmation wait.** A detection that a partner could make more severe (HORN, VOICE,
  VEHICLE, PERSON) waits `confirm_wait_seconds` (1 s) before it is announced. A
  detection that is already high (SIREN, IMPACT, ALARM, FIRE) is announced immediately.
- **Late confirmation.** If the partner arrives after the event was announced, the same
  event is upgraded to MULTIMODAL. It is announced again only if the severity went up
  (`realert_on_escalation`).
- **Severity policy.** A rule's `severity` is used when set. Otherwise, with
  `escalate_on_confirmation` on, the combined event is one level above its higher part.
  With it off, the event takes the higher part's severity.

## Tests

`python3 -m unittest discover -s tests -v` runs 20 tests in `tests/test_fusion.py`, all
using simulated detections. They cover CLAUDE.md tests 1 to 5 and the severity policy.

---

# Phase 3: alerting on unified events

## Pi (`hat_detect.py`)

- `AlertLogic` (per-group cooldown, plus the hard-coded horn or siren with vehicle
  pairing) is replaced by `EventPipeline`. It wraps `fusion.FusionEngine`: workers call
  `handle()` for each raw detection, and the main loop calls `tick()` every 0.2 s.
  `close()` flushes open events on shutdown.
- `EVENT` lines and `event_log` (`~/hatdetect/events.jsonl`) now carry **unified event
  records**: one `open` record, an `update` record if the event is upgraded, and one
  `closed` record per event. Raw detections go to `detection_log`
  (`~/hatdetect/detections.jsonl`, set to `null` to turn it off).
- The buzzer pulses only for records with `notify: true`, which means once per event,
  plus once more if the event is upgraded to a higher severity.
- `alerts.cooldown_seconds` and `alerts.combine_window_seconds` are removed from the
  config. Their jobs moved to `fusion.episode_gap_seconds` and each rule's
  `window_seconds`. The `COMBINED` severity entry is gone too, because rules now set the
  severity of combined events.
- The model modules (numpy, tflite) are imported inside the workers and `main()`, so the
  event pipeline can be tested on a laptop.
- **Deploying:** the Pi now needs `fusion.py` next to `hat_detect.py`:
  `scp fusion.py hat_detect.py hat_config.json qnxpi:hatdetect/`.

## Laptop (`scripts/laptop_companion.py`)

- The laptop speaks only records with `notify: true`, so it never decides alerts itself
  and nothing is announced by two parts of the system independently. The duplicate
  `SEVERITY` map is removed, and each record's own `severity` is used.
- New phrases:

  | Event | Phrase |
  |---|---|
  | `VEHICLE_HAZARD` | "Warning! Vehicle hazard on your left" |
  | `WORKER_DISTRESS` | "Warning! Worker in distress ahead" |
  | Any other multimodal hazard | "Warning! <hazard name>" |
  | Audio-only | "Siren", "Someone shouting" (no direction) |
  | Vision-only | "Person on your right", "Truck ahead" |

- `Speaker`:
  - Uses a priority queue, so the most severe alert plays first.
  - A **more severe alert interrupts** the tone or speech that is playing, and the
    interrupted alert is dropped. An equal or lower severity alert waits its turn.
  - Alerts older than 4 s are dropped.
- Speech backends:
  - macOS: `say`, one process per phrase, terminated when cancelled.
  - Windows: one persistent PowerShell System.Speech process, killed when cancelled and
    restarted for the next phrase.
  - Linux: `espeak-ng` or `espeak`.
  - Otherwise the phrase is printed.

## Tests

`tests/test_alerting.py` (11 tests) checks:

- The Pi pipeline: horn plus car gives one record stream and one buzz.
- `close()` flushes open events.
- Alerts and logging continue when the laptop disconnects.
- The phrases.
- The Speaker: interruption, no interruption at equal severity, severity order, and
  stale alerts being dropped.

All of this uses fakes for the buzzer, voice and clock.

## Known gap until Phase 4

`tiger_bridge.to_row()` still expects the old raw-detection fields (`source`, `score`).
Unified records fail to insert, so `TigerWriter` treats this as being offline and keeps
them in `.tiger_spool.jsonl`. No data is lost, and Phase 4 makes them insertable.

---

# Phase 4: unified events in Tiger Cloud

- **Schema (`db/schema.sql`):** `detections` now holds one row per unified event. Columns
  added:
  - `event_id`, `fusion`, `status`, `last_time`
  - Audio part: `audio_group`, `audio_label`, `audio_score`, `audio_count`
  - Video part: `video_group`, `video_label`, `video_score`, `video_count`
  - Multimodal only: `correlation_window`, `explanation`

  Other changes:
  - A new unique index on `(event_id, time)`.
  - The old dedup index is now partial (`WHERE event_id IS NULL`), so it can't reject an
    event.
  - New `hazard_types` rows for `VEHICLE_HAZARD` and `WORKER_DISTRESS`.
  - A new `hazard_events` view.

  `init` migrates the existing database in place with `ADD COLUMN IF NOT EXISTS`. Old
  rows are kept, and retention (90 days), the per-minute rollup and `safety_alerts` are
  unchanged. Recommended actions still come only from `hazard_types`.
- **Bridge (`scripts/tiger_bridge.py`):**
  - `to_row()` maps unified records to rows and validates `fusion`, `status` and
    `severity`.
  - `INSERT_SQL` upserts on `(event_id, time)`. Its `WHERE` clause on status rank means a
    replayed `open` never overwrites `closed`, and `alerted` is OR-ed.
  - Batches are collapsed to the latest record per event.
  - Only connection errors count as offline (spool and retry). Data errors fall back to
    inserting row by row, and the bad rows go to `.tiger_rejected.jsonl`.
  - When idle, the writer still retries a leftover spool.
  - The writer's connection, spool path and retry interval can be injected for tests.
  - `ingest` and `ingest-pi` skip pre-fusion lines and report them.
  - `demo` builds sample events with `fusion.FusionEngine`, and replaces old demo rows
    instead of adding to them.
- **Unchanged:** `.env` / `TIGER_DATABASE_URL`, the spool file, the Pi never connecting to
  the database, and `laptop_companion.py`'s `TigerWriter` use.
- **Tests (`tests/test_tiger_bridge.py`, 12):** use an in-memory database with the same
  upsert rule. They cover:
  - Audio-only and multimodal events stored as one row each.
  - An upgrade updating the same row.
  - Re-sending a whole log, or stale records, changing nothing.
  - Test 6: offline, then spooled, then reconnected, then sent once, with `add()` never
    blocking.
  - Replaying a spool when idle, and keeping it on a shutdown while offline.
  - Malformed records and database-rejected rows being isolated.
  - Pre-fusion lines being skipped.
  - Demo events being insertable.
- **Not yet run against real Postgres or TimescaleDB:** the SQL text (migration, upsert
  and view). See the "verify" step in the Phase 4 summary.

---

# Phase 5: dashboard on unified events

Same stack as before: a stdlib HTTP server, one HTML page, Chart.js 4.4.1 from cdnjs, and
polling every 5 s. No new dependencies.

- **Server (`scripts/dashboard.py`):**
  - `parse_filters()` validates the query string. It handles range presets and a custom
    `from`/`to` (clamped to 90 days and to now), plus device, hazard, severity, fusion
    and source. Unknown values are dropped.
  - `where()` builds one parameterised `WHERE` clause that every query shares.
  - `gather()` runs seven small queries on `hazard_events` and `hazard_types`, and
    `shape()` turns the rows into JSON.
  - Timeline buckets scale with the span: 1 min, 10 min, 30 min, 6 h or 1 day. They are
    aligned the way `time_bucket` aligns them, and empty buckets are filled with zero.
  - The per-minute rollup is no longer used by the dashboard, because it can't filter by
    fusion type. Event volumes are small, so querying the view directly is fine.
- **Page (`scripts/dashboard.html`):** the panels are listed in `docs/TIGER_DATA.md`.
  Colors follow the data-viz method:
  - Event type uses fixed categorical slots: multimodal is blue, audio only is orange,
    vision only is aqua. These were checked with the palette validator, which passes
    light and dark mode. Aqua is below 3:1 contrast in light mode, so every type always
    carries a text label and the table is the fallback.
  - Severity uses the status palette, always with an icon (▲, ◆, ●) and a label.
  - Text never uses a data color.
  - Light and dark mode are each chosen explicitly, and `data-theme` overrides the
    system setting.
  - Every bar has a hover tooltip, the event table works from the keyboard, and the
    layout is responsive.
- **Tests (`tests/test_dashboard.py`, 18):** run `gather()` against
  `tests/fake_events_db.py`, which answers the same queries in Python from demo events
  made by the real fusion engine. They cover CLAUDE.md test 7:
  - The summary numbers match the events, and the timeline adds up to the total.
  - Multimodal events carry both scores and an explanation.
  - Each filter, and combinations of filters, narrow the results correctly.
  - Custom windows, and bad values being rejected.
  - Filter values are bound as parameters, never pasted into SQL.
  - The HTTP endpoints.
- **Checked visually:** with headless Chrome against the fake database, in light mode
  (with an event selected) and dark mode, at 1440 px. There is no horizontal overflow at
  500, 600 or 900 px. Headless Chrome can't lay out narrower than 500 px.
- **Not yet checked:** the dashboard's SQL against a real TimescaleDB.

---

# Phase 6: testing

| Suite | Tests | Needs |
|---|---|---|
| `tests/test_fusion.py` | 21 | nothing |
| `tests/test_alerting.py` | 11 | nothing |
| `tests/test_tiger_bridge.py` | 12 | nothing (in-memory database) |
| `tests/test_dashboard.py` | 18 | nothing (`tests/fake_events_db.py`) |
| `tests/test_integration_db.py` | 9 | `HATDETECT_TEST_DATABASE_URL`: a local PostgreSQL + TimescaleDB, and psycopg |

How these map to CLAUDE.md's tests:

| CLAUDE.md test | Where |
|---|---|
| 1. Audio-only | `Test1AudioOnly`, plus the storage check in `TestStoringEvents` |
| 2. Vision-only | `Test2VisionOnly` |
| 3. Multimodal | `Test3Multimodal`, `test_horn_and_car_give_one_alert` and `test_multimodal_event_is_one_row_with_both_scores` |
| 4. Correlation timeout | `Test4CorrelationTimeout` |
| 5. Duplicate suppression | `Test5DuplicateSuppression` |
| 6. Database disconnection | `TestDatabaseDisconnection` with the fake database, plus `test_t6_offline_then_reconnect_sends_once` with the real driver and database |
| 7. Dashboard | `tests/test_dashboard.py`, plus `test_t9_dashboard_sql_matches_the_reference` (real SQL equals the reference for 9 filter combinations) and a manual check in headless Chrome |

What the integration suite found:

- The migration from the pre-fusion schema (commit `f9548d0`) works on TimescaleDB
  2.32.1, including when old chunks are compressed.
- Upserts into a compressed chunk work.
- `init` is safe to run twice.

Results: 71 of 71 pass with the database, and 62 pass with 9 skipped without it (macOS,
Python 3.14, TimescaleDB 2.32.1 / Postgres 17 in Docker).

Not covered, because it needs the hardware:

- `hat_detect.py` with real models and camera on QNX.
- `laptop_companion.py`'s SSH loop.
- The Windows speech backend.

# Phase 7: documentation

- `README.md` is rewritten from the code, the schema and the test results. It has two
  Mermaid diagrams (hardware, and software and data flow), both checked by rendering them
  with Mermaid 11.
- Every command in the README was run or checked against its script's `--help`, apart
  from the Pi-side and hardware commands, which come from the existing scripts' own usage
  text.
- `.env.example` no longer lists `DEVICE_ID`, which no code reads. The device ID is
  `device_id` in `hat_config.json`.
