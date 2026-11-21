# Tiger Data: hazard log and safety dashboard

Every hazard event from every hat is stored in a Tiger Cloud (TimescaleDB) service, and a
local dashboard shows the hazards found on site. An event is what the hat's fusion engine
(`fusion.py`) makes of one or more related model detections: a horn heard while a car is
in view is one `VEHICLE_HAZARD` event, not two separate detections.

```
Pi (QNX) ──EVENT lines over SSH──► laptop_companion.py ──► Tiger Cloud ◄── dashboard.py (localhost:8050)
                                    (tiger_bridge.py)        detections hypertable
                                                             detections_per_minute rollup
                                                             hazard_types register
```

The Pi never talks to the database: QNX has no Postgres client, and the password stays
on the laptop.

## 1. Create the service (once)

1. Sign up at <https://console.cloud.tigerdata.com/> and click **New service**.
2. When it's created, **download the config**. This is the only time the password is shown.
3. Copy `.env.example` to `.env` and paste the service URL:
   ```
   TIGER_DATABASE_URL=postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require
   ```
   `.env` is git-ignored. Never commit it.

## 2. Create the tables

```powershell
pip install "psycopg[binary]" python-dotenv
python scripts/tiger_bridge.py init      # safe to run again; migrates an older database
python scripts/tiger_bridge.py check
```

`db/schema.sql` creates:

| Object | What it holds |
|---|---|
| `detections` (hypertable) | One row per unified hazard event, upserted on `(event_id, time)` (see below) |
| `hazard_types` | Hazard register: severity, description and recommended control for each hazard, including the multimodal `VEHICLE_HAZARD` and `WORKER_DISTRESS` |
| `hazard_events` (view) | Unified events joined with the register's description and control. The dashboard reads this |
| `detections_per_minute` (continuous aggregate, real-time) | Counts and alerts per minute by device, hazard and label |
| `safety_alerts` (view) | Alerted rows joined with the register |
| Policies | Rollup refresh every minute; events kept 90 days |

### `detections` columns

| Column | Meaning |
|---|---|
| `event_id`, `time` | Unique key. `time` is when the event was first detected and never changes |
| `last_time` | Last detection that belonged to the event |
| `device_id` | Which hat |
| `fusion` | `AUDIO_ONLY`, `VISION_ONLY` or `MULTIMODAL` |
| `source` | `audio`, `video` or `combined` (the same split, kept for older queries) |
| `hazard`, `severity` | Hazard category and its configured severity, joined to `hazard_types` for the control |
| `label` | The labels in the event, e.g. `Siren + truck` |
| `audio_group`, `audio_label`, `audio_score`, `audio_count` | Audio part: group, YAMNet class, peak confidence, raw detections merged |
| `video_group`, `video_label`, `video_score`, `video_count`, `position`, `bbox` | Video part, the same, plus where it was in the frame |
| `score` | Strongest confidence of either part (the per-modality scores are the real record) |
| `correlation_window`, `explanation` | Multimodal only: the rule's window and why the two were combined |
| `alerted` | The wearer was alerted |
| `status` | Latest record received: `open`, `update` or `closed` |

Rows from before fusion (one per raw model detection) are kept, with `event_id` NULL.
`hazard_events` and the dashboard ignore them.

## 3. Record live data

`python scripts/laptop_companion.py` stores every event record as it arrives. Each event's
`open`, `update` and `closed` records upsert the same row, and an older record never
overwrites a newer one, so anything can be re-sent safely.

- **Offline:** if Tiger Cloud can't be reached, records go to `.tiger_spool.jsonl`. The
  bridge retries every 10 s, including while idle, and replays the spool before newer
  events. The hat's alerts never wait on the database.
- **Invalid records:** a record the database rejects goes to `.tiger_rejected.jsonl` with
  the reason, so it can't block the queue.
- **The Pi's own log:** the Pi keeps `~/hatdetect/events.jsonl`. Load it at any time
  without duplicates. Lines from before fusion are skipped and counted:

```powershell
python scripts/tiger_bridge.py ingest-pi
```

## 4. Dashboard

```powershell
python scripts/dashboard.py              # http://localhost:8050
```

It reads the `hazard_events` view, so it shows unified events and ignores rows from
before fusion. The page polls every 5 s, and events that arrive since the last poll are
highlighted.

**Filters** (one row, kept across reloads):

- Time range: 1 h, 8 h shift, 24 h, 7 days, or a custom from/to window.
- Hat, hazard, severity and event type (multimodal, audio only, vision only).
- Detection source: heard (has an audio part) or seen (has a video part).

Clicking a summary tile or a breakdown bar applies that filter.

**Panels:**

- **Summary:** hazard events, high severity, multimodal, audio only and vision only, each
  with its share of all events.
- **Events over time:** stacked by event type.
- **Breakdown:** by severity, by detection source, and multimodal versus single-modality.
- **Event timeline:** the latest 100 events with hazard, severity, type, audio and vision
  confidence, and the recommended action. Click a row, or use the arrow keys and Enter.
- **Event details:** the event ID and every field. Shows the audio and vision
  classifications, each with its peak confidence and detection count. Also shows the
  fusion explanation (why audio and video were combined, or why the event is
  single-modality), the recommended action and the alert status.
- **Events by hazard**, and **where events were around the wearer** (left, centre or
  right, from the camera).
- **The hazard register,** with event counts.

To try it before the hat is running, `python scripts/tiger_bridge.py demo` replaces the
`demo-01` sample data with 8 fresh hours (`--hours N` to change). It makes the samples by
running simulated detections through the real fusion engine and rules. `purge-demo`
removes them.
