# Tiger Data: hazard log and safety dashboard

Every detection from every hat is stored in a Tiger Cloud (TimescaleDB) service, and a
local dashboard shows the hazards found on site.

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
python scripts/tiger_bridge.py init      # safe to run again
python scripts/tiger_bridge.py check
```

`db/schema.sql` creates:

| Object | What it holds |
|---|---|
| `detections` (hypertable) | Every detection: time, device_id, source, hazard, severity, label, score, position, bbox, alerted |
| `hazard_types` | Hazard register: severity, description and recommended control for each hazard |
| `detections_per_minute` (continuous aggregate, real-time) | Counts and alerts per minute by device, hazard and label |
| `safety_alerts` (view) | Alerted detections joined with the register |
| Policies | Rollup refresh every minute; raw detections kept 90 days |

## 3. Record live data

`python scripts/laptop_companion.py` now stores every event as it arrives. If Tiger Cloud
can't be reached, events go to `.tiger_spool.jsonl` and are sent when the connection
comes back. The Pi also keeps its own log (`~/hatdetect/events.jsonl`); load it at any
time without duplicates:

```powershell
python scripts/tiger_bridge.py ingest-pi
```

## 4. Dashboard

```powershell
python scripts/dashboard.py              # http://localhost:8050
```

It refreshes every 5 s and has filters for time range (1 h, 8 h shift, 24 h, 7 days)
and hat. Panels:

- Headline numbers: detections, alerts, high-severity alerts, hazard types, last alert.
- Alerts over time, by severity.
- Hazards identified.
- Where alerts came from around the wearer (left/centre/right).
- Recent alerts, each with its recommended control.
- The hazard register.

To try it before the hat is running: `python scripts/tiger_bridge.py demo` adds 8 hours
of sample data as device `demo-01`; `purge-demo` removes it.
