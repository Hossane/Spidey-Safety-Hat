# Tiger Data: hazard log and safety dashboard

Every detection from every hat is stored in a Tiger Cloud service (managed PostgreSQL with
TimescaleDB). A dashboard on the laptop reads it and shows what is happening across the
site: who has had the most alerts, which zones are busiest, what happened to each worker,
and what supervisors have done about it.

```
Pi (QNX) ──EVENT lines over SSH──► laptop_companion.py ──► Tiger Cloud ◄── dashboard.py (localhost:8050)
  hat_detect.py                     (tiger_bridge.py)        detections hypertable
  ~/hatdetect/events.jsonl                                   detections_per_minute rollup
                                                             hazard_types · workers · activity log
```

**The Pi never talks to the database.** QNX has no PostgreSQL client library, and keeping
the password on the laptop means a lost hat doesn't leak it. The hat prints each event as
an `EVENT {json}` line, and the laptop companion inserts it.

## 1. Create the service (once)

1. Sign up at <https://console.cloud.tigerdata.com/> and click **New service**.
2. When it's created, **download the config**. This is the only time the password is shown.
3. Copy `.env.example` to `.env` and paste the service URL:
   ```
   TIGER_DATABASE_URL=postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require
   ```
   `.env` is git-ignored. Never commit it, and never copy it to the Pi.

## 2. Create the tables

```powershell
pip install "psycopg[binary]" python-dotenv
python scripts/tiger_bridge.py init      # applies db/schema.sql; safe to run again
python scripts/tiger_bridge.py check     # connects, checks TimescaleDB and the tables, shows rows per hat
```

To add the activity log tables (also safe to run again):

```powershell
python -c "import sys; sys.path.insert(0, 'scripts'); import tiger_bridge as t; t.connect(t.database_url(), autocommit=True).execute(open('db/activity_log.sql').read())"
```

### What gets stored

| Object | What it holds |
|---|---|
| `detections` (hypertable) | Every detection from every hat, partitioned on time. |
| `detections_per_minute` (continuous aggregate) | Detections and alerts per minute, by hat, hazard and label. The dashboard reads this, so it stays fast as history grows. It's real-time, so the last few minutes are included before they're rolled up. |
| `hazard_types` | The hazard register: severity, a description and the recommended control for each hazard, such as "Go to them now; call first aid and secure the area" for a person down. |
| `safety_alerts` (view) | Alerted detections joined with the register, used for the recent-alerts feed. |
| `workers` | Who wears each hat: name, role, site zone, and whether it's a simulated worker. |
| `supervisor_activity`, `supervisor_activity_history` | The supervisor activity log and every change to each entry (`db/activity_log.sql`). |
| Policies | The rollup refreshes every minute over the last 7 days. Raw detections are kept for 90 days. |

Columns of `detections`:

| Column | Example | Notes |
|---|---|---|
| `time` | `2026-11-22 05:23:01+00` | When the hat detected it (the Pi's clock) |
| `device_id` | `hat-01` | Which hat. Simulated hats are `sim-01`, `sim-02`, ... |
| `source` | `video` | `audio`, `video` or `combined` |
| `hazard` | `PERSON_DOWN` | The alert group; matches `hazard_types` |
| `severity` | `high` | `high`, `medium` or `low` |
| `label` | `person` | What the model saw or heard, e.g. `Siren`, `truck` |
| `score` | `0.84` | Model confidence, 0–1 |
| `position` | `left` | Where in the camera image (video only) |
| `bbox` | `[129, 270, 637, 443]` | Box in pixels (video only) |
| `alerted` | `true` | Whether the wearer was warned (false while that alert is in its cooldown) |

A unique index on `(device_id, source, label, time)` means the same event can be sent
twice without creating a duplicate. That's what makes re-sending after a network drop safe.

## 3. Record live data

`python scripts/laptop_companion.py` stores every event as it arrives, in batches of up to
one second. If Tiger Cloud can't be reached, events go to `.tiger_spool.jsonl` on the
laptop and are retried every 10 s until the connection comes back.

The Pi also keeps its own log (`~/hatdetect/events.jsonl`), so nothing is lost even if the
laptop wasn't connected. Load it at any time without duplicates:

```powershell
python scripts/tiger_bridge.py ingest-pi --host qnxpi-eth   # read the Pi's log over SSH
python scripts/tiger_bridge.py ingest events.jsonl          # or a log file you copied
```

**Check the Pi's clock first.** The Pi has no clock battery. If it can't reach the internet
after a reboot, it starts at 1 January 1970, and every event is stored with a 1970
timestamp that the dashboard never shows. From Git Bash, check with `ssh qnxpi-eth date`
and set it from the laptop with `ssh qnxpi-eth "sudo date -u -s @$(date +%s)"`.

## 4. Dashboard

```powershell
python scripts/dashboard.py              # http://localhost:8050
python scripts/dashboard.py --port 9000  # another port
```

It refreshes every 5 s, has a time range filter (1 h, 8 h shift, 24 h, 7 days) and a
**Simulated crew** switch. The dot next to "live" turns amber if updates stop for 15 s
and red if Tiger Cloud can't be reached. By default it shows only real detections from the hat.

### Site overview

- **Headline numbers:** workers (and how many are active), alerts, high-severity alerts,
  workers with high alerts, and the number of rows stored in Tiger with how long the
  queries took.
- **Site alerts over time**, stacked by severity.
- **Alerts by site zone.**
- **Workers, highest risk first.** Risk is 5 × high + 2 × medium + 1 × low alerts. Each
  row has a sparkline, the top hazard and the last alert. Search by name, role or zone,
  and click a worker to open their own view.
- **What the data suggests:** the zone with the most alerts, the worker with the most
  alerts, and whether alerts rose or fell between the first and second half of the time
  range.

Register a new hat:

```sql
INSERT INTO workers (device_id, name, role, zone) VALUES ('hat-02', 'Name', 'Role', 'Zone');
```

A hat that reports before it's registered still appears, under its `device_id`.

### Worker tab

The same view for one hat:
- **Panels:**
  - headline numbers and alerts over time;
  - alerts by hazard;
  - where alerts came from around the wearer (left, centre or right of the camera);
  - recent alerts, each with its recommended control;
  - the hazard register.
- **Incident timeline:** alerts less than 20 s apart are grouped into one incident,
  showing what the camera and mic each caught, in order.
- **What the data suggests:** the hazard this hat records most, which side of the image
  most alerts come from, and the first-half / second-half change.
- **Emergency banner:** shown on both tabs. Any SOS or simulated 911 call from the last 15
  minutes appears in red at the top.

The insight cards describe recorded alert counts only. They don't measure hazard rates or
rank workers' behaviour, and their wording says so.

### Activity log

The **Activity log** tab is where supervisors record what they did about the alerts:
inspections, corrective actions, status updates and notes.
- **Linking:** each entry can be linked to a zone, a hat and a hazard.
- **Status:** each entry is Open, In Progress or Resolved. Changing the status needs a
  note, and every change is kept with who made it and when.
- **Finding entries:** the list has filters by status and type, plus a search box.
- **Simulated data:** entries marked as relating to it only show with **Simulated crew** on.
- **Typing is safe:** the 5 s refresh doesn't wipe an update note you're typing.

The dashboard only accepts new entries sent as JSON from its own page. A browser won't let
another website send that without the server's permission, which it never gives, so other
websites can't add entries.

## 5. Simulated crew (proof of concept for scale)

There is only one real hat, so a simulated crew shows what a full site looks like:

```powershell
python scripts/sim_crew.py add                      # 12 workers, last 8 h (device ids sim-01...)
python scripts/sim_crew.py add --workers 40 --hours 24
python scripts/sim_crew.py add --live               # ...then keep adding detections live
python scripts/sim_crew.py topup                    # fill in from where their data stopped up to now
python scripts/sim_crew.py remove                   # delete all simulated workers and data
```

- **The crew:** each simulated worker has a role and zone, such as a forklift operator on
  the loading dock or a welder in the fabrication shop. Their detections follow a realistic
  hazard mix for that role, in short bursts like the real hat produces.
- **Only with the switch on:** simulated workers only appear when the dashboard's
  **Simulated crew** switch is on. They're badged SIM, and the real hat is badged LIVE.
- **Kept current:** while the switch is on, the dashboard tops the crew up every 30 s,
  adding detections from where they stopped up to now, at most 8 h back. So the overview
  and the insight cards always include them.
- **Real data is never touched:** it only does this when simulated workers exist (after
  `sim_crew.py add`), and it only writes `sim-` rows.

## 6. Example queries

Run these in the Tiger Cloud console's SQL editor, or with `psql`:

```sql
-- Alerts per hazard on one hat today
SELECT hazard, severity, sum(alerts) AS alerts
FROM detections_per_minute
WHERE device_id = 'hat-01' AND bucket > now() - interval '1 day'
GROUP BY hazard, severity ORDER BY alerts DESC;

-- Every high-severity alert in the last hour, with what to do about it
SELECT time, device_id, hazard, label, control
FROM safety_alerts
WHERE severity = 'high' AND time > now() - interval '1 hour'
ORDER BY time DESC;

-- The five busiest 10-minute periods across the site
SELECT time_bucket('10 minutes', bucket) AS t, sum(alerts) AS alerts
FROM detections_per_minute
GROUP BY t ORDER BY alerts DESC LIMIT 5;
```
