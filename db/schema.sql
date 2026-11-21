-- Hat Detect: workplace-safety hazard log on Tiger Cloud (TimescaleDB).
-- Apply with:  python scripts/tiger_bridge.py init      (safe to run again)

-- Hazard register: one row per hazard category the hat can identify, with the
-- control the dashboard recommends. Severity matches the alert severity on the hat.
CREATE TABLE IF NOT EXISTS hazard_types (
    hazard       TEXT PRIMARY KEY,           -- ALARM, SIREN, PERSON, VEHICLE, ...
    source       TEXT NOT NULL,              -- audio | video | combined
    severity     TEXT NOT NULL CHECK (severity IN ('high', 'medium', 'low')),
    description  TEXT NOT NULL,
    control      TEXT NOT NULL               -- recommended action / control measure
);

INSERT INTO hazard_types (hazard, source, severity, description, control) VALUES
  ('COMBINED', 'combined', 'high',   'Horn or siren heard while a vehicle is in view',
                                     'Stop and step clear of the vehicle path; make eye contact with the operator'),
  ('ALARM',    'audio',    'high',   'Alarm, smoke detector, reversing beeper or buzzer',
                                     'Identify the alarm source; follow the evacuation or muster procedure'),
  ('SIREN',    'audio',    'high',   'Emergency vehicle or civil-defence siren',
                                     'Clear access routes; follow site emergency procedure'),
  ('IMPACT',   'audio',    'high',   'Crash, breaking glass, explosion or loud bang',
                                     'Check for falling objects or injured workers; secure the area'),
  ('FIRE',     'audio',    'high',   'Fire or crackling sounds',
                                     'Raise the fire alarm; evacuate and use an extinguisher only if trained'),
  ('HORN',     'audio',    'medium', 'Vehicle or air horn',
                                     'Look for moving plant or vehicles; stay in pedestrian zones'),
  ('VOICE',    'audio',    'medium', 'Shouting, screaming or a person in distress',
                                     'Check on co-workers; respond to calls for help'),
  ('VEHICLE',  'video',    'medium', 'Car, truck, bus, motorcycle, bicycle or train in view',
                                     'Keep exclusion zones around moving plant; use spotters'),
  ('SHARP',    'video',    'medium', 'Knife or scissors in view',
                                     'Store sharp tools safely; wear cut-resistant gloves'),
  ('PERSON',   'video',    'low',    'Person in the work area',
                                     'Confirm they are authorised and wearing PPE')
ON CONFLICT (hazard) DO UPDATE SET source = EXCLUDED.source, severity = EXCLUDED.severity,
  description = EXCLUDED.description, control = EXCLUDED.control;

-- Every detection from every hat. Hypertable partitioned on time.
CREATE TABLE IF NOT EXISTS detections (
    time       TIMESTAMPTZ NOT NULL,         -- when the hat detected it
    device_id  TEXT        NOT NULL,         -- which hat, e.g. hat-01
    source     TEXT        NOT NULL CHECK (source IN ('audio', 'video', 'combined')),
    hazard     TEXT        NOT NULL,         -- category (hazard_types.hazard)
    severity   TEXT        NOT NULL,         -- high | medium | low
    label      TEXT        NOT NULL,         -- model label, e.g. Siren, car, person
    score      REAL        NOT NULL,         -- model confidence 0..1
    position   TEXT,                         -- left | centre | right (video)
    bbox       JSONB,                        -- [x0, y0, x1, y1] in pixels (video)
    alerted    BOOLEAN     NOT NULL DEFAULT false  -- wearer was alerted (not in cooldown)
) WITH (
    tsdb.hypertable,
    tsdb.segmentby = 'device_id',
    tsdb.orderby = 'time DESC'
);

-- Lets the bridge re-send events (after a network drop, or from the Pi's log) without duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS detections_dedup ON detections (device_id, source, label, time);
CREATE INDEX IF NOT EXISTS detections_hazard_time ON detections (hazard, time DESC);

-- Per-minute counts for the dashboard. Real-time: includes rows not yet materialized.
CREATE MATERIALIZED VIEW IF NOT EXISTS detections_per_minute
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 minute', time) AS bucket,
       device_id, source, hazard, severity, label,
       count(*)                          AS detections,
       count(*) FILTER (WHERE alerted)   AS alerts,
       max(score)                        AS max_score
FROM detections
GROUP BY bucket, device_id, source, hazard, severity, label
WITH NO DATA;

SELECT add_continuous_aggregate_policy('detections_per_minute',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => true);

-- Keep raw detections for 90 days; the per-minute rollup keeps the long-term trend.
SELECT add_retention_policy('detections', INTERVAL '90 days', if_not_exists => true);

-- Alerts only, joined with the hazard register (what the wearer was warned about, and what to do).
CREATE OR REPLACE VIEW safety_alerts AS
SELECT d.time, d.device_id, d.source, d.hazard, d.severity, d.label, d.score, d.position,
       h.description, h.control
FROM detections d
LEFT JOIN hazard_types h USING (hazard)
WHERE d.alerted;
