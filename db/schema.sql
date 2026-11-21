-- Hat Detect: workplace-safety hazard log on Tiger Cloud (TimescaleDB).
-- Apply with:  python scripts/tiger_bridge.py init      (safe to run again; it also
-- migrates a database created by the earlier raw-detection schema)

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
  ('VEHICLE_HAZARD',  'combined', 'high', 'Horn or siren heard while a vehicle is in view',
                                     'Stop and step clear of the vehicle path; make eye contact with the operator'),
  ('WORKER_DISTRESS', 'combined', 'high', 'Shouting, screaming or a crash while a person is in view',
                                     'Check on the worker; call for first aid and secure the area'),
  ('COMBINED', 'combined', 'high',   'Horn or siren heard while a vehicle is in view (before fusion rules; now VEHICLE_HAZARD)',
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

-- One row per unified hazard event (fusion.py on the hat), upserted on (event_id, time)
-- as the event is opened, upgraded and closed. Hypertable partitioned on time.
-- Rows written before fusion (one per raw model detection) have event_id NULL; the
-- dashboard and the hazard_events view ignore them.
CREATE TABLE IF NOT EXISTS detections (
    time        TIMESTAMPTZ NOT NULL,        -- when the event was first detected (fixed)
    device_id   TEXT        NOT NULL,        -- which hat, e.g. hat-01
    source      TEXT        NOT NULL CHECK (source IN ('audio', 'video', 'combined')),
    hazard      TEXT        NOT NULL,        -- category (hazard_types.hazard)
    severity    TEXT        NOT NULL,        -- high | medium | low
    label       TEXT        NOT NULL,        -- e.g. "Siren + truck", "person"
    score       REAL        NOT NULL,        -- strongest model confidence in the event, 0..1
    position    TEXT,                        -- left | centre | right (video)
    bbox        JSONB,                       -- [x0, y0, x1, y1] in pixels (video)
    alerted     BOOLEAN     NOT NULL DEFAULT false,  -- the wearer was alerted
    event_id    UUID,                        -- unified event id (NULL on pre-fusion rows)
    fusion      TEXT,                        -- AUDIO_ONLY | VISION_ONLY | MULTIMODAL
    status      TEXT,                        -- open | update | closed (latest record seen)
    last_time   TIMESTAMPTZ,                 -- last detection that belonged to the event
    audio_group TEXT,                        -- audio part: group, YAMNet class, peak score,
    audio_label TEXT,                        --   number of raw detections merged
    audio_score REAL,
    audio_count INTEGER,
    video_group TEXT,                        -- video part: group, COCO class, peak score,
    video_label TEXT,                        --   number of raw detections merged
    video_score REAL,
    video_count INTEGER,
    correlation_window REAL,                 -- multimodal: the rule's window, seconds
    explanation TEXT                         -- multimodal: why the two were combined
) WITH (
    tsdb.hypertable,
    tsdb.segmentby = 'device_id',
    tsdb.orderby = 'time DESC'
);

-- Migration from the raw-detection schema (no-ops on a fresh database). Plain nullable
-- columns only: constraints can't be added to a hypertable with the columnstore on, so
-- tiger_bridge.py validates fusion and status before inserting.
ALTER TABLE detections ADD COLUMN IF NOT EXISTS event_id UUID;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS fusion TEXT;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS last_time TIMESTAMPTZ;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS audio_group TEXT;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS audio_label TEXT;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS audio_score REAL;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS audio_count INTEGER;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS video_group TEXT;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS video_label TEXT;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS video_score REAL;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS video_count INTEGER;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS correlation_window REAL;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS explanation TEXT;

-- Idempotent ingestion: the bridge upserts on this, so re-sent events (after a network
-- drop, or from the Pi's log) never duplicate. Hypertable unique indexes must include time.
CREATE UNIQUE INDEX IF NOT EXISTS detections_event ON detections (event_id, time);
-- The old dedup key now only applies to pre-fusion rows, so it can't reject an event.
DROP INDEX IF EXISTS detections_dedup;
CREATE UNIQUE INDEX IF NOT EXISTS detections_dedup_legacy ON detections (device_id, source, label, time)
    WHERE event_id IS NULL;
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

-- Keep events for 90 days; the per-minute rollup keeps the long-term trend.
SELECT add_retention_policy('detections', INTERVAL '90 days', if_not_exists => true);

-- Alerts only, joined with the hazard register (what the wearer was warned about, and what to do).
CREATE OR REPLACE VIEW safety_alerts AS
SELECT d.time, d.device_id, d.source, d.hazard, d.severity, d.label, d.score, d.position,
       h.description, h.control
FROM detections d
LEFT JOIN hazard_types h USING (hazard)
WHERE d.alerted;

-- Unified hazard events with their recommended control: what the dashboard reads.
CREATE OR REPLACE VIEW hazard_events AS
SELECT d.event_id, d.time, d.last_time, d.device_id, d.hazard, d.severity, d.fusion, d.source,
       d.label, d.score, d.audio_group, d.audio_label, d.audio_score, d.audio_count,
       d.video_group, d.video_label, d.video_score, d.video_count, d.position, d.bbox,
       d.correlation_window, d.explanation, d.alerted, d.status,
       h.description, h.control
FROM detections d
LEFT JOIN hazard_types h USING (hazard)
WHERE d.event_id IS NOT NULL;
