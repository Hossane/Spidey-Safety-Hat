-- Hat Detect supervisor activity log (additive migration; safe to run more than once).
-- Run against the same Tiger Data database used by scripts/dashboard.py.
-- This does not alter detections, workers, existing views, or alert calculations.

CREATE TABLE IF NOT EXISTS supervisor_activity (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    supervisor   TEXT NOT NULL CHECK (length(trim(supervisor)) BETWEEN 1 AND 100),
    action_type  TEXT NOT NULL CHECK (action_type IN ('Inspection', 'Corrective action', 'Status update', 'Note')),
    status       TEXT NOT NULL DEFAULT 'Open' CHECK (status IN ('Open', 'In Progress', 'Resolved')),
    zone         TEXT,
    device_id    TEXT,
    hazard       TEXT,
    summary      TEXT NOT NULL CHECK (length(trim(summary)) BETWEEN 1 AND 180),
    details      TEXT,
    follow_up    TEXT,
    simulated    BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS supervisor_activity_created_at_idx
    ON supervisor_activity (created_at DESC);

CREATE INDEX IF NOT EXISTS supervisor_activity_status_idx
    ON supervisor_activity (status, created_at DESC);

CREATE INDEX IF NOT EXISTS supervisor_activity_zone_idx
    ON supervisor_activity (zone, created_at DESC);

CREATE TABLE IF NOT EXISTS supervisor_activity_history (
    id           BIGSERIAL PRIMARY KEY,
    activity_id  BIGINT NOT NULL REFERENCES supervisor_activity(id) ON DELETE CASCADE,
    changed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    changed_by   TEXT NOT NULL CHECK (length(trim(changed_by)) BETWEEN 1 AND 100),
    status       TEXT NOT NULL CHECK (status IN ('Open', 'In Progress', 'Resolved')),
    note         TEXT NOT NULL CHECK (length(trim(note)) BETWEEN 1 AND 1000)
);

CREATE INDEX IF NOT EXISTS supervisor_activity_history_activity_idx
    ON supervisor_activity_history (activity_id, changed_at DESC);
