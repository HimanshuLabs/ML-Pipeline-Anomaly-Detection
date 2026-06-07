-- Project 4: ML Pipeline & Anomaly Detection
-- Schema foundation for ML metadata, monitoring events, and audit history.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS audit;

COMMENT ON SCHEMA ml IS
'Stores Project 4 machine learning metadata including feature snapshots, training datasets, model registry records, metrics, baselines, and predictions.';

COMMENT ON SCHEMA monitoring IS
'Stores Project 4 runtime monitoring events including drift detections, alert events, latency breaches, and anomaly monitoring signals.';

COMMENT ON SCHEMA audit IS
'Stores Project 4 operational audit history including rollback events and production model change evidence.';
