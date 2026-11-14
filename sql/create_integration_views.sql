-- Project 4: ML Pipeline & Anomaly Detection
-- Read-only integration surface for Project 5 (role project5_reader).
-- Every view exposes the Project 5 source metadata columns first, then business columns.
-- Re-runnable: CREATE OR REPLACE keeps the same column list.

CREATE SCHEMA IF NOT EXISTS integration;

COMMENT ON SCHEMA integration IS
'Read-only views over Project 4 metadata for downstream consumers (Project 5). No base tables.';

CREATE OR REPLACE VIEW integration.model_metadata AS
SELECT
    'project4'::text AS source_project,
    'postgresql'::text AS source_system,
    'model_metadata'::text AS source_entity,
    r.model_name || ':' || r.model_version AS source_record_id,
    COALESCE(r.promoted_at, r.created_at) AS event_timestamp,
    'project4.integration.v1'::text AS schema_version,
    'ml_model_registry'::text AS domain_tag,
    r.created_at AS recorded_at,
    r.model_id,
    r.model_name,
    r.model_version,
    r.algorithm,
    r.status,
    r.approved_for_prod,
    (p.model_version IS NOT NULL) AS is_active,
    p.pointer_version AS active_pointer_version,
    r.feature_schema_version,
    r.snapshot_id AS dataset_snapshot_id,
    r.training_dataset_id,
    r.training_started_at,
    r.training_finished_at,
    r.baseline_anomaly_rate,
    r.threshold_used,
    r.promoted_at,
    r.archived_at,
    r.notes
FROM ml.model_registry AS r
LEFT JOIN ml.active_model_pointer AS p
    ON p.model_name = r.model_name
   AND p.model_version = r.model_version;

CREATE OR REPLACE VIEW integration.prediction_outputs AS
SELECT
    'project4'::text AS source_project,
    'postgresql'::text AS source_system,
    'prediction_outputs'::text AS source_entity,
    o.prediction_id::text AS source_record_id,
    o.request_timestamp AS event_timestamp,
    'project4.integration.v1'::text AS schema_version,
    'anomaly_predictions'::text AS domain_tag,
    o.created_at AS recorded_at,
    'online'::text AS prediction_source,
    NULL::uuid AS batch_run_id,
    o.model_name,
    o.model_version,
    o.feature_schema_version,
    o.dataset_snapshot_id,
    o.entity_type,
    o.entity_id,
    o.anomaly_score,
    o.is_anomaly,
    o.threshold_used,
    o.drift_status,
    o.prediction_status,
    o.latency_ms,
    o.feature_payload_hash
FROM ml.online_predictions AS o
UNION ALL
SELECT
    'project4'::text,
    'postgresql'::text,
    'prediction_outputs'::text,
    b.prediction_id::text,
    b.score_timestamp,
    'project4.integration.v1'::text,
    'anomaly_predictions'::text,
    b.created_at,
    'batch'::text,
    b.batch_run_id,
    b.model_name,
    b.model_version,
    b.feature_schema_version,
    b.dataset_snapshot_id,
    b.entity_type,
    b.entity_id,
    b.anomaly_score,
    b.is_anomaly,
    b.threshold_used,
    NULL::text,
    b.prediction_status,
    b.inference_latency_ms,
    b.feature_payload_hash
FROM ml.batch_predictions AS b;

CREATE OR REPLACE VIEW integration.drift_validation_rollback AS
SELECT
    'project4'::text AS source_project,
    'postgresql'::text AS source_system,
    'drift_validation_rollback'::text AS source_entity,
    d.drift_event_id::text AS source_record_id,
    d.detected_at AS event_timestamp,
    'project4.integration.v1'::text AS schema_version,
    'ml_observability'::text AS domain_tag,
    d.created_at AS recorded_at,
    'drift'::text AS event_type,
    d.model_name,
    d.model_version,
    d.drift_status AS status,
    NULL::text AS severity,
    d.feature_name AS subject,
    d.notes AS message,
    'mean_delta_percent'::text AS metric_name,
    d.mean_delta_percent AS metric_value,
    d.mean_critical_threshold AS threshold_value,
    NULL::text AS from_model_version,
    NULL::text AS to_model_version,
    NULL::text AS triggered_by
FROM monitoring.drift_events AS d
UNION ALL
SELECT
    'project4'::text,
    'postgresql'::text,
    'drift_validation_rollback'::text,
    a.alert_event_id::text,
    a.triggered_at,
    'project4.integration.v1'::text,
    'ml_observability'::text,
    a.created_at,
    'alert'::text,
    a.model_name,
    a.model_version,
    a.alert_status,
    a.severity,
    a.entity_id,
    a.message,
    a.metric_name,
    a.metric_value,
    a.threshold_value,
    NULL::text,
    NULL::text,
    a.alert_source
FROM monitoring.alert_events AS a
UNION ALL
SELECT
    'project4'::text,
    'postgresql'::text,
    'drift_validation_rollback'::text,
    rb.rollback_id,
    rb.triggered_at_utc,
    'project4.integration.v1'::text,
    'ml_observability'::text,
    rb.created_at,
    'rollback'::text,
    rb.model_name,
    rb.to_model_version,
    rb.validation_status,
    NULL::text,
    rb.from_model_version || ' -> ' || rb.to_model_version,
    rb.rollback_reason,
    NULL::text,
    NULL::numeric,
    NULL::numeric,
    rb.from_model_version,
    rb.to_model_version,
    rb.triggered_by
FROM audit.rollback_events AS rb;

COMMENT ON VIEW integration.model_metadata IS
'One row per registered model version, with whether it is the active one (ml.active_model_pointer).';

COMMENT ON VIEW integration.prediction_outputs IS
'Online and batch anomaly predictions, prediction_source = online | batch.';

COMMENT ON VIEW integration.drift_validation_rollback IS
'Drift events, alert events, and rollback events, event_type = drift | alert | rollback.';
