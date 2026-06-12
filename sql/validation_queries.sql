-- Project 4: ML Pipeline & Anomaly Detection
-- Metadata validation query pack.
-- This file validates schemas, tables, keys, constraints, required columns, and queryability.

\echo '=============================='
\echo 'Validate required schemas'
\echo '=============================='

WITH expected_schemas(schema_name) AS (
    VALUES
        ('ml'),
        ('monitoring'),
        ('audit')
),
missing_schemas AS (
    SELECT e.schema_name
    FROM expected_schemas e
    LEFT JOIN information_schema.schemata s
        ON s.schema_name = e.schema_name
    WHERE s.schema_name IS NULL
)
SELECT
    'required_schemas_exist' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*) AS missing_count,
    COALESCE(jsonb_agg(schema_name) FILTER (WHERE schema_name IS NOT NULL), '[]'::jsonb) AS missing_items
FROM missing_schemas;

\echo '=============================='
\echo 'Validate required tables'
\echo '=============================='

WITH expected_tables(table_schema, table_name) AS (
    VALUES
        ('ml', 'feature_snapshots'),
        ('ml', 'training_datasets'),
        ('ml', 'feature_baselines'),
        ('ml', 'feature_current_stats'),
        ('ml', 'model_registry'),
        ('ml', 'model_metrics'),
        ('ml', 'batch_predictions'),
        ('ml', 'online_predictions'),
        ('monitoring', 'drift_events'),
        ('monitoring', 'alert_events'),
        ('audit', 'rollback_events')
),
missing_tables AS (
    SELECT e.table_schema, e.table_name
    FROM expected_tables e
    LEFT JOIN information_schema.tables t
        ON t.table_schema = e.table_schema
       AND t.table_name = e.table_name
    WHERE t.table_name IS NULL
)
SELECT
    'required_tables_exist' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*) AS missing_count,
    COALESCE(
        jsonb_agg(table_schema || '.' || table_name) FILTER (WHERE table_name IS NOT NULL),
        '[]'::jsonb
    ) AS missing_items
FROM missing_tables;

\echo '=============================='
\echo 'Validate primary keys'
\echo '=============================='

WITH expected_tables(table_schema, table_name) AS (
    VALUES
        ('ml', 'feature_snapshots'),
        ('ml', 'training_datasets'),
        ('ml', 'feature_baselines'),
        ('ml', 'feature_current_stats'),
        ('ml', 'model_registry'),
        ('ml', 'model_metrics'),
        ('ml', 'batch_predictions'),
        ('ml', 'online_predictions'),
        ('monitoring', 'drift_events'),
        ('monitoring', 'alert_events'),
        ('audit', 'rollback_events')
),
missing_primary_keys AS (
    SELECT e.table_schema, e.table_name
    FROM expected_tables e
    LEFT JOIN information_schema.table_constraints tc
        ON tc.table_schema = e.table_schema
       AND tc.table_name = e.table_name
       AND tc.constraint_type = 'PRIMARY KEY'
    WHERE tc.constraint_name IS NULL
)
SELECT
    'required_primary_keys_exist' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*) AS missing_count,
    COALESCE(
        jsonb_agg(table_schema || '.' || table_name) FILTER (WHERE table_name IS NOT NULL),
        '[]'::jsonb
    ) AS missing_items
FROM missing_primary_keys;

\echo '=============================='
\echo 'Validate required lifecycle constraints'
\echo '=============================='

WITH expected_constraints(table_schema, table_name, constraint_name) AS (
    VALUES
        ('ml', 'model_registry', 'model_registry_model_version_unique'),
        ('ml', 'model_registry', 'model_registry_status_check'),
        ('ml', 'model_registry', 'model_registry_training_time_chk'),
        ('ml', 'model_metrics', 'model_metrics_model_version_fk'),
        ('ml', 'model_metrics', 'model_metrics_metric_scope_check'),
        ('ml', 'batch_predictions', 'batch_predictions_model_version_fk'),
        ('ml', 'online_predictions', 'online_predictions_model_version_fk'),
        ('ml', 'online_predictions', 'online_predictions_success_score_chk'),
        ('monitoring', 'drift_events', 'drift_events_model_version_fk'),
        ('monitoring', 'drift_events', 'drift_events_drift_status_check'),
        ('monitoring', 'alert_events', 'alert_events_alert_type_check'),
        ('monitoring', 'alert_events', 'alert_events_severity_check'),
        ('audit', 'rollback_events', 'rollback_events_from_model_version_fk'),
        ('audit', 'rollback_events', 'rollback_events_to_model_version_fk'),
        ('audit', 'rollback_events', 'rollback_events_versions_different_chk')
),
missing_constraints AS (
    SELECT e.table_schema, e.table_name, e.constraint_name
    FROM expected_constraints e
    LEFT JOIN information_schema.table_constraints tc
        ON tc.table_schema = e.table_schema
       AND tc.table_name = e.table_name
       AND tc.constraint_name = e.constraint_name
    WHERE tc.constraint_name IS NULL
)
SELECT
    'required_lifecycle_constraints_exist' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*) AS missing_count,
    COALESCE(
        jsonb_agg(table_schema || '.' || table_name || '.' || constraint_name)
            FILTER (WHERE constraint_name IS NOT NULL),
        '[]'::jsonb
    ) AS missing_items
FROM missing_constraints;

\echo '=============================='
\echo 'Validate required operational columns'
\echo '=============================='

WITH expected_columns(table_schema, table_name, column_name) AS (
    VALUES
        ('ml', 'feature_snapshots', 'snapshot_id'),
        ('ml', 'feature_snapshots', 'feature_schema_version'),
        ('ml', 'feature_snapshots', 'row_count'),
        ('ml', 'feature_snapshots', 'data_quality_status'),
        ('ml', 'training_datasets', 'training_dataset_id'),
        ('ml', 'training_datasets', 'snapshot_id'),
        ('ml', 'training_datasets', 'dataset_version'),
        ('ml', 'training_datasets', 'feature_columns'),
        ('ml', 'feature_baselines', 'baseline_mean'),
        ('ml', 'feature_baselines', 'baseline_variance'),
        ('ml', 'feature_current_stats', 'current_mean'),
        ('ml', 'feature_current_stats', 'current_variance'),
        ('ml', 'model_registry', 'model_id'),
        ('ml', 'model_registry', 'model_name'),
        ('ml', 'model_registry', 'model_version'),
        ('ml', 'model_registry', 'artifact_path'),
        ('ml', 'model_registry', 'feature_schema_version'),
        ('ml', 'model_registry', 'status'),
        ('ml', 'model_registry', 'approved_for_prod'),
        ('ml', 'model_metrics', 'metric_id'),
        ('ml', 'model_metrics', 'metric_scope'),
        ('ml', 'model_metrics', 'metric_name'),
        ('ml', 'model_metrics', 'metric_value'),
        ('ml', 'batch_predictions', 'prediction_id'),
        ('ml', 'batch_predictions', 'batch_run_id'),
        ('ml', 'batch_predictions', 'model_version'),
        ('ml', 'batch_predictions', 'anomaly_score'),
        ('ml', 'batch_predictions', 'is_anomaly'),
        ('ml', 'batch_predictions', 'threshold_used'),
        ('ml', 'online_predictions', 'prediction_id'),
        ('ml', 'online_predictions', 'request_id'),
        ('ml', 'online_predictions', 'model_version'),
        ('ml', 'online_predictions', 'latency_ms'),
        ('ml', 'online_predictions', 'drift_status'),
        ('monitoring', 'drift_events', 'drift_event_id'),
        ('monitoring', 'drift_events', 'model_version'),
        ('monitoring', 'drift_events', 'feature_name'),
        ('monitoring', 'drift_events', 'mean_delta'),
        ('monitoring', 'drift_events', 'variance_delta'),
        ('monitoring', 'drift_events', 'drift_status'),
        ('monitoring', 'alert_events', 'alert_event_id'),
        ('monitoring', 'alert_events', 'alert_type'),
        ('monitoring', 'alert_events', 'severity'),
        ('monitoring', 'alert_events', 'alert_status'),
        ('audit', 'rollback_events', 'rollback_id'),
        ('audit', 'rollback_events', 'from_model_version'),
        ('audit', 'rollback_events', 'to_model_version'),
        ('audit', 'rollback_events', 'rollback_reason'),
        ('audit', 'rollback_events', 'rollback_status')
),
missing_columns AS (
    SELECT e.table_schema, e.table_name, e.column_name
    FROM expected_columns e
    LEFT JOIN information_schema.columns c
        ON c.table_schema = e.table_schema
       AND c.table_name = e.table_name
       AND c.column_name = e.column_name
    WHERE c.column_name IS NULL
)
SELECT
    'required_operational_columns_exist' AS check_name,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    COUNT(*) AS missing_count,
    COALESCE(
        jsonb_agg(table_schema || '.' || table_name || '.' || column_name)
            FILTER (WHERE column_name IS NOT NULL),
        '[]'::jsonb
    ) AS missing_items
FROM missing_columns;

\echo '=============================='
\echo 'Validate metadata table row counts'
\echo '=============================='

SELECT 'ml.feature_snapshots' AS table_name, COUNT(*) AS row_count FROM ml.feature_snapshots
UNION ALL
SELECT 'ml.training_datasets' AS table_name, COUNT(*) AS row_count FROM ml.training_datasets
UNION ALL
SELECT 'ml.feature_baselines' AS table_name, COUNT(*) AS row_count FROM ml.feature_baselines
UNION ALL
SELECT 'ml.feature_current_stats' AS table_name, COUNT(*) AS row_count FROM ml.feature_current_stats
UNION ALL
SELECT 'ml.model_registry' AS table_name, COUNT(*) AS row_count FROM ml.model_registry
UNION ALL
SELECT 'ml.model_metrics' AS table_name, COUNT(*) AS row_count FROM ml.model_metrics
UNION ALL
SELECT 'ml.batch_predictions' AS table_name, COUNT(*) AS row_count FROM ml.batch_predictions
UNION ALL
SELECT 'ml.online_predictions' AS table_name, COUNT(*) AS row_count FROM ml.online_predictions
UNION ALL
SELECT 'monitoring.drift_events' AS table_name, COUNT(*) AS row_count FROM monitoring.drift_events
UNION ALL
SELECT 'monitoring.alert_events' AS table_name, COUNT(*) AS row_count FROM monitoring.alert_events
UNION ALL
SELECT 'audit.rollback_events' AS table_name, COUNT(*) AS row_count FROM audit.rollback_events
ORDER BY table_name;

\echo '=============================='
\echo 'Final validation summary'
\echo '=============================='

WITH validation_checks AS (
    SELECT
        'schemas' AS check_group,
        CASE WHEN (
            SELECT COUNT(*)
            FROM information_schema.schemata
            WHERE schema_name IN ('ml', 'monitoring', 'audit')
        ) = 3 THEN 1 ELSE 0 END AS passed

    UNION ALL

    SELECT
        'tables' AS check_group,
        CASE WHEN (
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE (table_schema = 'ml' AND table_name IN (
                    'feature_snapshots',
                    'training_datasets',
                    'feature_baselines',
                    'feature_current_stats',
                    'model_registry',
                    'model_metrics',
                    'batch_predictions',
                    'online_predictions'
                ))
               OR (table_schema = 'monitoring' AND table_name IN (
                    'drift_events',
                    'alert_events'
                ))
               OR (table_schema = 'audit' AND table_name = 'rollback_events')
        ) = 11 THEN 1 ELSE 0 END AS passed

    UNION ALL

    SELECT
        'primary_keys' AS check_group,
        CASE WHEN (
            SELECT COUNT(*)
            FROM information_schema.table_constraints
            WHERE constraint_type = 'PRIMARY KEY'
              AND (
                    (table_schema = 'ml' AND table_name IN (
                        'feature_snapshots',
                        'training_datasets',
                        'feature_baselines',
                        'feature_current_stats',
                        'model_registry',
                        'model_metrics',
                        'batch_predictions',
                        'online_predictions'
                    ))
                 OR (table_schema = 'monitoring' AND table_name IN (
                        'drift_events',
                        'alert_events'
                    ))
                 OR (table_schema = 'audit' AND table_name = 'rollback_events')
              )
        ) = 11 THEN 1 ELSE 0 END AS passed

    UNION ALL

    SELECT
        'foreign_keys' AS check_group,
        CASE WHEN (
            SELECT COUNT(*)
            FROM information_schema.table_constraints
            WHERE constraint_type = 'FOREIGN KEY'
              AND table_schema IN ('ml', 'monitoring', 'audit')
        ) >= 20 THEN 1 ELSE 0 END AS passed
)
SELECT
    CASE WHEN SUM(passed) = COUNT(*) THEN 'PASS' ELSE 'FAIL' END AS overall_status,
    SUM(passed) AS passed_checks,
    COUNT(*) AS total_checks
FROM validation_checks;
