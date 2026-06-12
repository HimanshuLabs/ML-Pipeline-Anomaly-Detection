-- Project 4: ML Pipeline & Anomaly Detection
-- Model registry and metrics metadata tables.

CREATE TABLE IF NOT EXISTS ml.model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_uri TEXT,
    snapshot_id UUID REFERENCES ml.feature_snapshots(snapshot_id) ON DELETE SET NULL,
    training_dataset_id UUID REFERENCES ml.training_datasets(training_dataset_id) ON DELETE SET NULL,
    feature_schema_version TEXT NOT NULL,
    training_started_at TIMESTAMPTZ,
    training_finished_at TIMESTAMPTZ,
    baseline_anomaly_rate NUMERIC CHECK (
        baseline_anomaly_rate IS NULL
        OR (baseline_anomaly_rate >= 0 AND baseline_anomaly_rate <= 1)
    ),
    baseline_precision_proxy NUMERIC CHECK (
        baseline_precision_proxy IS NULL
        OR (baseline_precision_proxy >= 0 AND baseline_precision_proxy <= 1)
    ),
    baseline_recall_proxy NUMERIC CHECK (
        baseline_recall_proxy IS NULL
        OR (baseline_recall_proxy >= 0 AND baseline_recall_proxy <= 1)
    ),
    baseline_f1_proxy NUMERIC CHECK (
        baseline_f1_proxy IS NULL
        OR (baseline_f1_proxy >= 0 AND baseline_f1_proxy <= 1)
    ),
    latency_p50_ms NUMERIC CHECK (latency_p50_ms IS NULL OR latency_p50_ms >= 0),
    latency_p95_ms NUMERIC CHECK (latency_p95_ms IS NULL OR latency_p95_ms >= 0),
    threshold_used NUMERIC,
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (
            status IN (
                'candidate',
                'staging',
                'production',
                'archived',
                'rolled_back',
                'failed_validation'
            )
        ),
    approved_for_prod BOOLEAN NOT NULL DEFAULT FALSE,
    promoted_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,
    notes TEXT,
    CONSTRAINT model_registry_model_version_unique UNIQUE (model_name, model_version),
    CONSTRAINT model_registry_training_time_chk
        CHECK (
            training_started_at IS NULL
            OR training_finished_at IS NULL
            OR training_finished_at >= training_started_at
        ),
    CONSTRAINT model_registry_archive_time_chk
        CHECK (
            archived_at IS NULL
            OR promoted_at IS NULL
            OR archived_at >= promoted_at
        )
);

CREATE TABLE IF NOT EXISTS ml.model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    metric_scope TEXT NOT NULL
        CHECK (
            metric_scope IN (
                'training',
                'validation',
                'baseline',
                'current',
                'backtest',
                'latency',
                'drift',
                'rollback_validation'
            )
        ),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_unit TEXT,
    threshold_used NUMERIC,
    comparison_operator TEXT
        CHECK (
            comparison_operator IS NULL
            OR comparison_operator IN ('<', '<=', '=', '>=', '>')
        ),
    passed_threshold BOOLEAN,
    dataset_snapshot_id UUID REFERENCES ml.feature_snapshots(snapshot_id) ON DELETE SET NULL,
    training_dataset_id UUID REFERENCES ml.training_datasets(training_dataset_id) ON DELETE SET NULL,
    observation_window_start TIMESTAMPTZ,
    observation_window_end TIMESTAMPTZ,
    metric_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT model_metrics_window_chk
        CHECK (
            observation_window_start IS NULL
            OR observation_window_end IS NULL
            OR observation_window_end >= observation_window_start
        ),
    CONSTRAINT model_metrics_model_version_fk
        FOREIGN KEY (model_name, model_version)
        REFERENCES ml.model_registry(model_name, model_version)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON ml.model_registry(status);

CREATE INDEX IF NOT EXISTS idx_model_registry_approved_for_prod
    ON ml.model_registry(approved_for_prod);

CREATE INDEX IF NOT EXISTS idx_model_registry_model_version
    ON ml.model_registry(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_model_registry_training_dataset_id
    ON ml.model_registry(training_dataset_id);

CREATE INDEX IF NOT EXISTS idx_model_metrics_model_id
    ON ml.model_metrics(model_id);

CREATE INDEX IF NOT EXISTS idx_model_metrics_scope_name
    ON ml.model_metrics(metric_scope, metric_name);

CREATE INDEX IF NOT EXISTS idx_model_metrics_model_version
    ON ml.model_metrics(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_model_metrics_computed_at
    ON ml.model_metrics(computed_at);

COMMENT ON TABLE ml.model_registry IS
'Stores versioned anomaly detection model metadata including artifact location, dataset lineage, baseline metrics, production approval, and lifecycle status.';

COMMENT ON TABLE ml.model_metrics IS
'Stores training, validation, baseline, current, drift, latency, and rollback validation metrics for registered model versions.';
