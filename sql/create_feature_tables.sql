-- Project 4: ML Pipeline & Anomaly Detection
-- Feature snapshot, training dataset, baseline, and current feature statistic tables.

CREATE TABLE IF NOT EXISTS ml.feature_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_name TEXT NOT NULL UNIQUE,
    source_project TEXT NOT NULL,
    source_layer TEXT NOT NULL,
    source_tables JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_date_start DATE,
    source_date_end DATE,
    feature_schema_version TEXT NOT NULL,
    feature_count INTEGER NOT NULL CHECK (feature_count >= 0),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    snapshot_path TEXT,
    data_quality_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (data_quality_status IN ('pending', 'passed', 'warning', 'failed')),
    validation_report_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,
    notes TEXT,
    CONSTRAINT feature_snapshot_date_range_chk
        CHECK (
            source_date_start IS NULL
            OR source_date_end IS NULL
            OR source_date_end >= source_date_start
        )
);

CREATE TABLE IF NOT EXISTS ml.training_datasets (
    training_dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name TEXT NOT NULL UNIQUE,
    snapshot_id UUID NOT NULL REFERENCES ml.feature_snapshots(snapshot_id) ON DELETE RESTRICT,
    dataset_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    train_row_count BIGINT NOT NULL CHECK (train_row_count >= 0),
    validation_row_count BIGINT NOT NULL DEFAULT 0 CHECK (validation_row_count >= 0),
    test_row_count BIGINT NOT NULL DEFAULT 0 CHECK (test_row_count >= 0),
    feature_columns JSONB NOT NULL DEFAULT '[]'::jsonb,
    label_column TEXT,
    dataset_path TEXT,
    data_quality_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (data_quality_status IN ('pending', 'passed', 'warning', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,
    notes TEXT,
    CONSTRAINT training_dataset_version_unique UNIQUE (dataset_name, dataset_version)
);

CREATE TABLE IF NOT EXISTS ml.feature_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL REFERENCES ml.feature_snapshots(snapshot_id) ON DELETE RESTRICT,
    training_dataset_id UUID REFERENCES ml.training_datasets(training_dataset_id) ON DELETE SET NULL,
    model_version TEXT,
    feature_schema_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_dtype TEXT NOT NULL,
    baseline_mean NUMERIC,
    baseline_variance NUMERIC,
    baseline_min NUMERIC,
    baseline_max NUMERIC,
    baseline_null_count BIGINT NOT NULL DEFAULT 0 CHECK (baseline_null_count >= 0),
    baseline_row_count BIGINT NOT NULL CHECK (baseline_row_count >= 0),
    baseline_computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT feature_baseline_unique UNIQUE (snapshot_id, feature_schema_version, feature_name),
    CONSTRAINT feature_baseline_variance_chk CHECK (baseline_variance IS NULL OR baseline_variance >= 0)
);

CREATE TABLE IF NOT EXISTS ml.feature_current_stats (
    current_stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID REFERENCES ml.feature_snapshots(snapshot_id) ON DELETE SET NULL,
    model_version TEXT,
    feature_schema_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_dtype TEXT NOT NULL,
    observation_window_start TIMESTAMPTZ NOT NULL,
    observation_window_end TIMESTAMPTZ NOT NULL,
    current_mean NUMERIC,
    current_variance NUMERIC,
    current_min NUMERIC,
    current_max NUMERIC,
    current_null_count BIGINT NOT NULL DEFAULT 0 CHECK (current_null_count >= 0),
    current_row_count BIGINT NOT NULL CHECK (current_row_count >= 0),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,
    CONSTRAINT feature_current_stats_window_chk
        CHECK (observation_window_end >= observation_window_start),
    CONSTRAINT feature_current_stats_variance_chk
        CHECK (current_variance IS NULL OR current_variance >= 0)
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_created_at
    ON ml.feature_snapshots(created_at);

CREATE INDEX IF NOT EXISTS idx_training_datasets_snapshot_id
    ON ml.training_datasets(snapshot_id);

CREATE INDEX IF NOT EXISTS idx_feature_baselines_feature_name
    ON ml.feature_baselines(feature_name);

CREATE INDEX IF NOT EXISTS idx_feature_baselines_model_version
    ON ml.feature_baselines(model_version);

CREATE INDEX IF NOT EXISTS idx_feature_current_stats_feature_window
    ON ml.feature_current_stats(feature_name, observation_window_start, observation_window_end);

CREATE INDEX IF NOT EXISTS idx_feature_current_stats_model_version
    ON ml.feature_current_stats(model_version);

COMMENT ON TABLE ml.feature_snapshots IS
'Tracks reproducible Project 4 feature snapshots built from Project 1 real-time signals and Project 2/3 warehouse or mart data.';

COMMENT ON TABLE ml.training_datasets IS
'Tracks training dataset versions derived from feature snapshots, including split counts and feature schema version.';

COMMENT ON TABLE ml.feature_baselines IS
'Stores baseline feature statistics from approved training snapshots for drift comparison.';

COMMENT ON TABLE ml.feature_current_stats IS
'Stores current runtime or batch feature statistics used for baseline-vs-current drift checks.';
