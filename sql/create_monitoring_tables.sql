-- Project 4: ML Pipeline & Anomaly Detection
-- Monitoring, alerting, and rollback audit tables.

CREATE TABLE IF NOT EXISTS monitoring.drift_events (
    drift_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    model_id UUID REFERENCES ml.model_registry(model_id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,

    feature_schema_version TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_dtype TEXT NOT NULL,

    baseline_id UUID REFERENCES ml.feature_baselines(baseline_id) ON DELETE SET NULL,
    current_stat_id UUID REFERENCES ml.feature_current_stats(current_stat_id) ON DELETE SET NULL,

    baseline_mean NUMERIC,
    current_mean NUMERIC,
    mean_delta NUMERIC CHECK (mean_delta IS NULL OR mean_delta >= 0),
    mean_delta_percent NUMERIC,

    baseline_variance NUMERIC CHECK (baseline_variance IS NULL OR baseline_variance >= 0),
    current_variance NUMERIC CHECK (current_variance IS NULL OR current_variance >= 0),
    variance_delta NUMERIC CHECK (variance_delta IS NULL OR variance_delta >= 0),
    variance_delta_percent NUMERIC,

    mean_warning_threshold NUMERIC CHECK (mean_warning_threshold IS NULL OR mean_warning_threshold >= 0),
    mean_critical_threshold NUMERIC CHECK (mean_critical_threshold IS NULL OR mean_critical_threshold >= 0),
    variance_warning_threshold NUMERIC CHECK (variance_warning_threshold IS NULL OR variance_warning_threshold >= 0),
    variance_critical_threshold NUMERIC CHECK (variance_critical_threshold IS NULL OR variance_critical_threshold >= 0),

    drift_status TEXT NOT NULL
        CHECK (drift_status IN ('normal', 'warning', 'critical')),

    detection_method TEXT NOT NULL DEFAULT 'mean_variance_threshold'
        CHECK (
            detection_method IN (
                'mean_variance_threshold',
                'manual_review',
                'backtest',
                'custom'
            )
        ),

    observation_window_start TIMESTAMPTZ NOT NULL,
    observation_window_end TIMESTAMPTZ NOT NULL,

    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT,

    CONSTRAINT drift_events_window_chk
        CHECK (observation_window_end >= observation_window_start),

    CONSTRAINT drift_events_model_version_fk
        FOREIGN KEY (model_name, model_version)
        REFERENCES ml.model_registry(model_name, model_version)
        ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS monitoring.alert_events (
    alert_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    drift_event_id UUID REFERENCES monitoring.drift_events(drift_event_id) ON DELETE SET NULL,

    model_id UUID REFERENCES ml.model_registry(model_id) ON DELETE SET NULL,
    model_name TEXT,
    model_version TEXT,

    alert_type TEXT NOT NULL
        CHECK (
            alert_type IN (
                'drift_warning',
                'drift_critical',
                'anomaly_rate_spike',
                'latency_budget_breach',
                'prediction_error_rate_breach',
                'model_degradation',
                'rollback_triggered',
                'manual_alert'
            )
        ),

    severity TEXT NOT NULL
        CHECK (severity IN ('info', 'warning', 'critical')),

    alert_status TEXT NOT NULL DEFAULT 'open'
        CHECK (alert_status IN ('open', 'acknowledged', 'resolved', 'suppressed')),

    alert_source TEXT NOT NULL DEFAULT 'project4_monitoring',

    metric_name TEXT,
    metric_value NUMERIC,
    threshold_value NUMERIC,
    comparison_operator TEXT
        CHECK (
            comparison_operator IS NULL
            OR comparison_operator IN ('<', '<=', '=', '>=', '>')
        ),

    entity_type TEXT,
    entity_id TEXT,

    message TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,

    triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT alert_events_ack_time_chk
        CHECK (
            acknowledged_at IS NULL
            OR acknowledged_at >= triggered_at
        ),

    CONSTRAINT alert_events_resolved_time_chk
        CHECK (
            resolved_at IS NULL
            OR resolved_at >= triggered_at
        ),

    CONSTRAINT alert_events_model_version_fk
        FOREIGN KEY (model_name, model_version)
        REFERENCES ml.model_registry(model_name, model_version)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit.rollback_events (
    rollback_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    from_model_version TEXT NOT NULL,
    to_model_version TEXT NOT NULL,
    rollback_reason TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    triggered_at_utc TIMESTAMPTZ NOT NULL,
    validation_status TEXT NOT NULL CHECK (
        validation_status IN (
            'dry_run_validated',
            'applied',
            'failed'
        )
    ),
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    active_model_path TEXT NOT NULL,
    rollback_event_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_events_model_version
    ON monitoring.drift_events(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_drift_events_feature_name
    ON monitoring.drift_events(feature_name);

CREATE INDEX IF NOT EXISTS idx_drift_events_status
    ON monitoring.drift_events(drift_status);

CREATE INDEX IF NOT EXISTS idx_drift_events_detected_at
    ON monitoring.drift_events(detected_at);

CREATE INDEX IF NOT EXISTS idx_alert_events_alert_type
    ON monitoring.alert_events(alert_type);

CREATE INDEX IF NOT EXISTS idx_alert_events_severity
    ON monitoring.alert_events(severity);

CREATE INDEX IF NOT EXISTS idx_alert_events_status
    ON monitoring.alert_events(alert_status);

CREATE INDEX IF NOT EXISTS idx_alert_events_triggered_at
    ON monitoring.alert_events(triggered_at);

CREATE INDEX IF NOT EXISTS idx_alert_events_model_version
    ON monitoring.alert_events(model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_rollback_events_model_name
    ON audit.rollback_events(model_name);

CREATE INDEX IF NOT EXISTS idx_rollback_events_from_to_versions
    ON audit.rollback_events(from_model_version, to_model_version);

CREATE INDEX IF NOT EXISTS idx_rollback_events_status
    ON audit.rollback_events(rollback_status);

CREATE INDEX IF NOT EXISTS idx_rollback_events_triggered_at
    ON audit.rollback_events(triggered_at);

COMMENT ON TABLE monitoring.drift_events IS
'Stores mean/variance feature drift events comparing approved baseline statistics against current runtime or batch feature statistics.';

COMMENT ON TABLE monitoring.alert_events IS
'Stores operational alerts for drift, anomaly-rate spikes, latency breaches, prediction errors, model degradation, and rollback triggers.';

COMMENT ON TABLE audit.rollback_events IS
'Stores rollback audit evidence when Project 4 reverts from an unhealthy model version to a previous stable model version.';


CREATE INDEX IF NOT EXISTS idx_rollback_events_model_time
    ON audit.rollback_events (model_name, triggered_at_utc DESC);

CREATE INDEX IF NOT EXISTS idx_rollback_events_versions
    ON audit.rollback_events (from_model_version, to_model_version);
