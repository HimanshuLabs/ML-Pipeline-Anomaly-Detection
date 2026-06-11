# Drift Monitoring — Project 4 ML Pipeline & Anomaly Detection

## Purpose

Project 4 monitors whether current feature behavior has moved materially away from the training baseline used to approve the active anomaly detection model.

The current implementation focuses on lightweight mean and variance drift checks. It is intentionally simple. This checkpoint does not pretend to be a full statistical drift platform. It adds an operational signal that answers one practical question:

Are the features being scored now still behaving like the features the model was trained on?

## Current implementation status

Implemented locally:

- Loads the active v002 baseline statistics from baseline_stats.json.
- Compares current numeric feature mean and variance against training baseline mean and variance.
- Supports three drift states: normal, warning, critical.
- Supports default drift thresholds.
- Supports feature-specific threshold overrides.
- Produces one drift event per evaluated feature.
- Produces an overall drift status for the evaluated window.
- Writes local drift evidence to logs/alerts/drift_events.jsonl.
- Includes deterministic tests for normal, warning, critical, variance-driven drift, zero-baseline handling, feature overrides, JSONL persistence, and bad-input failure paths.

Database-ready but not wired yet:

- monitoring.drift_events exists in sql/create_monitoring_tables.sql.
- The JSONL drift event shape is aligned with the SQL table columns.
- Direct PostgreSQL insert wiring is planned for a later hardening pass.

Planned later:

- Prometheus drift metrics.
- Grafana drift dashboard panels.
- Alert events from critical drift.
- Rollback review trigger when critical drift persists.
- Scheduled batch drift evaluation over scored production windows.

## Active model baseline

| Field | Value |
|---|---|
| Model name | isolation_forest |
| Model version | v002 |
| Feature schema version | feature_schema_v001 |
| Snapshot type | real_source_extract |
| Dataset snapshot ID | real_source_20260610T060738Z |
| Feature count | 51 |
| Baseline source | Project 1 + Project 2/3 local source extracts |

Baseline file:

    artifacts/models/isolation_forest/model_version=v002/baseline_stats.json

Active model pointer:

    configs/active_model.yaml

## Drift method

Implemented method:

    mean_variance_threshold

For each overlapping numeric feature, Project 4 calculates:

    current_mean
    current_variance
    mean_delta
    variance_delta
    mean_delta_percent
    variance_delta_percent

Baseline values come from the approved training baseline:

    baseline_mean
    baseline_variance

## Relative drift logic

For non-zero baselines:

    mean_delta_percent = abs(current_mean - baseline_mean) / abs(baseline_mean)
    variance_delta_percent = abs(current_variance - baseline_variance) / abs(baseline_variance)

For zero baselines, relative percentage is undefined. The implementation falls back to absolute delta:

    mean_delta_percent = abs(current_mean - baseline_mean)
    variance_delta_percent = abs(current_variance - baseline_variance)

This avoids divide-by-zero behavior and keeps zero-baseline features evaluable.

## Threshold configuration

Threshold file:

    configs/drift_thresholds.yaml

Default thresholds:

| Threshold | Value |
|---|---:|
| mean_delta_warning | 0.15 |
| mean_delta_critical | 0.30 |
| variance_delta_warning | 0.25 |
| variance_delta_critical | 0.50 |

Feature overrides currently exist for:

    avg_api_latency_ms
    page_load_p95_ms

These features receive wider thresholds because latency features are naturally more volatile than stable business aggregates.

## Drift status rules

A feature is marked critical when either condition is true:

    mean_delta_percent >= mean_delta_critical
    variance_delta_percent >= variance_delta_critical

A feature is marked warning when either condition is true and no critical threshold is crossed:

    mean_delta_percent >= mean_delta_warning
    variance_delta_percent >= variance_delta_warning

Otherwise, the feature is marked normal.

Overall drift status:

| Overall status | Condition |
|---|---|
| critical | At least one feature is critical |
| warning | No critical features, but at least one warning feature |
| normal | All evaluated features are normal |

## Drift event output

Local drift events are written to:

    logs/alerts/drift_events.jsonl

Each event includes:

    drift_event_id
    model_name
    model_version
    feature_schema_version
    feature_name
    feature_dtype
    baseline_mean
    current_mean
    mean_delta
    mean_delta_percent
    baseline_variance
    current_variance
    variance_delta
    variance_delta_percent
    mean_warning_threshold
    mean_critical_threshold
    variance_warning_threshold
    variance_critical_threshold
    drift_status
    detection_method
    observation_window_start
    observation_window_end
    detected_at
    notes

## Why JSONL first

JSONL is used as the first persistence layer because it is local, inspectable, testable, and does not require a running database.

This is not the final production storage target. It is the local fallback path.

The production-oriented table already exists:

    monitoring.drift_events

Direct database writes should reuse the same event contract so the API and batch jobs do not need a new drift payload shape later.

## Database alignment

The SQL table monitoring.drift_events already includes the same operational fields emitted by the Python drift event contract:

    model_name
    model_version
    feature_schema_version
    feature_name
    feature_dtype
    baseline_mean
    current_mean
    mean_delta
    mean_delta_percent
    baseline_variance
    current_variance
    variance_delta
    variance_delta_percent
    mean_warning_threshold
    mean_critical_threshold
    variance_warning_threshold
    variance_critical_threshold
    drift_status
    detection_method
    observation_window_start
    observation_window_end
    detected_at
    notes

The Python drift event contract intentionally mirrors these fields.

## Current limitations

Known limitations:

- It checks only numeric features.
- It compares only mean and variance.
- It does not run population stability index.
- It does not run Kolmogorov-Smirnov tests.
- It does not persist directly to PostgreSQL yet.
- It does not trigger alert events yet.
- It does not automatically roll back a model.

These limitations are acceptable at this checkpoint because Project 4 is building the monitoring spine incrementally.

## Operational interpretation

A normal result means current feature statistics are within configured tolerance.

A warning result means the feature movement should be reviewed, but the model is not automatically considered invalid.

A critical result means the feature movement may invalidate the active model assumptions. It should trigger operational review and, later, alert and rollback workflows.

Critical drift does not automatically prove the model is wrong. It proves the model is now scoring data that no longer resembles its approved baseline.

## Failure modes

The drift evaluator fails deliberately when:

- current feature dataframe is empty
- there are no numeric features
- no current numeric feature overlaps with the baseline
- required baseline fields are missing
- threshold configuration is missing
- critical thresholds are lower than warning thresholds
- observation window end is before observation window start
- drift event log is requested but does not exist

Failing loudly is correct here. Silent drift failure would make monitoring decorative and dangerous.

## Validation

Run drift tests:

    cd ~/Desktop/Project-4-ML-Pipeline-Anomaly-Detection
    source .venv/bin/activate
    PYTHONPATH=src pytest tests/test_drift.py -q

Expected result:

    15 passed

Run nearby regression tests:

    PYTHONPATH=src pytest tests/test_prediction_logging.py tests/test_batch_inference.py tests/test_api.py -q

Expected result:

    24 passed

Run final Checkpoint 12 validation:

    PYTHONPATH=src pytest tests/test_drift.py tests/test_prediction_logging.py tests/test_batch_inference.py tests/test_api.py -q

    PYTHONPATH=src python -m compileall -q src/anomaly_detection/drift.py tests/test_drift.py

Expected result:

    39 passed

## Fit inside Project 4

This checkpoint sits after prediction logging and before Prometheus metrics.

Flow:

    v002 training baseline
        ↓
    current feature batch/window
        ↓
    mean/variance drift evaluation
        ↓
    normal/warning/critical status
        ↓
    logs/alerts/drift_events.jsonl
        ↓
    future Prometheus metrics
        ↓
    future Grafana panels
        ↓
    future alert events
        ↓
    future rollback review

This makes Project 4 more than a model-serving demo. It gives the platform a way to know when the data underneath the model has changed.
