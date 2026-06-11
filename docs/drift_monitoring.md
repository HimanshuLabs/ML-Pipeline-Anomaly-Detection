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

This checkpoint sits after prediction logging and feeds the Prometheus metrics layer.

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
    Prometheus metrics
        ↓
    future Grafana panels
        ↓
    future alert events
        ↓
    future rollback review

This makes Project 4 more than a model-serving demo. It gives the platform a way to know when the data underneath the model has changed.

## Prometheus metrics

Project 4 exposes Prometheus metrics through the FastAPI `/metrics` endpoint.

Implemented metrics:

| Metric | Type | Purpose |
|---|---|---|
| `prediction_requests_total` | Counter | Counts online prediction requests by endpoint, model version, and status. |
| `prediction_errors_total` | Counter | Counts failed prediction attempts by endpoint and model version. |
| `anomalies_detected_total` | Counter | Counts anomaly predictions by endpoint and model version. |
| `prediction_latency_ms` | Histogram | Tracks online prediction latency in milliseconds against the service latency budget. |
| `drift_detected_total` | Counter | Counts warning and critical drift events by model version, feature, and drift status. |
| `feature_mean_delta` | Gauge | Publishes the latest absolute mean delta for each evaluated feature. |
| `feature_variance_delta` | Gauge | Publishes the latest absolute variance delta for each evaluated feature. |
| `active_model_version` | Gauge | Publishes the currently loaded model version as a labeled gauge value of `1`. |
| `model_rollback_total` | Counter | Counts rollback attempts by source version, target version, and status. |

Current implementation status:

- `/metrics` is implemented in `api/main.py`.
- Metric definitions live in `src/anomaly_detection/metrics.py`.
- Online inference records request, error, anomaly, latency, and active model metrics.
- Drift metrics can be published from drift evaluation results through `publish_drift_evaluation_metrics`.
- Rollback metrics are exposed as a contract now and will be wired to the real rollback mechanism in the rollback checkpoint.
- `monitoring/prometheus.yml` provides a local scrape configuration for the FastAPI service on port `8004`.

Operational interpretation:

- `prediction_requests_total` and `prediction_errors_total` prove API reliability.
- `anomalies_detected_total` shows anomaly volume over time.
- `prediction_latency_ms` proves whether online inference stays within the p95 latency budget.
- `feature_mean_delta` and `feature_variance_delta` make drift visible instead of burying it in logs.
- `active_model_version` connects runtime behavior to the model registry.
- `model_rollback_total` will become rollback evidence once rollback controls are implemented.

Limitations:

- Metrics are in-memory process metrics. Restarting the API resets counters unless Prometheus has already scraped them.
- PostgreSQL persistence for current metric snapshots is not implemented in this checkpoint.
- Grafana visualization is planned for the dashboard checkpoint.

## Grafana dashboard

Project 4 includes a Grafana dashboard definition at:

~~~text
monitoring/grafana/dashboards/anomaly_detection_dashboard.json
~~~

The dashboard is designed for local Grafana on:

~~~text
http://localhost:3004
~~~

It uses Prometheus as the datasource and expects Prometheus to scrape the Project 4 FastAPI metrics endpoint:

~~~text
http://localhost:8004/metrics
~~~

### Dashboard status

Implemented:

- Dashboard JSON exists and validates with `python -m json.tool`.
- Dashboard panels use the Prometheus metrics exposed by the FastAPI service.
- Panels are grouped around model health, traffic, anomaly behavior, latency, drift, alert proxy, and rollback evidence.
- The dashboard is local-first and does not claim cloud-managed Grafana deployment.

Planned in later checkpoints:

- Explicit alert-event metrics are handled in the alert events checkpoint.
- Rollback controls are handled in the rollback checkpoint.
- Until those checkpoints are complete, the dashboard includes alert and rollback panels that are wired to available metrics but may show zero or proxy values.

### Dashboard panels

| Panel | Metric/query family | What it proves |
|---|---|---|
| Active model version | `active_model_version` | Shows which model version is currently loaded by the online inference service. |
| Prediction requests — last 5m | `prediction_requests_total` | Shows recent online scoring volume. |
| Anomalies detected — last 5m | `anomalies_detected_total` | Shows recent anomaly volume. |
| Anomaly rate — last 5m | `anomalies_detected_total / prediction_requests_total` | Shows whether live anomaly behavior is stable or spiking. |
| Prediction request rate | `rate(prediction_requests_total[5m])` | Shows request throughput by endpoint and status. |
| Anomaly rate over time | `rate(anomalies_detected_total) / rate(prediction_requests_total)` | Shows anomaly-rate trend over time. |
| Prediction latency p50 | `histogram_quantile(0.50, prediction_latency_ms_bucket)` | Tracks median online inference latency. |
| Prediction latency p95 | `histogram_quantile(0.95, prediction_latency_ms_bucket)` | Tracks the service latency budget. The Project 4 target is p95 below 200 ms. |
| Feature mean delta | `feature_mean_delta` | Shows latest mean drift magnitude by feature. |
| Feature variance delta | `feature_variance_delta` | Shows latest variance drift magnitude by feature. |
| Drift events — last 15m | `drift_detected_total` | Shows recent drift detections. |
| Prediction errors — last 15m | `prediction_errors_total` | Shows online inference reliability issues. |
| Alert proxy — drift events last 15m | `drift_detected_total` | Temporary alert proxy until explicit alert event metrics are implemented. |
| Rollback count — last 24h | `model_rollback_total` | Shows rollback evidence once rollback controls are implemented. |
| Drift events by feature | `rate(drift_detected_total[5m]) by feature_name` | Shows which features are repeatedly drifting. |
| Rollback attempts by status | `rate(model_rollback_total[5m]) by status` | Shows rollback attempt outcomes once rollback execution exists. |

### Import flow

1. Confirm the API is running on port `8004`.
2. Confirm Prometheus is running and scraping `localhost:8004`.
3. Open Grafana on port `3004`.
4. Add or confirm a Prometheus datasource.
5. Import `monitoring/grafana/dashboards/anomaly_detection_dashboard.json`.
6. Select the Prometheus datasource when Grafana asks for `${DS_PROMETHEUS}`.
7. Generate a few `/predict` requests and refresh the dashboard.

### Operational interpretation

Healthy dashboard behavior:

- `active_model_version` shows the expected production model, currently `v002`.
- Request count increases when `/predict` or `/predict/batch` is called.
- Anomaly count increases only when the model returns anomaly predictions.
- p95 latency stays below the configured online inference budget.
- Feature mean and variance deltas stay within expected drift thresholds.
- Prediction errors remain near zero.
- Rollback count stays zero unless a rollback is intentionally triggered.

Unhealthy dashboard behavior:

- Active model version is missing or does not match the active model config.
- Request count is flat while the API is receiving traffic.
- Prediction errors increase.
- p95 latency crosses 200 ms.
- Drift events rise repeatedly for the same features.
- Anomaly rate materially exceeds the approved baseline anomaly rate.
- Rollback count increases unexpectedly.

### Current limitations

- Prometheus metrics are process-local. API restarts reset counters unless Prometheus has already scraped them.
- The dashboard reads metrics from Prometheus, not directly from PostgreSQL.
- Alert proxy panels use drift metrics until explicit alert-event metrics are implemented.
- Rollback panels are wired but only become operational evidence after rollback controls emit `model_rollback_total`.
- This is a local monitoring implementation, not a managed cloud Grafana deployment.
