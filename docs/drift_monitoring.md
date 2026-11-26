# Drift Monitoring — Project 4 ML Pipeline & Anomaly Detection

## Purpose

Project 4 monitors whether current feature behavior has moved materially away from the training baseline used to approve the active anomaly detection model.

The current implementation focuses on lightweight mean and variance drift checks. It is intentionally simple. This checkpoint does not pretend to be a full statistical drift platform. It adds an operational signal that answers one practical question:

Are the features being scored now still behaving like the features the model was trained on?

## Current state (2026-11-26)

Implemented:

- Drift is measured against the ACTIVE model's training baseline (the version named by
  the shared pointer: `ml.active_model_pointer` in database mode, else
  `configs/active_model.yaml`), read from its `baseline_stats.json`.
- A segmented model is evaluated per segment (entity type) against that segment's
  baseline and features; a single model on its whole feature set.
- Mean and variance drift per numeric feature; three states: normal, warning, critical;
  default thresholds plus feature-specific overrides (`configs/drift_thresholds.yaml`);
  a segment with fewer rows than the minimum is refused instead of producing noise.
- One drift event per evaluated feature, written to `logs/alerts/drift_events.jsonl` and,
  in database mode, to `monitoring.drift_events`; critical drift also creates alert
  events (JSONL and `monitoring.alert_events`).
- The drift job (`python -m anomaly_detection.drift_job --source batch-predictions
  --hours 24`, or `--input <file>`) evaluates the last hours of `ml.batch_predictions`
  for the active version. In Kubernetes it runs hourly (`k8s/cronjob-drift.yaml`).
  It reads the window in chunks into per-segment columns: peak 0.51 GB at 11,85,050
  rows, inside the CronJob's 512 MiB limit, with the same drift events as a one-piece
  read.
- The API reports the latest drift status of the request's segment in each online
  prediction (`drift_status`), or `not_evaluated` when there is none.
- Prometheus drift metrics and Grafana panels (see below).

Active model baseline (2026-11-26): `isolation_forest` v007, segmented
(`campaign`, `customer`, `funnel_day`, `product`, `user`, `web_customer`),
`feature_schema_v003`, snapshot `real_source_20261126T120904Z`:

    artifacts/models/isolation_forest/model_version=v007/baseline_stats.json

## Nothing to evaluate: a stalled pipeline or a quiet one

With no batch predictions for the active version in the window, the drift job reports
`not_evaluated`, records nothing, and exits 0. Its log line (stage 14c-2) separates three
cases by the model name's last successful batch prediction (any version):

| Last successful batch prediction | Log | Meaning |
|---|---|---|
| never | INFO `No current rows for … ; drift not_evaluated` | Nothing was ever batch-scored (e.g. a fresh cluster). |
| before the window | WARNING `… batch scoring stalled? last batch prediction at <ts> (model <version>), none in the last <h>h` | Batch scoring ran before but not recently: check it. |
| inside the window, another version | INFO `… no rows for <active> yet; last batch run used <version> at <ts>` | Usually right after a promote: the next batch run scores the new version. |

To tell a stalled batch pipeline from a quiet one, compare the latest batch score with
the batch schedule:

    SELECT model_version,
           max(score_timestamp) AS last_scored,
           count(*) FILTER (WHERE score_timestamp > now() - interval '24 hours') AS rows_24h
    FROM ml.batch_predictions
    WHERE model_name = 'isolation_forest' AND prediction_status = 'success'
    GROUP BY model_version
    ORDER BY last_scored DESC;

If `last_scored` is older than the batch schedule, scoring has stalled: check the batch
job's last run and its logs. If scoring ran on schedule and the window is still empty
for the active version, there was no traffic for it (or the last run used another
version).

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

## Persistence

Drift events are always appended to `logs/alerts/drift_events.jsonl` (local, inspectable,
no database needed) and, in database mode, inserted into `monitoring.drift_events` with
the same event contract. The API reads the latest run per segment from the database to
report `drift_status`.

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
- It evaluates batch-scored rows only; online requests are not part of the drift window.
- It does not automatically roll back a model.

These limitations are deliberate: drift is an operational signal, not a statistical
drift platform.

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

Run the drift and drift-job tests from the repository root with the dev image (the
`dev()` function in the README's "Local runbook"):

    dev env -u PYTHONPATH python -m pytest -p no:cacheprovider -o addopts= -q tests/test_drift.py tests/test_drift_job.py

The database-backed drift-job tests (`-m postgres`) need a disposable test database; see
the README's "Local runbook".

## Fit inside Project 4

This checkpoint sits after prediction logging and feeds the Prometheus metrics layer.

Flow:

    active model's training baseline (per segment)
        ↓
    batch predictions of the last N hours (or a feature file)
        ↓
    mean/variance drift evaluation
        ↓
    normal/warning/critical status
        ↓
    logs/alerts/drift_events.jsonl + monitoring.drift_events
        ↓
    Prometheus metrics
        ↓
    Grafana panels
        ↓
    alert events (critical drift)
        ↓
    rollback review

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
- Rollback metrics are wired to the local rollback mechanism.
- `monitoring/prometheus.yml` provides a local scrape configuration for the FastAPI service on port `8004`.

Operational interpretation:

- `prediction_requests_total` and `prediction_errors_total` prove API reliability.
- `anomalies_detected_total` shows anomaly volume over time.
- `prediction_latency_ms` proves whether online inference stays within the p95 latency budget.
- `feature_mean_delta` and `feature_variance_delta` make drift visible instead of burying it in logs.
- `active_model_version` connects runtime behavior to the model registry.
- `model_rollback_total` is rollback evidence emitted by the local rollback mechanism.

Limitations:

- Metrics are in-memory process metrics. Restarting the API resets counters unless Prometheus has already scraped them.
- PostgreSQL persistence for current metric snapshots is not implemented in this checkpoint.
- Grafana visualization is implemented locally through dashboard JSON files.

## Grafana dashboards

Project 4 now includes two Grafana dashboard definitions:

~~~text
monitoring/grafana/dashboards/anomaly_detection_dashboard.json
monitoring/grafana/dashboards/anomaly_detection_executive_dashboard.json
~~~

Both dashboards are designed for local Grafana on:

~~~text
http://localhost:3004
~~~

Both dashboards use Prometheus as the datasource and expect Prometheus to scrape the Project 4 FastAPI metrics endpoint:

~~~text
http://anomaly-api:8004/metrics   (compose network; from the host: http://127.0.0.1:8014/metrics)
~~~

### Dashboard inventory

| Dashboard | File | Audience | Purpose |
|---|---|---|---|
| Operator anomaly monitoring dashboard | `monitoring/grafana/dashboards/anomaly_detection_dashboard.json` | Data engineer, ML platform engineer, SRE-style reviewer | Detailed operational monitoring for model version, request volume, anomaly count, anomaly rate, latency, feature drift, prediction errors, rollback evidence, and drift status. |
| Executive ML health dashboard | `monitoring/grafana/dashboards/anomaly_detection_executive_dashboard.json` | Technical reviewer, technical reviewer, business-facing stakeholder | Polished summary dashboard showing active model, request volume, anomaly rate, p95 latency, drift events, prediction errors, rollback count, and a business-readable operational readout. |

### Operator dashboard

The operator dashboard is the deeper engineering view.

It contains panels for:

- active production model version
- prediction request count
- anomaly count
- anomaly rate
- request rate by endpoint and status
- p50 latency
- p95 latency
- feature mean delta
- feature variance delta
- drift events
- prediction errors
- alert proxy
- rollback count
- drift events by feature
- rollback attempts by status

This dashboard proves the ML service is observable at the level expected from a production-style MLOps system. It is meant to answer:

- What model is serving?
- Is traffic flowing?
- Is anomaly behavior stable?
- Is latency inside budget?
- Are features drifting?
- Are prediction errors rising?
- Has rollback happened?

### Executive dashboard

The executive dashboard is the cleaner demo view.

It contains panels for:

- project summary
- active model
- requests in the last 15 minutes
- anomaly rate in the last 15 minutes
- prediction errors in the last 15 minutes
- p95 latency
- drift events
- rollback count
- anomaly count
- latency trend against SLA
- drift signal trend
- operational readout

This dashboard is intentionally less noisy. It is useful when the goal is to explain the system quickly without dragging the viewer through every operational detail.

### Dashboard status

Implemented:

- Both dashboard JSON files exist.
- Both dashboard JSON files validate with `python -m json.tool`.
- Both dashboards use Prometheus metrics exposed by the FastAPI service.
- Both dashboards are local-first and do not claim managed cloud Grafana deployment.
- The operator dashboard is designed for technical debugging.
- The executive dashboard is designed for demo/storytelling clarity.

Notes:

- Alert events are written as JSONL evidence and, in database mode, to `monitoring.alert_events`.
- Rollback runs through `POST /admin/rollback` (X-Admin-Token required).
- Alert and rollback panels may show zero values until local traffic, drift evaluation, alerts, or rollback actions are generated.

### Import flow

1. Confirm the compose API is running (`curl -s http://127.0.0.1:8014/health`).
2. Confirm Prometheus (`127.0.0.1:9094`) is up and scraping `anomaly-api:8004` inside the compose network.
3. Open Grafana on `127.0.0.1:3004`.
4. Add or confirm a Prometheus datasource.
5. Import `monitoring/grafana/dashboards/anomaly_detection_dashboard.json`.
6. Import `monitoring/grafana/dashboards/anomaly_detection_executive_dashboard.json`.
7. Select the Prometheus datasource when Grafana asks for `${DS_PROMETHEUS}`.
8. Generate a few `/predict` requests and refresh both dashboards.

### Operational interpretation

Healthy behavior:

- `active_model_version` shows the expected production model (the version `GET /model/active` reports).
- Request count increases when `/predict` or `/predict/batch` is called.
- Anomaly count increases only when the model returns anomaly predictions.
- p95 latency stays below the configured online inference budget.
- Feature mean and variance deltas stay within expected drift thresholds.
- Prediction errors remain near zero.
- Rollback count stays zero unless a rollback is intentionally triggered.

Unhealthy behavior:

- Active model version is missing or does not match the active model config.
- Request count is flat while the API is receiving traffic.
- Prediction errors increase.
- p95 latency crosses 200 ms.
- Drift events rise repeatedly for the same features.
- Anomaly rate materially exceeds the approved baseline anomaly rate.
- Rollback count increases unexpectedly.

### Current limitations

- Prometheus metrics are process-local. API restarts reset counters unless Prometheus has already scraped them.
- The dashboards read metrics from Prometheus, not directly from PostgreSQL.
- Alert panels depend on emitted local alert evidence and Prometheus metrics.
- Rollback panels become operational evidence when rollback controls emit `model_rollback_total`.
- This is a local monitoring implementation, not a managed cloud Grafana deployment.

## Alert events

Alert events are a monitoring contract persisted to JSONL and, in database mode, to `monitoring.alert_events`.

Implemented:

- Critical drift alerts are generated from drift evaluation results.
- Anomaly-rate spike alerts are generated when current anomaly rate exceeds the configured baseline multiplier.
- Latency budget alerts are generated when online p95 latency exceeds the 200 ms service budget.
- Prediction error-rate alerts are generated when runtime error rate exceeds the configured threshold.
- Manual alert test events are supported for smoke validation.
- Alert records are persisted to `logs/alerts/anomaly_alerts.jsonl`.

Database:

- `sql/create_monitoring_tables.sql` defines `monitoring.alert_events`.
- The JSONL alert shape matches the table; in database mode every alert is also inserted there.

Alert event fields include:

- `alert_event_id`
- `drift_event_id`
- `model_name`
- `model_version`
- `alert_type`
- `severity`
- `alert_status`
- `alert_source`
- `metric_name`
- `metric_value`
- `threshold_value`
- `comparison_operator`
- `entity_type`
- `entity_id`
- `message`
- `details`
- `triggered_at`

Supported alert types:

- `drift_critical`
- `anomaly_rate_spike`
- `latency_budget_breach`
- `prediction_error_rate_breach`
- `manual_alert`

Current operational behavior:

Critical drift creates operational alert evidence. Warning drift remains visible through drift events and Prometheus metrics but does not create an alert event in this checkpoint. This is intentional: alerts should be meaningful, not noisy.

Alert outputs:

- `logs/alerts/anomaly_alerts.jsonl`
- `monitoring.alert_events`

Both are written: JSONL always, the table in database mode.

Failure-mode notes:

- Alert generation failure should not rewrite or invalidate the original drift result.
- Alerts are evidence for operator review and future rollback decisions.
- Alert existence does not automatically prove the model is wrong.
- Critical drift proves the current scored data has moved materially away from the approved training baseline.
- Latency and prediction-error alerts indicate service reliability risk.
- Anomaly-rate spikes indicate scoring behavior has materially changed against the approved baseline.

Rollback relationship:

Alert events are one input evidence layer for rollback review. Rollback is not blind automation. It consumes alert evidence, registry state, active model state, and previous stable model metadata before changing the active production pointer.
