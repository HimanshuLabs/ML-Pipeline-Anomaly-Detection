# Inference Design — Project 4 ML Pipeline & Anomaly Detection

## Purpose

Project 4 supports anomaly inference over validated feature batches and, later, online API requests.

The inference layer is designed around one rule:

> Every prediction must be traceable to the model version, feature payload, threshold, timestamp, and source entity.

This keeps Project 4 aligned with the wider platform:

- Project 1 provides real-time behavioral and system-performance signals.
- Project 2/3 provides trusted historical warehouse, marts, and customer/campaign/product aggregates.
- Project 4 scores those features with a versioned anomaly model.

---

## Current implementation status

| Area | Status | Notes |
|---|---|---|
| Active production model loading | Implemented | Batch scorer reads `configs/active_model.yaml`. |
| Production model version | Implemented | Active model version is `v002`. |
| Batch scoring | Implemented | `src/anomaly_detection/batch_inference.py` scores pandas feature batches. |
| Prediction contract | Implemented | Each record includes model version, entity, score, anomaly flag, threshold, timestamp, feature hash, and feature payload. |
| Local JSONL persistence | Implemented | Fallback path is `logs/predictions/batch_predictions.jsonl`. |
| PostgreSQL prediction schema | Implemented | `sql/create_prediction_tables.sql` defines `ml.batch_predictions` and `ml.online_predictions`. |
| Direct PostgreSQL insert from Python | Planned | Schema exists, but this checkpoint uses JSONL fallback first for deterministic local validation. |
| Online FastAPI inference | Implemented | FastAPI service loads active `v002` model in memory and supports health, active model, single prediction, batch prediction, metrics, and rollback stub endpoints. |
| Prometheus metrics | Implemented | FastAPI exposes Prometheus-format runtime metrics through `/metrics`. |
| Drift checks | Implemented | Mean/variance drift checks compare current feature windows against the active `v002` baseline. |
| Grafana dashboards | Implemented locally | Operator and executive dashboard JSON files are stored under `monitoring/grafana/dashboards/`. |
| Alert events | Implemented locally | Alert records are written for critical drift, anomaly-rate spikes, latency breaches, prediction-error breaches, and manual alerts. |
| Rollback integration | Implemented locally | `/admin/rollback` supports dry-run and applied rollback to the previous stable model. |

---

## Batch inference path

Batch inference is the first implemented inference mode.

It is used for:

- periodic scoring over feature snapshots,
- historical backfills,
- warehouse/mart anomaly scans,
- low-cost validation of model behavior,
- generating prediction evidence before adding online serving.

Current flow:

```text
Validated Project 4 feature dataframe
    ↓
Load active production model pointer
    ↓
Load v002 model artifact + metadata + schema + baseline stats
    ↓
Validate model feature columns and numeric values
    ↓
Generate anomaly scores with Isolation Forest
    ↓
Mark anomalies using model prediction output
    ↓
Write prediction evidence to JSONL fallback
```

Current active model:

```text
model_name: isolation_forest
model_version: v002
feature_schema_version: feature_schema_v001
snapshot_type: real_source_extract
source_projects: project_1, project_2_3
```

---

## Batch prediction record contract

Each batch prediction record contains:

```text
prediction_id
batch_run_id
model_name
model_version
dataset_snapshot_id
training_dataset_id
feature_schema_version
entity_type
entity_id
source_project
source_table
source_record_id
score_timestamp
anomaly_score
is_anomaly
threshold_used
feature_payload_hash
feature_payload
inference_latency_ms
prediction_status
error_message
```

The minimum checkpoint-required fields are present:

```text
prediction_id
model_version
entity_id
anomaly_score
is_anomaly
threshold_used
score_timestamp
feature_payload_hash
```

This matters because prediction logs are not just output records. They are model evidence.

---

## Threshold behavior

For the active Isolation Forest model, batch inference uses:

```text
decision_function threshold = 0.0
```

Interpretation:

```text
score < 0.0   => anomalous
score >= 0.0  => normal
```

The scorer also supports a fallback path for model objects exposing only `score_samples`, but the current production model uses the standard Isolation Forest prediction interface.

---

## Persistence design

The PostgreSQL table schema already exists:

```text
ml.batch_predictions
ml.online_predictions
```

Checkpoint 9 intentionally writes to a local JSONL fallback first:

```text
logs/predictions/batch_predictions.jsonl
```

Reason:

- deterministic local validation,
- no dependency on database availability,
- easy inspection during development,
- safe replay into PostgreSQL later,
- stable prediction contract before adding database insert logic.

This is a staged implementation. It does not claim that direct PostgreSQL persistence from Python is complete.

---

## Batch vs online inference tradeoff

### Batch inference

Batch inference is cheaper and simpler.

Best use cases:

- hourly or daily anomaly scans,
- warehouse backfills,
- scoring historical snapshots,
- batch reporting,
- drift/current-metric calculations,
- validating model behavior before online serving.

Strengths:

- lower infrastructure pressure,
- easier retry logic,
- easier backfills,
- less strict latency requirement,
- better for broad historical analysis.

Weaknesses:

- delayed detection,
- not suitable for immediate intervention,
- cannot support real-time user-facing decisions.

### Online inference

Online inference is implemented through FastAPI and loads the active `v002` model in memory.

Best use cases:

- immediate anomaly scoring,
- live feature API integration,
- low-latency operational decisions,
- serving current model state to downstream applications.

Strengths:

- immediate response,
- useful for real-time detection,
- better for interactive systems.

Weaknesses:

- tighter latency budget,
- API reliability matters,
- model must be preloaded in memory,
- request validation must be lightweight,
- operational monitoring becomes mandatory.

---

## Failure modes

Batch inference explicitly protects against:

| Failure mode | Current behavior |
|---|---|
| Missing active model pointer | Raises `BatchInferenceError`. |
| Missing model artifact | Raises `BatchInferenceError`. |
| Missing metadata file | Raises `BatchInferenceError`. |
| Missing baseline stats file | Raises `BatchInferenceError`. |
| Missing feature schema file | Raises `BatchInferenceError`. |
| Metadata/model-version mismatch | Raises `BatchInferenceError`. |
| Baseline/model-version mismatch | Raises `BatchInferenceError`. |
| Feature schema mismatch | Raises `BatchInferenceError`. |
| Missing model feature columns | Raises `BatchInferenceError`. |
| Non-numeric model feature values | Raises `BatchInferenceError`. |
| Empty input batch | Raises `BatchInferenceError`. |
| Unsupported input file format | Raises `BatchInferenceError`. |

This is intentional. Silent scoring with broken features would poison prediction evidence.

---

## Validation evidence

Checkpoint 9 validation confirms:

```text
active model artifacts load successfully
model version resolves to v002
feature count resolves to 51
baseline-like row scores successfully
batch inference unit tests pass
JSONL prediction persistence works
missing feature validation fails safely
non-numeric feature validation fails safely
feature payload hash is stable
```

Smoke validation produced a normal prediction for a baseline-like row:

```text
model_version = v002
entity_id = smoke_customer_001
is_anomaly = False
threshold_used = 0.0
```

That result is expected because the row was built from baseline feature means.

---

## Current limitation

Implemented:

```text
Python batch scoring
active v002 model loading
model metadata validation
feature schema validation
local JSONL prediction evidence
prediction table schema
unit tests
inference design documentation
```

Planned:

```text
direct PostgreSQL insert helper
online FastAPI inference
Prometheus metrics
drift checks
Grafana dashboard
alert events
rollback integration with inference health evidence
```

Checkpoint 9 is complete only when the batch scorer, unit tests, and this document validate together.

## Online FastAPI inference

### Implemented status

Online inference is implemented as a local FastAPI service that loads the active production model from `configs/active_model.yaml`.

The active model is currently:

| Field | Value |
|---|---|
| Model name | `isolation_forest` |
| Active version | `v002` |
| Artifact path | `artifacts/models/isolation_forest/model_version=v002/model.joblib` |
| Feature schema version | `feature_schema_v001` |
| Feature count | `51` |
| Snapshot type | `real_source_extract` |
| Online p95 latency budget | `200 ms` |

The service loads the model once into memory through `OnlineInferenceService`. It does not read the model artifact from disk for every prediction request.

### Implemented endpoints

| Endpoint | Status | Purpose |
|---|---|---|
| `GET /health` | Implemented | Returns API health and active model load status. |
| `GET /model/active` | Implemented | Returns active model metadata, feature count, baseline anomaly rate, threshold, and load timestamp. |
| `POST /predict` | Implemented | Scores one model-ready feature payload with the active in-memory model. |
| `POST /predict/batch` | Implemented | Scores multiple model-ready payloads through the same in-memory service. |
| `GET /metrics` | Implemented | Exposes Prometheus-format counters and latency histogram. |
| `POST /admin/rollback` | Implemented locally | Supports dry-run validation and applied rollback to the previous stable model. |

### Prediction response contract

| Field | Meaning |
|---|---|
| `prediction_id` | Unique prediction event ID. |
| `model_name` | Active model family. |
| `model_version` | Active model version used for scoring. |
| `dataset_snapshot_id` | Training snapshot lineage for the active model. |
| `feature_schema_version` | Feature contract version used by the model. |
| `entity_type` | Scored entity class, currently usually `user`. |
| `entity_id` | Scored entity identifier. |
| `prediction_timestamp` | UTC timestamp for the score event. |
| `anomaly_score` | Isolation Forest decision score. |
| `is_anomaly` | Boolean anomaly decision. |
| `threshold_used` | Threshold used for classification. |
| `drift_status` | Current drift evaluation state. Currently `not_evaluated` in the online path. |
| `feature_payload_hash` | Stable SHA-256 hash of the feature payload. |
| `latency_ms` | Measured scoring latency inside the inference service. |

### Threshold behavior

The `v002` baseline artifact does not currently include an explicit threshold field.

For online scoring, the service uses `0.0` as the fallback threshold. This matches the standard Isolation Forest decision boundary where negative `decision_function` values are treated as more anomalous.

This is intentionally documented instead of hidden. Later threshold tuning can replace this fallback with an explicit configured threshold after enough production-like score distributions are observed.

### Prometheus metrics exposed

| Metric | Purpose |
|---|---|
| `project4_prediction_requests_total` | Counts successful and failed prediction requests by endpoint and model version. |
| `project4_prediction_errors_total` | Counts prediction errors by endpoint and model version. |
| `project4_anomalies_detected_total` | Counts online anomaly decisions by endpoint and model version. |
| `project4_prediction_latency_seconds` | Tracks request latency by endpoint and model version. |

### Current limitations

| Area | Current state |
|---|---|
| Drift status | Returned by the online response contract; full per-request online drift evaluation remains planned hardening. Batch/current-window drift checks are implemented separately. |
| Prediction persistence | Online and batch prediction evidence is persisted locally through JSONL logs. Direct PostgreSQL inserts remain planned hardening. |
| Rollback | `/admin/rollback` supports dry-run and applied rollback. Applied rollback updates the active model pointer and writes rollback evidence. |
| Threshold tuning | Uses fallback `0.0` because `v002` has no explicit threshold artifact yet. |
| Authentication | Not implemented locally. This is acceptable for the local project build, but would be required before real deployment. |

### Validation evidence

Local validation currently covers:

    PYTHONPATH=src pytest tests/test_api.py tests/test_batch_inference.py -q

Expected result:

    15 passed

Manual API validation should include:

    uvicorn api.main:app --host 0.0.0.0 --port 8004
    curl http://localhost:8004/health
    curl http://localhost:8004/model/active
    curl http://localhost:8004/metrics

Online inference is considered partially complete when health and active model endpoints work.

It is considered complete for this checkpoint when `/predict` and `/predict/batch` score against the active in-memory `v002` model.

## Prediction evidence logging

Prediction logging is implemented as a shared local-first contract used by both inference paths.

Implemented:

- Batch inference appends prediction evidence to `logs/predictions/batch_predictions.jsonl`.
- Online FastAPI inference appends prediction evidence to `logs/predictions/online_predictions.jsonl`.
- Both paths use `src/anomaly_detection/prediction_logging.py` for JSON-safe serialization and JSONL persistence.
- Batch inference keeps the existing `write_predictions_jsonl` wrapper, but delegates to the shared prediction logging module.
- Online `/predict` persists one evidence record per successful request.
- Online `/predict/batch` persists one evidence record per scored payload.
- Prediction logging failure is logged but does not fail the API response path.

Prediction evidence includes:

- `prediction_id`
- `prediction_source`
- `model_name`
- `model_version`
- `entity_type`
- `entity_id`
- `anomaly_score`
- `is_anomaly`
- `threshold_used`
- `prediction_status`
- `feature_payload_hash`
- latency field from the inference path
- `logged_at`

Database readiness:

- `sql/create_prediction_tables.sql` already defines `ml.batch_predictions`.
- `sql/create_prediction_tables.sql` already defines `ml.online_predictions`.
- Direct PostgreSQL insert wiring is planned for a later hardening pass.
- The JSONL record shape is intentionally aligned with the SQL tables so database persistence can be added without changing the inference response contract.

Operational reason:

Prediction evidence makes anomaly scoring auditable. A prediction can be traced back to the model version, scored entity, timestamp, anomaly score, threshold, and payload hash. This is required before drift checks, alert events, current metrics, and rollback decisions can be trusted.
