# Inference Design — Project 4 ML Pipeline & Anomaly Detection

## Purpose

Project 4 supports anomaly inference over validated feature batches and, later, online API requests.

The inference layer is designed around one rule:

> Every prediction must be traceable to the model version, feature payload, threshold, timestamp, and source entity.

This keeps Project 4 aligned with the wider platform:

- Project 1 provides real-time behavioral and system-performance signals.
- Project 2/3 provides trusted historical warehouse, marts, and customer/campaign/product aggregates.
- Project 4 scores those features with a versioned anomaly model.
- Project 5 can later query model versions, prediction evidence, drift events, and rollback history through the AI decision layer.

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
| Online FastAPI inference | Planned | This belongs to the next checkpoint. |
| Prometheus metrics | Planned | Added after online/batch prediction paths stabilize. |
| Drift checks | Planned | Drift comparison comes after prediction evidence exists. |
| Grafana dashboard | Planned | Dashboard panels will use prediction, drift, latency, and model-version metrics later. |
| Alert events | Planned | Alerts will be generated after drift and runtime metrics exist. |
| Rollback integration | Planned | Rollback controls already exist separately and will later connect to inference health signals. |

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

Online inference is planned for the next checkpoint through FastAPI.

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
rollback integration with inference health
```

Checkpoint 9 is complete only when the batch scorer, unit tests, and this document validate together.
