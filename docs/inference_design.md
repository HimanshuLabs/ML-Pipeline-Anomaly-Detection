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
| Active production model loading | Implemented | The version named by the shared pointer: `ml.active_model_pointer` in database mode, else `configs/active_model.yaml`. |
| Production model version | Implemented | `v007` on 2026-11-26: segmented, one Isolation Forest per entity type, `feature_schema_v003`. |
| Batch scoring | Implemented | `src/anomaly_detection/batch_inference.py` scores pandas feature batches; segmented models route each row by `entity_type`. |
| Prediction contract | Implemented | Each record includes model version, entity, score, anomaly flag, threshold, timestamp, feature hash, and feature payload. |
| JSONL persistence | Implemented | `logs/predictions/batch_predictions.jsonl` and `logs/predictions/online_predictions.jsonl`. |
| PostgreSQL persistence | Implemented | In database mode, batch runs insert `ml.batch_predictions` and the API inserts `ml.online_predictions` (after the response). |
| Online FastAPI inference | Implemented | Serves the pointer's version in memory, swaps it in-process when the pointer changes; `/livez`, `/health`, `/model/active`, `/predict`, `/predict/batch`, `/metrics`, `/admin/rollback`. |
| Prometheus metrics | Implemented | FastAPI exposes Prometheus-format runtime metrics through `/metrics`. |
| Drift checks | Implemented | Mean/variance drift against the active model's baseline, per segment; hourly drift job. |
| Grafana dashboards | Implemented locally | Operator and executive dashboard JSON files are stored under `monitoring/grafana/dashboards/`. |
| Alert events | Implemented | JSONL and `monitoring.alert_events` for critical drift, anomaly-rate spikes, latency breaches, prediction-error breaches, and manual alerts. |
| Rollback integration | Implemented | `POST /admin/rollback` (X-Admin-Token) with dry run; target rules in `docs/rollback_runbook.md`. |

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
Load the active version's model artifact + metadata + schema + baseline stats
    ↓
Validate model feature columns and numeric values (per segment for segmented models)
    ↓
Generate anomaly scores with Isolation Forest
    ↓
Mark anomalies using model prediction output
    ↓
Write prediction evidence to JSONL (and ml.batch_predictions in database mode)
```

Current active model (2026-11-26):

```text
model_name: isolation_forest
model_version: v007
model_kind: segmented (campaign, customer, funnel_day, product, user, web_customer)
feature_schema_version: feature_schema_v003
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

Prediction evidence goes to two sinks:

```text
logs/predictions/batch_predictions.jsonl    logs/predictions/online_predictions.jsonl   (always)
ml.batch_predictions                        ml.online_predictions                       (database mode)
```

A batch run inserts its rows in one transaction, then runs `ANALYZE ml.batch_predictions`
in its own transaction (a failed `ANALYZE` is logged; the run is kept). The API writes online evidence as a
background task after the response, so a persistence failure is counted
(`prediction_persist_failures_total{sink=...}`) and logged but never fails the request.
`integration.prediction_outputs` exposes both tables to Project 5.

## feature_payload_hash and the 14c-5 cutover

`feature_payload_hash` is the sha256 of exactly the features the model scored for the
record (for a segmented model, the segment features of its entity type), with values
cast to float (NaN and inf as null), in sorted-key compact canonical JSON. Extra request
keys (`entity_id`, `entity_type`, unknown keys) never enter it. Online and batch scoring
use one shared function, so the same row hashes identically on both paths.

This canonical form starts at commit `a57058a` (stage 14c-5). Rows written before it differ:

| Rows | Before `a57058a` | From `a57058a` |
|---|---|---|
| Online | sha256 of the whole request payload as sent: every key, ints kept as ints (`4` and `4.0` hashed differently) | Only the scored features, as floats |
| Batch | Only the scored features, but values as pandas produced them: int columns stayed ints in frames with a text column | Only the scored features, always as floats |

Old rows are not rewritten. The stored `feature_payload` JSON of batch rows is unchanged
(raw values); only the hash computation changed. Joining or deduplicating on
`feature_payload_hash` across the cutover can therefore miss matches; from the cutover
on, online and batch hashes of the same row match.

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

Online inference is implemented through FastAPI and keeps the active version (the shared pointer's) in memory.

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

## History: checkpoint 9 (batch scoring with v002)

The two subsections below record checkpoint 9 as it was written. Everything listed under
"Planned" there is implemented now (see "Current implementation status").

### Validation evidence

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

### Checkpoint 9 scope

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

Online inference is a FastAPI service that serves the version named by the shared
pointer: `ml.active_model_pointer` in database mode (`pointer_source: postgres`), else
`configs/active_model.yaml` (`pointer_source: file`). It polls the database pointer every
`PROJECT4_POINTER_POLL_SECONDS` (default 30) and swaps the model in-process when the
pointer changes.

The active model on 2026-11-26:

| Field | Value |
|---|---|
| Model name | `isolation_forest` |
| Active version | `v007` |
| Model kind | segmented (6 entity types; each request is scored by its `entity_type`'s segment) |
| Artifact path | `artifacts/models/isolation_forest/model_version=v007/model.joblib` |
| Feature schema version | `feature_schema_v003` |
| Feature count | `52` in total (each segment scores its own subset) |
| Snapshot type | `real_source_extract` |
| Online p95 latency budget | `200 ms` |

The service loads the model once into memory through `OnlineInferenceService`. It does not read the model artifact from disk for every prediction request.

### Implemented endpoints

| Endpoint | Status | Purpose |
|---|---|---|
| `GET /livez` | Implemented | Liveness: the process is up (never touches the model). |
| `GET /health` | Implemented | API health, model-load status, and `pointer_source`; in database mode also `pointer_model_version` and `model_reload_status` (`current`, or `failed: …` while the previous model keeps serving). |
| `GET /model/active` | Implemented | Returns active model metadata, feature count, baseline anomaly rate, threshold, and load timestamp. |
| `POST /predict` | Implemented | Scores one feature payload with the active in-memory model; a segmented model needs its `entity_type` (default `user`). |
| `POST /predict/batch` | Implemented | Scores multiple payloads of one entity type through the same in-memory service. |
| `GET /metrics` | Implemented | Exposes Prometheus-format counters and the latency histogram. |
| `POST /admin/rollback` | Implemented | Dry-run and applied rollback; requires `X-Admin-Token` (503 when no token is configured, 401 when it is missing or wrong). |

### Prediction response contract

| Field | Meaning |
|---|---|
| `prediction_id` | Unique prediction event ID. |
| `model_name` | Active model family. |
| `model_version` | Active model version used for scoring. |
| `dataset_snapshot_id` | Training snapshot lineage for the active model. |
| `feature_schema_version` | Feature contract version used by the model. |
| `entity_type` | Scored entity class; selects the segment of a segmented model. |
| `entity_id` | Scored entity identifier. |
| `prediction_timestamp` | UTC timestamp for the score event. |
| `anomaly_score` | Isolation Forest decision score. |
| `is_anomaly` | Boolean anomaly decision. |
| `threshold_used` | Threshold used for classification. |
| `drift_status` | Latest drift-job status of the request's segment (`normal`, `warning`, `critical`), or `not_evaluated` when there is none. |
| `feature_payload_hash` | SHA-256 of the scored features in canonical form (see "feature_payload_hash and the 14c-5 cutover"). |
| `latency_ms` | Measured scoring latency inside the inference service. |

### Threshold behavior

The v004 baseline artifact (like v002 before it) does not include an explicit threshold field.

For online scoring, the service uses `0.0` as the fallback threshold. This matches the standard Isolation Forest decision boundary where negative `decision_function` values are treated as more anomalous.

This is intentionally documented instead of hidden. Later threshold tuning can replace this fallback with an explicit configured threshold after enough production-like score distributions are observed.

### Prometheus metrics exposed

| Metric | Purpose |
|---|---|
| `prediction_requests_total` | Counts successful and failed prediction requests by endpoint and model version. |
| `prediction_errors_total` | Counts prediction errors by endpoint and model version. |
| `predictions_scored_total` | Counts scored payloads by endpoint and model version. |
| `anomalies_detected_total` | Counts online anomaly decisions by endpoint and model version. |
| `prediction_latency_ms` | Histogram of in-handler scoring latency (milliseconds) by endpoint and model version; the evidence writes run after the response and are not included. |
| `prediction_persist_failures_total` | Counts evidence writes that failed, by sink (`jsonl`, `postgres`). |

The full list of metric families is in the README ("Metrics and monitoring").

### Current limitations

| Area | Current state |
|---|---|
| Drift status | The response carries the latest drift-job status of the request's segment; there is no per-request drift evaluation. |
| Prediction persistence | JSONL always and `ml.online_predictions` in database mode, written after the response; failures are counted, never returned to the client. |
| Rollback | `/admin/rollback` supports dry-run and applied rollback through the shared pointer; applied end to end on 2026-11-16 (v005 → v004 in compose and on both kind replicas). v007 is active; its rollback target is v006 (dry run 2026-11-26). |
| Threshold tuning | Uses fallback `0.0`; no version has an explicit threshold artifact yet. |
| Authentication | One shared admin token for `/admin/*`; no users or roles. Prediction endpoints are open on the local network. |

### Validation

Run the API and inference tests from the repository root with the dev image (the
`dev()` function in the README's "Local runbook"):

    dev env -u PYTHONPATH python -m pytest -p no:cacheprovider -o addopts= -q tests/test_api.py tests/test_batch_inference.py tests/test_payload_hash.py

Manual API validation against the compose API:

    curl -s http://127.0.0.1:8014/health
    curl -s http://127.0.0.1:8014/model/active
    curl -s http://127.0.0.1:8014/metrics | head -50

## Prediction evidence logging

Prediction logging is one shared contract used by both inference paths (JSONL always, plus the database in database mode).

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

Database:

- `sql/create_prediction_tables.sql` defines `ml.batch_predictions` and `ml.online_predictions`.
- In database mode both paths also insert there (batch in one transaction per run, online after the response).
- The JSONL record shape matches the tables, so both sinks carry the same contract.

Operational reason:

Prediction evidence makes anomaly scoring auditable. A prediction can be traced back to the model version, scored entity, timestamp, anomaly score, threshold, and payload hash. This is required before drift checks, alert events, current metrics, and rollback decisions can be trusted.
