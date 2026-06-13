# ML Pipeline & Anomaly Detection Platform

A local-first anomaly detection platform that turns upstream behavioral and warehouse data into validated ML features, trains a versioned Isolation Forest model, serves anomaly scores through batch and online inference paths, monitors drift and runtime behavior, emits alert evidence, and supports rollback to a previous stable model.

This project treats the model as a production artifact, not a notebook output.

The core system proves:

- source extraction from upstream data platforms
- feature validation before training and inference
- reproducible training snapshots
- model versioning and active model control
- baseline metrics and feature statistics
- batch anomaly scoring
- online FastAPI inference
- prediction evidence logging
- drift monitoring
- Prometheus metrics
- Grafana dashboards
- alert events
- rollback controls
- latency-budget validation
- Dockerized API and batch runtimes
- CI/CD validation

## System architecture

```text
Project 1 behavioral and runtime signals
+ Project 2/3 warehouse and mart signals
        ↓
source extraction
        ↓
unified anomaly feature dataframe
        ↓
feature validation
        ↓
training snapshot
        ↓
Isolation Forest training
        ↓
versioned local artifacts
        ↓
model registry
        ↓
active production model pointer
        ↓
batch inference + online FastAPI inference
        ↓
prediction evidence logs
        ↓
baseline-vs-current monitoring
        ↓
mean/variance drift checks
        ↓
Prometheus metrics
        ↓
Grafana dashboards
        ↓
alert events
        ↓
rollback to previous stable model
```

## Current implementation status

| Component | Status | Notes |
|---|---|---|
| Repository foundation | Implemented | Source, API, SQL, monitoring, Docker, tests, and docs structure exists. |
| Runtime configuration | Implemented | Config files and environment examples are separated from code. |
| SQL metadata contracts | Implemented | ML, monitoring, and audit schemas are defined under `sql/`. |
| Source extraction | Implemented locally | Extracts Project 1 and Project 2/3 local source data into Project 4 features. |
| Feature validation | Implemented | Validates schema, required columns, numeric types, nulls, ranges, and timestamp behavior. |
| Training snapshots | Implemented | Writes reproducible feature snapshots and metadata under `data/features/training/`. |
| Isolation Forest training | Implemented | Trains anomaly detection models from validated training snapshots. |
| Model registry | Implemented locally | Tracks model versions, artifact paths, lineage, status, and active production version. |
| Baseline metrics | Implemented | Stores anomaly rate and per-feature baseline statistics. |
| Batch inference | Implemented | Scores model-ready feature batches and writes prediction evidence. |
| Online inference | Implemented | FastAPI service scores single and batch payloads using the active model. |
| Prediction logging | Implemented locally | Batch and online paths write JSONL prediction evidence. |
| Drift checks | Implemented | Compares current feature mean/variance against approved training baselines. |
| Prometheus metrics | Implemented | Exposes request, error, anomaly, latency, drift, active-model, and rollback metrics. |
| Grafana dashboards | Implemented locally | Operator and executive dashboards are stored under `monitoring/grafana/dashboards/`. |
| Alert events | Implemented locally | Writes alert evidence for drift, anomaly-rate, latency, and prediction-error conditions. |
| Rollback controls | Implemented locally | Supports dry-run and applied rollback to a previous stable model. |
| Latency validation | Implemented | Includes benchmark script and p95 latency target. |
| Docker API runtime | Implemented | Builds and runs the FastAPI anomaly inference service. |
| Docker batch runtime | Implemented | Builds and runs the batch anomaly scoring runtime. |
| CI/CD validation | Implemented | GitHub Actions validates tests, SQL contracts, Docker builds, and runtime metadata. |

## Model lifecycle

The platform currently has two model versions.

| Version | Status | Source | Purpose |
|---|---|---|---|
| `v001` | Archived and rollback-eligible | Project 4 demo training snapshot | Proved training, artifact layout, registry, baseline metrics, and promotion mechanics. |
| `v002` | Active production model | Real local extracts from Project 1 and Project 2/3 | Active anomaly model used by batch and online inference. |

Current active model:

| Field | Value |
|---|---|
| Model name | `isolation_forest` |
| Active version | `v002` |
| Snapshot type | `real_source_extract` |
| Feature schema version | `feature_schema_v001` |
| Training row count | `16,750` |
| Feature count | `51` |
| Baseline anomaly rate | `0.05002985074626866` |
| Label availability | `unlabeled_proxy_metrics` |
| Artifact path | `artifacts/models/isolation_forest/model_version=v002/model.joblib` |

Model artifacts are stored locally under:

```text
artifacts/models/isolation_forest/model_version=<version>/
```

Each version stores:

```text
model.joblib
metadata.json
feature_schema.json
baseline_stats.json
```

## Feature contract

Project 4 does not train directly on raw upstream tables.

It builds a controlled anomaly feature dataframe first. The validated model feature contract includes:

- behavioral/session features
- commerce/cart features
- risk proxy features
- probability-style behavioral features
- campaign and funnel features
- customer-value features
- latency and system-performance features
- lineage columns for source traceability

The active feature schema is:

```text
feature_schema_v001
```

The validation layer protects the model from:

- missing required columns
- invalid numeric types
- null spikes
- negative values where impossible
- probability values outside `[0, 1]`
- invalid timestamps
- feature-schema mismatch
- malformed source lineage

This matters because bad upstream data can look like a business anomaly. The contract prevents broken data from being silently treated as model signal.

## Training snapshots

Training is snapshot-based for reproducibility.

A training snapshot records:

- snapshot ID
- snapshot type
- source projects
- source tables or paths
- feature schema version
- row count
- feature count
- creation timestamp
- source lineage

Current real-source snapshot type:

```text
real_source_extract
```

Training snapshots are written under:

```text
data/features/training/
```

The model registry links each trained model back to the snapshot used to train it.

## Batch inference

Batch inference scores feature files offline.

Primary use cases:

- scheduled anomaly scoring
- backfills
- offline monitoring windows
- validation against known feature snapshots
- cheaper non-real-time scoring

Batch prediction evidence is written to:

```text
logs/predictions/batch_predictions.jsonl
```

The SQL contract for database persistence exists at:

```text
sql/create_prediction_tables.sql
```

The implemented local path is JSONL-first. Direct PostgreSQL inserts are planned hardening, not currently claimed as implemented.

## Online FastAPI inference

Online inference serves anomaly scores through FastAPI.

Implemented endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns API health and model-load status. |
| `GET /model/active` | Returns active model metadata. |
| `POST /predict` | Scores one model-ready feature payload. |
| `POST /predict/batch` | Scores multiple model-ready payloads. |
| `GET /metrics` | Exposes Prometheus metrics. |
| `POST /admin/rollback` | Supports dry-run and applied rollback requests. |

Online prediction evidence is written to:

```text
logs/predictions/online_predictions.jsonl
```

Online inference uses the active model pointer:

```text
configs/active_model.yaml
```

## Prediction evidence

Each prediction record includes:

- prediction ID
- model name
- model version
- entity type
- entity ID
- prediction timestamp
- anomaly score
- anomaly flag
- threshold used
- feature payload hash
- prediction status
- latency
- source path: batch or online

Prediction evidence exists so anomalies can be audited after scoring. A record can be traced back to the model version, scored entity, score, threshold, and feature payload hash.

## Drift monitoring

Drift monitoring compares current feature behavior against the approved `v002` training baseline.

Implemented method:

```text
mean drift = current feature mean versus baseline feature mean
variance drift = current feature variance versus baseline feature variance
```

Supported drift statuses:

```text
normal
warning
critical
```

Drift evidence is written locally to:

```text
logs/alerts/drift_events.jsonl
```

The PostgreSQL table contract exists in:

```text
sql/create_monitoring_tables.sql
```

Drift checks are intentionally lightweight. This is not presented as a full statistical drift platform. It is an operational signal that shows whether current scoring data has moved away from the approved training baseline.

## Alert events

Alert events are implemented as local operational evidence.

Alert types include:

- critical drift
- anomaly-rate spike
- latency budget breach
- prediction error-rate breach
- manual alert

Alert evidence is written to:

```text
logs/alerts/anomaly_alerts.jsonl
```

The SQL table contract exists as:

```text
monitoring.alert_events
```

Alert events are evidence for operator review. They do not automatically prove the model is wrong.

## Rollback

Rollback is implemented locally.

Validated rollback path:

```text
current production model: v002
previous stable target: v001
```

Rollback supports:

- dry-run validation
- applied rollback
- active model pointer update
- rollback event logging
- rollback metric emission
- audit table contract

Rollback evidence is written to:

```text
logs/alerts/rollback_events.jsonl
```

The rollback audit table contract exists as:

```text
audit.rollback_events
```

Rollback is a recovery control, not blind automation. It should be used after reviewing drift events, alert events, prediction evidence, active model metadata, and registry state.

## Metrics and monitoring

Prometheus metrics are exposed from:

```text
GET /metrics
```

Core metric families:

```text
prediction_requests_total
prediction_errors_total
anomalies_detected_total
prediction_latency_ms
drift_detected_total
feature_mean_delta
feature_variance_delta
active_model_version
model_rollback_total
```

Grafana dashboard files:

```text
monitoring/grafana/dashboards/anomaly_detection_dashboard.json
monitoring/grafana/dashboards/anomaly_detection_executive_dashboard.json
```

Dashboard coverage:

- active model version
- request volume
- anomaly count
- anomaly rate
- prediction errors
- p50 latency
- p95 latency
- drift events
- feature drift deltas
- rollback count

The dashboards are local Grafana dashboard definitions. This repository does not claim managed cloud Grafana deployment.

## Latency budget

Online inference target:

```text
p95 latency < 200 ms
```

Benchmark script:

```text
scripts/benchmark_latency.py
```

Current local benchmark result:

```text
model_version=v002
p50_latency_ms=22.823
p95_latency_ms=23.530
max_latency_ms=24.274
mean_latency_ms=22.800
api_p95_latency_ms=19.994
pass_latency_budget=true
```

This benchmark proves local single-client scoring latency. It does not claim cloud latency, production network latency, external feature-store latency, or database-write latency.

## False-positive and false-negative position

The active model is unsupervised.

The project currently does not have verified ground-truth anomaly labels.

Therefore, this repository does not claim:

- real precision
- real recall
- real F1
- real false-positive rate
- real false-negative rate

Implemented today:

- baseline anomaly rate
- feature baseline statistics
- current anomaly-rate comparison support
- mean/variance drift checks
- prediction evidence
- alert events
- rollback evidence
- latency validation

Real supervised quality metrics require one of the following:

- delayed business labels
- human review labels
- incident-linked labels
- replay labels
- verified backtesting labels

Until then, anomaly-quality discussion remains proxy-based and explicitly labeled as such.

## Dockerized runtimes

Project 4 provides two local Docker application images.

| Image | Purpose |
|---|---|
| `project4-anomaly-api:local` | Online FastAPI anomaly inference runtime |
| `project4-anomaly-batch:local` | Batch/offline anomaly scoring runtime |

The API container runs FastAPI internally on port `8004`.

Docker Compose publishes the API on host port `8014` by default to avoid collision with local `uvicorn` runs on port `8004`.

Build both images:

```bash
docker build -f docker/Dockerfile.api -t project4-anomaly-api:local .
docker build -f docker/Dockerfile.batch -t project4-anomaly-batch:local .
```

Run the API container:

```bash
docker compose up -d --build anomaly-api
```

Validate the API runtime:

```bash
docker compose ps
curl http://localhost:8014/health
curl http://localhost:8014/model/active
curl http://localhost:8014/metrics | head -50
```

Run the batch runtime help contract:

```bash
docker compose --profile batch run --rm anomaly-batch
```

Run batch scoring with a mounted feature file:

```bash
docker compose --profile batch run --rm anomaly-batch \
  python -m anomaly_detection.batch_inference \
  --input-path /app/data/features/batch_scoring/features.jsonl \
  --output-path /app/logs/predictions/batch_predictions.jsonl \
  --entity-type customer \
  --source-project project_4 \
  --source-table local_batch_scoring
```

Stop the Docker runtime:

```bash
docker compose down --remove-orphans
```

Docker scope:

| Capability | Status |
|---|---|
| API image build | Implemented |
| Batch image build | Implemented |
| Docker Compose API runtime | Implemented |
| Docker Compose batch profile | Implemented |
| Active `v002` model packaged into images | Implemented |
| Prometheus metrics from containerized API | Implemented |
| CI validation for API image build | Implemented |
| CI validation for batch image build | Implemented |
| Kubernetes runtime manifests | Implemented locally; live cluster rollout optional / not performed |
| Cloud deployment | Not implemented |

## CI/CD validation

GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

Workflow name:

```text
ML Platform Validation
```

Implemented CI gates:

- Python import validation
- Ruff linting
- Full pytest suite
- JUnit pytest artifact upload
- SQL file smoke validation
- source extraction smoke validation
- baseline metrics audit
- active model metadata validation
- drift tests
- alert tests
- rollback tests
- FastAPI tests
- metrics tests
- Docker API image build
- Docker batch image build
- basic secret-pattern scan

Latest validated CI evidence:

```text
tests=112
passed=112
failures=0
errors=0
skipped=0
```

## Local runbook

Activate the environment:

```bash
cd ~/Desktop/Project-4-ML-Pipeline-Anomaly-Detection
source .venv/bin/activate
```

Run the full test suite:

```bash
PYTHONPATH=src pytest -q
```

Run linting:

```bash
ruff check .
```

Run the API locally:

```bash
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8004
```

Validate local API:

```bash
curl http://localhost:8004/health
curl http://localhost:8004/model/active
curl http://localhost:8004/metrics | head -50
```

Run latency benchmark:

```bash
PYTHONPATH=src python scripts/benchmark_latency.py \
  --url http://localhost:8004/predict \
  --requests 100
```

Build Docker images:

```bash
docker build -f docker/Dockerfile.api -t project4-anomaly-api:local .
docker build -f docker/Dockerfile.batch -t project4-anomaly-batch:local .
```

Run Docker API:

```bash
docker compose up -d --build anomaly-api
curl http://localhost:8014/health
```

Stop Docker runtime:

```bash
docker compose down --remove-orphans
```

## Repository structure

```text
.
├── api/
│   ├── main.py
│   └── schemas.py
├── artifacts/
│   └── models/
├── configs/
│   ├── active_model.yaml
│   ├── database.yaml.example
│   ├── drift_thresholds.yaml
│   ├── model_config.yaml
│   └── source_extract.yaml
├── data/
│   └── features/
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.batch
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── drift_monitoring.md
│   ├── false_positive_false_negative_analysis.md
│   ├── feature_engineering.md
│   ├── inference_design.md
│   ├── latency_budget.md
│   ├── model_lifecycle.md
│   ├── rollback_runbook.md
│   └── source_integration.md
├── logs/
│   ├── alerts/
│   └── predictions/
├── monitoring/
│   ├── grafana/
│   └── prometheus.yml
├── scripts/
│   └── benchmark_latency.py
├── sql/
├── src/
│   └── anomaly_detection/
├── tests/
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Documentation map

| Document | Purpose |
|---|---|
| `docs/architecture.md` | End-to-end Project 4 architecture and source-to-model flow |
| `docs/source_integration.md` | Real-source extraction from upstream local systems |
| `docs/data_contract.md` | Feature schema, required columns, validation rules, rejected fields |
| `docs/feature_engineering.md` | Feature definitions and entity-grain logic |
| `docs/model_lifecycle.md` | Training snapshots, model versions, registry, baseline metrics |
| `docs/inference_design.md` | Batch and online inference contracts |
| `docs/drift_monitoring.md` | Drift checks, Prometheus metrics, Grafana dashboards, alert events |
| `docs/latency_budget.md` | Online inference latency target and benchmark evidence |
| `docs/false_positive_false_negative_analysis.md` | Threshold behavior and label limitations |
| `docs/rollback_runbook.md` | Dry-run and applied rollback procedure |

## Known limitations

Implemented locally, not production-deployed:

- local artifact registry instead of managed model registry
- JSONL persistence before direct database insert wiring
- local Prometheus and Grafana configuration
- local Docker runtime
- local rollback controls
- no authentication on local administrative API endpoints
- no managed cloud deployment
- no real production traffic
- no verified anomaly labels
- no real external incident-management integration

These are intentional boundaries. The project is built to prove production ML platform mechanics locally before adding cloud cost or managed services.

## Completion definition

Project 4 is complete when:

- full local validation passes
- Docker API and batch images build
- API health and model endpoints respond
- Prometheus metrics are exposed
- Grafana dashboard JSON validates
- active model metadata points to `v002`
- rollback path remains recoverable
- GitHub Actions passes on the remote branch
- final documentation does not exaggerate implementation state

## Kubernetes runtime manifests

Project 4 now includes local Kubernetes runtime manifests for the anomaly inference API.

Implemented Kubernetes files:

| File | Purpose |
|---|---|
| `k8s/namespace.yaml` | Creates the `project4-ml` namespace |
| `k8s/configmap.yaml` | Provides runtime configuration for model `v002` |
| `k8s/deployment.yaml` | Runs two replicas of the FastAPI anomaly inference service |
| `k8s/service.yaml` | Exposes the API internally as a `ClusterIP` service |
| `k8s/hpa.yaml` | Defines CPU-based autoscaling from 2 to 5 replicas |
| `k8s/kustomization.yaml` | Renders the Kubernetes resources together for offline validation |

Validation status:

- Implemented: Kubernetes manifests for local runtime deployment design.
- Passed: YAML parsing validation.
- Passed: cross-manifest consistency audit.
- Passed: `kubectl kustomize k8s/` offline render validation.
- Blocked: `kubectl apply --dry-run=client` because no Kubernetes context is configured.
- Not performed: live cluster rollout.
- Not implemented: cloud Kubernetes deployment.

Run offline render validation:

    kubectl kustomize k8s/ > /tmp/project4_k8s_rendered.yaml

Run live deployment only after a local cluster such as kind, minikube, Docker Desktop Kubernetes, or another Kubernetes context is configured:

    kubectl apply -f k8s/namespace.yaml
    kubectl apply -k k8s/
    kubectl -n project4-ml get pods,svc,hpa
