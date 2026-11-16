# ML Pipeline & Anomaly Detection Platform

A local-first anomaly detection platform that turns upstream behavioral and warehouse data into validated ML features, trains versioned Isolation Forest models (one per entity type for segmented models), serves anomaly scores through batch and online inference paths, monitors drift and runtime behavior, emits alert evidence, and supports rollback to a previous stable model.

This project treats the model as a production artifact, not a notebook output.

The core system proves:

- source extraction from upstream data platforms
- feature validation before training and inference
- reproducible training snapshots
- model versioning in a Postgres model registry, with one shared active-model pointer
- segmented models: one Isolation Forest per entity type
- baseline metrics and feature statistics
- batch anomaly scoring
- online FastAPI inference
- prediction evidence logging
- drift monitoring
- Prometheus metrics
- Grafana dashboards
- alert events
- token-protected rollback controls
- latency-budget validation
- Docker images for the API, tools and tests
- MLflow experiment tracking (a record of training, not the source of truth)
- hardened Kubernetes manifests, validated live on a local kind cluster
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
Isolation Forest training (one model per entity type)
        ↓
versioned local artifacts
        ↓
model registry (Postgres, with a JSON file mirror)
        ↓
active model pointer (database; every API replica follows it)
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
| Model registry | Implemented | `ml.model_registry` and `ml.active_model_pointer` in Postgres (JSON file mirror in the repository): versions, artifact paths, lineage, status, and the active version. |
| Baseline metrics | Implemented | Stores anomaly rate and per-feature baseline statistics. |
| Batch inference | Implemented | Scores model-ready feature batches and writes prediction evidence. |
| Online inference | Implemented | FastAPI service scores single and batch payloads using the active model. |
| Prediction logging | Implemented | JSONL evidence, plus `ml.online_predictions` / `ml.batch_predictions` when the database is configured. |
| Drift checks | Implemented | Compares current feature mean/variance against the active model's baseline, per segment; hourly drift job. |
| Prometheus metrics | Implemented | Exposes request, error, anomaly, latency, drift, active-model, and rollback metrics. |
| Grafana dashboards | Implemented locally | Operator and executive dashboards are stored under `monitoring/grafana/dashboards/`. |
| Alert events | Implemented | Writes alert evidence (JSONL, plus `monitoring.alert_events`) for drift, anomaly-rate, latency, and prediction-error conditions. |
| Rollback controls | Implemented | `POST /admin/rollback` (X-Admin-Token) with dry run; an applied rollback moves the shared pointer and every API replica follows. |
| Latency validation | Implemented | Includes benchmark script and p95 latency target. |
| Docker API runtime | Implemented | Builds and runs the FastAPI anomaly inference service. |
| Docker batch runtime | Implemented | Builds and runs the batch anomaly scoring runtime. |
| CI/CD validation | Implemented | GitHub Actions validates tests, SQL contracts, Docker builds, and runtime metadata. |

## Model lifecycle

Registered versions on 2026-11-16:

| Version | Status | Kind | Feature schema | Why |
|---|---|---|---|---|
| `v001` | retired | single | `feature_schema_v001` | First model, trained on a Project 4 demo snapshot to prove training, artifacts, registry, and promotion. |
| `v002` | archived | single | `feature_schema_v001` | First real-source model (Project 1 + Project 2/3 extracts, 51 features); production until v004. |
| `v003` | retired | single | `feature_schema_v002` | Superseded: the single model learned the entity type rather than anomalies (see its `evaluation_report.json`). |
| `v004` | production | segmented | `feature_schema_v003` | One Isolation Forest per entity type (6 segments, 52 features in total). |
| `v005` | rolled_back | segmented | `feature_schema_v003` | Retrained from a fresh read-only extract (same data as v004); `training_dataset_id` `64e7a3c0-bd07-5f74-95f5-71cc9e494e1b`, MLflow run `93a91a6a69834659befd027b2496b953`. Promoted, then rolled back to v004: the first real rollback (compose and kind). |

Current active model:

| Field | Value |
|---|---|
| Model name | `isolation_forest` |
| Active version | `v004` |
| Model kind | segmented: `campaign`, `customer`, `funnel_day`, `product`, `user`, `web_customer` |
| Snapshot | `real_source_20261115T010701Z` (`real_source_extract`) |
| Feature schema version | `feature_schema_v003` |
| Feature count | `52` (each segment scores its own subset) |
| Baseline anomaly rate | `0.0507` |
| Label availability | `unlabeled_proxy_metrics` |
| Artifact path | `artifacts/models/isolation_forest/model_version=v004/model.joblib` |

The table is a dated snapshot. The live state comes from the API and the registry:

```bash
curl -s http://127.0.0.1:8014/model/active
docker compose exec -T anomaly-postgres psql -U manshu -d project4_ml -c "
  SELECT r.model_version, r.status, r.approved_for_prod, r.feature_schema_version,
         p.pointer_version
  FROM ml.model_registry AS r
  LEFT JOIN ml.active_model_pointer AS p
    ON p.model_name = r.model_name AND p.model_version = r.model_version
  ORDER BY r.model_version"
```

Where the registry lives: with `PROJECT4_POSTGRES_*` configured, `ml.model_registry` and
`ml.active_model_pointer` are the source of truth, and `artifacts/models/_registry/model_registry.json`
plus `configs/active_model.yaml` are a file mirror (written unless
`PROJECT4_REGISTRY_FILE_MIRROR=false`, as in Kubernetes). Without a database, the files are
the registry.

Lifecycle: training and retraining register a version as `candidate`; `promote` makes it
`production` after a serving check and archives the previous production version; `retire`
takes a non-active version out of use (its files are kept); a rollback marks the version
it leaves `rolled_back`. `promote` accepts a `rolled_back` version as any other, but
refuses a `retired` one before the serving check unless `--allow-retired` is given, which
needs `--notes` saying why.

```bash
# tools() runs the tools image (see "Local runbook"); retrain never promotes
tools python -m anomaly_detection.retrain --snapshot latest --model-version v005 --compare-with v004
tools python -m anomaly_detection.registry promote --model-version v005 --notes "why"
tools python -m anomaly_detection.registry promote --model-version v003 --allow-retired --notes "why"
tools python -m anomaly_detection.registry retire --model-version v003 --notes "why"
tools python -m anomaly_detection.registry register --metadata-path artifacts/models/isolation_forest/model_version=v005/metadata.json
tools python -m anomaly_detection.registry mirror
```

- `retrain --compare-with <version>` scores the new snapshot with that version and the new
  one and adds a `comparison` section to `evaluation_report.json`: per entity type the
  flag rate of each, the Jaccard overlap of the flagged entities, and the `content_sha256`
  of both training snapshots (the previous one is `null` when its bundle does not record
  it and its snapshot directory is not under the training root). It is evidence, not a
  gate; the only hard check is that the version has the same segment spec (what makes
  the two rollback targets of each other), refused before anything is trained.
- `register --metadata-path` adds an already trained bundle as `candidate` after the
  same serving check as `promote`: its snapshot row, its `ml.training_datasets` row (when
  the bundle records a `training_dataset_id`) and its registry row, in that order, in one
  transaction. It works as `project4_app`, never promotes, changes nothing when the same
  bundle is registered again, and refuses a different bundle under a registered version.
  It is how a database that already has a pointer (a seeded cluster) learns a new version.
- `mirror` rewrites `artifacts/models/_registry/model_registry.json` and
  `configs/active_model.yaml` from the database (it only reads the database), exactly as a
  promote or rollback that left the database in that state writes them. Run it after a
  change made where the repository is not mounted, such as an `/admin/rollback` handled by
  the compose API (its mirror is written inside its own container).

Registration (`register_model_from_artifacts`, used by training and retraining) takes a
relative `metadata.json` path as relative to the repository root, never to the current
directory.

Model artifacts are stored under:

```text
artifacts/models/isolation_forest/model_version=<version>/
```

Each version stores:

```text
model.joblib
metadata.json
feature_schema.json
baseline_stats.json
evaluation_report.json   (versions trained by retrain)
```

New versions store their bundle paths in `metadata.json` relative to the repository root;
v001–v004 keep the absolute paths they were written with, and registration finds their
files next to `metadata.json` wherever the bundle is.

## Experiment tracking (MLflow)

**MLflow is not the source of truth.** The Postgres model registry and the active-model
pointer decide promotion, rollback and serving. MLflow keeps a record of how each model
version was trained, and its model registry is not used.

What is recorded per run (experiment `project4-isolation-forest`, run name
`<model_version>-<entrypoint>`):

- params: model parameters, `segments`, `snapshot_id`, `feature_set_fingerprint`,
  `feature_schema_version`, runtime library versions
- metrics: `train.*`, `segment.<entity_type>.*` (rows, anomaly rate, score p5/p50/p95),
  and for retrain `eval.overall.*` / `eval.<entity_type>.*` from `evaluation_report.json`
- artifacts under `bundle/`: `metadata.json`, `feature_schema.json`, `baseline_stats.json`,
  `evaluation_report.json`, `model.joblib`
- tags: `p4.model_name`, `p4.model_version`, `p4.snapshot_id`, `p4.feature_set_fingerprint`,
  `p4.entrypoint`, `p4.registry_status_at_training=candidate`, `p4.model_sha256`

New model versions record `mlflow_run_id` and `mlflow_experiment_id` in `metadata.json`
(null when tracking was off or unavailable). The registry, database tables and
integration views are unchanged.

### Tracking server (Postgres backend)

The compose service `mlflow` runs the MLflow server from the tools image. Runs are stored
in the dedicated database `project4_mlflow`, owned by the dedicated role
`project4_mlflow` (no superuser, no database creation, no access for `PUBLIC`), inside
the `anomaly-postgres` instance. It never uses `project4_ml`. Artifacts are served by the
server from the `mlflow_artifacts` volume, mounted at `/var/lib/mlflow` (artifacts in
`/var/lib/mlflow/mlflow-artifacts`, owned by the image's non-root `app` user).

### One-time setup

1. Generate the role's password into `.env`. Nobody types or sees it, and nothing
   happens when `.env` already has a value:

   ```bash
   grep -q '^PROJECT4_MLFLOW_DB_PASSWORD=.' .env || {
     sed -i '/^PROJECT4_MLFLOW_DB_PASSWORD=/d' .env
     printf 'PROJECT4_MLFLOW_DB_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> .env
   }
   ```

2. Create the role and the database (re-running is safe and re-applies the password from
   `.env`):

   ```bash
   docker compose up -d anomaly-postgres
   PROJECT4_MLFLOW_DB_PASSWORD="$(grep '^PROJECT4_MLFLOW_DB_PASSWORD=' .env | cut -d= -f2-)" \
     docker compose exec -T -e PROJECT4_MLFLOW_DB_PASSWORD anomaly-postgres \
     psql -v ON_ERROR_STOP=1 -U manshu -d postgres < sql/create_mlflow_database.sql
   ```

   The script refuses when `PROJECT4_MLFLOW_DB_PASSWORD` is empty.

### Start the tracking server

```bash
docker compose up -d mlflow
curl -s http://127.0.0.1:5004/health      # OK
```

The UI is at <http://127.0.0.1:5004> (bound to 127.0.0.1 only). Without
`PROJECT4_MLFLOW_DB_PASSWORD` in `.env` the `mlflow` container refuses to start; other
services are unaffected.

### Tracking settings

| Variable | Meaning |
|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5004` from the host (`--network host` containers too). Unset or empty: tracking is off and mlflow is never imported. |
| `PROJECT4_MLFLOW_REQUIRED` | `false` (default): an unreachable server logs a warning and training goes on with `mlflow_run_id` null. `true`: training and retrain refuse before writing any file. If logging fails after the bundle is written (strict mode), the bundle folder stays on disk unregistered and that version number is used; delete it by hand or train the next version. |
| `MLFLOW_DISABLE_TELEMETRY`, `GIT_PYTHON_REFRESH` | Set to `true` / `quiet` by the tracking code itself before mlflow is imported. |

### Backfill existing versions

Records v002 and v004 from their existing bundle files as runs tagged
`p4.backfilled=true`. The bundles are only read, and re-running skips versions that
already have a backfilled run:

```bash
docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5004 -v "$PWD":/app:ro -w /app \
  --entrypoint "" project4-anomaly-tools:local \
  python -m anomaly_detection.tracking backfill --model-version v002 --model-version v004
```

## Database access

Who connects to the `anomaly-postgres` instance (127.0.0.1:5435), and as what:

| Who | Role | Database | Rights |
|---|---|---|---|
| Project 4 API, CLIs and drift job (compose) | `manshu` (the instance owner, a superuser) | `project4_ml` | everything |
| Project 4 API, registry CLI and drift job (Kubernetes) | `project4_app`, created by `apply-schema` | `project4_ml` | CONNECT; USAGE on `ml`, `monitoring`, `audit`; exactly the table privileges they use (`database.APP_TABLE_PRIVILEGES`); no DDL; LOGIN only when `PROJECT4_APP_PASSWORD` is set |
| Project 5 today | `manshu`, the same owner role | `project4_ml` | everything (to be replaced by `project5_reader`) |
| Project 5 (intended) | `project5_reader`, created by `apply-schema` | `project4_ml` | CONNECT plus SELECT on the `integration.*` views only; LOGIN only when `PROJECT4_P5_READER_PASSWORD` is set |
| MLflow server | `project4_mlflow` | `project4_mlflow` only | owner of its own database; cannot connect to `project4_ml` once PUBLIC's CONNECT is revoked |

`sql/restrict_project4_ml_connect.sql` revokes CONNECT and TEMPORARY on `project4_ml`
from PUBLIC and grants CONNECT to `project5_reader`, so no other role can connect.
Superusers are not affected. It is re-runnable, refuses (changing nothing) when
`project5_reader` does not exist yet, and ends by listing every login role with whether
it can connect. `apply-schema` also grants `project5_reader` CONNECT, so a fresh
database keeps working. Run it once:

```bash
docker compose exec -T anomaly-postgres psql -v ON_ERROR_STOP=1 -U manshu -d project4_ml < sql/restrict_project4_ml_connect.sql
```

Undo it:

```bash
docker compose exec -T anomaly-postgres psql -v ON_ERROR_STOP=1 -U manshu -d project4_ml -c 'GRANT CONNECT, TEMPORARY ON DATABASE project4_ml TO PUBLIC'
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

The active feature schema (from `configs/active_model.yaml`) is:

```text
feature_schema_v003
```

`feature_schema_v001` is the contract of v001/v002 and `feature_schema_v002` that of v003.

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

Each real-source snapshot written since stage 15a also records:

- `content_sha256`: sha256 of the data. Rows are sorted by every column in a fixed order
  (entity type and id first) before both the hash and `features.parquet`, so the same
  source data gives the same hash and the same file whatever order the sources return.
  `extracted_at` (when the run happened) is kept in the file but not hashed.
- `training_dataset_id`: `uuid5(NAMESPACE_URL, "project4.training_dataset|<snapshot_id>|<content_sha256>|<feature_set_fingerprint>")`,
  only for a snapshot that passed validation. Registering a model trained on it inserts
  one `ml.training_datasets` row (`ON CONFLICT DO NOTHING`, so re-running adds nothing),
  and the id appears in `integration.model_metadata`, `/model/active` and predictions.
  v002–v004 were trained before this and keep `training_dataset_id` null; their rows
  are never rewritten.

Extraction reads Project 1 (127.0.0.1:5433) and Project 2/3 (127.0.0.1:5434) read-only.
Every source session is opened with
`-c default_transaction_read_only=on -c statement_timeout=120000` (Project 4's own
database engine is not), and before each table is read the extract checks
`SHOW transaction_read_only` is `on` and refuses otherwise. It issues only `SELECT *` per
configured table (plus `warehouse.dim_customer` for web events). It runs as a
`--network host` container against the sources' loopback ports, so they never need to
listen beyond 127.0.0.1. The two passwords are read singly from `.env` and passed by
name; never `--env-file .env`, which would hand the extract every Project 4 secret too:

```bash
PROJECT1_POSTGRES_PASSWORD="$(grep '^PROJECT1_POSTGRES_PASSWORD=' .env | cut -d= -f2-)" \
PROJECT23_POSTGRES_PASSWORD="$(grep '^PROJECT23_POSTGRES_PASSWORD=' .env | cut -d= -f2-)" \
docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PROJECT1_POSTGRES_PASSWORD -e PROJECT23_POSTGRES_PASSWORD \
  -v "$PWD":/app -w /app --entrypoint "" project4-anomaly-tools:local \
  python -m anomaly_detection.source_extract --config configs/source_extract.yaml
```

The assignments in front of `docker run` set the variables for that one command only;
`-e NAME` without a value passes each one on from it, so neither password appears on a
command line or on screen.

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

With `PROJECT4_POSTGRES_*` configured, every batch run is also inserted into
`ml.batch_predictions` (table contract: `sql/create_prediction_tables.sql`), which the
drift job reads.

## Online FastAPI inference

Online inference serves anomaly scores through FastAPI.

Implemented endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /livez` | Liveness: the process is up (never touches the model). |
| `GET /health` | API health, model-load status, and `pointer_source` (`postgres` or `file`). |
| `GET /model/active` | Returns active model metadata. |
| `POST /predict` | Scores one feature payload; a segmented model needs its `entity_type`. |
| `POST /predict/batch` | Scores multiple payloads of one entity type. |
| `GET /metrics` | Exposes Prometheus metrics. |
| `POST /admin/rollback` | Dry-run and applied rollback; needs `X-Admin-Token`. |

Online prediction evidence is written to `logs/predictions/online_predictions.jsonl`
and, with the database configured, to `ml.online_predictions`.

The API serves the version named by the shared pointer: `ml.active_model_pointer` when
`PROJECT4_POSTGRES_*` is configured (`pointer_source: postgres`), else
`configs/active_model.yaml` (`pointer_source: file`). It polls the database pointer every
`PROJECT4_POINTER_POLL_SECONDS` (default 30) and swaps the model in-process, so a promote
or rollback through any replica reaches every replica without a restart.

### Admin authentication

`/admin/*` requires the header `X-Admin-Token`, compared in constant time with the
server's `ADMIN_TOKEN` (at least 16 characters). Without a configured token every admin
call returns **503** (admin disabled); a missing or wrong header returns **401**. There
are no users or roles.

Compose passes `PROJECT4_ADMIN_TOKEN` from `.env` to the API as `ADMIN_TOKEN`. Generate it
once; nobody types or sees it, and nothing happens when `.env` already has a value:

```bash
grep -q '^PROJECT4_ADMIN_TOKEN=.' .env || {
  sed -i '/^PROJECT4_ADMIN_TOKEN=/d' .env
  printf 'PROJECT4_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" >> .env
}
docker compose up -d anomaly-api
```

Call an admin endpoint by reading the token into a variable (never onto the screen) and
unsetting it afterwards. `printf` is a shell builtin and curl reads the header from stdin
(`-H @-`), so the token never appears in any process's arguments (`ps`):

```bash
ADMIN_TOKEN="$(grep '^PROJECT4_ADMIN_TOKEN=' .env | cut -d= -f2-)"
printf 'X-Admin-Token: %s\n' "$ADMIN_TOKEN" | curl -s -X POST http://127.0.0.1:8014/admin/rollback \
  -H @- -H 'Content-Type: application/json' \
  -d '{"dry_run": true, "rollback_reason": "manual check", "triggered_by": "operator"}'
unset ADMIN_TOKEN
```

In Kubernetes the token lives in the secret `project4-admin`; see
[the Kubernetes runtime validation](docs/kubernetes_runtime_validation.md), step 7.

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

`feature_payload_hash` is the sha256 of exactly the features the model scored for the
record (a segmented model's segment features for its entity type), as floats, in
sorted-key canonical JSON; extra request keys never enter it. Since stage 14c-5 the same
row hashes identically online and in batch. Earlier online rows hashed the whole request
payload as sent, so hashes do not match across that cutover.

## Drift monitoring

Drift monitoring compares current feature behavior against the active model's training
baseline; a segmented model is evaluated per segment against that segment's baseline.

The drift job evaluates the last hours of batch predictions (or a feature file):

```bash
tools python -m anomaly_detection.drift_job --source batch-predictions --hours 24
```

With no batch predictions in the window it reports `not_evaluated` and exits 0. It logs a
WARNING when the model was batch-scored before but not in the window (batch scoring may
have stalled), and INFO when it was never batch-scored or the last run used another
version (e.g. right after a promote).

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

Drift evidence is written to `logs/alerts/drift_events.jsonl` and, with the database
configured, to `monitoring.drift_events` (`sql/create_monitoring_tables.sql`).

Drift checks are intentionally lightweight. This is not presented as a full statistical drift platform. It is an operational signal that shows whether current scoring data has moved away from the approved training baseline.

## Alert events

Alert events are operational evidence (JSONL, plus `monitoring.alert_events` with the database configured).

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

`POST /admin/rollback` (see "Admin authentication") moves the active model back to the
previous stable version. The target must:

- have status `archived` and be `approved_for_prod`
- be loadable and serve the same feature contract as the active model: the same ordered
  features for single models, the same segment spec for segmented models; a single and a
  segmented model are never compatible

Among eligible versions the most recently archived one is chosen. The first real
rollbacks, v005 → v004, ran on 2026-11-16 in compose and on both replicas of the kind
cluster (see [the rollback runbook](docs/rollback_runbook.md)). **v004 is active and has no
compatible target today**: v005 is `rolled_back` (not a target), v001 and v003 are
retired, and v002 (single) is never compatible with the segmented v004, so a dry run
returns HTTP 400 `No compatible rollback target for isolation_forest: …` until a version
with v004's segment spec is promoted.

Rollback supports:

- dry-run validation (always first; it changes nothing)
- applied rollback: registry and pointer change together (one transaction in database
  mode), the API swaps the model in-process, and every replica follows the shared pointer
- rollback event logging and the `model_rollback_total` metric

Rollback evidence is written to `logs/alerts/rollback_events.jsonl` and, in the same
transaction as the pointer change, to `audit.rollback_events`.

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
predictions_scored_total
anomalies_detected_total
prediction_latency_ms
prediction_persist_failures_total
drift_detected_total
feature_drift_status
feature_mean_delta
feature_variance_delta
active_model_version
model_reload_total
model_rollback_total
active_pointer_poll_failures_total
registry_mirror_write_failures_total
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

Live measurement of the active v004 (2026-11-15): 1,644 `/predict` requests built from
real feature rows, sent through the compose API, read from the Prometheus histogram
`prediction_latency_ms` (5 ms buckets):

```text
model_version=v004
requests=1644
errors=0
p50_latency_ms≈47.6
p95_latency_ms≈54.4
pass_latency_budget=true
```

About 29% of the requests fell in the 15–20 ms bucket, so the latency has two groups;
the percentiles above are bucket estimates, not exact values.

What the histogram measures: inside the `/predict` handler, from just before the model
call to just after the request counters, i.e. model scoring, the cached drift-status
lookup, and the metric updates. It excludes request parsing, response serialization,
and the JSONL and database evidence writes, which run as a background task after the
response is sent.

Earlier v002 measurement (single model, `scripts/benchmark_latency.py`, kept for
history):

```text
model_version=v002
p50_latency_ms=22.823
p95_latency_ms=23.530
max_latency_ms=24.274
mean_latency_ms=22.800
api_p95_latency_ms=19.994
pass_latency_budget=true
```

The two measurements use different methods: the v002 benchmark timed requests client-side
(`scripts/benchmark_latency.py`), the v004 figures come from the server-side histogram,
so they are not directly comparable.

These measure local single-client scoring latency. They do not claim cloud latency,
production network latency, or external feature-store latency.

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

## Evaluation limits

The data is unlabeled, so no precision, recall, or F1 can be computed. Isolation Forest
flags the lowest-scoring `contamination` share of the training rows (0.05), so the
training anomaly rate (v004: 0.0507 overall, 0.050–0.054 per entity type) is about 5% by
construction and says nothing about quality. `evaluation_report.json` is computed on the
training snapshot itself, not on a hold-out set. The only signals are proxy and
operational ones: score distributions, the anomaly rate over time against that baseline,
mean/variance drift, and latency. Real quality figures need labelled outcomes for scored
records (human review, incident links, or delayed business labels) collected after
scoring; precision and recall are then measured on a held-out, later time window at the
deployed threshold.

## Dockerized runtimes

Project 4 builds three local images from one `docker/Dockerfile` (base
`python:3.12.16-slim`, pinned by digest), each with its own dependency set:

| Image | Target | Requirements | Purpose |
|---|---|---|---|
| `project4-anomaly-api:local` | `runtime` | `requirements-runtime.txt` | Online FastAPI anomaly inference only. The build refuses mlflow, pyarrow and test tools. |
| `project4-anomaly-tools:local` | `tools` | `requirements-tools.txt` | Batch scoring, source extraction, training, retrain, drift job, MLflow tracking and the MLflow server |
| `project4-anomaly-dev:local` | `dev` | `requirements-dev.txt` | Tests and lint (pytest, ruff), run with the repository mounted at `/app` |

All three install against `constraints.txt` (frozen from the dev image).

The API container runs FastAPI internally on port `8004`.

Docker Compose publishes the API on `127.0.0.1:8014` by default (host-only, like every
Project 4 service) to avoid collision with local `uvicorn` runs on port `8004`.

Build the images:

```bash
docker build -f docker/Dockerfile --target runtime -t project4-anomaly-api:local .
docker build -f docker/Dockerfile --target tools -t project4-anomaly-tools:local .
docker build -f docker/Dockerfile --target dev -t project4-anomaly-dev:local .
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
| Runtime (API), tools, and dev image builds | Implemented |
| Docker Compose API runtime | Implemented |
| Docker Compose batch profile | Implemented |
| Every tracked model bundle packaged into the images; the pointer selects one | Implemented |
| Prometheus metrics from containerized API | Implemented |
| CI builds of all three images and an image content check | Implemented |
| Kubernetes runtime manifests | Implemented; validated live on kind on 2026-11-15 and 2026-11-16 (real rollback) |
| Cloud deployment | Not implemented |

## Kubernetes runtime manifests

Project 4 runs in its own kind cluster `project4-local` (context `kind-project4-local`).
Every command below names that context explicitly; never rely on the current kubectl
context, which may point at another project's cluster.

| File | Purpose |
|---|---|
| `k8s/namespace.yaml` | Creates the `project4-ml` namespace |
| `k8s/configmap.yaml` | `LOG_LEVEL` and `PROJECT4_REGISTRY_FILE_MIRROR=false` (database mode, read-only root filesystem) |
| `k8s/deployment.yaml` | Two hardened API replicas in database mode as `project4_app` (non-root uid 999, read-only root, no capabilities, RuntimeDefault seccomp, no service-account token) |
| `k8s/service.yaml` | Exposes the API internally as a `ClusterIP` service |
| `k8s/hpa.yaml` | CPU-based autoscaling from 2 to 5 replicas |
| `k8s/pdb.yaml` | Keeps at least one API pod during voluntary disruptions |
| `k8s/cronjob-drift.yaml` | Hourly drift job from the tools image, hardened like the API, as `project4_app` |
| `k8s/postgres.yaml` | In-cluster Postgres: StatefulSet and headless Service, `postgres:16.14` (the compose image), 1Gi volume |
| `k8s/networkpolicy.yaml` | 5 NetworkPolicies: default deny, then DNS, API ingress, and Postgres ingress/egress allowed explicitly |
| `k8s/seed-job.yaml` | One-off seed (schema, registry rows, `project4_app` role); applied by the runbook once Postgres is ready, not in the kustomization |
| `k8s/secret.example.yaml` | Example secrets (placeholders only; never applied, not in the kustomization) |
| `k8s/kustomization.yaml` | Renders the resources above except the seed job and the secrets |

The API and the drift job read `PROJECT4_POSTGRES_*` from the secret `project4-app-db`
(the least-privilege role `project4_app`); only the in-cluster Postgres and its seed job
use the owner secret `project4-postgres-owner`.

Validation status:

- Passed: YAML parsing and manifest tests (`tests/test_k8s.py`: hardening, images, secrets, kustomization).
- Passed in CI: kubeconform v0.8.0, strict, against Kubernetes 1.36.1.
- Validated live on 2026-11-15 on the dedicated kind cluster `project4-local`: seed job, both replicas on the database pointer, promote propagation across replicas, network-policy deny checks, drift job, non-root read-only pods, and HPA. See [the live validation](docs/kubernetes_runtime_validation.md#live-validation-2026-11-15).
- Validated live again on 2026-11-16 (stage 15e) with images built from `1f855c5`: the seed reproduced the committed registry state exactly, v005 was promoted and then really rolled back to v004 through `/admin/rollback`, both replicas following each change within one poll, and the integration views matched the compose database. See [the 2026-11-16 record](docs/kubernetes_runtime_validation.md#2026-11-16-live-record-stage-15e).
- Not implemented: cloud Kubernetes deployment, public ingress, managed load balancer, or managed cluster autoscaler.

Render the manifests offline (no cluster is contacted):

    kubectl --context kind-project4-local kustomize k8s/ > /tmp/project4_k8s_rendered.yaml

The tested, step-by-step procedure is
[the Kubernetes runtime validation](docs/kubernetes_runtime_validation.md): images,
cluster creation in its own kubeconfig file (`$KCFG`, so the current context is never
changed), generated secrets, apply, seed, a promote and a real rollback, checks, and
teardown. On a running cluster:

    kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get pods,svc,hpa,pdb,cronjob,networkpolicy
    kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml port-forward svc/project4-anomaly-api 8015:8004

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

- **Install Python dependencies**: `requirements-dev.txt` with `constraints.txt`
- **Run Ruff quality gate**
- **Run secret scan basics**: private keys and common token patterns
- **Apply PostgreSQL schema twice (idempotency) and check**: against a disposable `postgres:16.14` service (the compose image)
- **Run PostgreSQL integration tests**: the database tests, required (never skipped)
- **Run training smoke tests**
- **Run source extraction smoke tests**
- **Run registry and baseline audit**: registry tests, then the active bundle's metadata, baseline, and schema, per segment for segmented models
- **Run drift and rollback tests**: drift, alert, and rollback tests
- **Run FastAPI and metrics tests**: API, metrics, and prediction logging tests
- **Run full pytest suite with JUnit report**: every test, with postgres-marked tests required
- **Publish pytest summary**: totals from the JUnit report in the job summary
- **Upload pytest report artifact**: `pytest-results`
- **Build runtime (API) Docker image**
- **Build tools Docker image**
- **Build dev Docker image**
- **Check image contents**: the runtime image has no mlflow, pyarrow, pytest, ruff, httpx, or `/app/sql` and imports `api.main`; the tools image carries `/app/sql`
- **Download and verify kubeconform v0.8.0**: sha256-checked release archive
- **Show `kubectl` client**: fails clearly when the binary is missing
- **Validate manifests with kubeconform (strict)**: the kustomize output, the seed job, and the secret example against Kubernetes 1.36.1

Each bullet is a step name in `.github/workflows/ci.yml`: the job `Validate ML platform`
runs everything up to "Check image contents", the job `Validate Kubernetes manifests` the
last three. Each job first checks out the repository; the first also sets up Python 3.12.

First CI run:
[run 43817679214](https://github.com/HimanshuLabs/ML-Pipeline-Anomaly-Detection/actions/runs/43817679214),
event `pull_request`, started 2026-11-16T06:34:33Z, on commit `2ce4918` (the branch head
before this README record was added, so it validated everything except this paragraph).

- `Validate ML platform`: success in 4m33s, every step successful, including
  "Run Ruff quality gate" (no output in its log) and "Run full pytest suite with
  JUnit report". The run's JUnit report for commit `2ce4918` recorded 621 tests,
  0 failures, 0 errors, 0 skipped (82.7 s). The per-segment
  baseline audit printed `OK` for all four checks of each of the six segments
  (`campaign`, `customer`, `funnel_day`, `product`, `user`, `web_customer`: feature
  baselines equal feature names, feature names within the model, anomaly rate in [0, 1],
  row count matches the baseline), then `OK: baseline metrics audit passed`. The image
  builds printed
  `OK: active model bundle is present and loads: artifacts/models/isolation_forest/model_version=v004`.
  "Check image contents" printed
  `OK: runtime image has no tools, tests or sql and imports api.main`, then listed nine
  `/app/sql/*.sql` files.
- `Validate Kubernetes manifests`: success in 10s. kubeconform v0.8.0 in strict mode
  against Kubernetes 1.36.1: kustomize output 14/14 valid; seed job and secret example
  4/4 valid.

Each run's job summary lists the pytest totals from the JUnit report, and the
`pytest-results` artifact holds the report itself. Postgres-marked tests are required in
CI, so the summary shows 0 skipped.

## Local runbook

The images run the same dependency sets as CI. Two shell functions wrap them, with the
repository mounted at `/app`; run everything from the repository root:

```bash
# Tests and lint: dev image, no network.
dev() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD":/app -w /app --entrypoint "" project4-anomaly-dev:local "$@"; }
# Training, retraining, registry, batch, and drift: tools image on the host network;
# .env supplies PROJECT4_POSTGRES_* (see .env.example).
tools() { docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp --env-file .env -v "$PWD":/app -w /app --entrypoint "" project4-anomaly-tools:local "$@"; }
```

```bash
dev env -u PYTHONPATH python -m pytest -p no:cacheprovider -o addopts= -q
dev ruff check --no-cache .
tools python -m anomaly_detection.registry promote --model-version v005 --notes "why"
```

Postgres-marked tests skip unless `PROJECT4_TEST_DATABASE_URL` points at a disposable
database; with `PROJECT4_REQUIRE_TEST_DATABASE=1` they fail instead of skipping (as in CI).
`dev()` runs with `--network none`, so it cannot reach any database. For the full suite,
start a throwaway Postgres on 127.0.0.1:5547 and use `dev_db()`, which is `dev()` on the
host network with the test database set:

```bash
docker run --rm -d --name project4_pg_test -p 127.0.0.1:5547:5432 \
  -e POSTGRES_USER=p4test -e POSTGRES_PASSWORD=test_only -e POSTGRES_DB=project4_ml postgres:16.14
timeout 90 sh -c 'until docker exec project4_pg_test pg_isready -h 127.0.0.1 -p 5432 -U p4test -q; do sleep 1; done'
dev_db() { docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PROJECT4_TEST_DATABASE_URL=postgresql+psycopg2://p4test:test_only@127.0.0.1:5547/project4_ml -e PROJECT4_REQUIRE_TEST_DATABASE=1 -v "$PWD":/app -w /app --entrypoint "" project4-anomaly-dev:local "$@"; }
dev_db env -u PYTHONPATH python -m pytest -p no:cacheprovider -o addopts= -q
docker stop project4_pg_test   # --rm removes the container (and its data) on stop
```

The container is disposable: its role `p4test` (the same role CI's Postgres service
creates) and its password exist only for this throwaway database, and nothing else ever
connects to 5547.

Outside Docker, a Python 3.12 virtual environment works too:
`python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt -c constraints.txt`.

Build Docker images:

```bash
docker build -f docker/Dockerfile --target runtime -t project4-anomaly-api:local .
docker build -f docker/Dockerfile --target tools -t project4-anomaly-tools:local .
docker build -f docker/Dockerfile --target dev -t project4-anomaly-dev:local .
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
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── main.py
│   └── schemas.py
├── artifacts/
│   └── models/
├── configs/
│   ├── active_model.yaml
│   ├── drift_thresholds.yaml
│   ├── model_config.yaml
│   └── source_extract.yaml
├── data/
│   └── features/
├── docker/
│   └── Dockerfile
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── drift_monitoring.md
│   ├── false_positive_false_negative_analysis.md
│   ├── feature_engineering.md
│   ├── inference_design.md
│   ├── kubernetes_runtime_validation.md
│   ├── latency_budget.md
│   ├── model_lifecycle.md
│   ├── rollback_runbook.md
│   └── source_integration.md
├── k8s/
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
├── LICENSE
├── pyproject.toml
├── constraints.txt
├── requirements.txt
├── requirements-runtime.txt
├── requirements-tools.txt
├── requirements-dev.txt
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
| `docs/kubernetes_runtime_validation.md` | Kubernetes runbook on a dedicated kind cluster, and the 2026-11-15 live validation |

## Known limitations

Implemented locally, not production-deployed:

- a self-hosted Postgres model registry, not a managed one
- local Prometheus and Grafana configuration
- local Docker runtime and a local kind cluster
- local rollback controls
- administrative endpoints protected by one shared token (no users or roles)
- `registry promote` accepts a registered version in any status, including `retired`;
  only the serving check gates it (hardening: refuse `retired` unless explicitly overridden)
- some tests read the committed registry state (`artifacts/models/_registry/model_registry.json`,
  `configs/active_model.yaml`), so committing new lifecycle state can break them (seen in
  stage 15f: a frozen-clock mirror test archived v004 before its committed `promoted_at`);
  hardening: fixture registries instead of the committed files
- the documented admin `curl` commands pass the token as a `-H` argument, which the same
  user can see in the process list; reading it on stdin
  (`printf 'X-Admin-Token: %s\n' "$ADMIN_TOKEN" | curl -H @- …`) avoids that
- no managed cloud deployment
- no real production traffic
- no verified anomaly labels
- no real external incident-management integration

These are intentional boundaries. The project is built to prove production ML platform mechanics locally before adding cloud cost or managed services.

## Completion definition

Project 4 is complete when:

- full local validation passes
- the runtime, tools and dev images build
- API health and model endpoints respond
- Prometheus metrics are exposed
- Grafana dashboard JSON validates
- the active pointer names a bundle that passes `bundle_check`
- rollback path remains recoverable
- GitHub Actions passes on the remote branch
- final documentation does not exaggerate implementation state
