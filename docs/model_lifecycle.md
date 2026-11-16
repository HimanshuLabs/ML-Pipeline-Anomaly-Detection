# Model Lifecycle

## Purpose

Project 4 treats the anomaly model as a production asset, not a notebook artifact.

Every model version must be traceable back to:

- exact training feature snapshot
- feature schema version
- row count used for training
- source timestamp range
- model artifact path
- baseline metrics captured at training time
- production promotion or rollback decision

This makes the system reproducible, auditable, and recoverable.

---

## Current implementation status

| Area | Status | Notes |
|---|---|---|
| Feature validation | Implemented | Feature frames are validated before training, batch scoring, or online inference. |
| Training snapshot generation | Implemented | Parquet snapshot and JSON metadata are written under data/features/training/. |
| Database metadata write | Implemented | In database mode, a registry change also inserts the version's snapshot into ml.feature_snapshots (and ml.training_datasets for demo snapshots), if absent, in the same transaction. |
| Isolation Forest training | Implemented | Training consumes frozen snapshot files and writes versioned artifacts under artifacts/models/; real-source snapshots train one model per entity type (segmented). |
| Model registry | Implemented | ml.model_registry and ml.active_model_pointer in Postgres, with the JSON registry and configs/active_model.yaml as file mirror (the store when no database is configured). |
| Baseline metrics | Implemented | Training writes baseline_stats.json with feature baselines, anomaly rate, score summary, latency placeholders, and label-availability notes; per segment for segmented models. |
| Promotion and rollback | Implemented | `registry promote` / `retire` CLIs, `POST /admin/rollback` (X-Admin-Token) with dry run, compatibility-checked targets, and rollback evidence in JSONL and audit.rollback_events. |

---

## Training snapshot contract

A training snapshot is a frozen, reproducible copy of validated model-ready features.

Snapshot feature path pattern:

    data/features/training/snapshot_date=YYYY-MM-DD/snapshot_id=<snapshot_id>/features.parquet

Snapshot metadata path pattern:

    data/features/training/snapshot_date=YYYY-MM-DD/snapshot_id=<snapshot_id>/metadata.json

The metadata records:

| Field | Purpose |
|---|---|
| snapshot_id | Stable deterministic ID for the feature contents. |
| training_dataset_id | Stable deterministic ID for the dataset derived from the snapshot. |
| snapshot_name | Human-readable snapshot name. |
| dataset_version | Dataset version label used by training and registry flows. |
| feature_schema_version | Feature contract version used to validate the dataframe. |
| row_count | Number of rows snapshotted. |
| feature_count | Number of model feature columns. |
| feature_columns | Ordered model feature list. |
| entity_key | Entity identifier column. |
| timestamp_column | Feature timestamp column. |
| source_min_timestamp | Earliest timestamp in the snapshot. |
| source_max_timestamp | Latest timestamp in the snapshot. |
| snapshot_path | Partition directory containing snapshot files. |
| features_path | Parquet feature file path. |
| metadata_path | JSON metadata file path. |
| data_quality_status | Validation result. |
| content_hash | SHA-256 hash of the normalized feature frame. |
| source_tables | Logical source tables or data assets used to build the snapshot. |
| source_project_versions | Project 1 and Project 2/3 source version notes. |
| created_at_utc | Snapshot creation timestamp. |

---

## Reproducibility rule

The same validated feature content should produce the same:

- snapshot_id
- training_dataset_id
- content_hash

Row ordering must not change snapshot identity.

The snapshot module sorts by:

    entity_id, feature_timestamp

before hashing and writing.

This prevents accidental model drift caused by unstable dataframe ordering.

---

## Validation rule

A feature frame must pass the Project 4 feature contract before it can be snapshotted.

The snapshot generator rejects:

- empty dataframes
- missing required feature columns
- schema version mismatches
- invalid timestamps
- invalid numeric feature values
- null or non-finite values beyond the feature validation contract

This keeps bad training data out of the model lifecycle.

---

## Snapshot metadata in files and in the database

Snapshot metadata is always written to JSON next to the snapshot. Snapshot generation
itself never needs a database.

In database mode, whenever the registry records a version (register, promote, retire,
rollback, `sync-from-files`), the version's snapshot is inserted into
`ml.feature_snapshots` (and, for demo snapshots, `ml.training_datasets`) if it is not there yet, in the same
transaction as the registry change. The rows come from the snapshot metadata described
by the bundle's `metadata.json` (`metadata_to_database_records()` for demo snapshots,
`real_source_snapshot_row()` for real-source extracts).

---

## Database mapping

Local snapshot metadata maps to database records as follows.

### ml.feature_snapshots

| Metadata field | Database purpose |
|---|---|
| snapshot_id | Primary snapshot identity. |
| snapshot_name | Unique snapshot label. |
| source_tables | Source assets used to build features. |
| source_min_timestamp | Lower source timestamp bound. |
| source_max_timestamp | Upper source timestamp bound. |
| feature_schema_version | Feature contract version. |
| row_count | Snapshot row count. |
| snapshot_path | Local or future artifact path. |
| data_quality_status | Validation status. |

### ml.training_datasets

| Metadata field | Database purpose |
|---|---|
| training_dataset_id | Primary training dataset identity. |
| snapshot_id | Link back to ml.feature_snapshots. |
| dataset_version | Human-readable dataset version. |
| feature_schema_version | Feature schema used for training. |
| feature_columns | Ordered model feature list. |
| row_count | Training row count. |

---

## Model version linkage

Model registry records include or are designed to include:

- model_version
- model_name
- algorithm
- artifact_path
- snapshot_id
- training_dataset_id
- feature_schema_version
- baseline_metrics
- training_started_at
- training_finished_at
- status
- approved_for_prod

A valid model registry record must answer this question:

Which exact dataset produced this model?

If that answer is missing, the model is not production-grade.

---

## Model registry

With `PROJECT4_POSTGRES_*` configured, the registry is `ml.model_registry` and the
active pointer is `ml.active_model_pointer` (with a `pointer_version` that increases on
every change). The JSON file

    artifacts/models/_registry/model_registry.json

and `configs/active_model.yaml` are a mirror of the database, written after each change
unless `PROJECT4_REGISTRY_FILE_MIRROR=false` (as in Kubernetes). Without a database they
are the registry. Both files are tracked in Git (`.gitignore` excepts them), so a fresh
database can be seeded from them with `python -m anomaly_detection.database
sync-from-files`.

Each registry entry records:

| Field group | Purpose |
|---|---|
| Model identity | model_id, model_name, model_version, algorithm |
| Artifact location | artifact_path, artifact_dir, metadata_path, feature_schema_path, baseline_stats_path |
| Dataset lineage | snapshot_id, training_dataset_id |
| Schema lineage | feature_schema_version |
| Training metadata | training_started_at_utc, training_finished_at_utc |
| Baseline metrics | baseline_anomaly_rate, latency placeholders, future precision/recall proxy fields |
| Lifecycle state | candidate, staging, production, archived, rolled_back, retired |
| Approval state | approved_for_prod, promoted_at_utc, archived_at_utc |

Lifecycle: training and retraining register a version as `candidate`; `registry promote`
makes it `production` after a serving check and archives the previous production
version; `registry retire` takes a non-active version out of use (files are kept); a
rollback marks the version it leaves `rolled_back`.

New versions (stage 14c-4 onwards) store their bundle paths in `metadata.json` relative
to the repository root. v001–v004 keep the absolute paths they were written with
(`/home/...` for v001/v002, `/app/...` for v003/v004); registration resolves them to the
files next to the `metadata.json` being read, wherever the bundle is.

---

## Active model pointer

The API, batch scoring, and the drift job serve the version named by the shared pointer:
`ml.active_model_pointer` in database mode (`/health` reports `pointer_source:
postgres`), else `configs/active_model.yaml` (`pointer_source: file`). The API polls the
database pointer every `PROJECT4_POINTER_POLL_SECONDS` (default 30) and swaps the model
in-process, so every replica follows a promote or rollback without a restart.

Current pointer (`configs/active_model.yaml`, mirroring the database):

    model_name: isolation_forest
    active_model_version: v004
    artifact_path: artifacts/models/isolation_forest/model_version=v004/model.joblib
    feature_schema_version: feature_schema_v003
    status: production

The artifact path is stored repo-relative.

---

## Current model state (2026-11-16)

| Version | Status | Kind | Feature schema | Notes |
|---|---|---|---|---|
| `v001` | retired | single | `feature_schema_v001` | Demo snapshot; proved training, artifacts, registry, and promotion. |
| `v002` | archived | single | `feature_schema_v001` | First real-source model (51 features); production until v004. |
| `v003` | retired | single | `feature_schema_v002` | Superseded: the single model learned the entity type rather than anomalies. |
| `v004` | production | segmented | `feature_schema_v003` | One Isolation Forest per entity type (6 segments, 52 features); baseline anomaly rate 0.0507. |
| `v005` | rolled_back | segmented | `feature_schema_v003` | Retrained on 2026-11-16 from a fresh read-only extract (`real_source_20261116T035043Z`, same data as v004); `training_dataset_id` `64e7a3c0-bd07-5f74-95f5-71cc9e494e1b`, MLflow run `93a91a6a69834659befd027b2496b953`. Promoted, then rolled back to v004 (the first real rollback). |

v004 is active and has no compatible rollback target: v005 is `rolled_back` (not a
target), v001 and v003 are retired, and v002 is single while v004 is segmented. Promoting a
version with v004's segment spec makes v004 its target again. The live state is in
`ml.model_registry` and `ml.active_model_pointer`, and `GET /model/active` reports the
served version.

### v005 evidence (stage 15)

- The fresh extract (15b) had the same `content_sha256`
  (`ad2c8596da09eae4b77875b3d96c698de948ac1508a42a482bb9e4f44f02f6f1`, 2504 rows) as
  v004's training data: the sources had not changed.
- `retrain --compare-with v004` (15c, `evaluation_report.json`): the same 6 segments and 52
  features, identical flag rates in every segment, Jaccard overlap of flagged entities
  0.924 overall (campaign, customer, product and web_customer 1.000; user 0.905;
  funnel_day 0.843). Same data, rows in a different order (v004's snapshot predates the
  sorted extract), so the Isolation Forest subsampler drew different samples.
- Known gap: `registry promote` accepts a version in any status, including `retired`
  (v001, v003). Only the serving check gates it; see the README's known limitations.

---

## Fingerprints and content hashes

- `feature_set_fingerprint` at the top of `metadata.json` uses two algorithms depending on
  the model kind. Single models: sha256 of the ordered feature names joined with
  newlines (`training.feature_set_fingerprint`). Segmented models: sha256 of the segment
  spec as sorted-key JSON (`segmented_model.segment_spec_fingerprint`); each segment's
  own `feature_set_fingerprint` uses the newline form. Nothing compares fingerprints
  across model kinds: rollback compatibility compares the feature spec itself, and
  retraining compares fingerprints computed by the same function. Changing either
  algorithm would change stored metadata of existing bundles, so both stay as they are.
- The demo snapshot `content_hash` (and the snapshot and dataset ids derived from it) is
  the sha256 of `DataFrame.to_csv(...)`, so its float formatting depends on the pandas
  version: the same frame could hash differently after a pandas upgrade. Real-source
  snapshots use timestamp ids (`real_source_<UTC timestamp>`) and are not affected.

---

## Implemented local training flow

The implemented local training flow is:

    validated feature dataframe
        -> training snapshot generation
        -> snapshot metadata persisted locally as JSON
        -> training reads features.parquet
        -> model trains on ordered feature columns
        -> baseline metrics are calculated
        -> artifact is saved under artifacts/models/
        -> registry record links model_version to snapshot_id

Training should not read from a loose CSV, random dataframe, or mutable source table directly.

---

## Failure modes

| Failure mode | Impact | Control |
|---|---|---|
| Feature schema mismatch | Model trains on incompatible columns. | Reject snapshot before writing. |
| Empty feature dataframe | Produces useless or broken model. | Raise snapshot error. |
| Invalid timestamps | Source range cannot be trusted. | Reject snapshot. |
| Non-deterministic row ordering | Same data produces different IDs. | Normalize and sort before hashing. |
| Missing metadata | Model version cannot be traced. | Always write metadata.json. |
| Database unavailable | Registry changes cannot be recorded. | Snapshots never need the database; a registry change refuses and changes nothing; the API keeps its model (or starts from the file pointer). |
| Mutable training source | Model cannot be reproduced later. | Train only from snapshot path. |

---

## History

The sections below record earlier checkpoints as they were written (local JSON
registry, v001 and then v002 in production). They are kept for history; the sections
above describe the current state.

### Checkpoint 7 lifecycle API update

The local registry now exposes an explicit lifecycle API:

- `ModelStatus`
- `ModelRegistryRecord`
- `register_model()`
- `promote_model()`
- `get_active_model()`

This keeps model lifecycle behavior testable from Python instead of relying only on manually edited JSON or YAML files.

Supported model statuses:

| Status | Meaning |
|---|---|
| candidate | Newly trained model awaiting review. |
| staging | Model approved for pre-production validation. |
| production | Active model serving batch or online inference. |
| archived | Previous production model no longer active. |
| rolled_back | Model version removed from active service due to rollback. |
| failed_validation | Model rejected because validation, drift, metric, or artifact checks failed. |

The active production model pointer must include:

- `model_name`
- `active_model_version`
- `status`
- `artifact_path`
- `dataset_snapshot_id`
- `training_dataset_id`
- `feature_schema_version`
- `updated_at`

The key rule is simple: the production pointer must identify not only which artifact is active, but also which exact training snapshot produced it.

This avoids a common MLOps failure mode where the serving layer knows the model file path but loses dataset lineage.

<!-- BEGIN REAL_SOURCE_MODEL_LIFECYCLE -->
### Real-source model lifecycle


#### Active production model

The active production anomaly detection model is now `isolation_forest` version `v002`.

| Field | Value |
|---|---|
| Active version | `v002` |
| Status | `production` |
| Artifact path | `artifacts/models/isolation_forest/model_version=v002/model.joblib` |
| Dataset snapshot ID | `real_source_20260610T060738Z` |
| Snapshot type | `real_source_extract` |
| Feature schema version | `feature_schema_v001` |
| Training rows | `16,750` |
| Feature count | `51` |
| Baseline anomaly rate | `0.05002985074626866` |
| Source projects | `project_1`, `project_2_3` |

#### Version history

| Version | Status | Source | Rows | Features | Notes |
|---|---|---|---:|---:|---|
| `v001` | `archived` | Project 4 generated demo snapshot | `240` | `11` | First local Isolation Forest baseline used to prove training, baseline metrics, and registry mechanics. |
| `v002` | `production` | Real Project 1 and Project 2/3 local PostgreSQL extracts | `16,750` | `51` | First production model trained from actual upstream project outputs. |

#### Metric interpretation

The model is an unsupervised Isolation Forest anomaly detector.

Implemented claims:

- baseline anomaly rate is captured
- feature baseline mean, variance, min, max, and missing counts are captured
- model version lineage is captured
- source snapshot lineage is captured
- rollback path exists through archived `v001`

Non-claims:

- no precision claim
- no recall claim
- no false positive rate claim
- no false negative rate claim

Those require delayed labels, backtesting labels, or a verified simulated-label evaluation set.

<!-- END REAL_SOURCE_MODEL_LIFECYCLE -->

<!-- BASELINE_METRICS_CLOSURE_START -->
### Baseline metrics closure for v002

The active production model is `isolation_forest` version `v002`.

`v002` is the first production model trained from real local source extracts from Project 1 and Project 2/3. Its training metadata records `snapshot_type = real_source_extract`, and `configs/active_model.yaml` points production traffic to `v002`.

Verified baseline state:

| Field | Value |
|---|---|
| Model version | `v002` |
| Active status | `production` |
| Snapshot type | `real_source_extract` |
| Training rows | `16,750` |
| Model feature count | `51` |
| Baseline anomaly rate | `0.05002985074626866` |
| Metric type | `unsupervised_training_baseline` |
| Label availability | `unlabeled_proxy_metrics` |

Every model feature in `metadata.json` has a corresponding baseline entry in `baseline_stats.json`. These feature baselines are the reference point for later current-vs-baseline monitoring and mean/variance drift checks.

This project does not claim real supervised precision, recall, false positive rate, or false negative rate for `v002`, because the current training data does not contain verified ground-truth anomaly labels. Any future precision/recall/FPR/FNR metrics must be explicitly marked as proxy, simulated, backtested, or delayed-truth metrics.

The current latency values in `baseline_stats.json` are placeholders from the offline training baseline and are not evidence of measured FastAPI serving latency. Online latency will be measured separately during latency-budget validation, with the target remaining p95 under 200 ms.
<!-- BASELINE_METRICS_CLOSURE_END -->
