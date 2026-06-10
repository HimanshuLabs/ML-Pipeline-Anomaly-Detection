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
| Training snapshot generation | Implemented locally | Parquet snapshot and JSON metadata are written under data/features/training/. |
| Database metadata write | Planned | Metadata maps to ml.feature_snapshots and ml.training_datasets, but DB writes are not wired yet. |
| Isolation Forest training | Implemented locally | Training consumes frozen snapshot files and writes versioned local artifacts under artifacts/models/. |
| Model registry | Implemented locally | Local JSON registry links model versions to artifact paths, snapshot_id, training_dataset_id, status, and approval state. |
| Baseline metrics | Implemented locally | Training writes baseline_stats.json with feature baselines, anomaly rate, score summary, latency placeholders, and label-availability notes. |
| Promotion and rollback | Partially implemented | Production promotion and active_model.yaml pointer are implemented locally. Rollback controls are planned for the rollback checkpoint. |

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

## Local metadata fallback

At this checkpoint, snapshot metadata is written locally to JSON.

This is intentional.

The SQL metadata layer already defines the target tables:

- ml.feature_snapshots
- ml.training_datasets

The snapshot module exposes database-shaped metadata records through:

    metadata_to_database_records()

Direct database writes will be added only after the database interface is confirmed and needed by the model registry flow.

This avoids coupling snapshot generation to a database connection too early.

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

## Local model registry implementation

Checkpoint 7 implements a local JSON model registry.

Registry path:

    artifacts/models/_registry/model_registry.json

The registry is generated runtime metadata and is intentionally ignored by Git.

Each local registry entry records:

| Field group | Purpose |
|---|---|
| Model identity | model_id, model_name, model_version, algorithm |
| Artifact location | artifact_path, artifact_dir, metadata_path, feature_schema_path, baseline_stats_path |
| Dataset lineage | snapshot_id, training_dataset_id |
| Schema lineage | feature_schema_version |
| Training metadata | training_started_at_utc, training_finished_at_utc |
| Baseline metrics | baseline_anomaly_rate, latency placeholders, future precision/recall proxy fields |
| Lifecycle state | candidate, staging, production, archived, rolled_back, failed_validation |
| Approval state | approved_for_prod, promoted_at_utc, archived_at_utc |

This local registry mirrors the PostgreSQL target tables without forcing a database dependency into the first model lifecycle checkpoint.

---

## Active model pointer

The active production model is tracked in:

    configs/active_model.yaml

Current local pointer shape:

    model_name: isolation_forest
    active_model_version: v001
    artifact_path: artifacts/models/isolation_forest/model_version=v001/model.joblib
    feature_schema_version: feature_schema_v001
    status: production

The artifact path is stored repo-relative. Absolute local paths are avoided because they make the repository machine-specific.

The future online inference service will read this pointer to load the active production model.

---

## Current local model state

The first promoted model is:

| Field | Value |
|---|---|
| model_name | isolation_forest |
| model_version | v001 |
| status | production |
| approved_for_prod | true |
| training mode | validated local demo snapshot |
| baseline anomaly rate | 0.05 |
| label availability | unlabeled_proxy_metrics |

Important limitation:

The current v001 model is trained from a validated demo snapshot generated inside Project 4. It proves the lifecycle mechanics, artifact layout, baseline metrics, registry, and promotion flow. It is not yet trained from a live extract of Project 1 and Project 2/3 sources.

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
        -> local model registry record links model_version to snapshot_id

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
| Database unavailable | Snapshot process blocked if DB is hard dependency. | Use local JSON fallback. |
| Mutable training source | Model cannot be reproduced later. | Train only from snapshot path. |

---


---

## Checkpoint 7 lifecycle API update

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
## Real-source model lifecycle


### Active production model

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

### Version history

| Version | Status | Source | Rows | Features | Notes |
|---|---|---|---:|---:|---|
| `v001` | `archived` | Project 4 generated demo snapshot | `240` | `11` | First local Isolation Forest baseline used to prove training, baseline metrics, and registry mechanics. |
| `v002` | `production` | Real Project 1 and Project 2/3 local PostgreSQL extracts | `16,750` | `51` | First production model trained from actual upstream project outputs. |

### Metric interpretation

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

