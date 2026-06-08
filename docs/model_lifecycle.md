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
| Isolation Forest training | Planned | Training will consume frozen snapshot files. |
| Model registry | Planned | Registry will link model versions to snapshot_id and training_dataset_id. |
| Baseline metrics | Planned | Baseline feature stats and anomaly metrics will be captured after training. |
| Promotion and rollback | Planned | Active model pointer and rollback controls are added later. |

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

Future model registry records must include:

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

## Planned training flow

The intended training flow is:

    validated feature dataframe
        -> training snapshot generation
        -> snapshot metadata persisted locally and later to SQL
        -> training reads features.parquet
        -> model trains on ordered feature columns
        -> baseline metrics are calculated
        -> artifact is saved under artifacts/models/
        -> model registry record links model_version to snapshot_id

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

