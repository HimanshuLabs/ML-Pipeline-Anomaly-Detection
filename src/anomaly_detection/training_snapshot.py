"""Training snapshot generation for the anomaly detection platform.

This module freezes validated model-ready features into reproducible Parquet
snapshots. Every model version trained later should reference one of these
snapshot IDs, not a mutable dataframe or ad-hoc CSV export.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.feature_validation import (
    FeatureValidationConfig,
    FeatureValidationError,
    FeatureValidationResult,
    load_feature_validation_config,
    validate_feature_dataframe,
)


DEFAULT_TRAINING_SNAPSHOT_ROOT = Path("data/features/training")
FEATURES_FILENAME = "features.parquet"
METADATA_FILENAME = "metadata.json"


class TrainingSnapshotError(ValueError):
    """Raised when a training snapshot cannot be generated safely."""


@dataclass(frozen=True)
class TrainingSnapshotConfig:
    """Runtime settings for writing training feature snapshots."""

    output_root: Path = DEFAULT_TRAINING_SNAPSHOT_ROOT
    features_filename: str = FEATURES_FILENAME
    metadata_filename: str = METADATA_FILENAME
    compression: str = "snappy"


@dataclass(frozen=True)
class TrainingSnapshotMetadata:
    """Metadata record describing a reproducible training feature snapshot."""

    snapshot_id: str
    training_dataset_id: str
    snapshot_name: str
    dataset_version: str
    feature_schema_version: str
    row_count: int
    feature_count: int
    feature_columns: list[str]
    entity_key: str
    timestamp_column: str
    source_min_timestamp: str
    source_max_timestamp: str
    snapshot_date: str
    snapshot_path: str
    features_path: str
    metadata_path: str
    data_quality_status: str
    validation_errors: list[str]
    validation_warnings: list[str]
    content_hash: str
    source_tables: list[str] = field(default_factory=list)
    source_project_versions: dict[str, str] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def create_training_snapshot(
    frame: pd.DataFrame,
    *,
    snapshot_config: TrainingSnapshotConfig | None = None,
    validation_config: FeatureValidationConfig | None = None,
    source_tables: list[str] | None = None,
    source_project_versions: dict[str, str] | None = None,
    created_at_utc: datetime | None = None,
    overwrite: bool = True,
) -> TrainingSnapshotMetadata:
    """Validate and write a reproducible training snapshot.

    Args:
        frame: Model-ready feature dataframe.
        snapshot_config: Output path and file settings.
        validation_config: Feature contract. Loaded from configs/model_config.yaml
            when omitted.
        source_tables: Optional logical source table names used to build the
            feature dataframe.
        source_project_versions: Optional source project version identifiers.
        created_at_utc: Optional fixed timestamp for deterministic testing.
        overwrite: Whether to overwrite an existing snapshot with the same ID.

    Returns:
        TrainingSnapshotMetadata describing the written snapshot.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise TrainingSnapshotError("Cannot create a training snapshot from an empty dataframe.")

    resolved_snapshot_config = snapshot_config or TrainingSnapshotConfig()
    resolved_validation_config = validation_config or load_feature_validation_config()

    validation_result = validate_feature_dataframe(
        frame,
        config=resolved_validation_config,
        strict=False,
    )

    if not validation_result.is_valid:
        raise FeatureValidationError(
            "Feature dataframe failed validation and cannot be snapshotted: "
            + "; ".join(validation_result.errors)
        )

    normalized_frame = _normalize_feature_frame(
        frame=frame,
        validation_config=resolved_validation_config,
    )

    source_min_timestamp, source_max_timestamp = _resolve_source_timestamp_range(
        normalized_frame,
        timestamp_column=resolved_validation_config.timestamp_column,
    )

    snapshot_date = source_max_timestamp.date().isoformat()
    content_hash = _hash_feature_frame(normalized_frame)

    snapshot_id = _build_deterministic_uuid(
        namespace="project4.training_snapshot",
        values=[
            resolved_validation_config.schema_version,
            str(len(normalized_frame)),
            source_min_timestamp.isoformat(),
            source_max_timestamp.isoformat(),
            content_hash,
        ],
    )

    training_dataset_id = _build_deterministic_uuid(
        namespace="project4.training_dataset",
        values=[snapshot_id, resolved_validation_config.schema_version],
    )

    snapshot_name = f"training_snapshot_{snapshot_date}_{snapshot_id}"
    dataset_version = f"dataset_{snapshot_date}_{snapshot_id[:8]}"

    snapshot_path = (
        resolved_snapshot_config.output_root
        / f"snapshot_date={snapshot_date}"
        / f"snapshot_id={snapshot_id}"
    )
    features_path = snapshot_path / resolved_snapshot_config.features_filename
    metadata_path = snapshot_path / resolved_snapshot_config.metadata_filename

    if snapshot_path.exists() and not overwrite:
        raise TrainingSnapshotError(f"Snapshot already exists: {snapshot_path}")

    snapshot_path.mkdir(parents=True, exist_ok=True)

    normalized_frame.to_parquet(
        features_path,
        index=False,
        compression=resolved_snapshot_config.compression,
    )

    metadata = TrainingSnapshotMetadata(
        snapshot_id=snapshot_id,
        training_dataset_id=training_dataset_id,
        snapshot_name=snapshot_name,
        dataset_version=dataset_version,
        feature_schema_version=resolved_validation_config.schema_version,
        row_count=validation_result.row_count,
        feature_count=validation_result.feature_count,
        feature_columns=list(resolved_validation_config.required_numeric_features),
        entity_key=resolved_validation_config.entity_key,
        timestamp_column=resolved_validation_config.timestamp_column,
        source_min_timestamp=source_min_timestamp.isoformat(),
        source_max_timestamp=source_max_timestamp.isoformat(),
        snapshot_date=snapshot_date,
        snapshot_path=snapshot_path.as_posix(),
        features_path=features_path.as_posix(),
        metadata_path=metadata_path.as_posix(),
        data_quality_status="passed",
        validation_errors=list(validation_result.errors),
        validation_warnings=list(validation_result.warnings),
        content_hash=content_hash,
        source_tables=source_tables or [],
        source_project_versions=source_project_versions or {},
        created_at_utc=(created_at_utc or datetime.now(UTC)).isoformat(),
    )

    _write_metadata_json(metadata_path, metadata)

    return metadata


def load_training_snapshot_metadata(metadata_path: str | Path) -> TrainingSnapshotMetadata:
    """Load local JSON metadata for a training snapshot."""

    path = Path(metadata_path)

    if not path.exists():
        raise FileNotFoundError(f"Training snapshot metadata not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_metadata: dict[str, Any] = json.load(file)

    return TrainingSnapshotMetadata(**raw_metadata)


def read_training_snapshot(features_path: str | Path) -> pd.DataFrame:
    """Read a previously written Parquet training snapshot."""

    path = Path(features_path)

    if not path.exists():
        raise FileNotFoundError(f"Training snapshot features not found: {path}")

    return pd.read_parquet(path)


def _normalize_feature_frame(
    *,
    frame: pd.DataFrame,
    validation_config: FeatureValidationConfig,
) -> pd.DataFrame:
    """Return a stable, ordered dataframe for hashing and snapshot writing."""

    required_columns = [
        validation_config.entity_key,
        validation_config.timestamp_column,
        validation_config.schema_version_column,
        *validation_config.required_numeric_features,
    ]

    missing_columns = [column for column in required_columns if column not in frame.columns]

    if missing_columns:
        raise TrainingSnapshotError(
            "Cannot normalize feature frame. Missing columns: "
            + ", ".join(missing_columns)
        )

    normalized = frame.loc[:, required_columns].copy()

    normalized[validation_config.entity_key] = normalized[
        validation_config.entity_key
    ].astype(str)

    normalized[validation_config.timestamp_column] = pd.to_datetime(
        normalized[validation_config.timestamp_column],
        errors="raise",
        utc=True,
        format="mixed",
    )

    normalized = normalized.sort_values(
        by=[validation_config.entity_key, validation_config.timestamp_column],
        kind="mergesort",
    ).reset_index(drop=True)

    return normalized


def _resolve_source_timestamp_range(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return min/max timestamps from the validated feature frame."""

    timestamps = pd.to_datetime(
        frame[timestamp_column],
        errors="raise",
        utc=True,
        format="mixed",
    )

    if timestamps.isna().any():
        raise TrainingSnapshotError("Feature timestamp column contains null timestamps.")

    return timestamps.min(), timestamps.max()


def _hash_feature_frame(frame: pd.DataFrame) -> str:
    """Build a stable SHA-256 hash from a normalized feature dataframe."""

    stable_csv = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(stable_csv.encode("utf-8")).hexdigest()


def _build_deterministic_uuid(*, namespace: str, values: list[str]) -> str:
    """Build a deterministic UUID from semantic snapshot inputs."""

    raw_key = "|".join([namespace, *values])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_key))


def _write_metadata_json(
    metadata_path: Path,
    metadata: TrainingSnapshotMetadata,
) -> None:
    """Write snapshot metadata as stable local JSON."""

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2, sort_keys=True)
        file.write("\n")


def metadata_to_database_records(
    metadata: TrainingSnapshotMetadata,
) -> dict[str, dict[str, Any]]:
    """Return DB-shaped records for future insertion into Project 4 SQL tables.

    The current checkpoint writes local JSON. This helper keeps the module ready
    for the later registry/database integration without depending on an
    unconfirmed database interface.
    """

    feature_snapshot_record = {
        "snapshot_id": metadata.snapshot_id,
        "snapshot_name": metadata.snapshot_name,
        "source_system": "project4_local_snapshot",
        "source_tables": metadata.source_tables,
        "source_min_timestamp": metadata.source_min_timestamp,
        "source_max_timestamp": metadata.source_max_timestamp,
        "feature_schema_version": metadata.feature_schema_version,
        "row_count": metadata.row_count,
        "snapshot_path": metadata.snapshot_path,
        "data_quality_status": metadata.data_quality_status,
        "created_at": metadata.created_at_utc,
    }

    training_dataset_record = {
        "training_dataset_id": metadata.training_dataset_id,
        "snapshot_id": metadata.snapshot_id,
        "dataset_version": metadata.dataset_version,
        "feature_schema_version": metadata.feature_schema_version,
        "feature_columns": metadata.feature_columns,
        "row_count": metadata.row_count,
        "created_at": metadata.created_at_utc,
    }

    return {
        "ml.feature_snapshots": feature_snapshot_record,
        "ml.training_datasets": training_dataset_record,
    }


__all__ = [
    "TrainingSnapshotConfig",
    "TrainingSnapshotError",
    "TrainingSnapshotMetadata",
    "create_training_snapshot",
    "load_training_snapshot_metadata",
    "metadata_to_database_records",
    "read_training_snapshot",
]
