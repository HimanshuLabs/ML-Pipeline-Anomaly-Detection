"""Isolation Forest training pipeline for the anomaly detection of this pipeline.

The training pipeline reads a frozen feature snapshot, validates the feature
contract, trains an Isolation Forest model, and writes versioned local artifacts.

Artifacts are local and versioned; the model registry decides what is served.
The training CLI records each run in MLflow when MLFLOW_TRACKING_URI is set
(see anomaly_detection.tracking); MLflow is a record, not the source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import IsolationForest

from anomaly_detection.evaluation import (
    build_baseline_stats_payload,
    calculate_feature_baseline_stats,
)
from anomaly_detection.feature_validation import get_model_feature_matrix
from anomaly_detection.registry import (
    DEFAULT_REGISTRY_PATH,
    _write_bytes_atomic,
    load_model_registry,
    project_relative_path,
)
from anomaly_detection.segmented_model import (
    SEGMENTED_MODEL_KIND,
    SINGLE_MODEL_KIND,
    SegmentedIsolationForest,
    SegmentedModelError,
    check_segment_row_counts,
    feature_spec_for,
    feature_spec_from_metadata,
    fit_segmented_isolation_forest,
    score_distribution,
    segment_spec_fingerprint,
)
from anomaly_detection.tracking import TrackingError, TrainingRun, training_run
from anomaly_detection.training_snapshot import (
    find_latest_snapshot,
    load_training_snapshot_metadata,
    read_training_snapshot,
    unfinished_snapshots_message,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


class TrainingError(ValueError):
    """Raised when model training cannot be completed safely."""


@dataclass(frozen=True)
class IsolationForestTrainingConfig:
    """Resolved training configuration."""

    model_name: str
    algorithm: str
    random_state: int
    contamination: float | str
    n_estimators: int
    max_samples: int | float | str
    artifact_root: Path
    feature_schema_version: str
    real_source_feature_schema_version: str
    min_rows_per_segment: int


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Metadata for a trained local model artifact."""

    model_name: str
    model_version: str
    algorithm: str
    artifact_dir: str
    model_path: str
    metadata_path: str
    feature_schema_path: str
    baseline_stats_path: str
    feature_schema_version: str
    feature_names: list[str]
    row_count: int
    training_started_at_utc: str
    training_finished_at_utc: str
    training_snapshot_metadata: dict[str, Any]
    model_parameters: dict[str, Any]
    baseline_anomaly_rate: float
    runtime_versions: dict[str, str]
    feature_set_fingerprint: str
    model_kind: str = SINGLE_MODEL_KIND
    # Segmented models: {entity_type: {feature_names, row_count, feature_set_fingerprint}}.
    segments: dict[str, Any] | None = None
    # MLflow run that recorded this training (null when tracking was off or unavailable).
    mlflow_run_id: str | None = None
    mlflow_experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""
        return asdict(self)


def runtime_versions() -> dict[str, str]:
    """Library versions of the running environment, recorded with each trained model."""

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def feature_set_fingerprint(feature_names: list[str]) -> str:
    """sha256 of the ordered feature names, UTF-8, joined with newlines."""

    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def ensure_feature_schema_version_is_unambiguous(
    *,
    feature_schema_version: str,
    feature_names: list[str],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    segment_features: dict[str, list[str]] | None = None,
) -> None:
    """Refuse a schema label that the registry already uses for a different feature spec.

    Single models compare the ordered feature_names, so the same names in a
    different order also count as a different feature set. Segmented models
    compare the whole segment spec (entity types and each type's ordered
    features), and a segmented and a single model never share a label.
    """

    new_spec = feature_spec_for(feature_names, segment_features)

    for entry in load_model_registry(registry_path):
        if entry.feature_schema_version != feature_schema_version:
            continue

        metadata_path = Path(entry.metadata_path)
        if not metadata_path.is_absolute():
            metadata_path = PROJECT_ROOT / metadata_path

        try:
            registered_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError(
                f"Cannot verify feature_schema_version {feature_schema_version!r}: "
                f"metadata for {entry.model_name} {entry.model_version} is unreadable "
                f"({entry.metadata_path})."
            ) from exc

        try:
            difference = feature_spec_from_metadata(registered_metadata).difference(new_spec)
        except ValueError as exc:
            difference = f"its metadata has no usable feature spec ({exc})"

        if difference is not None:
            raise TrainingError(
                f"feature_schema_version {feature_schema_version!r} is already used by "
                f"{entry.model_name} {entry.model_version} for a different ordered "
                f"feature_names list or segment spec: {difference}. Configure a new "
                "feature_schema_version for this feature set."
            )


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _resolve_path(path_value: str | Path, *, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise TrainingError(f"Model config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise TrainingError(f"Model config must be a YAML mapping: {config_path}")

    return loaded


def load_training_config(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    artifact_root_override: str | Path | None = None,
) -> IsolationForestTrainingConfig:
    """Load Isolation Forest training settings from model_config.yaml."""
    raw_config = _load_yaml(config_path)

    model_config = raw_config.get("model") or {}
    features_config = raw_config.get("features") or {}
    training_config = raw_config.get("training") or {}

    model_name = str(model_config.get("name", "isolation_forest"))
    algorithm = str(model_config.get("algorithm", "IsolationForest"))

    if algorithm != "IsolationForest":
        raise TrainingError(
            "Checkpoint 6 supports only IsolationForest. "
            f"Configured algorithm: {algorithm}"
        )

    contamination = model_config.get("contamination", 0.05)
    n_estimators = int(model_config.get("n_estimators", 200))
    max_samples = model_config.get("max_samples", "auto")
    random_state = int(model_config.get("random_state", 42))

    feature_schema_version = str(
        features_config.get("schema_version", "feature_schema_v001")
    )
    # The demo contract and the real-source feature set are different feature
    # lists, so they carry different labels.
    real_source_feature_schema_version = str(
        features_config.get("real_source_schema_version", feature_schema_version)
    )

    artifact_root_raw = artifact_root_override or training_config.get(
        "artifact_root",
        "artifacts/models",
    )
    artifact_root = _resolve_path(artifact_root_raw)

    min_rows_per_segment = training_config.get("min_rows_per_segment", 30)
    if isinstance(min_rows_per_segment, bool) or not isinstance(min_rows_per_segment, int):
        raise TrainingError(
            f"training.min_rows_per_segment must be an integer; got {min_rows_per_segment!r}"
        )
    if min_rows_per_segment < 1:
        raise TrainingError("training.min_rows_per_segment must be at least 1")

    if isinstance(contamination, str) and contamination != "auto":
        raise TrainingError(
            "IsolationForest contamination must be a float or 'auto'. "
            f"Got: {contamination}"
        )

    if not isinstance(max_samples, str | int | float):
        raise TrainingError(
            "IsolationForest max_samples must be 'auto', int, or float. "
            f"Got: {type(max_samples).__name__}"
        )

    return IsolationForestTrainingConfig(
        model_name=model_name,
        algorithm=algorithm,
        random_state=random_state,
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        artifact_root=artifact_root,
        feature_schema_version=feature_schema_version,
        real_source_feature_schema_version=real_source_feature_schema_version,
        min_rows_per_segment=min_rows_per_segment,
    )


def resolve_feature_schema_version(
    config: IsolationForestTrainingConfig,
    snapshot_metadata: dict[str, Any],
) -> str:
    """Label for the trained feature set: real-source snapshots have their own."""

    if snapshot_metadata.get("snapshot_type") == "real_source_extract":
        return config.real_source_feature_schema_version
    return config.feature_schema_version


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}

    if is_dataclass(metadata):
        return asdict(metadata)

    if isinstance(metadata, dict):
        return metadata

    if hasattr(metadata, "model_dump"):
        return metadata.model_dump()

    if hasattr(metadata, "dict"):
        return metadata.dict()

    raise TrainingError(
        "Unsupported training snapshot metadata object. "
        f"Type: {type(metadata).__name__}"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return _isoformat_utc(value)

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically: a crash leaves the previous file or none, never half of one."""
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    _write_bytes_atomic(path, text.encode("utf-8"))


def _load_optional_snapshot_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        raise TrainingError(f"Snapshot metadata file does not exist: {path}")

    try:
        metadata = load_training_snapshot_metadata(path)
        return _metadata_to_dict(metadata)
    except Exception:
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        if not isinstance(loaded, dict):
            raise TrainingError(f"Snapshot metadata must be a JSON object: {path}") from None

        return loaded


def _existing_model_versions(model_root: Path) -> list[int]:
    if not model_root.exists():
        return []

    versions: list[int] = []
    pattern = re.compile(r"^model_version=v(\d+)$")

    for child in model_root.iterdir():
        if not child.is_dir():
            continue

        match = pattern.match(child.name)
        if match:
            versions.append(int(match.group(1)))

    return sorted(versions)


def resolve_next_model_version(model_root: Path) -> str:
    """Resolve the next local model version in v001 format."""
    existing_versions = _existing_model_versions(model_root)
    next_version = existing_versions[-1] + 1 if existing_versions else 1
    return f"v{next_version:03d}"


def _prepare_artifact_dir(
    *,
    artifact_root: Path,
    model_name: str,
    model_version: str,
    overwrite: bool,
) -> Path:
    """The bundle folder this training will publish; refuses an existing one unless overwrite.

    Nothing is created here: the bundle is written beside it and renamed into place whole.
    """
    artifact_dir = artifact_root / model_name / f"model_version={model_version}"

    if artifact_dir.exists() and not overwrite:
        raise TrainingError(
            "Model artifact directory already exists. "
            f"Use overwrite=True or choose a new model version: {artifact_dir}"
        )
    return artifact_dir


def _staging_dir(artifact_dir: Path) -> Path:
    """A hidden folder beside the bundle; readers only look at model_version=* names."""
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = artifact_dir.parent / f".{artifact_dir.name}.partial-{uuid.uuid4().hex[:12]}"
    staging_dir.mkdir()
    return staging_dir


def _publish_bundle(staging_dir: Path, artifact_dir: Path) -> None:
    """Rename the complete bundle into place; an overwritten bundle is removed only now."""
    replaced = None
    if artifact_dir.exists():
        replaced = artifact_dir.parent / f".{artifact_dir.name}.replaced-{uuid.uuid4().hex[:12]}"
        artifact_dir.rename(replaced)
    try:
        staging_dir.rename(artifact_dir)
    except BaseException:
        if replaced is not None:
            replaced.rename(artifact_dir)
        raise
    if replaced is not None:
        shutil.rmtree(replaced)


def _build_model(config: IsolationForestTrainingConfig) -> IsolationForest:
    return IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        max_samples=config.max_samples,
        random_state=config.random_state,
        n_jobs=-1,
    )


def build_training_feature_matrix(
    snapshot_frame: pd.DataFrame,
    *,
    snapshot_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Resolve the numeric training feature matrix for demo or real-source snapshots.

    Older v001/demo snapshots use this pipeline's feature validation contract.
    Real-source snapshots generated by source_extract.py store their feature list
    in metadata.json under ``feature_columns``.
    """
    metadata = snapshot_metadata or {}

    if metadata.get("snapshot_type") == "real_source_extract":
        feature_columns = metadata.get("feature_columns")

        if not isinstance(feature_columns, list) or not feature_columns:
            raise TrainingError(
                "Real-source snapshot metadata must contain non-empty feature_columns."
            )

        missing_columns = [
            column for column in feature_columns if column not in snapshot_frame.columns
        ]
        if missing_columns:
            raise TrainingError(
                "Real-source snapshot is missing feature columns: "
                f"{missing_columns}"
            )

        feature_matrix = snapshot_frame.loc[:, feature_columns].copy()

        for column in feature_matrix.columns:
            feature_matrix[column] = pd.to_numeric(feature_matrix[column], errors="coerce")

        null_counts = feature_matrix.isna().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            raise TrainingError(
                "Real-source training feature matrix contains null values: "
                f"{columns_with_nulls.to_dict()}"
            )

        return feature_matrix.astype("float64")

    return get_model_feature_matrix(snapshot_frame, strict=True)


def resolve_snapshot_paths(
    snapshot: str | Path,
    *,
    training_root: str | Path = PROJECT_ROOT / "data" / "features" / "training",
) -> tuple[Path, Path | None]:
    """Resolve snapshot CLI input into features.parquet and metadata.json paths."""
    snapshot_value = str(snapshot)
    root = Path(training_root)

    if snapshot_value == "latest":
        latest = find_latest_snapshot(root)
        if latest.unfinished:
            raise TrainingError(unfinished_snapshots_message(latest.unfinished))
        if latest.folder is None:
            raise TrainingError(f"No training snapshot metadata found under: {root}")

        metadata_path = latest.folder / "metadata.json"
        features_path = latest.folder / "features.parquet"

        if not features_path.exists():
            raise TrainingError(f"Latest snapshot is missing features.parquet: {features_path}")

        return features_path, metadata_path

    snapshot_path = Path(snapshot)

    if snapshot_path.is_dir():
        features_path = snapshot_path / "features.parquet"
        metadata_path = snapshot_path / "metadata.json"
        if not features_path.exists():
            raise TrainingError(f"Snapshot directory is missing features.parquet: {snapshot_path}")
        return features_path, metadata_path if metadata_path.exists() else None

    if snapshot_path.is_file():
        metadata_path = snapshot_path.parent / "metadata.json"
        return snapshot_path, metadata_path if metadata_path.exists() else None

    matching_metadata = sorted(
        root.glob(f"snapshot_date=*/snapshot_id={snapshot_value}/metadata.json"),
    )
    if not matching_metadata:
        matching_metadata = sorted(
            root.glob(f"snapshot_date=*/snapshot_id=*{snapshot_value}*/metadata.json"),
        )

    if not matching_metadata:
        raise TrainingError(f"Could not resolve training snapshot: {snapshot_value}")

    metadata_path = matching_metadata[-1]
    features_path = metadata_path.parent / "features.parquet"

    if not features_path.exists():
        raise TrainingError(f"Resolved snapshot is missing features.parquet: {features_path}")

    return features_path, metadata_path


def train_isolation_forest_from_snapshot(
    *,
    features_path: str | Path,
    snapshot_metadata_path: str | Path | None = None,
    model_version: str | None = None,
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    artifact_root_override: str | Path | None = None,
    overwrite: bool = False,
    training_started_at_utc: datetime | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    training_run: TrainingRun | None = None,
) -> ModelArtifactMetadata:
    """Train Isolation Forest from a frozen training snapshot.

    Args:
        features_path: Path to a Parquet feature snapshot.
        snapshot_metadata_path: Optional JSON metadata path from Checkpoint 5.
        model_version: Optional explicit model version such as ``v001``.
        config_path: Path to model config YAML.
        artifact_root_override: Optional artifact root for tests/local overrides.
        overwrite: Whether to replace an existing artifact directory.
        training_started_at_utc: Optional fixed timestamp for deterministic tests.
        registry_path: Registry checked so a feature_schema_version label is never
            reused for a different ordered feature list.
        training_run: MLflow run started by the caller (training CLI, retrain); its
            ids go into metadata.json and the written bundle is logged to it.

    Returns:
        ModelArtifactMetadata describing written artifacts.
    """
    started_at = training_started_at_utc or _utc_now()
    config = load_training_config(
        config_path=config_path,
        artifact_root_override=artifact_root_override,
    )

    snapshot_path = Path(features_path)
    if not snapshot_path.exists():
        raise TrainingError(f"Training snapshot features file does not exist: {snapshot_path}")

    LOGGER.info("Reading training snapshot from %s", snapshot_path)
    snapshot_frame = read_training_snapshot(snapshot_path)

    if not isinstance(snapshot_frame, pd.DataFrame):
        raise TrainingError(
            "Training snapshot reader must return a pandas DataFrame. "
            f"Got: {type(snapshot_frame).__name__}"
        )

    if snapshot_frame.empty:
        raise TrainingError("Training snapshot must not be empty.")

    snapshot_metadata = _load_optional_snapshot_metadata(snapshot_metadata_path)

    feature_matrix = build_training_feature_matrix(
        snapshot_frame,
        snapshot_metadata=snapshot_metadata,
    )
    if feature_matrix.empty:
        raise TrainingError("Validated model feature matrix must not be empty.")

    feature_names = [str(column) for column in feature_matrix.columns]
    feature_schema_version = resolve_feature_schema_version(config, snapshot_metadata)

    # Real-source snapshots train one model per entity type on that type's own features.
    segment_features: dict[str, list[str]] | None = None
    entity_types = None
    if snapshot_metadata.get("snapshot_type") == "real_source_extract":
        segment_features, entity_types = _resolve_segment_features(
            snapshot_frame,
            feature_names=feature_names,
            min_rows_per_segment=config.min_rows_per_segment,
        )
        feature_names = [
            name
            for name in feature_names
            if any(name in names for names in segment_features.values())
        ]
        fingerprint = segment_spec_fingerprint(segment_features)
    else:
        fingerprint = feature_set_fingerprint(feature_names)

    ensure_feature_schema_version_is_unambiguous(
        feature_schema_version=feature_schema_version,
        feature_names=feature_names,
        registry_path=registry_path,
        segment_features=segment_features,
    )

    model_root = config.artifact_root / config.model_name
    resolved_model_version = model_version or resolve_next_model_version(model_root)
    artifact_dir = _prepare_artifact_dir(
        artifact_root=config.artifact_root,
        model_name=config.model_name,
        model_version=resolved_model_version,
        overwrite=overwrite,
    )

    LOGGER.info(
        "Training %s model_version=%s on %s rows and %s features",
        config.model_name,
        resolved_model_version,
        len(feature_matrix),
        len(feature_matrix.columns),
    )

    segments_metadata: dict[str, Any] | None = None
    if segment_features is None:
        model: IsolationForest | SegmentedIsolationForest = _build_model(config)
        model.fit(feature_matrix)
        anomaly_scores = model.decision_function(feature_matrix)
        predictions = model.predict(feature_matrix)
    else:
        feature_matrix = feature_matrix.loc[:, feature_names]
        training_frame = feature_matrix.assign(entity_type=entity_types)
        try:
            model = fit_segmented_isolation_forest(
                training_frame,
                segment_features=segment_features,
                build_model=lambda: _build_model(config),
                min_rows_per_segment=config.min_rows_per_segment,
            )
        except SegmentedModelError as exc:
            raise TrainingError(str(exc)) from exc
        anomaly_scores = model.decision_function(training_frame)
        predictions = model.predict(training_frame)
        segments_metadata = {
            entity_type: {
                "feature_names": segment.feature_names,
                "row_count": segment.row_count,
                "feature_set_fingerprint": feature_set_fingerprint(segment.feature_names),
            }
            for entity_type, segment in model.segments.items()
        }

    baseline_stats = build_baseline_stats_payload(
        model_name=config.model_name,
        model_version=resolved_model_version,
        feature_schema_version=feature_schema_version,
        feature_matrix=feature_matrix,
        anomaly_scores=anomaly_scores,
        predictions=predictions,
    )
    if isinstance(model, SegmentedIsolationForest):
        baseline_stats["model_kind"] = SEGMENTED_MODEL_KIND
        baseline_stats["segments"] = _segment_baselines(
            model, training_frame, anomaly_scores, predictions
        )

    finished_at = _utc_now()

    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    feature_schema_path = artifact_dir / "feature_schema.json"
    baseline_stats_path = artifact_dir / "baseline_stats.json"

    # Describe where the features actually came from: the training snapshot.
    feature_schema_payload = {
        "feature_schema_version": feature_schema_version,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "feature_set_fingerprint": fingerprint,
        "entity_key": "entity_id",
        "timestamp_column": snapshot_metadata.get("timestamp_column"),
        "source": snapshot_metadata.get("snapshot_id"),
    }
    if segments_metadata is not None:
        feature_schema_payload["model_kind"] = SEGMENTED_MODEL_KIND
        feature_schema_payload["segments"] = segments_metadata

    artifact_metadata = ModelArtifactMetadata(
        model_name=config.model_name,
        model_version=resolved_model_version,
        algorithm=config.algorithm,
        # Repo-relative inside the repository, absolute for a bundle outside it.
        artifact_dir=project_relative_path(artifact_dir),
        model_path=project_relative_path(model_path),
        metadata_path=project_relative_path(metadata_path),
        feature_schema_path=project_relative_path(feature_schema_path),
        baseline_stats_path=project_relative_path(baseline_stats_path),
        feature_schema_version=feature_schema_version,
        feature_names=feature_names,
        row_count=int(len(feature_matrix)),
        training_started_at_utc=_isoformat_utc(started_at),
        training_finished_at_utc=_isoformat_utc(finished_at),
        training_snapshot_metadata=snapshot_metadata,
        model_parameters={
            "n_estimators": config.n_estimators,
            "contamination": config.contamination,
            "max_samples": config.max_samples,
            "random_state": config.random_state,
        },
        baseline_anomaly_rate=float(baseline_stats["baseline_anomaly_rate"]),
        runtime_versions=runtime_versions(),
        feature_set_fingerprint=fingerprint,
        model_kind=SEGMENTED_MODEL_KIND if segments_metadata is not None else SINGLE_MODEL_KIND,
        segments=segments_metadata,
        mlflow_run_id=training_run.run_id if training_run else None,
        mlflow_experiment_id=training_run.experiment_id if training_run else None,
    )

    # Written beside the bundle and renamed into place whole: a crash never leaves a
    # half bundle (model.joblib without its metadata) under a model_version= name.
    staging_dir = _staging_dir(artifact_dir)
    try:
        joblib.dump(model, staging_dir / model_path.name)
        _write_json(staging_dir / feature_schema_path.name, feature_schema_payload)
        _write_json(staging_dir / baseline_stats_path.name, baseline_stats)
        _write_json(staging_dir / metadata_path.name, artifact_metadata.to_dict())
        _publish_bundle(staging_dir, artifact_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    LOGGER.info("Wrote trained model artifacts to %s", artifact_dir)

    if training_run is not None:
        training_run.log_bundle(
            metadata=artifact_metadata.to_dict(),
            artifact_dir=artifact_dir,
            baseline_stats=baseline_stats,
            extra_params={"model.min_rows_per_segment": config.min_rows_per_segment},
        )

    return artifact_metadata


def load_trained_model(model_path: str | Path) -> IsolationForest | SegmentedIsolationForest:
    """Load a trained (single or segmented) Isolation Forest model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise TrainingError(f"Model artifact does not exist: {path}")

    model = joblib.load(path)

    if not isinstance(model, IsolationForest | SegmentedIsolationForest):
        raise TrainingError(
            "Loaded model is not an IsolationForest or SegmentedIsolationForest instance. "
            f"Got: {type(model).__name__}"
        )

    return model


def _resolve_segment_features(
    snapshot_frame: pd.DataFrame,
    *,
    feature_names: list[str],
    min_rows_per_segment: int,
) -> tuple[dict[str, list[str]], Any]:
    """Segment spec for the entity types present; refuses before anything is written."""

    from anomaly_detection.source_extract import SEGMENT_FEATURE_COLUMNS

    if "entity_type" not in snapshot_frame.columns:
        raise TrainingError("Real-source snapshot has no entity_type column to segment by.")
    if snapshot_frame["entity_type"].isna().any():
        raise TrainingError("Real-source snapshot has null entity_type values.")
    entity_types = snapshot_frame["entity_type"].astype(str).to_numpy()

    try:
        counts = check_segment_row_counts(
            entity_types,
            segment_features=SEGMENT_FEATURE_COLUMNS,
            min_rows_per_segment=min_rows_per_segment,
        )
    except SegmentedModelError as exc:
        raise TrainingError(str(exc)) from exc

    segment_features = {t: list(SEGMENT_FEATURE_COLUMNS[t]) for t in sorted(counts)}
    missing = sorted(
        {name for names in segment_features.values() for name in names} - set(feature_names)
    )
    if missing:
        raise TrainingError(f"Snapshot feature_columns lack segment feature(s): {missing}")
    return segment_features, entity_types


def _segment_baselines(
    model: SegmentedIsolationForest,
    training_frame: pd.DataFrame,
    anomaly_scores: Any,
    predictions: Any,
) -> dict[str, Any]:
    """Per segment: anomaly rate, p5/p50/p95 scores, and its features' baselines."""

    entity_types = training_frame["entity_type"].to_numpy()
    scores = numpy.asarray(anomaly_scores, dtype=float)
    is_anomaly = numpy.asarray(predictions) == -1
    baselines: dict[str, Any] = {}
    for entity_type, segment in model.segments.items():
        mask = entity_types == entity_type
        baselines[entity_type] = {
            **score_distribution(scores[mask], is_anomaly[mask]),
            "feature_baselines": calculate_feature_baseline_stats(
                training_frame.loc[mask, segment.feature_names]
            ),
        }
    return baselines

def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train an Isolation Forest anomaly model from a Project 4 training snapshot."
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Snapshot path, snapshot id, or 'latest'.",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Explicit model version such as v002. If omitted, the next version is used.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_MODEL_CONFIG_PATH),
        help="Path to model_config.yaml.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Optional artifact root override.",
    )
    parser.add_argument(
        "--training-root",
        default=str(PROJECT_ROOT / "data" / "features" / "training"),
        help="Training snapshot root used when --snapshot is 'latest' or a snapshot id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the model artifact directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for local model training."""
    from anomaly_detection.logging_config import configure_logging

    configure_logging()

    args = parse_args()
    features_path, metadata_path = resolve_snapshot_paths(
        args.snapshot,
        training_root=args.training_root,
    )
    # The version is fixed up front so the MLflow run can be named after it.
    config = load_training_config(args.config, artifact_root_override=args.artifact_root)
    model_version = args.model_version or resolve_next_model_version(
        config.artifact_root / config.model_name
    )

    try:
        with training_run(
            entrypoint="train", model_version=model_version, model_name=config.model_name
        ) as run:
            artifact_metadata = train_isolation_forest_from_snapshot(
                features_path=features_path,
                snapshot_metadata_path=metadata_path,
                model_version=model_version,
                config_path=args.config,
                artifact_root_override=args.artifact_root,
                overwrite=args.overwrite,
                training_run=run,
            )
    except TrackingError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    dataset_snapshot_id = artifact_metadata.training_snapshot_metadata.get(
        "snapshot_id",
        "unknown",
    )
    dataset_snapshot_type = artifact_metadata.training_snapshot_metadata.get(
        "snapshot_type",
        "unknown",
    )

    print("OK: trained Isolation Forest model")
    print(f"model_name={artifact_metadata.model_name}")
    print(f"model_version={artifact_metadata.model_version}")
    print(f"dataset_snapshot_id={dataset_snapshot_id}")
    print(f"dataset_snapshot_type={dataset_snapshot_type}")
    print(f"row_count={artifact_metadata.row_count}")
    print(f"feature_count={len(artifact_metadata.feature_names)}")
    print(f"baseline_anomaly_rate={artifact_metadata.baseline_anomaly_rate:.6f}")
    print(f"artifact_dir={artifact_metadata.artifact_dir}")
    print(f"model_path={artifact_metadata.model_path}")
    print(f"metadata_path={artifact_metadata.metadata_path}")
    print(f"feature_schema_path={artifact_metadata.feature_schema_path}")
    print(f"baseline_stats_path={artifact_metadata.baseline_stats_path}")
    print(f"mlflow_run_id={artifact_metadata.mlflow_run_id}")


if __name__ == "__main__":
    main()

