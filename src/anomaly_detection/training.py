"""Isolation Forest training pipeline for Project 4 anomaly detection.

The training pipeline reads a frozen feature snapshot, validates the feature
contract, trains an Isolation Forest model, and writes versioned local artifacts.

This is intentionally local-first. The model registry/MLflow/S3 layer comes
later; this checkpoint proves the first reproducible training artifact.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

from anomaly_detection.evaluation import build_baseline_stats_payload
from anomaly_detection.feature_validation import get_model_feature_matrix
from anomaly_detection.training_snapshot import (
    load_training_snapshot_metadata,
    read_training_snapshot,
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

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""
        return asdict(self)


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

    artifact_root_raw = artifact_root_override or training_config.get(
        "artifact_root",
        "artifacts/models",
    )
    artifact_root = _resolve_path(artifact_root_raw)

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
    )


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)
        file.write("\n")


def _load_optional_snapshot_metadata(metadata_path: str | Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        raise TrainingError(f"Snapshot metadata file does not exist: {path}")

    metadata = load_training_snapshot_metadata(path)
    return _metadata_to_dict(metadata)


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
    artifact_dir = artifact_root / model_name / f"model_version={model_version}"

    if artifact_dir.exists():
        if not overwrite:
            raise TrainingError(
                "Model artifact directory already exists. "
                f"Use overwrite=True or choose a new model version: {artifact_dir}"
            )
        shutil.rmtree(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _build_model(config: IsolationForestTrainingConfig) -> IsolationForest:
    return IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        max_samples=config.max_samples,
        random_state=config.random_state,
        n_jobs=-1,
    )


def train_isolation_forest_from_snapshot(
    *,
    features_path: str | Path,
    snapshot_metadata_path: str | Path | None = None,
    model_version: str | None = None,
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    artifact_root_override: str | Path | None = None,
    overwrite: bool = False,
    training_started_at_utc: datetime | None = None,
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

    feature_matrix = get_model_feature_matrix(snapshot_frame, strict=True)
    if feature_matrix.empty:
        raise TrainingError("Validated model feature matrix must not be empty.")

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

    model = _build_model(config)
    model.fit(feature_matrix)

    anomaly_scores = model.decision_function(feature_matrix)
    predictions = model.predict(feature_matrix)

    baseline_stats = build_baseline_stats_payload(
        model_name=config.model_name,
        model_version=resolved_model_version,
        feature_schema_version=config.feature_schema_version,
        feature_matrix=feature_matrix,
        anomaly_scores=anomaly_scores,
        predictions=predictions,
    )

    snapshot_metadata = _load_optional_snapshot_metadata(snapshot_metadata_path)
    finished_at = _utc_now()

    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    feature_schema_path = artifact_dir / "feature_schema.json"
    baseline_stats_path = artifact_dir / "baseline_stats.json"

    joblib.dump(model, model_path)

    feature_schema_payload = {
        "feature_schema_version": config.feature_schema_version,
        "feature_names": list(feature_matrix.columns),
        "feature_count": len(feature_matrix.columns),
        "entity_key": "entity_id",
        "timestamp_column": "feature_timestamp",
        "source": "configs/model_config.yaml",
    }

    artifact_metadata = ModelArtifactMetadata(
        model_name=config.model_name,
        model_version=resolved_model_version,
        algorithm=config.algorithm,
        artifact_dir=str(artifact_dir),
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        baseline_stats_path=str(baseline_stats_path),
        feature_schema_version=config.feature_schema_version,
        feature_names=list(feature_matrix.columns),
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
    )

    _write_json(feature_schema_path, feature_schema_payload)
    _write_json(baseline_stats_path, baseline_stats)
    _write_json(metadata_path, artifact_metadata.to_dict())

    LOGGER.info("Wrote trained model artifacts to %s", artifact_dir)

    return artifact_metadata


def load_trained_model(model_path: str | Path) -> IsolationForest:
    """Load a trained Isolation Forest model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise TrainingError(f"Model artifact does not exist: {path}")

    model = joblib.load(path)

    if not isinstance(model, IsolationForest):
        raise TrainingError(
            "Loaded model is not an IsolationForest instance. "
            f"Got: {type(model).__name__}"
        )

    return model
