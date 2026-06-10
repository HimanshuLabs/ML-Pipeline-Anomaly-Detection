"""Isolation Forest training pipeline for Project 4 anomaly detection.

The training pipeline reads a frozen feature snapshot, validates the feature
contract, trains an Isolation Forest model, and writes versioned local artifacts.

This is intentionally local-first. The model registry/MLflow/S3 layer comes
later; this checkpoint proves the first reproducible training artifact.
"""

from __future__ import annotations

import argparse
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

    try:
        metadata = load_training_snapshot_metadata(path)
        return _metadata_to_dict(metadata)
    except Exception:
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        if not isinstance(loaded, dict):
            raise TrainingError(f"Snapshot metadata must be a JSON object: {path}")

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


def build_training_feature_matrix(
    snapshot_frame: pd.DataFrame,
    *,
    snapshot_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Resolve the numeric training feature matrix for demo or real-source snapshots.

    Older v001/demo snapshots use the Project 4 feature validation contract.
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
        metadata_files = sorted(root.glob("snapshot_date=*/snapshot_id=*/metadata.json"))
        if not metadata_files:
            raise TrainingError(f"No training snapshot metadata found under: {root}")

        metadata_path = metadata_files[-1]
        features_path = metadata_path.parent / "features.parquet"

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

    matching_metadata = sorted(root.glob(f"snapshot_date=*/snapshot_id={snapshot_value}/metadata.json"))
    if not matching_metadata:
        matching_metadata = sorted(root.glob(f"snapshot_date=*/snapshot_id=*{snapshot_value}*/metadata.json"))

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

    snapshot_metadata = _load_optional_snapshot_metadata(snapshot_metadata_path)

    feature_matrix = build_training_feature_matrix(
        snapshot_frame,
        snapshot_metadata=snapshot_metadata,
    )
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    args = parse_args()
    features_path, metadata_path = resolve_snapshot_paths(
        args.snapshot,
        training_root=args.training_root,
    )

    artifact_metadata = train_isolation_forest_from_snapshot(
        features_path=features_path,
        snapshot_metadata_path=metadata_path,
        model_version=args.model_version,
        config_path=args.config,
        artifact_root_override=args.artifact_root,
        overwrite=args.overwrite,
    )

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


if __name__ == "__main__":
    main()

