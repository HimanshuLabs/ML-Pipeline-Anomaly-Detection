"""Batch anomaly scoring for Project 4.

This module loads the active production anomaly model, scores a batch of
validated feature rows, and persists prediction evidence to a local JSONL
fallback file.

Database persistence to ml.batch_predictions is represented by the SQL schema
and can be wired later without changing the prediction contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml

from anomaly_detection.prediction_logging import write_batch_predictions_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_PREDICTION_LOG_PATH = (
    PROJECT_ROOT / "logs" / "predictions" / "batch_predictions.jsonl"
)


class BatchInferenceError(RuntimeError):
    """Raised when batch inference cannot safely score input records."""


@dataclass(frozen=True)
class ActiveModelArtifacts:
    """Resolved active model artifact bundle."""

    model_name: str
    model_version: str
    model_path: Path
    metadata_path: Path
    baseline_stats_path: Path
    feature_schema_path: Path
    feature_schema_version: str
    dataset_snapshot_id: str | None
    training_dataset_id: str | None
    model: Any
    metadata: dict[str, Any]
    baseline_stats: dict[str, Any]
    feature_schema: dict[str, Any]


@dataclass(frozen=True)
class BatchPredictionRecord:
    """Serializable batch prediction record."""

    prediction_id: str
    batch_run_id: str
    model_name: str
    model_version: str
    dataset_snapshot_id: str | None
    training_dataset_id: str | None
    feature_schema_version: str
    entity_type: str
    entity_id: str
    source_project: str | None
    source_table: str | None
    source_record_id: str | None
    score_timestamp: str
    anomaly_score: float
    is_anomaly: bool
    threshold_used: float
    feature_payload_hash: str
    feature_payload: dict[str, Any]
    inference_latency_ms: float
    prediction_status: str = "success"
    error_message: str | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BatchInferenceError(f"Missing YAML file: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise BatchInferenceError(f"YAML file did not contain an object: {path}")

    return loaded


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BatchInferenceError(f"Missing JSON file: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise BatchInferenceError(f"JSON file did not contain an object: {path}")

    return loaded


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def hash_feature_payload(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for the scored feature payload."""

    normalized_payload = {
        key: _json_safe(payload[key])
        for key in sorted(payload)
    }
    encoded = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_active_model_artifacts(
    active_model_path: Path = DEFAULT_ACTIVE_MODEL_PATH,
) -> ActiveModelArtifacts:
    """Load the active production model and its metadata bundle."""

    active_model = _read_yaml(active_model_path)

    model_name = str(active_model.get("model_name") or "").strip()
    model_version = str(active_model.get("active_model_version") or "").strip()
    model_path_raw = active_model.get("artifact_path")
    feature_schema_version = str(
        active_model.get("feature_schema_version") or ""
    ).strip()

    if not model_name:
        raise BatchInferenceError("active_model.yaml is missing model_name")

    if not model_version:
        raise BatchInferenceError(
            "active_model.yaml is missing active_model_version"
        )

    if not model_path_raw:
        raise BatchInferenceError("active_model.yaml is missing artifact_path")

    model_path = _resolve_project_path(str(model_path_raw))
    if not model_path.exists():
        raise BatchInferenceError(f"Active model artifact not found: {model_path}")

    model_dir = model_path.parent
    metadata_path = model_dir / "metadata.json"
    baseline_stats_path = model_dir / "baseline_stats.json"
    feature_schema_path = model_dir / "feature_schema.json"

    metadata = _read_json(metadata_path)
    baseline_stats = _read_json(baseline_stats_path)
    feature_schema = _read_json(feature_schema_path)

    metadata_version = metadata.get("model_version")
    baseline_version = baseline_stats.get("model_version")
    schema_version = feature_schema.get("feature_schema_version")

    if metadata_version != model_version:
        raise BatchInferenceError(
            "metadata.json model_version does not match active model: "
            f"{metadata_version!r} != {model_version!r}"
        )

    if baseline_version != model_version:
        raise BatchInferenceError(
            "baseline_stats.json model_version does not match active model: "
            f"{baseline_version!r} != {model_version!r}"
        )

    if feature_schema_version and schema_version != feature_schema_version:
        raise BatchInferenceError(
            "feature_schema.json feature_schema_version does not match active "
            f"pointer: {schema_version!r} != {feature_schema_version!r}"
        )

    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise BatchInferenceError("metadata.json is missing feature_names")

    if feature_schema.get("feature_count") != len(feature_names):
        raise BatchInferenceError(
            "feature_schema.json feature_count does not match metadata feature_names"
        )

    model = joblib.load(model_path)

    return ActiveModelArtifacts(
        model_name=model_name,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        baseline_stats_path=baseline_stats_path,
        feature_schema_path=feature_schema_path,
        feature_schema_version=feature_schema_version or str(schema_version),
        dataset_snapshot_id=active_model.get("dataset_snapshot_id"),
        training_dataset_id=active_model.get("training_dataset_id"),
        model=model,
        metadata=metadata,
        baseline_stats=baseline_stats,
        feature_schema=feature_schema,
    )


def get_required_feature_names(artifacts: ActiveModelArtifacts) -> list[str]:
    """Return ordered model feature names."""

    feature_names = artifacts.metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise BatchInferenceError("Active model metadata has no feature_names")

    return [str(feature_name) for feature_name in feature_names]


def prepare_feature_matrix(
    features: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Select and validate numeric model features."""

    if features.empty:
        raise BatchInferenceError("Cannot score an empty feature batch")

    missing_features = [
        feature_name
        for feature_name in feature_names
        if feature_name not in features.columns
    ]
    if missing_features:
        raise BatchInferenceError(
            "Input batch is missing required model features: "
            + ", ".join(missing_features)
        )

    matrix = features.loc[:, feature_names].copy()

    for column in feature_names:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")

    null_counts = matrix.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        details = ", ".join(
            f"{column}={count}"
            for column, count in columns_with_nulls.items()
        )
        raise BatchInferenceError(
            "Input batch contains null/non-numeric model feature values: "
            + details
        )

    return matrix.astype(float)


def _get_entity_id(
    row: pd.Series,
    row_index: int,
    entity_key: str,
) -> str:
    if entity_key in row and pd.notna(row[entity_key]):
        return str(row[entity_key])

    if "entity_id" in row and pd.notna(row["entity_id"]):
        return str(row["entity_id"])

    if "user_id" in row and pd.notna(row["user_id"]):
        return str(row["user_id"])

    return f"row_{row_index}"


def _get_source_record_id(row: pd.Series) -> str | None:
    for column in ("source_record_id", "event_id", "session_id"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return None


def score_feature_batch(
    features: pd.DataFrame,
    artifacts: ActiveModelArtifacts,
    *,
    entity_type: str = "customer",
    source_project: str | None = None,
    source_table: str | None = None,
    batch_run_id: str | None = None,
) -> list[BatchPredictionRecord]:
    """Score a batch of features with the active model."""

    batch_id = batch_run_id or str(uuid4())
    feature_names = get_required_feature_names(artifacts)
    feature_matrix = prepare_feature_matrix(features, feature_names)

    started = time.perf_counter()

    if hasattr(artifacts.model, "decision_function"):
        scores = artifacts.model.decision_function(feature_matrix)
        threshold_used = 0.0
    elif hasattr(artifacts.model, "score_samples"):
        scores = artifacts.model.score_samples(feature_matrix)
        threshold_used = _infer_score_threshold(artifacts.baseline_stats)
    else:
        raise BatchInferenceError(
            "Active model must expose decision_function or score_samples"
        )

    if hasattr(artifacts.model, "predict"):
        model_predictions = artifacts.model.predict(feature_matrix)
        anomaly_flags = np.asarray(model_predictions) == -1
    else:
        anomaly_flags = np.asarray(scores) < threshold_used

    elapsed_ms = (time.perf_counter() - started) * 1000
    per_record_latency_ms = elapsed_ms / max(len(feature_matrix), 1)

    score_timestamp = datetime.now(UTC).isoformat()
    entity_key = str(artifacts.feature_schema.get("entity_key") or "entity_id")

    records: list[BatchPredictionRecord] = []

    for row_number, (row_index, row) in enumerate(features.iterrows()):
        feature_payload = {
            feature_name: _json_safe(row[feature_name])
            for feature_name in feature_names
        }

        entity_id = _get_entity_id(row, row_number, entity_key)

        records.append(
            BatchPredictionRecord(
                prediction_id=str(uuid4()),
                batch_run_id=batch_id,
                model_name=artifacts.model_name,
                model_version=artifacts.model_version,
                dataset_snapshot_id=artifacts.dataset_snapshot_id,
                training_dataset_id=artifacts.training_dataset_id,
                feature_schema_version=artifacts.feature_schema_version,
                entity_type=entity_type,
                entity_id=entity_id,
                source_project=source_project,
                source_table=source_table,
                source_record_id=_get_source_record_id(row),
                score_timestamp=score_timestamp,
                anomaly_score=float(scores[row_number]),
                is_anomaly=bool(anomaly_flags[row_number]),
                threshold_used=float(threshold_used),
                feature_payload_hash=hash_feature_payload(feature_payload),
                feature_payload=feature_payload,
                inference_latency_ms=float(per_record_latency_ms),
            )
        )

    return records


def _infer_score_threshold(baseline_stats: dict[str, Any]) -> float:
    """Infer a threshold from baseline stats for score_samples fallback.

    IsolationForest decision_function uses 0.0 as the anomaly threshold.
    This fallback exists for model objects that expose score_samples only.
    """

    prediction_summary = baseline_stats.get("prediction_summary")
    if isinstance(prediction_summary, dict):
        for key in (
            "threshold_used",
            "score_threshold",
            "anomaly_score_threshold",
            "decision_threshold",
            "contamination_threshold",
            "max_anomaly_score",
            "min_normal_score",
        ):
            value = prediction_summary.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

    for key in ("threshold_used", "score_threshold", "anomaly_score_threshold"):
        value = baseline_stats.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    return 0.0


def write_predictions_jsonl(
    predictions: list[BatchPredictionRecord],
    output_path: Path = DEFAULT_PREDICTION_LOG_PATH,
) -> Path:
    """Persist batch prediction evidence using the shared logging contract."""

    return write_batch_predictions_jsonl(
        list(predictions),
        output_path,
    )

def load_feature_batch(input_path: Path) -> pd.DataFrame:
    """Load a batch feature file from parquet, csv, json, or jsonl."""

    if not input_path.exists():
        raise BatchInferenceError(f"Input feature file does not exist: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_path)

    if suffix == ".csv":
        return pd.read_csv(input_path)

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(input_path, lines=True)

    if suffix == ".json":
        return pd.read_json(input_path)

    raise BatchInferenceError(
        "Unsupported input format. Use .parquet, .csv, .json, .jsonl, or .ndjson"
    )


def run_batch_inference(
    features: pd.DataFrame,
    *,
    active_model_path: Path = DEFAULT_ACTIVE_MODEL_PATH,
    output_path: Path = DEFAULT_PREDICTION_LOG_PATH,
    entity_type: str = "customer",
    source_project: str | None = None,
    source_table: str | None = None,
    persist_jsonl: bool = True,
) -> list[BatchPredictionRecord]:
    """Load active model, score features, and optionally persist JSONL records."""

    artifacts = load_active_model_artifacts(active_model_path)
    predictions = score_feature_batch(
        features,
        artifacts,
        entity_type=entity_type,
        source_project=source_project,
        source_table=source_table,
    )

    if persist_jsonl:
        write_predictions_jsonl(predictions, output_path)

    return predictions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a feature batch with the active anomaly model."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        type=Path,
        help="Path to a parquet/csv/json/jsonl feature batch.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_PREDICTION_LOG_PATH,
        type=Path,
        help="Path to append JSONL prediction records.",
    )
    parser.add_argument(
        "--active-model-path",
        default=DEFAULT_ACTIVE_MODEL_PATH,
        type=Path,
        help="Path to configs/active_model.yaml.",
    )
    parser.add_argument(
        "--entity-type",
        default="customer",
        help="Entity type for scored rows, for example customer, session, campaign.",
    )
    parser.add_argument(
        "--source-project",
        default=None,
        help="Optional source project label.",
    )
    parser.add_argument(
        "--source-table",
        default=None,
        help="Optional source table label.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    feature_batch = load_feature_batch(args.input_path)
    predictions = run_batch_inference(
        feature_batch,
        active_model_path=args.active_model_path,
        output_path=args.output_path,
        entity_type=args.entity_type,
        source_project=args.source_project,
        source_table=args.source_table,
        persist_jsonl=True,
    )

    anomaly_count = sum(prediction.is_anomaly for prediction in predictions)
    print(
        json.dumps(
            {
                "status": "success",
                "records_scored": len(predictions),
                "anomalies_detected": anomaly_count,
                "output_path": str(args.output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
