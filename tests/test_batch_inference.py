from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from anomaly_detection.batch_inference import (
    BatchInferenceError,
    BatchPredictionRecord,
    hash_feature_payload,
    load_active_model_artifacts,
    prepare_feature_matrix,
    score_feature_batch,
    write_predictions_jsonl,
)


class DummyAnomalyModel:
    """Small deterministic model used only for batch inference unit tests."""

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([0.25, -0.40], dtype=float)[: len(features)]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([1, -1], dtype=int)[: len(features)]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_artifact_bundle(tmp_path: Path) -> Path:
    model_dir = tmp_path / "artifacts" / "models" / "isolation_forest" / "model_version=vtest"
    model_dir.mkdir(parents=True)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    baseline_path = model_dir / "baseline_stats.json"
    schema_path = model_dir / "feature_schema.json"
    active_path = tmp_path / "active_model.yaml"

    joblib.dump(DummyAnomalyModel(), model_path)

    _write_json(
        metadata_path,
        {
            "model_name": "isolation_forest",
            "model_version": "vtest",
            "feature_schema_version": "feature_schema_test",
            "feature_names": ["feature_a", "feature_b"],
            "baseline_stats_path": str(baseline_path),
            "feature_schema_path": str(schema_path),
            "model_path": str(model_path),
        },
    )

    _write_json(
        baseline_path,
        {
            "model_name": "isolation_forest",
            "model_version": "vtest",
            "feature_schema_version": "feature_schema_test",
            "baseline_anomaly_rate": 0.50,
            "label_availability": "unlabeled_proxy_metrics",
            "metric_type": "unit_test_proxy",
            "prediction_summary": {
                "threshold_used": 0.0,
            },
            "feature_baselines": {
                "feature_a": {"mean": 1.0, "variance": 0.1},
                "feature_b": {"mean": 2.0, "variance": 0.2},
            },
        },
    )

    _write_json(
        schema_path,
        {
            "feature_schema_version": "feature_schema_test",
            "feature_count": 2,
            "feature_names": ["feature_a", "feature_b"],
            "entity_key": "entity_id",
            "source": "unit_test",
            "timestamp_column": "event_timestamp",
        },
    )

    active_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "isolation_forest",
                "active_model_version": "vtest",
                "artifact_path": str(model_path),
                "feature_schema_version": "feature_schema_test",
                "status": "production",
                "dataset_snapshot_id": "snapshot_test",
                "training_dataset_id": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return active_path


def test_load_active_model_artifacts_from_active_pointer(tmp_path: Path) -> None:
    active_path = _build_artifact_bundle(tmp_path)

    artifacts = load_active_model_artifacts(active_path)

    assert artifacts.model_name == "isolation_forest"
    assert artifacts.model_version == "vtest"
    assert artifacts.feature_schema_version == "feature_schema_test"
    assert artifacts.dataset_snapshot_id == "snapshot_test"
    assert artifacts.training_dataset_id is None
    assert artifacts.metadata["feature_names"] == ["feature_a", "feature_b"]


def test_prepare_feature_matrix_selects_ordered_numeric_features() -> None:
    features = pd.DataFrame(
        [
            {
                "entity_id": "customer_001",
                "feature_b": "2.5",
                "feature_a": 1,
                "ignored_column": "not_model_input",
            }
        ]
    )

    matrix = prepare_feature_matrix(features, ["feature_a", "feature_b"])

    assert list(matrix.columns) == ["feature_a", "feature_b"]
    assert matrix.iloc[0].to_dict() == {"feature_a": 1.0, "feature_b": 2.5}


def test_prepare_feature_matrix_fails_for_missing_feature() -> None:
    features = pd.DataFrame([{"feature_a": 1.0}])

    with pytest.raises(BatchInferenceError, match="missing required model features"):
        prepare_feature_matrix(features, ["feature_a", "feature_b"])


def test_prepare_feature_matrix_fails_for_non_numeric_feature() -> None:
    features = pd.DataFrame(
        [
            {
                "feature_a": 1.0,
                "feature_b": "bad_value",
            }
        ]
    )

    with pytest.raises(BatchInferenceError, match="null/non-numeric"):
        prepare_feature_matrix(features, ["feature_a", "feature_b"])


def test_hash_feature_payload_is_stable_for_key_order() -> None:
    first_hash = hash_feature_payload({"feature_b": 2.0, "feature_a": 1.0})
    second_hash = hash_feature_payload({"feature_a": 1.0, "feature_b": 2.0})

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_score_feature_batch_returns_prediction_contract(tmp_path: Path) -> None:
    active_path = _build_artifact_bundle(tmp_path)
    artifacts = load_active_model_artifacts(active_path)

    features = pd.DataFrame(
        [
            {
                "entity_id": "customer_001",
                "source_record_id": "source_001",
                "feature_a": 1.0,
                "feature_b": 2.0,
            },
            {
                "entity_id": "customer_002",
                "source_record_id": "source_002",
                "feature_a": 10.0,
                "feature_b": 20.0,
            },
        ]
    )

    predictions = score_feature_batch(
        features,
        artifacts,
        entity_type="customer",
        source_project="unit_test_project",
        source_table="unit_test_table",
        batch_run_id="batch_test_001",
    )

    assert len(predictions) == 2
    assert all(isinstance(prediction, BatchPredictionRecord) for prediction in predictions)

    first, second = predictions

    assert first.batch_run_id == "batch_test_001"
    assert first.model_name == "isolation_forest"
    assert first.model_version == "vtest"
    assert first.entity_id == "customer_001"
    assert first.source_record_id == "source_001"
    assert first.anomaly_score == 0.25
    assert first.is_anomaly is False
    assert first.threshold_used == 0.0
    assert first.prediction_status == "success"
    assert len(first.feature_payload_hash) == 64

    assert second.entity_id == "customer_002"
    assert second.source_record_id == "source_002"
    assert second.anomaly_score == -0.40
    assert second.is_anomaly is True


def test_write_predictions_jsonl_appends_records(tmp_path: Path) -> None:
    active_path = _build_artifact_bundle(tmp_path)
    artifacts = load_active_model_artifacts(active_path)

    features = pd.DataFrame(
        [
            {
                "entity_id": "customer_001",
                "feature_a": 1.0,
                "feature_b": 2.0,
            }
        ]
    )

    predictions = score_feature_batch(features, artifacts)
    output_path = tmp_path / "batch_predictions.jsonl"

    written_path = write_predictions_jsonl(predictions, output_path)

    assert written_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["model_version"] == "vtest"
    assert record["entity_id"] == "customer_001"
    assert record["prediction_status"] == "success"
    assert record["threshold_used"] == 0.0
    assert len(record["feature_payload_hash"]) == 64
