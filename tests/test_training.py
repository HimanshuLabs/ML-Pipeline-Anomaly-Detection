"""Tests for Isolation Forest training pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection.training import (
    TrainingError,
    load_trained_model,
    resolve_next_model_version,
    train_isolation_forest_from_snapshot,
)
from anomaly_detection.training_snapshot import (
    TrainingSnapshotConfig,
    create_training_snapshot,
)


def _sample_feature_frame(row_count: int = 40) -> pd.DataFrame:
    """Build deterministic, contract-valid feature data for training tests."""
    rows = []

    for index in range(row_count):
        is_outlier = index >= row_count - 2
        multiplier = 8.0 if is_outlier else 1.0

        rows.append(
            {
                "entity_id": f"user_{index:03d}",
                "feature_timestamp": pd.Timestamp("2026-06-01T00:00:00Z")
                + pd.Timedelta(minutes=index),
                "schema_version": "feature_schema_v001",
                "avg_cart_value_7d": 100.0 * multiplier + index,
                "event_count_1h": 3.0 * multiplier + (index % 4),
                "avg_api_latency_ms": 120.0 * multiplier + index,
                "fraud_score_avg": min(0.99, 0.02 * multiplier + (index % 3) * 0.01),
                "purchase_probability_delta": min(
                    0.99,
                    0.05 * multiplier + (index % 5) * 0.01,
                ),
                "cart_abandonment_rate": min(
                    0.99,
                    0.10 * multiplier + (index % 4) * 0.02,
                ),
                "campaign_roas": 2.0 * multiplier + (index % 5) * 0.1,
                "conversion_rate": min(0.99, 0.20 + (index % 5) * 0.03),
                "customer_lifetime_value": 500.0 * multiplier + index * 2.0,
                "discount_sensitivity": 0.40 * multiplier + (index % 3) * 0.05,
                "page_load_p95_ms": 250.0 * multiplier + index,
            }
        )

    return pd.DataFrame(rows)


def test_train_isolation_forest_from_snapshot_writes_versioned_artifacts(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    artifact_root = tmp_path / "artifacts"

    snapshot_metadata = create_training_snapshot(
        _sample_feature_frame(),
        snapshot_config=TrainingSnapshotConfig(output_root=snapshot_root),
        source_tables=["project4_test_features"],
        source_project_versions={
            "project1": "test",
            "project2_3": "test",
        },
        created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )

    artifact_metadata = train_isolation_forest_from_snapshot(
        features_path=snapshot_metadata.features_path,
        snapshot_metadata_path=snapshot_metadata.metadata_path,
        model_version="v001",
        artifact_root_override=artifact_root,
        overwrite=True,
        training_started_at_utc=datetime(2026, 6, 8, 12, 5, 0, tzinfo=UTC),
        registry_path=tmp_path / "empty_registry.json",
    )

    artifact_dir = artifact_root / "isolation_forest" / "model_version=v001"

    expected_files = {
        artifact_dir / "model.joblib",
        artifact_dir / "metadata.json",
        artifact_dir / "feature_schema.json",
        artifact_dir / "baseline_stats.json",
    }

    for path in expected_files:
        assert path.exists(), f"Missing artifact: {path}"

    assert artifact_metadata.model_name == "isolation_forest"
    assert artifact_metadata.model_version == "v001"
    assert artifact_metadata.algorithm == "IsolationForest"
    assert artifact_metadata.row_count == 40
    assert artifact_metadata.feature_schema_version == "feature_schema_v001"
    assert artifact_metadata.baseline_anomaly_rate >= 0.0
    assert artifact_metadata.baseline_anomaly_rate <= 1.0
    assert (
        artifact_metadata.training_snapshot_metadata["snapshot_id"]
        == snapshot_metadata.snapshot_id
    )

    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata_payload = json.load(file)

    assert metadata_payload["model_name"] == "isolation_forest"
    assert metadata_payload["model_version"] == "v001"
    assert metadata_payload["row_count"] == 40
    assert metadata_payload["model_parameters"]["n_estimators"] == 200
    assert metadata_payload["model_parameters"]["random_state"] == 42

    with (artifact_dir / "feature_schema.json").open("r", encoding="utf-8") as file:
        schema_payload = json.load(file)

    assert schema_payload["feature_schema_version"] == "feature_schema_v001"
    assert schema_payload["feature_count"] == 11
    assert schema_payload["feature_names"] == [
        "avg_cart_value_7d",
        "event_count_1h",
        "avg_api_latency_ms",
        "fraud_score_avg",
        "purchase_probability_delta",
        "cart_abandonment_rate",
        "campaign_roas",
        "conversion_rate",
        "customer_lifetime_value",
        "discount_sensitivity",
        "page_load_p95_ms",
    ]

    with (artifact_dir / "baseline_stats.json").open("r", encoding="utf-8") as file:
        baseline_payload = json.load(file)

    assert baseline_payload["model_name"] == "isolation_forest"
    assert baseline_payload["model_version"] == "v001"
    assert baseline_payload["metric_type"] == "unsupervised_training_baseline"
    assert baseline_payload["label_availability"] == "unlabeled_proxy_metrics"
    assert baseline_payload["prediction_summary"]["row_count"] == 40
    assert baseline_payload["baseline_anomaly_rate"] >= 0.0
    assert baseline_payload["baseline_anomaly_rate"] <= 1.0
    assert set(baseline_payload["feature_baselines"]) == set(schema_payload["feature_names"])

    loaded_model = load_trained_model(artifact_dir / "model.joblib")
    predictions = loaded_model.predict(
        _sample_feature_frame().loc[:, schema_payload["feature_names"]],
    )
    assert len(predictions) == 40


def test_resolve_next_model_version_uses_existing_artifacts(tmp_path: Path) -> None:
    model_root = tmp_path / "artifacts" / "isolation_forest"
    (model_root / "model_version=v001").mkdir(parents=True)
    (model_root / "model_version=v002").mkdir(parents=True)

    assert resolve_next_model_version(model_root) == "v003"


def test_training_fails_for_missing_snapshot_file(tmp_path: Path) -> None:
    missing_features_path = tmp_path / "missing" / "features.parquet"

    with pytest.raises(TrainingError, match="does not exist"):
        train_isolation_forest_from_snapshot(
            features_path=missing_features_path,
            artifact_root_override=tmp_path / "artifacts",
        )


def test_build_training_feature_matrix_uses_real_source_metadata_columns() -> None:
    from anomaly_detection.training import build_training_feature_matrix

    frame = pd.DataFrame(
        {
            "entity_id": ["user_1", "user_2"],
            "source_project": ["project_1", "project_2_3"],
            "total_events": [10, 20],
            "avg_api_latency_ms": [120.0, 180.0],
            "avg_fraud_score": [0.02, 0.08],
        }
    )

    matrix = build_training_feature_matrix(
        frame,
        snapshot_metadata={
            "snapshot_type": "real_source_extract",
            "feature_columns": [
                "total_events",
                "avg_api_latency_ms",
                "avg_fraud_score",
            ],
        },
    )

    assert list(matrix.columns) == [
        "total_events",
        "avg_api_latency_ms",
        "avg_fraud_score",
    ]
    assert len(matrix) == 2


def test_resolve_snapshot_paths_supports_latest_snapshot(tmp_path: Path) -> None:
    from anomaly_detection.training import resolve_snapshot_paths

    snapshot_dir = (
        tmp_path
        / "snapshot_date=2026-06-10"
        / "snapshot_id=real_source_20260610T060738Z"
    )
    snapshot_dir.mkdir(parents=True)

    features_path = snapshot_dir / "features.parquet"
    metadata_path = snapshot_dir / "metadata.json"

    pd.DataFrame({"value": [1]}).to_parquet(features_path, index=False)
    metadata_path.write_text('{"snapshot_id": "real_source_20260610T060738Z"}', encoding="utf-8")

    resolved_features_path, resolved_metadata_path = resolve_snapshot_paths(
        "latest",
        training_root=tmp_path,
    )

    assert resolved_features_path == features_path
    assert resolved_metadata_path == metadata_path


# --- reproducibility: runtime versions, fingerprints, schema labels ----------


def _feature_set_fingerprint(feature_names: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def _demo_snapshot(tmp_path: Path):
    return create_training_snapshot(
        _sample_feature_frame(),
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path / "snapshots"),
        source_tables=["project4_test_features"],
        created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )


def test_training_records_runtime_versions_and_feature_set_fingerprint(
    tmp_path: Path,
) -> None:
    import platform

    import joblib
    import numpy
    import sklearn

    snapshot = _demo_snapshot(tmp_path)

    artifact = train_isolation_forest_from_snapshot(
        features_path=snapshot.features_path,
        snapshot_metadata_path=snapshot.metadata_path,
        model_version="v001",
        artifact_root_override=tmp_path / "artifacts",
        registry_path=tmp_path / "empty_registry.json",
    )

    metadata = json.loads(Path(artifact.metadata_path).read_text(encoding="utf-8"))
    schema = json.loads(Path(artifact.feature_schema_path).read_text(encoding="utf-8"))
    expected_fingerprint = _feature_set_fingerprint(metadata["feature_names"])

    assert metadata["runtime_versions"] == {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    assert metadata["feature_set_fingerprint"] == expected_fingerprint
    assert schema["feature_set_fingerprint"] == expected_fingerprint
    assert schema["source"] == snapshot.snapshot_id
    assert schema["timestamp_column"] == snapshot.timestamp_column


def test_feature_schema_source_and_timestamp_come_from_snapshot_metadata(
    tmp_path: Path,
) -> None:
    from anomaly_detection.source_extract import SEGMENT_FEATURE_COLUMNS

    # A one-entity-type real-source snapshot (real-source training is segmented).
    feature_columns = list(SEGMENT_FEATURE_COLUMNS["user"])
    snapshot_dir = tmp_path / "snapshot_id=real_source_test_001"
    snapshot_dir.mkdir()
    pd.DataFrame(
        {
            "entity_type": ["user"] * 40,
            "entity_id": [f"entity_{index}" for index in range(40)],
            **{
                column: [float((index * (position + 3)) % 11) for index in range(40)]
                for position, column in enumerate(feature_columns)
            },
        }
    ).to_parquet(snapshot_dir / "features.parquet", index=False)
    (snapshot_dir / "metadata.json").write_text(
        json.dumps(
            {
                "snapshot_id": "real_source_test_001",
                "snapshot_type": "real_source_extract",
                "feature_columns": feature_columns,
            }
        ),
        encoding="utf-8",
    )

    artifact = train_isolation_forest_from_snapshot(
        features_path=snapshot_dir / "features.parquet",
        snapshot_metadata_path=snapshot_dir / "metadata.json",
        model_version="v001",
        artifact_root_override=tmp_path / "artifacts",
        registry_path=tmp_path / "empty_registry.json",
    )

    schema = json.loads(Path(artifact.feature_schema_path).read_text(encoding="utf-8"))

    assert schema["source"] == "real_source_test_001"
    assert schema["timestamp_column"] is None
    assert schema["feature_names"] == feature_columns


def _registry_with_schema_label(
    tmp_path: Path,
    feature_names: list[str],
    feature_schema_version: str = "feature_schema_v001",
) -> Path:
    import joblib

    from anomaly_detection.registry import register_model

    registry_path = tmp_path / "registry" / "model_registry.json"
    artifact_dir = tmp_path / "existing_model"
    artifact_dir.mkdir()
    joblib.dump({"fake_model": "v900"}, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"feature_names": feature_names}),
        encoding="utf-8",
    )
    register_model(
        model_name="isolation_forest",
        model_version="v900",
        artifact_path=str(artifact_dir / "model.joblib"),
        dataset_snapshot_id="snapshot_v900",
        feature_schema_version=feature_schema_version,
        training_timestamp="2026-06-10T10:00:00+00:00",
        registry_path=registry_path,
    )
    return registry_path


def _configured_feature_names() -> list[str]:
    from anomaly_detection.feature_validation import load_feature_validation_config

    return list(load_feature_validation_config().required_numeric_features)


@pytest.mark.parametrize(
    "existing_feature_names",
    [
        pytest.param(["some_other_feature", "another_feature"], id="different_names"),
        pytest.param("reversed", id="same_names_different_order"),
    ],
)
def test_training_refuses_schema_label_used_for_different_feature_names(
    tmp_path: Path,
    existing_feature_names,
) -> None:
    if existing_feature_names == "reversed":
        existing_feature_names = list(reversed(_configured_feature_names()))
    registry_path = _registry_with_schema_label(tmp_path, existing_feature_names)
    snapshot = _demo_snapshot(tmp_path)
    artifact_root = tmp_path / "artifacts"

    with pytest.raises(TrainingError, match="feature_schema_v001.*v900"):
        train_isolation_forest_from_snapshot(
            features_path=snapshot.features_path,
            snapshot_metadata_path=snapshot.metadata_path,
            model_version="v001",
            artifact_root_override=artifact_root,
            registry_path=registry_path,
        )

    assert not (artifact_root / "isolation_forest" / "model_version=v001").exists()


def test_training_allows_schema_label_used_for_identical_feature_names(
    tmp_path: Path,
) -> None:
    registry_path = _registry_with_schema_label(tmp_path, _configured_feature_names())
    snapshot = _demo_snapshot(tmp_path)

    artifact = train_isolation_forest_from_snapshot(
        features_path=snapshot.features_path,
        snapshot_metadata_path=snapshot.metadata_path,
        model_version="v001",
        artifact_root_override=tmp_path / "artifacts",
        registry_path=registry_path,
    )

    assert artifact.feature_schema_version == "feature_schema_v001"
