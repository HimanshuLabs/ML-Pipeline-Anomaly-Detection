from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection.feature_validation import FeatureValidationError
from anomaly_detection.training_snapshot import (
    TrainingSnapshotConfig,
    create_training_snapshot,
    load_training_snapshot_metadata,
    metadata_to_database_records,
    read_training_snapshot,
)


def _sample_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity_id": "user_002",
                "feature_timestamp": "2026-06-07T11:00:00+00:00",
                "schema_version": "feature_schema_v001",
                "avg_cart_value_7d": 1200.50,
                "event_count_1h": 4.0,
                "avg_api_latency_ms": 95.0,
                "fraud_score_avg": 0.12,
                "purchase_probability_delta": 0.05,
                "cart_abandonment_rate": 0.20,
                "discount_sensitivity": 0.35,
                "page_load_p95_ms": 320.0,
                "campaign_roas": 2.8,
                "conversion_rate": 0.14,
                "customer_lifetime_value": 18000.0,
            },
            {
                "entity_id": "user_001",
                "feature_timestamp": "2026-06-07T10:00:00+00:00",
                "schema_version": "feature_schema_v001",
                "avg_cart_value_7d": 850.00,
                "event_count_1h": 2.0,
                "avg_api_latency_ms": 110.0,
                "fraud_score_avg": 0.08,
                "purchase_probability_delta": 0.03,
                "cart_abandonment_rate": 0.30,
                "discount_sensitivity": 0.22,
                "page_load_p95_ms": 410.0,
                "campaign_roas": 1.9,
                "conversion_rate": 0.10,
                "customer_lifetime_value": 9500.0,
            },
        ]
    )


def test_create_training_snapshot_writes_parquet_and_metadata(tmp_path: Path) -> None:
    frame = _sample_feature_frame()

    metadata = create_training_snapshot(
        frame,
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
        source_tables=["project1.user_events_gold", "project2.marts.customer_360"],
        source_project_versions={
            "project1": "streaming_features_v001",
            "project2_3": "warehouse_marts_v001",
        },
    )

    features_path = Path(metadata.features_path)
    metadata_path = Path(metadata.metadata_path)

    assert features_path.exists()
    assert metadata_path.exists()

    assert metadata.row_count == 2
    assert metadata.feature_count == 11
    assert metadata.feature_schema_version == "feature_schema_v001"
    assert metadata.snapshot_date == "2026-06-07"
    assert metadata.data_quality_status == "passed"
    assert metadata.source_min_timestamp == "2026-06-07T10:00:00+00:00"
    assert metadata.source_max_timestamp == "2026-06-07T11:00:00+00:00"
    assert metadata.source_tables == [
        "project1.user_events_gold",
        "project2.marts.customer_360",
    ]

    written_frame = read_training_snapshot(features_path)
    assert len(written_frame) == 2
    assert list(written_frame["entity_id"]) == ["user_001", "user_002"]

    loaded_metadata = load_training_snapshot_metadata(metadata_path)
    assert loaded_metadata.snapshot_id == metadata.snapshot_id
    assert loaded_metadata.training_dataset_id == metadata.training_dataset_id
    assert loaded_metadata.content_hash == metadata.content_hash


def test_training_snapshot_id_is_reproducible_for_same_content(tmp_path: Path) -> None:
    frame = _sample_feature_frame()
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)

    first_metadata = create_training_snapshot(
        frame,
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
    )

    second_metadata = create_training_snapshot(
        reversed_frame,
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
    )

    assert second_metadata.snapshot_id == first_metadata.snapshot_id
    assert second_metadata.training_dataset_id == first_metadata.training_dataset_id
    assert second_metadata.content_hash == first_metadata.content_hash


def test_training_snapshot_rejects_invalid_feature_schema(tmp_path: Path) -> None:
    frame = _sample_feature_frame()
    frame.loc[0, "schema_version"] = "wrong_schema"

    with pytest.raises(FeatureValidationError):
        create_training_snapshot(
            frame,
            snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
        )


def test_training_snapshot_metadata_maps_to_database_records(tmp_path: Path) -> None:
    metadata = create_training_snapshot(
        _sample_feature_frame(),
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
    )

    records = metadata_to_database_records(metadata)

    assert set(records) == {"ml.feature_snapshots", "ml.training_datasets"}

    feature_snapshot = records["ml.feature_snapshots"]
    training_dataset = records["ml.training_datasets"]

    assert feature_snapshot["snapshot_id"] == metadata.snapshot_id
    assert feature_snapshot["row_count"] == 2
    assert feature_snapshot["feature_schema_version"] == "feature_schema_v001"
    assert feature_snapshot["snapshot_path"] == metadata.snapshot_path

    assert training_dataset["training_dataset_id"] == metadata.training_dataset_id
    assert training_dataset["snapshot_id"] == metadata.snapshot_id
    assert training_dataset["row_count"] == 2
    assert training_dataset["feature_columns"] == metadata.feature_columns
