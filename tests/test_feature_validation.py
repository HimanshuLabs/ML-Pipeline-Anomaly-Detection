from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anomaly_detection.feature_validation import (  # noqa: E402
    FeatureValidationConfig,
    FeatureValidationError,
    get_model_feature_matrix,
    load_feature_validation_config,
    validate_feature_dataframe,
)


def _validation_config() -> FeatureValidationConfig:
    return FeatureValidationConfig(
        schema_version="feature_schema_v001",
        entity_key="entity_id",
        timestamp_column="feature_timestamp",
        required_numeric_features=(
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
        ),
    )


def _valid_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["user_001", "user_002", "user_003"],
            "feature_timestamp": [
                "2026-06-08T05:00:00+05:30",
                "2026-06-08T05:01:00+05:30",
                "2026-06-08T05:02:00+05:30",
            ],
            "schema_version": [
                "feature_schema_v001",
                "feature_schema_v001",
                "feature_schema_v001",
            ],
            "avg_cart_value_7d": [1200.0, 800.0, 1500.0],
            "event_count_1h": [12, 8, 20],
            "avg_api_latency_ms": [85.0, 95.0, 110.0],
            "fraud_score_avg": [0.10, 0.20, 0.35],
            "purchase_probability_delta": [0.05, 0.10, 0.15],
            "cart_abandonment_rate": [0.30, 0.25, 0.40],
            "campaign_roas": [2.5, 1.8, 3.2],
            "conversion_rate": [0.08, 0.05, 0.11],
            "customer_lifetime_value": [15000.0, 9000.0, 22000.0],
            "discount_sensitivity": [0.20, 0.10, 0.30],
            "page_load_p95_ms": [1200.0, 980.0, 1500.0],
        }
    )


def test_loads_feature_validation_config_from_model_config() -> None:
    config = load_feature_validation_config(PROJECT_ROOT / "configs/model_config.yaml")

    assert config.schema_version == "feature_schema_v001"
    assert config.entity_key == "entity_id"
    assert config.timestamp_column == "feature_timestamp"
    assert "avg_cart_value_7d" in config.required_numeric_features


def test_valid_feature_dataframe_passes_contract() -> None:
    result = validate_feature_dataframe(
        _valid_feature_frame(),
        config=_validation_config(),
    )

    assert result.is_valid is True
    assert result.row_count == 3
    assert result.feature_count == 11
    assert result.schema_version == "feature_schema_v001"
    assert result.errors == ()


def test_missing_required_feature_fails_validation() -> None:
    frame = _valid_feature_frame().drop(columns=["campaign_roas"])

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("Missing required feature columns" in error for error in result.errors)
    assert any("campaign_roas" in error for error in result.errors)


def test_schema_version_mismatch_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame.loc[0, "schema_version"] = "feature_schema_v999"

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("Schema version mismatch" in error for error in result.errors)


def test_null_explosion_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame.loc[:, "avg_api_latency_ms"] = [np.nan, np.nan, 110.0]

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("null fraction" in error for error in result.errors)


def test_non_numeric_model_feature_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame["avg_cart_value_7d"] = frame["avg_cart_value_7d"].astype("object")
    frame.loc[1, "avg_cart_value_7d"] = "bad-cart-value"

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("non-numeric" in error for error in result.errors)


def test_infinite_value_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame.loc[1, "page_load_p95_ms"] = np.inf

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("infinite" in error for error in result.errors)


def test_negative_impossible_value_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame.loc[1, "event_count_1h"] = -3

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("negative values" in error for error in result.errors)


def test_probability_out_of_range_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame.loc[1, "fraud_score_avg"] = 1.5

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("probability range" in error for error in result.errors)


def test_invalid_timestamp_fails_validation() -> None:
    frame = _valid_feature_frame()
    frame.loc[1, "feature_timestamp"] = "not-a-timestamp"

    result = validate_feature_dataframe(frame, config=_validation_config())

    assert result.is_valid is False
    assert any("Timestamp column" in error for error in result.errors)


def test_strict_mode_raises_validation_error() -> None:
    frame = _valid_feature_frame().drop(columns=["avg_cart_value_7d"])

    with pytest.raises(FeatureValidationError):
        validate_feature_dataframe(
            frame,
            config=_validation_config(),
            strict=True,
        )


def test_model_feature_matrix_returns_numeric_features_in_contract_order() -> None:
    config = _validation_config()
    matrix = get_model_feature_matrix(
        _valid_feature_frame(),
        config=config,
        strict=True,
    )

    assert list(matrix.columns) == list(config.required_numeric_features)
    assert matrix.shape == (3, 11)


def _raw_project1_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["evt_001", "evt_002", "evt_003", "evt_004"],
            "session_id": ["s1", "s1", "s2", "s3"],
            "user_id": ["user_001", "user_001", "user_002", "user_003"],
            "event_timestamp": [
                "2026-06-08T04:45:00+05:30",
                "2026-06-08T04:55:00+05:30",
                "2026-06-08T04:35:00+05:30",
                "2026-06-07T03:00:00+05:30",
            ],
            "event_type": ["view", "cart_add", "purchase", "view"],
            "cart_value": [1000.0, 1500.0, 800.0, 500.0],
            "api_latency_ms": [80.0, 100.0, 120.0, 90.0],
            "page_load_time_ms": [900.0, 1100.0, 1400.0, 1000.0],
            "fraud_score": [0.10, 0.20, 0.30, 0.05],
            "purchase_probability": [0.30, 0.45, 0.60, 0.20],
            "cart_abandonment_probability": [0.25, 0.35, 0.20, 0.50],
            "discount_percent": [10.0, 20.0, 5.0, 0.0],
            "schema_version": ["v1", "v1", "v1", "v1"],
            "source": ["project1", "project1", "project1", "project1"],
        }
    )


def _warehouse_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["user_001", "user_002", "user_003"],
            "campaign_roas": [2.5, 1.5, 3.0],
            "conversion_rate": [0.10, 0.08, 0.12],
            "customer_lifetime_value": [15000.0, 9000.0, 22000.0],
        }
    )


def test_build_anomaly_features_from_events_and_warehouse_signals() -> None:
    from anomaly_detection.feature_builder import build_anomaly_features

    config = _validation_config()

    features = build_anomaly_features(
        _raw_project1_events(),
        _warehouse_signals(),
        validation_config=config,
        reference_timestamp="2026-06-08T05:00:00+05:30",
    )

    assert list(features.columns) == [
        "entity_id",
        "feature_timestamp",
        "schema_version",
        *config.required_numeric_features,
    ]

    assert features.shape == (3, 14)

    result = validate_feature_dataframe(features, config=config, strict=False)

    assert result.is_valid is True
    assert result.errors == ()


def test_build_anomaly_features_counts_only_last_hour_events() -> None:
    from anomaly_detection.feature_builder import build_anomaly_features

    features = build_anomaly_features(
        _raw_project1_events(),
        _warehouse_signals(),
        validation_config=_validation_config(),
        reference_timestamp="2026-06-08T05:00:00+05:30",
    )

    indexed = features.set_index("entity_id")

    assert indexed.loc["user_001", "event_count_1h"] == 2.0
    assert indexed.loc["user_002", "event_count_1h"] == 1.0
    assert indexed.loc["user_003", "event_count_1h"] == 0.0


def test_build_anomaly_features_fails_on_missing_entity_column() -> None:
    from anomaly_detection.feature_builder import FeatureBuildError, build_anomaly_features

    bad_events = _raw_project1_events().drop(columns=["user_id"])

    with pytest.raises(FeatureBuildError):
        build_anomaly_features(
            bad_events,
            _warehouse_signals(),
            validation_config=_validation_config(),
        )


def test_build_anomaly_features_fails_on_invalid_event_timestamp() -> None:
    from anomaly_detection.feature_builder import FeatureBuildError, build_anomaly_features

    bad_events = _raw_project1_events()
    bad_events.loc[0, "event_timestamp"] = "not-a-real-timestamp"

    with pytest.raises(FeatureBuildError):
        build_anomaly_features(
            bad_events,
            _warehouse_signals(),
            validation_config=_validation_config(),
        )
