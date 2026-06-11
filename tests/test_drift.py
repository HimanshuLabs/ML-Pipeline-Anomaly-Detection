"""Tests for mean/variance drift monitoring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection.drift import (
    DriftMonitoringError,
    DriftThresholds,
    FeatureCurrentStats,
    append_drift_events_jsonl,
    calculate_current_feature_stats,
    evaluate_drift,
    evaluate_feature_drift,
    get_thresholds_for_feature,
    read_drift_events_jsonl,
)


def _baseline_stats() -> dict:
    return {
        "model_name": "isolation_forest",
        "model_version": "vtest",
        "feature_schema_version": "feature_schema_test",
        "feature_baselines": {
            "cart_value": {
                "mean": 100.0,
                "variance": 100.0,
                "min": 50.0,
                "max": 150.0,
                "missing_count": 0,
            },
            "avg_api_latency_ms": {
                "mean": 200.0,
                "variance": 400.0,
                "min": 100.0,
                "max": 500.0,
                "missing_count": 0,
            },
            "zero_baseline_feature": {
                "mean": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "missing_count": 0,
            },
        },
    }


def _threshold_config() -> dict:
    return {
        "thresholds": {
            "default": {
                "mean_delta_warning": 0.15,
                "mean_delta_critical": 0.30,
                "variance_delta_warning": 0.25,
                "variance_delta_critical": 0.50,
            },
            "feature_overrides": {
                "avg_api_latency_ms": {
                    "mean_delta_warning": 0.20,
                    "mean_delta_critical": 0.40,
                    "variance_delta_warning": 0.30,
                    "variance_delta_critical": 0.60,
                },
            },
        },
    }


def _window() -> tuple[datetime, datetime]:
    start = datetime(2026, 6, 11, 9, 0, tzinfo=UTC)
    end = datetime(2026, 6, 11, 10, 0, tzinfo=UTC)
    return start, end


def test_calculate_current_feature_stats_for_numeric_columns_only() -> None:
    frame = pd.DataFrame(
        {
            "cart_value": [90.0, 100.0, 110.0, None],
            "event_type": ["view", "cart", "purchase", "view"],
        }
    )

    stats = calculate_current_feature_stats(frame)

    assert set(stats) == {"cart_value"}
    assert stats["cart_value"].current_mean == 100.0
    assert round(stats["cart_value"].current_variance, 6) == round(200 / 3, 6)
    assert stats["cart_value"].non_null_count == 3
    assert stats["cart_value"].missing_count == 1


def test_calculate_current_feature_stats_rejects_empty_dataframe() -> None:
    with pytest.raises(DriftMonitoringError, match="empty"):
        calculate_current_feature_stats(pd.DataFrame())


def test_get_thresholds_for_feature_uses_default_thresholds() -> None:
    thresholds = get_thresholds_for_feature(
        "cart_value",
        _threshold_config(),
    )

    assert thresholds == DriftThresholds(
        mean_delta_warning=0.15,
        mean_delta_critical=0.30,
        variance_delta_warning=0.25,
        variance_delta_critical=0.50,
    )


def test_get_thresholds_for_feature_uses_feature_overrides() -> None:
    thresholds = get_thresholds_for_feature(
        "avg_api_latency_ms",
        _threshold_config(),
    )

    assert thresholds == DriftThresholds(
        mean_delta_warning=0.20,
        mean_delta_critical=0.40,
        variance_delta_warning=0.30,
        variance_delta_critical=0.60,
    )


def test_evaluate_feature_drift_returns_normal_status() -> None:
    start, end = _window()

    event = evaluate_feature_drift(
        feature_name="cart_value",
        baseline_feature_stats={"mean": 100.0, "variance": 100.0},
        current_feature_stats=FeatureCurrentStats(
            feature_name="cart_value",
            feature_dtype="float64",
            current_mean=110.0,
            current_variance=110.0,
            non_null_count=10,
            missing_count=0,
        ),
        thresholds=get_thresholds_for_feature("cart_value", _threshold_config()),
        model_name="isolation_forest",
        model_version="vtest",
        feature_schema_version="feature_schema_test",
        observation_window_start=start,
        observation_window_end=end,
    )

    assert event.drift_status == "normal"
    assert event.mean_delta == 10.0
    assert event.mean_delta_percent == 0.10
    assert event.variance_delta_percent == 0.10


def test_evaluate_feature_drift_returns_warning_status() -> None:
    start, end = _window()

    event = evaluate_feature_drift(
        feature_name="cart_value",
        baseline_feature_stats={"mean": 100.0, "variance": 100.0},
        current_feature_stats=FeatureCurrentStats(
            feature_name="cart_value",
            feature_dtype="float64",
            current_mean=120.0,
            current_variance=110.0,
            non_null_count=10,
            missing_count=0,
        ),
        thresholds=get_thresholds_for_feature("cart_value", _threshold_config()),
        model_name="isolation_forest",
        model_version="vtest",
        feature_schema_version="feature_schema_test",
        observation_window_start=start,
        observation_window_end=end,
    )

    assert event.drift_status == "warning"
    assert event.mean_delta_percent == 0.20


def test_evaluate_feature_drift_returns_critical_status() -> None:
    start, end = _window()

    event = evaluate_feature_drift(
        feature_name="cart_value",
        baseline_feature_stats={"mean": 100.0, "variance": 100.0},
        current_feature_stats=FeatureCurrentStats(
            feature_name="cart_value",
            feature_dtype="float64",
            current_mean=140.0,
            current_variance=110.0,
            non_null_count=10,
            missing_count=0,
        ),
        thresholds=get_thresholds_for_feature("cart_value", _threshold_config()),
        model_name="isolation_forest",
        model_version="vtest",
        feature_schema_version="feature_schema_test",
        observation_window_start=start,
        observation_window_end=end,
    )

    assert event.drift_status == "critical"
    assert event.mean_delta_percent == 0.40


def test_evaluate_feature_drift_can_be_variance_driven() -> None:
    start, end = _window()

    event = evaluate_feature_drift(
        feature_name="cart_value",
        baseline_feature_stats={"mean": 100.0, "variance": 100.0},
        current_feature_stats=FeatureCurrentStats(
            feature_name="cart_value",
            feature_dtype="float64",
            current_mean=100.0,
            current_variance=160.0,
            non_null_count=10,
            missing_count=0,
        ),
        thresholds=get_thresholds_for_feature("cart_value", _threshold_config()),
        model_name="isolation_forest",
        model_version="vtest",
        feature_schema_version="feature_schema_test",
        observation_window_start=start,
        observation_window_end=end,
    )

    assert event.drift_status == "critical"
    assert event.mean_delta_percent == 0.0
    assert event.variance_delta_percent == 0.60


def test_evaluate_feature_drift_handles_zero_baseline_with_absolute_fallback() -> None:
    start, end = _window()

    event = evaluate_feature_drift(
        feature_name="zero_baseline_feature",
        baseline_feature_stats={"mean": 0.0, "variance": 0.0},
        current_feature_stats=FeatureCurrentStats(
            feature_name="zero_baseline_feature",
            feature_dtype="float64",
            current_mean=0.20,
            current_variance=0.0,
            non_null_count=10,
            missing_count=0,
        ),
        thresholds=get_thresholds_for_feature(
            "zero_baseline_feature",
            _threshold_config(),
        ),
        model_name="isolation_forest",
        model_version="vtest",
        feature_schema_version="feature_schema_test",
        observation_window_start=start,
        observation_window_end=end,
    )

    assert event.drift_status == "warning"
    assert event.mean_delta == 0.20
    assert event.mean_delta_percent == 0.20


def test_evaluate_drift_returns_overall_warning_status() -> None:
    start, end = _window()

    frame = pd.DataFrame(
        {
            "cart_value": [110.0, 110.0, 130.0, 130.0],
            "unknown_numeric_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = evaluate_drift(
        current_features=frame,
        baseline_stats=_baseline_stats(),
        threshold_config=_threshold_config(),
        observation_window_start=start,
        observation_window_end=end,
    )

    assert result.model_name == "isolation_forest"
    assert result.model_version == "vtest"
    assert result.feature_schema_version == "feature_schema_test"
    assert result.evaluated_feature_count == 1
    assert result.warning_feature_count == 1
    assert result.critical_feature_count == 0
    assert result.overall_drift_status == "warning"
    assert result.drift_events[0].feature_name == "cart_value"


def test_evaluate_drift_returns_overall_critical_status() -> None:
    start, end = _window()

    frame = pd.DataFrame(
        {
            "cart_value": [130.0, 130.0, 150.0, 150.0],
            "avg_api_latency_ms": [180.0, 180.0, 220.0, 220.0],
        }
    )

    result = evaluate_drift(
        current_features=frame,
        baseline_stats=_baseline_stats(),
        threshold_config=_threshold_config(),
        observation_window_start=start,
        observation_window_end=end,
    )

    assert result.evaluated_feature_count == 2
    assert result.critical_feature_count == 1
    assert result.normal_feature_count == 1
    assert result.overall_drift_status == "critical"


def test_evaluate_drift_rejects_window_end_before_start() -> None:
    start = datetime(2026, 6, 11, 10, 0, tzinfo=UTC)
    end = datetime(2026, 6, 11, 9, 0, tzinfo=UTC)

    with pytest.raises(DriftMonitoringError, match="observation_window_end"):
        evaluate_drift(
            current_features=pd.DataFrame({"cart_value": [100.0]}),
            baseline_stats=_baseline_stats(),
            threshold_config=_threshold_config(),
            observation_window_start=start,
            observation_window_end=end,
        )


def test_evaluate_drift_rejects_no_overlapping_numeric_features() -> None:
    start, end = _window()

    with pytest.raises(DriftMonitoringError, match="no overlapping numeric"):
        evaluate_drift(
            current_features=pd.DataFrame({"unknown_feature": [1.0, 2.0]}),
            baseline_stats=_baseline_stats(),
            threshold_config=_threshold_config(),
            observation_window_start=start,
            observation_window_end=end,
        )


def test_append_and_read_drift_events_jsonl(tmp_path: Path) -> None:
    start, end = _window()

    result = evaluate_drift(
        current_features=pd.DataFrame({"cart_value": [110.0, 110.0, 130.0, 130.0]}),
        baseline_stats=_baseline_stats(),
        threshold_config=_threshold_config(),
        observation_window_start=start,
        observation_window_end=end,
    )

    output_path = tmp_path / "drift_events.jsonl"

    written_path = append_drift_events_jsonl(result.drift_events, output_path)
    records = read_drift_events_jsonl(written_path)

    assert written_path == output_path
    assert len(records) == 1
    assert records[0]["model_name"] == "isolation_forest"
    assert records[0]["model_version"] == "vtest"
    assert records[0]["feature_name"] == "cart_value"
    assert records[0]["drift_status"] == "warning"
    assert records[0]["detection_method"] == "mean_variance_threshold"


def test_read_drift_events_jsonl_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DriftMonitoringError, match="does not exist"):
        read_drift_events_jsonl(tmp_path / "missing.jsonl")
