"""Tests for Project 4 alert event generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from anomaly_detection.alerts import (
    AlertEvent,
    AlertingError,
    append_alert_events_jsonl,
    create_alert_event,
    create_anomaly_rate_alert,
    create_critical_drift_alerts,
    create_latency_budget_alert,
    create_manual_alert,
    create_prediction_error_rate_alert,
    read_alert_events_jsonl,
)


def _critical_drift_result() -> dict:
    return {
        "model_name": "isolation_forest",
        "model_version": "vtest",
        "feature_schema_version": "feature_schema_test",
        "overall_drift_status": "critical",
        "drift_events": [
            {
                "drift_event_id": "drift_critical_001",
                "feature_name": "cart_value",
                "feature_dtype": "float64",
                "baseline_mean": 100.0,
                "current_mean": 145.0,
                "mean_delta": 45.0,
                "mean_delta_percent": 0.45,
                "baseline_variance": 100.0,
                "current_variance": 120.0,
                "variance_delta": 20.0,
                "variance_delta_percent": 0.20,
                "mean_critical_threshold": 0.30,
                "variance_critical_threshold": 0.50,
                "drift_status": "critical",
                "detection_method": "mean_variance_threshold",
                "observation_window_start": "2026-06-11T09:00:00+00:00",
                "observation_window_end": "2026-06-11T10:00:00+00:00",
            },
            {
                "drift_event_id": "drift_warning_001",
                "feature_name": "avg_api_latency_ms",
                "feature_dtype": "float64",
                "baseline_mean": 200.0,
                "current_mean": 245.0,
                "mean_delta": 45.0,
                "mean_delta_percent": 0.225,
                "baseline_variance": 400.0,
                "current_variance": 410.0,
                "variance_delta": 10.0,
                "variance_delta_percent": 0.025,
                "mean_critical_threshold": 0.40,
                "variance_critical_threshold": 0.60,
                "drift_status": "warning",
                "detection_method": "mean_variance_threshold",
                "observation_window_start": "2026-06-11T09:00:00+00:00",
                "observation_window_end": "2026-06-11T10:00:00+00:00",
            },
        ],
    }


def test_create_manual_alert_event() -> None:
    alert = create_manual_alert(
        model_name="isolation_forest",
        model_version="vtest",
        triggered_at="2026-06-11T10:00:00+00:00",
    )

    assert alert.alert_type == "manual_alert"
    assert alert.severity == "info"
    assert alert.alert_status == "open"
    assert alert.alert_source == "project4_monitoring"
    assert alert.model_name == "isolation_forest"
    assert alert.model_version == "vtest"
    assert alert.details["reason"] == "manual_test"
    assert alert.triggered_at == "2026-06-11T10:00:00+00:00"


def test_create_alert_event_rejects_invalid_type() -> None:
    with pytest.raises(AlertingError, match="invalid alert_type"):
        create_alert_event(
            alert_type="fake_alert",
            severity="warning",
            message="bad alert",
        )


def test_create_alert_event_rejects_invalid_severity() -> None:
    with pytest.raises(AlertingError, match="invalid severity"):
        create_alert_event(
            alert_type="manual_alert",
            severity="emergency",
            message="bad severity",
        )


def test_create_critical_drift_alerts_only_emits_critical_events() -> None:
    alerts = create_critical_drift_alerts(_critical_drift_result())

    assert len(alerts) == 1

    alert = alerts[0]

    assert alert.alert_type == "drift_critical"
    assert alert.severity == "critical"
    assert alert.drift_event_id == "drift_critical_001"
    assert alert.model_name == "isolation_forest"
    assert alert.model_version == "vtest"
    assert alert.entity_type == "feature"
    assert alert.entity_id == "cart_value"
    assert alert.metric_name == "feature_drift_delta_percent"
    assert alert.metric_value == 0.45
    assert alert.threshold_value == 0.50
    assert alert.comparison_operator == ">="
    assert alert.details["feature_name"] == "cart_value"
    assert alert.details["mean_delta_percent"] == 0.45


def test_create_anomaly_rate_alert_when_current_rate_exceeds_multiplier() -> None:
    alert = create_anomaly_rate_alert(
        model_name="isolation_forest",
        model_version="vtest",
        baseline_anomaly_rate=0.05,
        current_anomaly_rate=0.13,
        multiplier=2.0,
        observation_window="last_15m",
    )

    assert alert is not None
    assert alert.alert_type == "anomaly_rate_spike"
    assert alert.severity == "warning"
    assert alert.metric_name == "current_anomaly_rate"
    assert alert.metric_value == 0.13
    assert alert.threshold_value == 0.10
    assert alert.comparison_operator == ">"
    assert alert.details["multiplier"] == 2.0
    assert alert.details["observation_window"] == "last_15m"


def test_create_anomaly_rate_alert_returns_none_when_inside_threshold() -> None:
    alert = create_anomaly_rate_alert(
        model_name="isolation_forest",
        model_version="vtest",
        baseline_anomaly_rate=0.05,
        current_anomaly_rate=0.09,
        multiplier=2.0,
    )

    assert alert is None


def test_create_latency_budget_alert_when_p95_exceeds_200_ms() -> None:
    alert = create_latency_budget_alert(
        model_name="isolation_forest",
        model_version="vtest",
        p95_latency_ms=225.5,
        budget_ms=200.0,
        endpoint="/predict",
    )

    assert alert is not None
    assert alert.alert_type == "latency_budget_breach"
    assert alert.severity == "warning"
    assert alert.metric_name == "prediction_latency_p95_ms"
    assert alert.metric_value == 225.5
    assert alert.threshold_value == 200.0
    assert alert.entity_type == "endpoint"
    assert alert.entity_id == "/predict"


def test_create_latency_budget_alert_returns_none_when_inside_budget() -> None:
    alert = create_latency_budget_alert(
        model_name="isolation_forest",
        model_version="vtest",
        p95_latency_ms=120.0,
        budget_ms=200.0,
        endpoint="/predict",
    )

    assert alert is None


def test_create_prediction_error_rate_alert_when_error_rate_exceeds_threshold() -> None:
    alert = create_prediction_error_rate_alert(
        model_name="isolation_forest",
        model_version="vtest",
        error_rate=0.08,
        threshold=0.05,
        endpoint="/predict/batch",
    )

    assert alert is not None
    assert alert.alert_type == "prediction_error_rate_breach"
    assert alert.severity == "warning"
    assert alert.metric_name == "prediction_error_rate"
    assert alert.metric_value == 0.08
    assert alert.threshold_value == 0.05
    assert alert.entity_id == "/predict/batch"


def test_create_prediction_error_rate_alert_returns_none_when_inside_threshold() -> None:
    alert = create_prediction_error_rate_alert(
        model_name="isolation_forest",
        model_version="vtest",
        error_rate=0.01,
        threshold=0.05,
        endpoint="/predict",
    )

    assert alert is None


def test_append_and_read_alert_events_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "anomaly_alerts.jsonl"

    first_alert = create_manual_alert(
        model_name="isolation_forest",
        model_version="vtest",
        triggered_at=datetime(2026, 6, 11, 10, 0, tzinfo=UTC),
    )
    second_alert = create_latency_budget_alert(
        model_name="isolation_forest",
        model_version="vtest",
        p95_latency_ms=250.0,
        budget_ms=200.0,
        endpoint="/predict",
    )

    assert isinstance(second_alert, AlertEvent)

    append_alert_events_jsonl([first_alert], output_path)
    append_alert_events_jsonl([second_alert], output_path)

    records = read_alert_events_jsonl(output_path)

    assert len(records) == 2
    assert records[0]["alert_type"] == "manual_alert"
    assert records[0]["triggered_at"] == "2026-06-11T10:00:00+00:00"
    assert records[1]["alert_type"] == "latency_budget_breach"
    assert records[1]["metric_value"] == 250.0
    assert records[1]["threshold_value"] == 200.0


def test_append_alert_events_jsonl_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(AlertingError, match="no alert events supplied"):
        append_alert_events_jsonl([], tmp_path / "alerts.jsonl")


def test_read_alert_events_jsonl_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AlertingError, match="does not exist"):
        read_alert_events_jsonl(tmp_path / "missing_alerts.jsonl")


def test_drift_result_generates_critical_alert_event() -> None:
    from anomaly_detection.drift import create_alerts_for_drift_result

    alerts = create_alerts_for_drift_result(_critical_drift_result())

    assert len(alerts) == 1
    assert alerts[0].alert_type == "drift_critical"
    assert alerts[0].severity == "critical"
    assert alerts[0].entity_type == "feature"
    assert alerts[0].entity_id == "cart_value"


def test_drift_result_without_critical_events_generates_no_alerts() -> None:
    from anomaly_detection.drift import create_alerts_for_drift_result

    drift_result = _critical_drift_result()
    drift_result["overall_drift_status"] = "warning"
    drift_result["drift_events"][0]["drift_status"] = "warning"

    alerts = create_alerts_for_drift_result(drift_result)

    assert alerts == []


def test_append_alerts_for_drift_result_jsonl_writes_critical_alert(
    tmp_path: Path,
) -> None:
    from anomaly_detection.drift import append_alerts_for_drift_result_jsonl

    output_path = tmp_path / "anomaly_alerts.jsonl"

    written_path = append_alerts_for_drift_result_jsonl(
        _critical_drift_result(),
        output_path,
    )

    assert written_path == output_path

    records = read_alert_events_jsonl(output_path)

    assert len(records) == 1
    assert records[0]["alert_type"] == "drift_critical"
    assert records[0]["severity"] == "critical"
    assert records[0]["entity_id"] == "cart_value"


def test_append_alerts_for_drift_result_jsonl_returns_none_without_alerts(
    tmp_path: Path,
) -> None:
    from anomaly_detection.drift import append_alerts_for_drift_result_jsonl

    drift_result = _critical_drift_result()
    drift_result["overall_drift_status"] = "warning"
    drift_result["drift_events"][0]["drift_status"] = "warning"

    output_path = tmp_path / "anomaly_alerts.jsonl"

    written_path = append_alerts_for_drift_result_jsonl(
        drift_result,
        output_path,
    )

    assert written_path is None
    assert not output_path.exists()
