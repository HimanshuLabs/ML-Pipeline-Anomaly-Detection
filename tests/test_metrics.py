"""Tests for Project 4 Prometheus metrics."""

from __future__ import annotations

from dataclasses import dataclass

from anomaly_detection.metrics import (
    observe_prediction_latency_ms,
    prometheus_content_type,
    publish_drift_evaluation_metrics,
    record_anomaly_detected,
    record_drift_event,
    record_model_rollback,
    record_prediction_error,
    record_prediction_request,
    render_prometheus_metrics,
    set_active_model_version,
)


@dataclass(frozen=True)
class DummyDriftEvent:
    """Small drift event shape matching the production drift dataclass contract."""

    feature_name: str
    drift_status: str
    mean_delta: float
    variance_delta: float


@dataclass(frozen=True)
class DummyDriftResult:
    """Small drift result shape matching DriftEvaluationResult enough for metrics."""

    model_version: str
    drift_events: list[DummyDriftEvent]


def _metrics_text() -> str:
    return render_prometheus_metrics().decode("utf-8")


def test_prediction_metrics_are_exposed() -> None:
    record_prediction_request(
        endpoint="/predict",
        model_version="vmetrics",
        status="success",
    )
    record_prediction_error(
        endpoint="/predict",
        model_version="vmetrics",
    )
    record_anomaly_detected(
        endpoint="/predict",
        model_version="vmetrics",
    )
    observe_prediction_latency_ms(
        endpoint="/predict",
        model_version="vmetrics",
        latency_ms=42.5,
    )

    metrics_text = _metrics_text()

    assert "prediction_requests_total" in metrics_text
    assert 'endpoint="/predict"' in metrics_text
    assert 'model_version="vmetrics"' in metrics_text
    assert 'status="success"' in metrics_text
    assert "prediction_errors_total" in metrics_text
    assert "anomalies_detected_total" in metrics_text
    assert "prediction_latency_ms" in metrics_text


def test_active_model_version_metric_is_exposed() -> None:
    set_active_model_version(
        model_name="isolation_forest",
        model_version="vmetrics",
    )

    metrics_text = _metrics_text()

    assert "active_model_version" in metrics_text
    assert 'model_name="isolation_forest"' in metrics_text
    assert 'model_version="vmetrics"' in metrics_text


def test_drift_metrics_are_exposed_for_warning_and_critical_events() -> None:
    record_drift_event(
        model_version="vmetrics",
        feature_name="api_latency_ms",
        drift_status="warning",
        mean_delta=7.25,
        variance_delta=13.5,
    )
    record_drift_event(
        model_version="vmetrics",
        feature_name="fraud_score_avg",
        drift_status="critical",
        mean_delta=0.4,
        variance_delta=0.8,
    )

    metrics_text = _metrics_text()

    assert "drift_detected_total" in metrics_text
    assert "feature_mean_delta" in metrics_text
    assert "feature_variance_delta" in metrics_text
    assert 'feature_name="api_latency_ms"' in metrics_text
    assert 'feature_name="fraud_score_avg"' in metrics_text
    assert 'drift_status="warning"' in metrics_text
    assert 'drift_status="critical"' in metrics_text


def test_normal_drift_updates_delta_gauges_without_incrementing_drift_counter() -> None:
    record_drift_event(
        model_version="vmetrics",
        feature_name="stable_feature",
        drift_status="normal",
        mean_delta=0.01,
        variance_delta=0.02,
    )

    metrics_text = _metrics_text()

    assert "feature_mean_delta" in metrics_text
    assert "feature_variance_delta" in metrics_text
    assert 'feature_name="stable_feature"' in metrics_text
    assert 'drift_status="normal"' not in metrics_text


def test_publish_drift_evaluation_metrics_accepts_dataclass_result() -> None:
    result = DummyDriftResult(
        model_version="vmetrics",
        drift_events=[
            DummyDriftEvent(
                feature_name="purchase_probability_delta",
                drift_status="critical",
                mean_delta=0.6,
                variance_delta=1.2,
            ),
        ],
    )

    publish_drift_evaluation_metrics(result)

    metrics_text = _metrics_text()

    assert "drift_detected_total" in metrics_text
    assert 'feature_name="purchase_probability_delta"' in metrics_text
    assert 'drift_status="critical"' in metrics_text


def test_model_rollback_metric_is_exposed() -> None:
    record_model_rollback(
        from_model_version="v003",
        to_model_version="v002",
        status="success",
    )

    metrics_text = _metrics_text()

    assert "model_rollback_total" in metrics_text
    assert 'from_model_version="v003"' in metrics_text
    assert 'to_model_version="v002"' in metrics_text
    assert 'status="success"' in metrics_text


def test_prometheus_content_type_is_text_plain() -> None:
    assert "text/plain" in prometheus_content_type()
