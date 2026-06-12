"""Prometheus metrics for Project 4 anomaly detection.

This module centralizes runtime metrics for online inference, drift monitoring,
active model visibility, and rollback evidence.

Metric names intentionally match the Project 4 monitoring contract:

- prediction_requests_total
- prediction_errors_total
- anomalies_detected_total
- prediction_latency_ms
- drift_detected_total
- feature_mean_delta
- feature_variance_delta
- active_model_version
- model_rollback_total
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

UNKNOWN_MODEL_VERSION = "unknown"

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of anomaly prediction requests.",
    ["endpoint", "model_version", "status"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of failed anomaly prediction requests.",
    ["endpoint", "model_version"],
)

ANOMALIES_DETECTED_TOTAL = Counter(
    "anomalies_detected_total",
    "Total number of anomalies detected by inference.",
    ["endpoint", "model_version"],
)

PREDICTION_LATENCY_MS = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds.",
    ["endpoint", "model_version"],
    buckets=(
        1,
        2.5,
        5,
        10,
        25,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1000,
        2500,
        5000,
    ),
)

DRIFT_DETECTED_TOTAL = Counter(
    "drift_detected_total",
    "Total number of drift events detected by feature and status.",
    ["model_version", "feature_name", "drift_status"],
)

FEATURE_MEAN_DELTA = Gauge(
    "feature_mean_delta",
    "Absolute mean delta between current feature stats and baseline.",
    ["model_version", "feature_name"],
)

FEATURE_VARIANCE_DELTA = Gauge(
    "feature_variance_delta",
    "Absolute variance delta between current feature stats and baseline.",
    ["model_version", "feature_name"],
)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version",
    "Active model version marker. The active model labelset is set to 1.",
    ["model_name", "model_version"],
)

MODEL_ROLLBACK_TOTAL = Counter(
    "model_rollback_total",
    "Total number of model rollback attempts.",
    ["from_model_version", "to_model_version", "status"],
)


def _clean_label(value: str | None) -> str:
    """Return a safe Prometheus label value."""

    if value is None:
        return UNKNOWN_MODEL_VERSION

    cleaned = str(value).strip()
    return cleaned if cleaned else UNKNOWN_MODEL_VERSION


def _to_mapping(value: Any) -> dict[str, Any]:
    """Convert dataclasses or mapping-like objects into a plain dictionary."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if isinstance(value, dict):
        return value

    result: dict[str, Any] = {}

    for attribute_name in dir(value):
        if attribute_name.startswith("_"):
            continue

        try:
            attribute_value = getattr(value, attribute_name)
        except AttributeError:
            continue

        if callable(attribute_value):
            continue

        result[attribute_name] = attribute_value

    return result


def record_prediction_request(
    *,
    endpoint: str,
    model_version: str | None,
    status: str,
) -> None:
    """Increment prediction request count."""

    PREDICTION_REQUESTS_TOTAL.labels(
        endpoint=_clean_label(endpoint),
        model_version=_clean_label(model_version),
        status=_clean_label(status),
    ).inc()


def record_prediction_error(
    *,
    endpoint: str,
    model_version: str | None,
) -> None:
    """Increment prediction error count."""

    PREDICTION_ERRORS_TOTAL.labels(
        endpoint=_clean_label(endpoint),
        model_version=_clean_label(model_version),
    ).inc()


def record_anomaly_detected(
    *,
    endpoint: str,
    model_version: str | None,
) -> None:
    """Increment anomaly count."""

    ANOMALIES_DETECTED_TOTAL.labels(
        endpoint=_clean_label(endpoint),
        model_version=_clean_label(model_version),
    ).inc()


def observe_prediction_latency_ms(
    *,
    endpoint: str,
    model_version: str | None,
    latency_ms: float,
) -> None:
    """Observe prediction latency in milliseconds."""

    PREDICTION_LATENCY_MS.labels(
        endpoint=_clean_label(endpoint),
        model_version=_clean_label(model_version),
    ).observe(max(float(latency_ms), 0.0))


def set_active_model_version(
    *,
    model_name: str,
    model_version: str | None,
) -> None:
    """Publish the active model version marker."""

    ACTIVE_MODEL_VERSION.labels(
        model_name=_clean_label(model_name),
        model_version=_clean_label(model_version),
    ).set(1)


def record_drift_event(
    *,
    model_version: str | None,
    feature_name: str,
    drift_status: str,
    mean_delta: float,
    variance_delta: float,
) -> None:
    """Publish drift counters and feature delta gauges for one drift event."""

    cleaned_model_version = _clean_label(model_version)
    cleaned_feature_name = _clean_label(feature_name)
    cleaned_status = _clean_label(drift_status)

    FEATURE_MEAN_DELTA.labels(
        model_version=cleaned_model_version,
        feature_name=cleaned_feature_name,
    ).set(float(mean_delta))

    FEATURE_VARIANCE_DELTA.labels(
        model_version=cleaned_model_version,
        feature_name=cleaned_feature_name,
    ).set(float(variance_delta))

    if cleaned_status in {"warning", "critical"}:
        DRIFT_DETECTED_TOTAL.labels(
            model_version=cleaned_model_version,
            feature_name=cleaned_feature_name,
            drift_status=cleaned_status,
        ).inc()


def publish_drift_evaluation_metrics(drift_result: Any) -> None:
    """Publish metrics from a DriftEvaluationResult-like object.

    The drift module owns drift evaluation logic. This function only translates
    drift events into Prometheus counters and gauges.
    """

    result = _to_mapping(drift_result)
    model_version = result.get("model_version")
    drift_events = result.get("drift_events", [])

    for event in drift_events:
        event_mapping = _to_mapping(event)

        record_drift_event(
            model_version=model_version,
            feature_name=str(event_mapping.get("feature_name", "unknown_feature")),
            drift_status=str(event_mapping.get("drift_status", "unknown")),
            mean_delta=float(event_mapping.get("mean_delta", 0.0)),
            variance_delta=float(event_mapping.get("variance_delta", 0.0)),
        )


def record_model_rollback(
    *,
    from_model_version: str,
    to_model_version: str,
    status: str = "success",
    model_name: str = "isolation_forest",
    triggered_by: str = "unknown",
) -> None:
    """Record a model rollback metric.

    Supports both the original metrics-test call shape:

        record_model_rollback(from_model_version=..., to_model_version=..., status=...)

    and the newer API call shape:

        record_model_rollback(
            model_name=...,
            from_model_version=...,
            to_model_version=...,
            triggered_by=...,
        )

    Detailed rollback evidence lives in logs/alerts/rollback_events.jsonl and
    the audit.rollback_events table contract.
    """

    label_attempts = [
        {
            "model_name": model_name,
            "from_model_version": from_model_version,
            "to_model_version": to_model_version,
            "triggered_by": triggered_by,
            "status": status,
        },
        {
            "model_name": model_name,
            "from_model_version": from_model_version,
            "to_model_version": to_model_version,
            "triggered_by": triggered_by,
        },
        {
            "from_model_version": from_model_version,
            "to_model_version": to_model_version,
            "status": status,
        },
        {
            "from_model_version": from_model_version,
            "to_model_version": to_model_version,
        },
    ]

    for labels in label_attempts:
        try:
            MODEL_ROLLBACK_TOTAL.labels(**labels).inc()
            return
        except ValueError:
            continue

    MODEL_ROLLBACK_TOTAL.inc()


def render_prometheus_metrics() -> bytes:
    """Render all registered Prometheus metrics in text exposition format."""

    return generate_latest()


def prometheus_content_type() -> str:
    """Return the Prometheus text exposition content type."""

    return CONTENT_TYPE_LATEST
