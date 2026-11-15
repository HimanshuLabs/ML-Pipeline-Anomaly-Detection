"""Prometheus metrics for Project 4 anomaly detection.

This module centralizes runtime metrics for online inference, drift monitoring,
active model visibility, and rollback evidence.

Metric names intentionally match the Project 4 monitoring contract:

- prediction_requests_total
- predictions_scored_total
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

PREDICTIONS_SCORED_TOTAL = Counter(
    "predictions_scored_total",
    "Rows scored successfully (1 per /predict call, N per /predict/batch call of N rows).",
    ["endpoint", "model_version"],
)

# 5 ms steps up to 100 ms and 10 ms steps up to 200 ms, so p50/p95 interpolate within
# narrow buckets; 200 ms (the p95 budget the alert rule checks) stays a bucket edge.
LATENCY_BUCKETS_MS = (
    *range(5, 101, 5),
    *range(110, 201, 10),
    250,
    300,
    400,
    500,
    750,
    1000,
    2500,
    5000,
)

PREDICTION_LATENCY_MS = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds.",
    ["endpoint", "model_version"],
    buckets=LATENCY_BUCKETS_MS,
)

# Segment label of a single (unsegmented) model's drift.
ALL_SEGMENTS = "all"
DRIFT_STATUS_VALUES = {"normal": 0, "warning": 1, "critical": 2}

DRIFT_DETECTED_TOTAL = Counter(
    "drift_detected_total",
    "Warning/critical features in newly loaded drift runs, by segment, feature and status.",
    ["model_version", "segment", "feature_name", "drift_status"],
)

FEATURE_MEAN_DELTA = Gauge(
    "feature_mean_delta",
    "Absolute mean delta between current feature stats and baseline (latest drift run).",
    ["model_version", "segment", "feature_name"],
)

FEATURE_VARIANCE_DELTA = Gauge(
    "feature_variance_delta",
    "Absolute variance delta between current feature stats and baseline (latest drift run).",
    ["model_version", "segment", "feature_name"],
)

FEATURE_DRIFT_STATUS = Gauge(
    "feature_drift_status",
    "Drift status of a feature in the latest drift run: 0 normal, 1 warning, 2 critical.",
    ["model_version", "segment", "feature_name"],
)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version",
    "Active model version marker. The active model labelset is set to 1.",
    ["model_name", "model_version"],
)

MODEL_ROLLBACK_TOTAL = Counter(
    "model_rollback_total",
    "Total number of model rollback attempts.",
    ["model_name", "from_model_version", "to_model_version", "triggered_by", "status"],
)


ACTIVE_POINTER_POLL_FAILURES_TOTAL = Counter(
    "active_pointer_poll_failures_total",
    "Polls of the shared active model pointer that could not reach the database.",
)

MODEL_RELOAD_TOTAL = Counter(
    "model_reload_total",
    "Reloads of the served model triggered by a changed shared pointer.",
    ["result"],
)

REGISTRY_MIRROR_WRITE_FAILURES_TOTAL = Counter(
    "registry_mirror_write_failures_total",
    "Failed writes of the registry/pointer file mirror after a committed database change.",
)


PREDICTION_PERSIST_FAILURES_TOTAL = Counter(
    "prediction_persist_failures_total",
    "Prediction evidence that could not be persisted, by sink.",
    ["sink"],
)


def record_prediction_persist_failure(*, sink: str) -> None:
    """Count a failed prediction evidence write to one sink ('jsonl' or 'postgres')."""

    PREDICTION_PERSIST_FAILURES_TOTAL.labels(sink=sink).inc()


def record_pointer_poll_failure() -> None:
    """Count a pointer poll that could not reach the database."""

    ACTIVE_POINTER_POLL_FAILURES_TOTAL.inc()


def record_model_reload(*, result: str) -> None:
    """Count a pointer-triggered model reload ('success' or 'failed')."""

    MODEL_RELOAD_TOTAL.labels(result=result).inc()


def record_registry_mirror_write_failure() -> None:
    """Count a failed file-mirror write after a committed database change."""

    REGISTRY_MIRROR_WRITE_FAILURES_TOTAL.inc()


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


def record_predictions_scored(
    *,
    endpoint: str,
    model_version: str | None,
    count: int,
) -> None:
    """Count rows scored successfully by one request."""

    PREDICTIONS_SCORED_TOTAL.labels(
        endpoint=_clean_label(endpoint),
        model_version=_clean_label(model_version),
    ).inc(count)


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


def replace_active_model_version(*, model_name: str, model_version: str | None) -> None:
    """Make model_version the only active_model_version series (stale ones are removed)."""

    ACTIVE_MODEL_VERSION.clear()
    set_active_model_version(model_name=model_name, model_version=model_version)


def _segment_label(segment: str | None) -> str:
    return _clean_label(segment) if segment else ALL_SEGMENTS


def record_drift_event(
    *,
    model_version: str | None,
    feature_name: str,
    drift_status: str,
    mean_delta: float,
    variance_delta: float,
    segment: str | None = None,
    count_detection: bool = True,
) -> None:
    """Publish feature drift gauges (and count warning/critical) for one drift event."""

    labels = {
        "model_version": _clean_label(model_version),
        "segment": _segment_label(segment),
        "feature_name": _clean_label(feature_name),
    }
    cleaned_status = _clean_label(drift_status)

    FEATURE_MEAN_DELTA.labels(**labels).set(float(mean_delta))
    FEATURE_VARIANCE_DELTA.labels(**labels).set(float(variance_delta))
    if cleaned_status in DRIFT_STATUS_VALUES:
        FEATURE_DRIFT_STATUS.labels(**labels).set(DRIFT_STATUS_VALUES[cleaned_status])

    if count_detection and cleaned_status in {"warning", "critical"}:
        DRIFT_DETECTED_TOTAL.labels(**labels, drift_status=cleaned_status).inc()


def publish_latest_drift(
    *,
    model_version: str,
    summaries: dict[str | None, Any],
    new_runs: set[str | None],
) -> None:
    """Expose the cached latest drift run per segment as gauges.

    The gauges show only the given model version's latest runs (stale series are
    removed). drift_detected_total counts the warning/critical features of the
    runs in new_runs once, when they are first loaded.
    """

    for gauge in (FEATURE_MEAN_DELTA, FEATURE_VARIANCE_DELTA, FEATURE_DRIFT_STATUS):
        gauge.clear()

    for segment, summary in summaries.items():
        for feature in summary.features:
            record_drift_event(
                model_version=model_version,
                feature_name=feature.feature_name,
                drift_status=feature.drift_status,
                mean_delta=feature.mean_delta,
                variance_delta=feature.variance_delta,
                segment=segment,
                count_detection=segment in new_runs,
            )


def publish_drift_evaluation_metrics(drift_result: Any) -> None:
    """Publish metrics from a DriftEvaluationResult-like object.

    The drift module owns drift evaluation logic. This function only translates
    drift events into Prometheus counters and gauges.
    """

    result = _to_mapping(drift_result)
    model_version = result.get("model_version")
    segment = result.get("entity_type")
    drift_events = result.get("drift_events", [])

    for event in drift_events:
        event_mapping = _to_mapping(event)

        record_drift_event(
            model_version=model_version,
            feature_name=str(event_mapping.get("feature_name", "unknown_feature")),
            drift_status=str(event_mapping.get("drift_status", "unknown")),
            mean_delta=float(event_mapping.get("mean_delta", 0.0)),
            variance_delta=float(event_mapping.get("variance_delta", 0.0)),
            segment=segment,
        )


def record_model_rollback(
    *,
    from_model_version: str,
    to_model_version: str,
    status: str = "success",
    model_name: str = "isolation_forest",
    triggered_by: str = "unknown",
) -> None:
    """Count a model rollback, labelled with every value the caller passes.

    Detailed rollback evidence lives in logs/alerts/rollback_events.jsonl and
    the audit.rollback_events table contract.
    """

    MODEL_ROLLBACK_TOTAL.labels(
        model_name=_clean_label(model_name),
        from_model_version=_clean_label(from_model_version),
        to_model_version=_clean_label(to_model_version),
        triggered_by=_clean_label(triggered_by),
        status=_clean_label(status),
    ).inc()


def render_prometheus_metrics() -> bytes:
    """Render all registered Prometheus metrics in text exposition format."""

    return generate_latest()


def prometheus_content_type() -> str:
    """Return the Prometheus text exposition content type."""

    return CONTENT_TYPE_LATEST
