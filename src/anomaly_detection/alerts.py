"""Alert event generation for Project 4 anomaly monitoring.

This module creates local-first alert evidence for operational conditions that
matter to the anomaly detection platform:

- critical drift
- anomaly-rate spikes
- p95 latency budget breaches
- prediction error-rate breaches
- manual alert test events

The PostgreSQL table `monitoring.alert_events` already exists in
sql/create_monitoring_tables.sql. This checkpoint writes deterministic JSONL
records to logs/alerts/anomaly_alerts.jsonl so the alert contract is testable
without pretending that external PagerDuty or Slack wiring exists.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALERT_EVENTS_PATH = PROJECT_ROOT / "logs" / "alerts" / "anomaly_alerts.jsonl"

VALID_ALERT_TYPES = {
    "drift_warning",
    "drift_critical",
    "anomaly_rate_spike",
    "latency_budget_breach",
    "prediction_error_rate_breach",
    "model_degradation",
    "rollback_triggered",
    "manual_alert",
}

VALID_SEVERITIES = {"info", "warning", "critical"}
VALID_ALERT_STATUSES = {"open", "acknowledged", "resolved", "suppressed"}
VALID_COMPARISON_OPERATORS = {"<", "<=", "=", ">=", ">"}

DEFAULT_ALERT_SOURCE = "project4_monitoring"


class AlertingError(RuntimeError):
    """Raised when alert events cannot be generated or persisted safely."""


@dataclass(frozen=True)
class AlertEvent:
    """Operational alert event aligned with monitoring.alert_events."""

    alert_event_id: str
    drift_event_id: str | None
    model_name: str | None
    model_version: str | None
    alert_type: str
    severity: str
    alert_status: str
    alert_source: str
    metric_name: str | None
    metric_value: float | None
    threshold_value: float | None
    comparison_operator: str | None
    entity_type: str | None
    entity_id: str | None
    message: str
    details: dict[str, Any]
    triggered_at: str


def _json_safe(value: Any) -> Any:
    """Convert common Python values into JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def _to_mapping(value: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping-like object into a dictionary."""

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


def _optional_finite_float(value: Any, *, field_name: str) -> float | None:
    """Return a finite float or None."""

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise AlertingError(f"{field_name} must be numeric") from exc

    if math.isnan(numeric_value) or math.isinf(numeric_value):
        raise AlertingError(f"{field_name} must be finite")

    return numeric_value


def _required_finite_float(value: Any, *, field_name: str) -> float:
    """Return a required finite float."""

    numeric_value = _optional_finite_float(value, field_name=field_name)

    if numeric_value is None:
        raise AlertingError(f"{field_name} is required")

    return numeric_value


def _validate_alert_event(event: AlertEvent) -> None:
    """Validate alert event fields before persistence."""

    if event.alert_type not in VALID_ALERT_TYPES:
        raise AlertingError(f"invalid alert_type: {event.alert_type}")

    if event.severity not in VALID_SEVERITIES:
        raise AlertingError(f"invalid severity: {event.severity}")

    if event.alert_status not in VALID_ALERT_STATUSES:
        raise AlertingError(f"invalid alert_status: {event.alert_status}")

    if (
        event.comparison_operator is not None
        and event.comparison_operator not in VALID_COMPARISON_OPERATORS
    ):
        raise AlertingError(
            f"invalid comparison_operator: {event.comparison_operator}"
        )

    if not event.message.strip():
        raise AlertingError("message must not be empty")


def create_alert_event(
    *,
    alert_type: str,
    severity: str,
    message: str,
    model_name: str | None = None,
    model_version: str | None = None,
    drift_event_id: str | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    threshold_value: float | None = None,
    comparison_operator: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    details: dict[str, Any] | None = None,
    alert_status: str = "open",
    alert_source: str = DEFAULT_ALERT_SOURCE,
    alert_event_id: str | None = None,
    triggered_at: datetime | str | None = None,
) -> AlertEvent:
    """Create and validate one alert event."""

    if triggered_at is None:
        triggered_at_value = datetime.now(UTC).isoformat()
    elif isinstance(triggered_at, datetime):
        triggered_at_value = triggered_at.isoformat()
    else:
        triggered_at_value = str(triggered_at)

    event = AlertEvent(
        alert_event_id=alert_event_id or str(uuid4()),
        drift_event_id=drift_event_id,
        model_name=model_name,
        model_version=model_version,
        alert_type=alert_type,
        severity=severity,
        alert_status=alert_status,
        alert_source=alert_source,
        metric_name=metric_name,
        metric_value=_optional_finite_float(
            metric_value,
            field_name="metric_value",
        ),
        threshold_value=_optional_finite_float(
            threshold_value,
            field_name="threshold_value",
        ),
        comparison_operator=comparison_operator,
        entity_type=entity_type,
        entity_id=entity_id,
        message=message,
        details=_json_safe(details or {}),
        triggered_at=triggered_at_value,
    )

    _validate_alert_event(event)
    return event


def create_manual_alert(
    *,
    message: str = "Manual Project 4 alert test event.",
    model_name: str | None = None,
    model_version: str | None = None,
    severity: str = "info",
    details: dict[str, Any] | None = None,
    triggered_at: datetime | str | None = None,
) -> AlertEvent:
    """Create a manual alert test event."""

    return create_alert_event(
        alert_type="manual_alert",
        severity=severity,
        model_name=model_name,
        model_version=model_version,
        message=message,
        details=details or {"reason": "manual_test"},
        triggered_at=triggered_at,
    )


def create_critical_drift_alerts(drift_result: Any) -> list[AlertEvent]:
    """Create alert events for critical feature drift records."""

    result = _to_mapping(drift_result)
    model_name = result.get("model_name")
    model_version = result.get("model_version")
    drift_events = result.get("drift_events", [])

    alerts: list[AlertEvent] = []

    for raw_event in drift_events:
        event = _to_mapping(raw_event)

        if event.get("drift_status") != "critical":
            continue

        feature_name = str(event.get("feature_name", "unknown_feature"))
        mean_delta_percent = _required_finite_float(
            event.get("mean_delta_percent", 0.0),
            field_name="mean_delta_percent",
        )
        variance_delta_percent = _required_finite_float(
            event.get("variance_delta_percent", 0.0),
            field_name="variance_delta_percent",
        )
        metric_value = max(mean_delta_percent, variance_delta_percent)

        mean_threshold = _optional_finite_float(
            event.get("mean_critical_threshold"),
            field_name="mean_critical_threshold",
        )
        variance_threshold = _optional_finite_float(
            event.get("variance_critical_threshold"),
            field_name="variance_critical_threshold",
        )
        threshold_candidates = [
            value
            for value in [mean_threshold, variance_threshold]
            if value is not None
        ]
        threshold_value = max(threshold_candidates) if threshold_candidates else None

        alerts.append(
            create_alert_event(
                alert_type="drift_critical",
                severity="critical",
                drift_event_id=event.get("drift_event_id"),
                model_name=model_name,
                model_version=model_version,
                metric_name="feature_drift_delta_percent",
                metric_value=metric_value,
                threshold_value=threshold_value,
                comparison_operator=">=",
                entity_type="feature",
                entity_id=feature_name,
                message=(
                    "Critical drift detected for feature "
                    f"{feature_name} on model {model_version}."
                ),
                details={
                    "feature_name": feature_name,
                    "feature_dtype": event.get("feature_dtype"),
                    "baseline_mean": event.get("baseline_mean"),
                    "current_mean": event.get("current_mean"),
                    "mean_delta": event.get("mean_delta"),
                    "mean_delta_percent": mean_delta_percent,
                    "baseline_variance": event.get("baseline_variance"),
                    "current_variance": event.get("current_variance"),
                    "variance_delta": event.get("variance_delta"),
                    "variance_delta_percent": variance_delta_percent,
                    "observation_window_start": event.get(
                        "observation_window_start"
                    ),
                    "observation_window_end": event.get("observation_window_end"),
                    "detection_method": event.get("detection_method"),
                },
            )
        )

    return alerts


def create_anomaly_rate_alert(
    *,
    model_name: str,
    model_version: str,
    baseline_anomaly_rate: float,
    current_anomaly_rate: float,
    multiplier: float,
    observation_window: str | None = None,
) -> AlertEvent | None:
    """Create an anomaly-rate spike alert when current rate exceeds threshold."""

    baseline_rate = _required_finite_float(
        baseline_anomaly_rate,
        field_name="baseline_anomaly_rate",
    )
    current_rate = _required_finite_float(
        current_anomaly_rate,
        field_name="current_anomaly_rate",
    )
    threshold_multiplier = _required_finite_float(
        multiplier,
        field_name="multiplier",
    )

    if baseline_rate < 0 or current_rate < 0:
        raise AlertingError("anomaly rates must be non-negative")

    if threshold_multiplier <= 0:
        raise AlertingError("multiplier must be positive")

    threshold_value = baseline_rate * threshold_multiplier

    if current_rate <= threshold_value:
        return None

    severity = "critical" if current_rate >= threshold_value * 2 else "warning"

    return create_alert_event(
        alert_type="anomaly_rate_spike",
        severity=severity,
        model_name=model_name,
        model_version=model_version,
        metric_name="current_anomaly_rate",
        metric_value=current_rate,
        threshold_value=threshold_value,
        comparison_operator=">",
        message=(
            "Anomaly rate spike detected: "
            f"current={current_rate:.6f}, threshold={threshold_value:.6f}."
        ),
        details={
            "baseline_anomaly_rate": baseline_rate,
            "current_anomaly_rate": current_rate,
            "multiplier": threshold_multiplier,
            "observation_window": observation_window,
        },
    )


def create_latency_budget_alert(
    *,
    model_name: str,
    model_version: str,
    p95_latency_ms: float,
    budget_ms: float = 200.0,
    endpoint: str | None = None,
) -> AlertEvent | None:
    """Create a p95 latency breach alert when latency exceeds budget."""

    latency_value = _required_finite_float(
        p95_latency_ms,
        field_name="p95_latency_ms",
    )
    budget_value = _required_finite_float(
        budget_ms,
        field_name="budget_ms",
    )

    if latency_value < 0 or budget_value <= 0:
        raise AlertingError("latency and budget values must be positive")

    if latency_value <= budget_value:
        return None

    severity = "critical" if latency_value >= budget_value * 2 else "warning"

    return create_alert_event(
        alert_type="latency_budget_breach",
        severity=severity,
        model_name=model_name,
        model_version=model_version,
        metric_name="prediction_latency_p95_ms",
        metric_value=latency_value,
        threshold_value=budget_value,
        comparison_operator=">",
        entity_type="endpoint" if endpoint else None,
        entity_id=endpoint,
        message=(
            "Online inference p95 latency breached budget: "
            f"p95={latency_value:.3f} ms, budget={budget_value:.3f} ms."
        ),
        details={
            "endpoint": endpoint,
            "p95_latency_ms": latency_value,
            "budget_ms": budget_value,
        },
    )


def create_prediction_error_rate_alert(
    *,
    model_name: str,
    model_version: str,
    error_rate: float,
    threshold: float,
    endpoint: str | None = None,
) -> AlertEvent | None:
    """Create a prediction error-rate breach alert."""

    current_error_rate = _required_finite_float(
        error_rate,
        field_name="error_rate",
    )
    threshold_value = _required_finite_float(
        threshold,
        field_name="threshold",
    )

    if current_error_rate < 0 or threshold_value < 0:
        raise AlertingError("error rate and threshold must be non-negative")

    if current_error_rate <= threshold_value:
        return None

    severity = (
        "critical"
        if threshold_value > 0 and current_error_rate >= threshold_value * 2
        else "warning"
    )

    return create_alert_event(
        alert_type="prediction_error_rate_breach",
        severity=severity,
        model_name=model_name,
        model_version=model_version,
        metric_name="prediction_error_rate",
        metric_value=current_error_rate,
        threshold_value=threshold_value,
        comparison_operator=">",
        entity_type="endpoint" if endpoint else None,
        entity_id=endpoint,
        message=(
            "Prediction error-rate breach detected: "
            f"error_rate={current_error_rate:.6f}, "
            f"threshold={threshold_value:.6f}."
        ),
        details={
            "endpoint": endpoint,
            "error_rate": current_error_rate,
            "threshold": threshold_value,
        },
    )


def append_alert_events_jsonl(
    alert_events: list[AlertEvent],
    output_path: Path = DEFAULT_ALERT_EVENTS_PATH,
) -> Path:
    """Append alert events to a JSONL file."""

    if not alert_events:
        raise AlertingError("no alert events supplied")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for alert_event in alert_events:
            _validate_alert_event(alert_event)
            payload = _json_safe(alert_event)

            if not isinstance(payload, dict):
                raise AlertingError("alert event must serialize to a JSON object")

            handle.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return output_path


def read_alert_events_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read alert events from a JSONL file."""

    if not path.exists():
        raise AlertingError(f"alert JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        loaded = json.loads(line)

        if not isinstance(loaded, dict):
            raise AlertingError(
                f"line {line_number} is not a JSON object: {path}"
            )

        records.append(loaded)

    return records
