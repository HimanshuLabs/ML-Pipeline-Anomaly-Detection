"""Mean/variance drift monitoring for Project 4.

This module compares current feature statistics against the approved training
baseline for the active production model.

The PostgreSQL table exists in sql/create_monitoring_tables.sql. This checkpoint
keeps persistence local-first by writing deterministic JSONL drift events under
logs/alerts/drift_events.jsonl. Database insert wiring can reuse the same event
shape later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_STATS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "isolation_forest"
    / "model_version=v002"
    / "baseline_stats.json"
)

DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_DRIFT_THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "drift_thresholds.yaml"
DEFAULT_DRIFT_EVENTS_PATH = PROJECT_ROOT / "logs" / "alerts" / "drift_events.jsonl"

DETECTION_METHOD = "mean_variance_threshold"
VALID_DRIFT_STATUSES = {"normal", "warning", "critical"}


class DriftMonitoringError(RuntimeError):
    """Raised when drift monitoring cannot be evaluated safely."""


@dataclass(frozen=True)
class DriftThresholds:
    """Warning and critical thresholds for feature drift."""

    mean_delta_warning: float
    mean_delta_critical: float
    variance_delta_warning: float
    variance_delta_critical: float

    def validate(self) -> None:
        values = asdict(self)

        for field_name, value in values.items():
            if value < 0:
                raise DriftMonitoringError(
                    f"threshold {field_name} must be non-negative"
                )

        if self.mean_delta_critical < self.mean_delta_warning:
            raise DriftMonitoringError(
                "mean_delta_critical must be >= mean_delta_warning"
            )

        if self.variance_delta_critical < self.variance_delta_warning:
            raise DriftMonitoringError(
                "variance_delta_critical must be >= variance_delta_warning"
            )


@dataclass(frozen=True)
class FeatureCurrentStats:
    """Current mean/variance statistics for one feature."""

    feature_name: str
    feature_dtype: str
    current_mean: float
    current_variance: float
    non_null_count: int
    missing_count: int


@dataclass(frozen=True)
class FeatureDriftEvent:
    """Drift event for one feature."""

    drift_event_id: str
    model_name: str
    model_version: str
    feature_schema_version: str
    feature_name: str
    feature_dtype: str
    baseline_mean: float
    current_mean: float
    mean_delta: float
    mean_delta_percent: float
    baseline_variance: float
    current_variance: float
    variance_delta: float
    variance_delta_percent: float
    mean_warning_threshold: float
    mean_critical_threshold: float
    variance_warning_threshold: float
    variance_critical_threshold: float
    drift_status: str
    detection_method: str
    observation_window_start: str
    observation_window_end: str
    detected_at: str
    notes: str


@dataclass(frozen=True)
class DriftEvaluationResult:
    """Complete drift evaluation result for a batch/window."""

    model_name: str
    model_version: str
    feature_schema_version: str
    observation_window_start: str
    observation_window_end: str
    evaluated_feature_count: int
    normal_feature_count: int
    warning_feature_count: int
    critical_feature_count: int
    overall_drift_status: str
    drift_events: list[FeatureDriftEvent]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DriftMonitoringError(f"missing JSON file: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(loaded, dict):
        raise DriftMonitoringError(f"JSON file must contain an object: {path}")

    return loaded


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DriftMonitoringError(f"missing YAML file: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(loaded, dict):
        raise DriftMonitoringError(f"YAML file must contain an object: {path}")

    return loaded


def _safe_float(value: Any, *, field_name: str) -> float:
    if value is None:
        raise DriftMonitoringError(f"{field_name} is required")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise DriftMonitoringError(
            f"{field_name} must be numeric; got {value!r}"
        ) from exc

    if np.isnan(numeric_value) or np.isinf(numeric_value):
        raise DriftMonitoringError(f"{field_name} must be finite")

    return numeric_value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            return None
        return numeric_value

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def _relative_delta(delta: float, baseline_value: float) -> float:
    """Return relative delta, with absolute fallback when baseline is zero.

    The drift threshold config uses compact values like 0.15 and 0.30. For
    non-zero baselines those represent 15% and 30% movement. For zero baselines,
    a relative percentage is undefined, so the absolute delta is used.
    """

    if baseline_value == 0:
        return abs(delta)

    return abs(delta) / abs(baseline_value)


def load_active_model_config(
    active_model_path: Path = DEFAULT_ACTIVE_MODEL_PATH,
) -> dict[str, Any]:
    """Load active production model pointer."""

    active_model = _read_yaml(active_model_path)

    required_fields = [
        "model_name",
        "active_model_version",
        "feature_schema_version",
    ]

    missing_fields = [
        field for field in required_fields if field not in active_model
    ]

    if missing_fields:
        raise DriftMonitoringError(
            "active model config missing required fields: "
            + ", ".join(missing_fields)
        )

    return active_model


def load_baseline_stats(
    baseline_stats_path: Path = DEFAULT_BASELINE_STATS_PATH,
) -> dict[str, Any]:
    """Load approved training baseline statistics."""

    baseline_stats = _read_json(baseline_stats_path)

    required_fields = [
        "model_name",
        "model_version",
        "feature_schema_version",
        "feature_baselines",
    ]

    missing_fields = [
        field for field in required_fields if field not in baseline_stats
    ]

    if missing_fields:
        raise DriftMonitoringError(
            "baseline stats missing required fields: "
            + ", ".join(missing_fields)
        )

    feature_baselines = baseline_stats["feature_baselines"]
    if not isinstance(feature_baselines, dict) or not feature_baselines:
        raise DriftMonitoringError("feature_baselines must be a non-empty object")

    return baseline_stats


def load_drift_thresholds(
    threshold_config_path: Path = DEFAULT_DRIFT_THRESHOLDS_PATH,
) -> dict[str, Any]:
    """Load drift threshold config."""

    config = _read_yaml(threshold_config_path)

    if "thresholds" not in config:
        raise DriftMonitoringError("drift config missing thresholds section")

    thresholds = config["thresholds"]

    if not isinstance(thresholds, dict):
        raise DriftMonitoringError("thresholds section must be an object")

    if "default" not in thresholds:
        raise DriftMonitoringError("thresholds.default section is required")

    return config


def get_thresholds_for_feature(
    feature_name: str,
    threshold_config: dict[str, Any],
) -> DriftThresholds:
    """Resolve default thresholds plus feature-specific overrides."""

    thresholds_section = threshold_config.get("thresholds", {})
    default_thresholds = thresholds_section.get("default", {})
    feature_overrides = thresholds_section.get("feature_overrides", {})

    if not isinstance(default_thresholds, dict):
        raise DriftMonitoringError("thresholds.default must be an object")

    if not isinstance(feature_overrides, dict):
        raise DriftMonitoringError("thresholds.feature_overrides must be an object")

    merged_thresholds = {
        **default_thresholds,
        **feature_overrides.get(feature_name, {}),
    }

    required_fields = [
        "mean_delta_warning",
        "mean_delta_critical",
        "variance_delta_warning",
        "variance_delta_critical",
    ]

    missing_fields = [
        field for field in required_fields if field not in merged_thresholds
    ]

    if missing_fields:
        raise DriftMonitoringError(
            f"thresholds missing for feature {feature_name}: "
            + ", ".join(missing_fields)
        )

    thresholds = DriftThresholds(
        mean_delta_warning=_safe_float(
            merged_thresholds["mean_delta_warning"],
            field_name=f"{feature_name}.mean_delta_warning",
        ),
        mean_delta_critical=_safe_float(
            merged_thresholds["mean_delta_critical"],
            field_name=f"{feature_name}.mean_delta_critical",
        ),
        variance_delta_warning=_safe_float(
            merged_thresholds["variance_delta_warning"],
            field_name=f"{feature_name}.variance_delta_warning",
        ),
        variance_delta_critical=_safe_float(
            merged_thresholds["variance_delta_critical"],
            field_name=f"{feature_name}.variance_delta_critical",
        ),
    )
    thresholds.validate()

    return thresholds


def calculate_current_feature_stats(
    current_features: pd.DataFrame,
) -> dict[str, FeatureCurrentStats]:
    """Calculate current mean/variance for numeric dataframe columns."""

    if current_features.empty:
        raise DriftMonitoringError("current feature dataframe is empty")

    numeric_features = current_features.select_dtypes(include=["number"])

    if numeric_features.empty:
        raise DriftMonitoringError(
            "current feature dataframe has no numeric features"
        )

    current_stats: dict[str, FeatureCurrentStats] = {}

    for feature_name in numeric_features.columns:
        series = numeric_features[feature_name]
        non_null_series = series.dropna()

        if non_null_series.empty:
            raise DriftMonitoringError(
                f"feature {feature_name} has no non-null numeric values"
            )

        current_stats[feature_name] = FeatureCurrentStats(
            feature_name=feature_name,
            feature_dtype=str(series.dtype),
            current_mean=float(non_null_series.mean()),
            current_variance=float(non_null_series.var(ddof=0)),
            non_null_count=int(non_null_series.shape[0]),
            missing_count=int(series.isna().sum()),
        )

    return current_stats


def evaluate_feature_drift(
    *,
    feature_name: str,
    baseline_feature_stats: dict[str, Any],
    current_feature_stats: FeatureCurrentStats,
    thresholds: DriftThresholds,
    model_name: str,
    model_version: str,
    feature_schema_version: str,
    observation_window_start: datetime,
    observation_window_end: datetime,
) -> FeatureDriftEvent:
    """Evaluate one feature against baseline mean/variance stats."""

    baseline_mean = _safe_float(
        baseline_feature_stats.get("mean"),
        field_name=f"{feature_name}.baseline_mean",
    )
    baseline_variance = _safe_float(
        baseline_feature_stats.get("variance"),
        field_name=f"{feature_name}.baseline_variance",
    )

    current_mean = current_feature_stats.current_mean
    current_variance = current_feature_stats.current_variance

    mean_delta = abs(current_mean - baseline_mean)
    variance_delta = abs(current_variance - baseline_variance)

    mean_delta_percent = _relative_delta(mean_delta, baseline_mean)
    variance_delta_percent = _relative_delta(variance_delta, baseline_variance)

    if (
        mean_delta_percent >= thresholds.mean_delta_critical
        or variance_delta_percent >= thresholds.variance_delta_critical
    ):
        drift_status = "critical"
    elif (
        mean_delta_percent >= thresholds.mean_delta_warning
        or variance_delta_percent >= thresholds.variance_delta_warning
    ):
        drift_status = "warning"
    else:
        drift_status = "normal"

    detected_at = datetime.now(UTC).isoformat()

    return FeatureDriftEvent(
        drift_event_id=str(uuid4()),
        model_name=model_name,
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        feature_name=feature_name,
        feature_dtype=current_feature_stats.feature_dtype,
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        mean_delta=mean_delta,
        mean_delta_percent=mean_delta_percent,
        baseline_variance=baseline_variance,
        current_variance=current_variance,
        variance_delta=variance_delta,
        variance_delta_percent=variance_delta_percent,
        mean_warning_threshold=thresholds.mean_delta_warning,
        mean_critical_threshold=thresholds.mean_delta_critical,
        variance_warning_threshold=thresholds.variance_delta_warning,
        variance_critical_threshold=thresholds.variance_delta_critical,
        drift_status=drift_status,
        detection_method=DETECTION_METHOD,
        observation_window_start=observation_window_start.isoformat(),
        observation_window_end=observation_window_end.isoformat(),
        detected_at=detected_at,
        notes=(
            "Compared current feature mean/variance against approved training "
            "baseline statistics."
        ),
    )


def evaluate_drift(
    *,
    current_features: pd.DataFrame,
    baseline_stats: dict[str, Any],
    threshold_config: dict[str, Any],
    observation_window_start: datetime | None = None,
    observation_window_end: datetime | None = None,
) -> DriftEvaluationResult:
    """Evaluate mean/variance drift for all overlapping numeric features."""

    observation_window_end = observation_window_end or datetime.now(UTC)
    observation_window_start = observation_window_start or observation_window_end

    if observation_window_end < observation_window_start:
        raise DriftMonitoringError(
            "observation_window_end must be >= observation_window_start"
        )

    feature_baselines = baseline_stats["feature_baselines"]
    current_stats_by_feature = calculate_current_feature_stats(current_features)

    drift_events: list[FeatureDriftEvent] = []

    for feature_name, current_feature_stats in current_stats_by_feature.items():
        if feature_name not in feature_baselines:
            continue

        thresholds = get_thresholds_for_feature(
            feature_name,
            threshold_config,
        )

        drift_events.append(
            evaluate_feature_drift(
                feature_name=feature_name,
                baseline_feature_stats=feature_baselines[feature_name],
                current_feature_stats=current_feature_stats,
                thresholds=thresholds,
                model_name=str(baseline_stats["model_name"]),
                model_version=str(baseline_stats["model_version"]),
                feature_schema_version=str(baseline_stats["feature_schema_version"]),
                observation_window_start=observation_window_start,
                observation_window_end=observation_window_end,
            )
        )

    if not drift_events:
        raise DriftMonitoringError(
            "no overlapping numeric features found between current data and baseline"
        )

    status_counts = {
        status: sum(event.drift_status == status for event in drift_events)
        for status in VALID_DRIFT_STATUSES
    }

    if status_counts["critical"] > 0:
        overall_status = "critical"
    elif status_counts["warning"] > 0:
        overall_status = "warning"
    else:
        overall_status = "normal"

    return DriftEvaluationResult(
        model_name=str(baseline_stats["model_name"]),
        model_version=str(baseline_stats["model_version"]),
        feature_schema_version=str(baseline_stats["feature_schema_version"]),
        observation_window_start=observation_window_start.isoformat(),
        observation_window_end=observation_window_end.isoformat(),
        evaluated_feature_count=len(drift_events),
        normal_feature_count=status_counts["normal"],
        warning_feature_count=status_counts["warning"],
        critical_feature_count=status_counts["critical"],
        overall_drift_status=overall_status,
        drift_events=drift_events,
    )


def evaluate_drift_from_paths(
    *,
    current_features: pd.DataFrame,
    baseline_stats_path: Path = DEFAULT_BASELINE_STATS_PATH,
    threshold_config_path: Path = DEFAULT_DRIFT_THRESHOLDS_PATH,
    observation_window_start: datetime | None = None,
    observation_window_end: datetime | None = None,
) -> DriftEvaluationResult:
    """Load baseline/config from disk and evaluate current feature drift."""

    baseline_stats = load_baseline_stats(baseline_stats_path)
    threshold_config = load_drift_thresholds(threshold_config_path)

    return evaluate_drift(
        current_features=current_features,
        baseline_stats=baseline_stats,
        threshold_config=threshold_config,
        observation_window_start=observation_window_start,
        observation_window_end=observation_window_end,
    )


def append_drift_events_jsonl(
    drift_events: list[FeatureDriftEvent],
    output_path: Path = DEFAULT_DRIFT_EVENTS_PATH,
) -> Path:
    """Append drift events to local JSONL fallback storage."""

    if not drift_events:
        raise DriftMonitoringError("no drift events supplied")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for event in drift_events:
            payload = _json_safe(asdict(event))
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    return output_path


def read_drift_events_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read drift event JSONL records for tests and smoke validation."""

    if not path.exists():
        raise DriftMonitoringError(f"drift event log does not exist: {path}")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        record = json.loads(line)
        if not isinstance(record, dict):
            raise DriftMonitoringError(
                f"line {line_number} is not a JSON object: {path}"
            )

        records.append(record)

    return records


def create_alerts_for_drift_result(
    drift_result: DriftEvaluationResult,
) -> list[Any]:
    """Create alert events from a drift evaluation result.

    Drift evaluation remains the source of truth. This helper only translates
    critical drift events into alert events. Warning drift is still persisted as
    drift evidence, but it does not generate an operational alert in this
    checkpoint.
    """

    from anomaly_detection.alerts import create_critical_drift_alerts

    return create_critical_drift_alerts(drift_result)


def append_alerts_for_drift_result_jsonl(
    drift_result: DriftEvaluationResult,
    output_path: Path | None = None,
) -> Path | None:
    """Persist alert events generated from a drift evaluation result.

    Returns:
        Path to the written alert JSONL file when alerts are generated.
        None when the drift result does not produce alert events.
    """

    from anomaly_detection.alerts import (
        DEFAULT_ALERT_EVENTS_PATH,
        append_alert_events_jsonl,
    )

    alert_events = create_alerts_for_drift_result(drift_result)

    if not alert_events:
        return None

    return append_alert_events_jsonl(
        alert_events,
        output_path or DEFAULT_ALERT_EVENTS_PATH,
    )
