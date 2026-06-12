"""Feature builder for the anomaly detection platform.

The builder converts raw event-level and warehouse-level inputs into the
model-ready feature contract used by Isolation Forest training and inference.

Input families:
- Project 1 event stream / replay exports
- Project 2/3 warehouse or mart aggregates

Output:
- one row per entity_id
- feature_timestamp
- schema_version
- numeric features defined in configs/model_config.yaml
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly_detection.feature_validation import (
    FeatureValidationConfig,
    load_feature_validation_config,
    validate_feature_dataframe,
)


@dataclass(frozen=True)
class FeatureBuilderConfig:
    """Configuration for building anomaly model features."""

    source_entity_columns: tuple[str, ...] = ("entity_id", "user_id", "customer_id")
    source_timestamp_columns: tuple[str, ...] = ("event_timestamp", "event_time")
    feature_schema_version: str = "feature_schema_v001"


class FeatureBuildError(ValueError):
    """Raised when features cannot be built from the provided inputs."""


def build_anomaly_features(
    event_frame: pd.DataFrame,
    warehouse_frame: pd.DataFrame | None = None,
    *,
    validation_config: FeatureValidationConfig | None = None,
    builder_config: FeatureBuilderConfig | None = None,
    reference_timestamp: str | pd.Timestamp | None = None,
    validate_output: bool = True,
) -> pd.DataFrame:
    """Build model-ready anomaly features.

    Args:
        event_frame: Raw behavioral/session/commerce/recommendation events.
        warehouse_frame: Optional warehouse or mart aggregates keyed by entity.
        validation_config: Optional validation contract. If not provided, it is
            loaded from configs/model_config.yaml.
        builder_config: Optional builder configuration.
        reference_timestamp: Optional point-in-time reference for lookback
            windows. Defaults to the max source event timestamp.
        validate_output: When True, validate the generated dataframe.

    Returns:
        A dataframe matching the Project 4 anomaly feature contract.
    """

    if not isinstance(event_frame, pd.DataFrame):
        raise TypeError("event_frame must be a pandas DataFrame.")

    if event_frame.empty:
        raise FeatureBuildError("event_frame is empty; cannot build features.")

    validation_contract = validation_config or load_feature_validation_config()
    build_config = builder_config or FeatureBuilderConfig(
        feature_schema_version=validation_contract.schema_version
    )

    working_events = _prepare_event_frame(event_frame, build_config)
    warehouse_signals = _prepare_warehouse_frame(warehouse_frame, build_config)

    reference_time = _resolve_reference_timestamp(
        working_events=working_events,
        reference_timestamp=reference_timestamp,
    )

    entity_index = pd.Index(
        sorted(working_events["entity_id"].dropna().astype(str).unique()),
        name="entity_id",
    )

    if entity_index.empty:
        raise FeatureBuildError("No valid entity identifiers found in event_frame.")

    features = pd.DataFrame(index=entity_index)
    features[validation_contract.timestamp_column] = reference_time.isoformat()
    features[validation_contract.schema_version_column] = validation_contract.schema_version

    one_hour_start = reference_time - pd.Timedelta(hours=1)
    seven_day_start = reference_time - pd.Timedelta(days=7)

    events_1h = working_events[working_events["feature_event_timestamp"] >= one_hour_start]
    events_7d = working_events[working_events["feature_event_timestamp"] >= seven_day_start]

    features["avg_cart_value_7d"] = _group_mean(
        events_7d,
        entity_index,
        "cart_value",
        default=0.0,
    )

    features["event_count_1h"] = (
        events_1h.groupby("entity_id").size().reindex(entity_index).fillna(0).astype(float)
    )

    features["avg_api_latency_ms"] = _group_mean(
        working_events,
        entity_index,
        "api_latency_ms",
        default=0.0,
    )

    features["fraud_score_avg"] = _group_mean(
        working_events,
        entity_index,
        "fraud_score",
        default=0.0,
    )

    features["purchase_probability_delta"] = _group_delta(
        working_events,
        entity_index,
        "purchase_probability",
        default=0.0,
    )

    features["cart_abandonment_rate"] = _group_mean(
        working_events,
        entity_index,
        "cart_abandonment_probability",
        default=0.0,
    )

    features["discount_sensitivity"] = _build_discount_sensitivity(
        working_events,
        entity_index,
    )

    features["page_load_p95_ms"] = _group_quantile(
        working_events,
        entity_index,
        "page_load_time_ms",
        quantile=0.95,
        default=0.0,
    )

    features["campaign_roas"] = _warehouse_signal(
        warehouse_signals,
        entity_index,
        "campaign_roas",
        default=0.0,
    )

    features["conversion_rate"] = _warehouse_signal(
        warehouse_signals,
        entity_index,
        "conversion_rate",
        default=0.0,
    )

    features["customer_lifetime_value"] = _warehouse_signal(
        warehouse_signals,
        entity_index,
        "customer_lifetime_value",
        default=0.0,
    )

    features = features.reset_index()

    ordered_columns = [
        validation_contract.entity_key,
        validation_contract.timestamp_column,
        validation_contract.schema_version_column,
        *validation_contract.required_numeric_features,
    ]

    features = features.rename(columns={"entity_id": validation_contract.entity_key})
    features = features.loc[:, ordered_columns]

    if validate_output:
        validate_feature_dataframe(
            features,
            config=validation_contract,
            strict=True,
        )

    return features


def _prepare_event_frame(
    event_frame: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    frame = event_frame.copy()

    entity_column = _first_existing_column(frame.columns, config.source_entity_columns)
    timestamp_column = _first_existing_column(frame.columns, config.source_timestamp_columns)

    if entity_column is None:
        raise FeatureBuildError(
            "event_frame must contain one entity column from: "
            + ", ".join(config.source_entity_columns)
        )

    if timestamp_column is None:
        raise FeatureBuildError(
            "event_frame must contain one timestamp column from: "
            + ", ".join(config.source_timestamp_columns)
        )

    frame["entity_id"] = frame[entity_column].astype(str)
    frame["feature_event_timestamp"] = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    if frame["feature_event_timestamp"].isna().any():
        raise FeatureBuildError("event_frame contains invalid event timestamps.")

    return frame


def _prepare_warehouse_frame(
    warehouse_frame: pd.DataFrame | None,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    if warehouse_frame is None:
        return pd.DataFrame(columns=["entity_id"])

    if not isinstance(warehouse_frame, pd.DataFrame):
        raise TypeError("warehouse_frame must be a pandas DataFrame or None.")

    if warehouse_frame.empty:
        return pd.DataFrame(columns=["entity_id"])

    frame = warehouse_frame.copy()
    entity_column = _first_existing_column(frame.columns, config.source_entity_columns)

    if entity_column is None:
        raise FeatureBuildError(
            "warehouse_frame must contain one entity column from: "
            + ", ".join(config.source_entity_columns)
        )

    frame["entity_id"] = frame[entity_column].astype(str)

    return frame


def _first_existing_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    existing = set(columns)

    for candidate in candidates:
        if candidate in existing:
            return candidate

    return None


def _resolve_reference_timestamp(
    working_events: pd.DataFrame,
    reference_timestamp: str | pd.Timestamp | None,
) -> pd.Timestamp:
    if reference_timestamp is not None:
        parsed = pd.Timestamp(reference_timestamp)

        parsed = (
            parsed.tz_localize("UTC")
            if parsed.tzinfo is None
            else parsed.tz_convert("UTC")
        )

        return parsed

    return working_events["feature_event_timestamp"].max()


def _numeric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(index=frame.index, dtype="float64")

    return pd.to_numeric(frame[column_name], errors="coerce")


def _group_mean(
    frame: pd.DataFrame,
    entity_index: pd.Index,
    column_name: str,
    *,
    default: float,
) -> pd.Series:
    if column_name not in frame.columns or frame.empty:
        return _default_series(entity_index, default)

    numeric_frame = frame.assign(_numeric_value=_numeric_series(frame, column_name))

    return (
        numeric_frame.groupby("entity_id")["_numeric_value"]
        .mean()
        .reindex(entity_index)
        .fillna(default)
        .astype(float)
    )


def _group_delta(
    frame: pd.DataFrame,
    entity_index: pd.Index,
    column_name: str,
    *,
    default: float,
) -> pd.Series:
    if column_name not in frame.columns or frame.empty:
        return _default_series(entity_index, default)

    numeric_frame = frame.assign(_numeric_value=_numeric_series(frame, column_name))
    grouped = numeric_frame.groupby("entity_id")["_numeric_value"]

    delta = grouped.max() - grouped.min()

    return delta.reindex(entity_index).fillna(default).astype(float)


def _group_quantile(
    frame: pd.DataFrame,
    entity_index: pd.Index,
    column_name: str,
    *,
    quantile: float,
    default: float,
) -> pd.Series:
    if column_name not in frame.columns or frame.empty:
        return _default_series(entity_index, default)

    numeric_frame = frame.assign(_numeric_value=_numeric_series(frame, column_name))

    return (
        numeric_frame.groupby("entity_id")["_numeric_value"]
        .quantile(quantile)
        .reindex(entity_index)
        .fillna(default)
        .astype(float)
    )


def _build_discount_sensitivity(
    frame: pd.DataFrame,
    entity_index: pd.Index,
) -> pd.Series:
    if "discount_percent" not in frame.columns or frame.empty:
        return _default_series(entity_index, 0.0)

    numeric_frame = frame.assign(
        _numeric_value=_numeric_series(frame, "discount_percent")
    )

    discount = (
        numeric_frame.groupby("entity_id")["_numeric_value"]
        .mean()
        .reindex(entity_index)
        .fillna(0.0)
        .astype(float)
    )

    # Project 1 events store discount as a percent-like field. Model features
    # use ratio-style numeric values to keep scale stable.
    return np.where(discount > 1.0, discount / 100.0, discount).astype(float)


def _warehouse_signal(
    warehouse_frame: pd.DataFrame,
    entity_index: pd.Index,
    column_name: str,
    *,
    default: float,
) -> pd.Series:
    if warehouse_frame.empty or column_name not in warehouse_frame.columns:
        return _default_series(entity_index, default)

    numeric_frame = warehouse_frame.assign(
        _numeric_value=_numeric_series(warehouse_frame, column_name)
    )

    return (
        numeric_frame.groupby("entity_id")["_numeric_value"]
        .mean()
        .reindex(entity_index)
        .fillna(default)
        .astype(float)
    )


def _default_series(entity_index: pd.Index, default: float) -> pd.Series:
    return pd.Series(default, index=entity_index, dtype="float64")
