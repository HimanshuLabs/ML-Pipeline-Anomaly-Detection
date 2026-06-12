"""Feature validation utilities for the anomaly detection platform.

This module validates model-ready feature frames before they are used for
training, batch scoring, or online inference. The checks are intentionally
strict because anomaly models are sensitive to schema drift, null spikes, and
silent type changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("configs/model_config.yaml")


class FeatureValidationError(ValueError):
    """Raised when feature validation fails in strict mode."""


@dataclass(frozen=True)
class FeatureValidationConfig:
    """Runtime validation contract for anomaly model features."""

    schema_version: str
    entity_key: str
    timestamp_column: str
    required_numeric_features: tuple[str, ...]
    schema_version_column: str = "schema_version"
    max_null_fraction: float = 0.05
    require_finite_values: bool = True
    non_negative_features: tuple[str, ...] = (
        "avg_cart_value_7d",
        "event_count_1h",
        "avg_api_latency_ms",
        "fraud_score_avg",
        "cart_abandonment_rate",
        "campaign_roas",
        "conversion_rate",
        "customer_lifetime_value",
        "discount_sensitivity",
        "page_load_p95_ms",
        "session_duration_sec",
        "items_viewed_in_session",
        "time_on_page_sec",
        "scroll_depth_percent",
        "hover_duration_ms",
        "api_latency_ms",
        "page_load_time_ms",
        "cart_value",
        "quantity",
        "discount_percent",
        "discounted_price",
    )
    probability_features: tuple[str, ...] = (
        "purchase_probability_delta",
        "cart_abandonment_rate",
        "conversion_rate",
        "fraud_score_avg",
        "recommendation_clicked_rate",
    )


@dataclass(frozen=True)
class FeatureValidationResult:
    """Structured output from feature validation."""

    is_valid: bool
    row_count: int
    feature_count: int
    schema_version: str | None
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_feature_validation_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> FeatureValidationConfig:
    """Load the feature validation contract from the model config YAML file."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    feature_config = raw_config.get("features", {})

    required_numeric_features = tuple(
        feature_config.get("required_numeric_features", [])
    )

    if not required_numeric_features:
        raise FeatureValidationError(
            "Config must define features.required_numeric_features."
        )

    schema_version = feature_config.get("schema_version")
    entity_key = feature_config.get("entity_key")
    timestamp_column = feature_config.get("timestamp_column")

    missing_config_keys = [
        key
        for key, value in {
            "features.schema_version": schema_version,
            "features.entity_key": entity_key,
            "features.timestamp_column": timestamp_column,
        }.items()
        if not value
    ]

    if missing_config_keys:
        raise FeatureValidationError(
            "Missing required feature config keys: "
            + ", ".join(missing_config_keys)
        )

    return FeatureValidationConfig(
        schema_version=str(schema_version),
        entity_key=str(entity_key),
        timestamp_column=str(timestamp_column),
        required_numeric_features=required_numeric_features,
    )


def validate_feature_dataframe(
    frame: pd.DataFrame,
    config: FeatureValidationConfig | None = None,
    *,
    strict: bool = False,
) -> FeatureValidationResult:
    """Validate a model-ready feature dataframe.

    Args:
        frame: Pandas dataframe containing entity, timestamp, schema version,
            and numeric model features.
        config: Optional explicit validation config. If omitted, the config is
            loaded from configs/model_config.yaml.
        strict: When True, raise FeatureValidationError if validation fails.

    Returns:
        FeatureValidationResult with errors and warnings.
    """

    validation_config = config or load_feature_validation_config()

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        errors.append("Feature dataframe is empty.")
        return _finish_validation(
            is_valid=False,
            row_count=0,
            feature_count=len(validation_config.required_numeric_features),
            schema_version=None,
            errors=errors,
            warnings=warnings,
            strict=strict,
        )

    required_columns = (
        validation_config.entity_key,
        validation_config.timestamp_column,
        validation_config.schema_version_column,
        *validation_config.required_numeric_features,
    )

    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required feature columns: " + ", ".join(missing_columns)
        )

    schema_version = _extract_schema_version(
        frame=frame,
        schema_version_column=validation_config.schema_version_column,
    )

    if validation_config.schema_version_column in frame.columns:
        invalid_schema_rows = frame[
            frame[validation_config.schema_version_column].astype(str)
            != validation_config.schema_version
        ]

        if not invalid_schema_rows.empty:
            errors.append(
                "Schema version mismatch. Expected "
                f"{validation_config.schema_version}, found "
                f"{sorted(frame[validation_config.schema_version_column].astype(str).unique())}."
            )

    _validate_entity_key(frame, validation_config, errors)
    _validate_timestamp(frame, validation_config, errors)

    present_numeric_features = [
        feature
        for feature in validation_config.required_numeric_features
        if feature in frame.columns
    ]

    _validate_numeric_features(
        frame=frame,
        feature_names=present_numeric_features,
        config=validation_config,
        errors=errors,
        warnings=warnings,
    )

    result_is_valid = not errors

    return _finish_validation(
        is_valid=result_is_valid,
        row_count=len(frame),
        feature_count=len(present_numeric_features),
        schema_version=schema_version,
        errors=errors,
        warnings=warnings,
        strict=strict,
    )


def get_model_feature_matrix(
    frame: pd.DataFrame,
    config: FeatureValidationConfig | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Return numeric model features in contract order after validation."""

    validation_config = config or load_feature_validation_config()
    validation_result = validate_feature_dataframe(
        frame=frame,
        config=validation_config,
        strict=strict,
    )

    if not validation_result.is_valid:
        raise FeatureValidationError(
            "Cannot build model feature matrix from invalid dataframe."
        )

    return frame.loc[:, list(validation_config.required_numeric_features)].copy()


def _extract_schema_version(
    frame: pd.DataFrame,
    schema_version_column: str,
) -> str | None:
    if schema_version_column not in frame.columns:
        return None

    unique_versions = frame[schema_version_column].dropna().astype(str).unique()

    if len(unique_versions) == 1:
        return str(unique_versions[0])

    if len(unique_versions) == 0:
        return None

    return ",".join(sorted(unique_versions))


def _validate_entity_key(
    frame: pd.DataFrame,
    config: FeatureValidationConfig,
    errors: list[str],
) -> None:
    if config.entity_key not in frame.columns:
        return

    if frame[config.entity_key].isna().any():
        errors.append(f"Entity key column {config.entity_key} contains nulls.")


def _validate_timestamp(
    frame: pd.DataFrame,
    config: FeatureValidationConfig,
    errors: list[str],
) -> None:
    if config.timestamp_column not in frame.columns:
        return

    parsed_timestamp = pd.to_datetime(
        frame[config.timestamp_column],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    if parsed_timestamp.isna().any():
        errors.append(
            f"Timestamp column {config.timestamp_column} contains invalid values."
        )


def _validate_numeric_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    config: FeatureValidationConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    for feature_name in feature_names:
        series = frame[feature_name]

        numeric_series = pd.to_numeric(series, errors="coerce")
        non_numeric_count = int(numeric_series.isna().sum() - series.isna().sum())

        if non_numeric_count > 0:
            errors.append(
                f"Feature {feature_name} contains {non_numeric_count} "
                "non-numeric values."
            )
            continue

        null_fraction = float(series.isna().mean())

        if null_fraction > config.max_null_fraction:
            errors.append(
                f"Feature {feature_name} null fraction {null_fraction:.2%} "
                f"exceeds limit {config.max_null_fraction:.2%}."
            )
        elif null_fraction > 0:
            warnings.append(
                f"Feature {feature_name} has null fraction {null_fraction:.2%}."
            )

        if config.require_finite_values:
            finite_mask = np.isfinite(numeric_series.dropna().to_numpy(dtype=float))

            if not bool(finite_mask.all()):
                errors.append(f"Feature {feature_name} contains infinite values.")

        if feature_name in config.non_negative_features:
            negative_count = int((numeric_series.dropna() < 0).sum())

            if negative_count > 0:
                errors.append(
                    f"Feature {feature_name} contains {negative_count} "
                    "negative values."
                )

        if feature_name in config.probability_features:
            out_of_range_count = int(
                (
                    (numeric_series.dropna() < 0)
                    | (numeric_series.dropna() > 1)
                ).sum()
            )

            if out_of_range_count > 0:
                errors.append(
                    f"Feature {feature_name} contains {out_of_range_count} "
                    "values outside probability range [0, 1]."
                )


def _finish_validation(
    *,
    is_valid: bool,
    row_count: int,
    feature_count: int,
    schema_version: str | None,
    errors: list[str],
    warnings: list[str],
    strict: bool,
) -> FeatureValidationResult:
    result = FeatureValidationResult(
        is_valid=is_valid,
        row_count=row_count,
        feature_count=feature_count,
        schema_version=schema_version,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

    if strict and not result.is_valid:
        raise FeatureValidationError("; ".join(result.errors))

    return result
