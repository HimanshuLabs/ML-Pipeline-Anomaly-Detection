"""Evaluation helpers for anomaly detection model training.

This module intentionally avoids pretending that unsupervised anomaly detection
has perfect ground-truth labels. The metrics here are baseline/proxy metrics
captured at training time so future runtime behavior can be compared against
the approved model baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class EvaluationError(ValueError):
    """Raised when model evaluation inputs are invalid."""


@dataclass(frozen=True)
class AnomalyPredictionSummary:
    """Training-time anomaly scoring summary."""

    row_count: int
    anomaly_count: int
    normal_count: int
    anomaly_rate: float
    score_min: float
    score_max: float
    score_mean: float
    score_std: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable summary dictionary."""
        return {
            "row_count": self.row_count,
            "anomaly_count": self.anomaly_count,
            "normal_count": self.normal_count,
            "anomaly_rate": self.anomaly_rate,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
        }


def _require_non_empty_dataframe(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty:
        raise EvaluationError(f"{name} must not be empty.")


def _to_float(value: Any) -> float:
    """Convert numpy/pandas values to JSON-safe floats."""
    if pd.isna(value):
        return 0.0
    return float(value)


def calculate_feature_baseline_stats(feature_matrix: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate baseline mean and variance for every model feature.

    Args:
        feature_matrix: Numeric model feature dataframe in contract order.

    Returns:
        Mapping of feature name to baseline statistics.

    Raises:
        EvaluationError: If the input is empty or contains non-numeric columns.
    """
    _require_non_empty_dataframe(feature_matrix, name="feature_matrix")

    non_numeric_columns = [
        column
        for column in feature_matrix.columns
        if not pd.api.types.is_numeric_dtype(feature_matrix[column])
    ]
    if non_numeric_columns:
        raise EvaluationError(
            "feature_matrix must contain numeric columns only. "
            f"Non-numeric columns: {non_numeric_columns}"
        )

    stats: dict[str, dict[str, float]] = {}
    for column in feature_matrix.columns:
        series = feature_matrix[column].astype(float)
        stats[column] = {
            "mean": _to_float(series.mean()),
            "variance": _to_float(series.var(ddof=0)),
            "min": _to_float(series.min()),
            "max": _to_float(series.max()),
            "missing_count": int(series.isna().sum()),
        }

    return stats


def summarize_anomaly_predictions(
    *,
    anomaly_scores: np.ndarray,
    predictions: np.ndarray,
) -> AnomalyPredictionSummary:
    """Summarize Isolation Forest training-time predictions.

    Isolation Forest returns predictions as:
    - 1 for normal records
    - -1 for anomalous records

    Args:
        anomaly_scores: Model score output for each row.
        predictions: Isolation Forest predicted labels.

    Returns:
        AnomalyPredictionSummary with anomaly counts and score distribution.

    Raises:
        EvaluationError: If arrays are empty, mismatched, or contain invalid labels.
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    labels = np.asarray(predictions, dtype=int)

    if scores.size == 0:
        raise EvaluationError("anomaly_scores must not be empty.")

    if labels.size == 0:
        raise EvaluationError("predictions must not be empty.")

    if scores.shape[0] != labels.shape[0]:
        raise EvaluationError(
            "anomaly_scores and predictions must have the same number of rows. "
            f"Got {scores.shape[0]} scores and {labels.shape[0]} predictions."
        )

    allowed_labels = {-1, 1}
    invalid_labels = sorted(set(labels.tolist()) - allowed_labels)
    if invalid_labels:
        raise EvaluationError(
            "predictions must use Isolation Forest labels -1 and 1 only. "
            f"Invalid labels: {invalid_labels}"
        )

    row_count = int(labels.shape[0])
    anomaly_count = int(np.sum(labels == -1))
    normal_count = int(np.sum(labels == 1))
    anomaly_rate = anomaly_count / row_count if row_count else 0.0

    return AnomalyPredictionSummary(
        row_count=row_count,
        anomaly_count=anomaly_count,
        normal_count=normal_count,
        anomaly_rate=float(anomaly_rate),
        score_min=float(np.min(scores)),
        score_max=float(np.max(scores)),
        score_mean=float(np.mean(scores)),
        score_std=float(np.std(scores)),
    )


def calculate_latency_summary(latencies_ms: list[float] | np.ndarray) -> dict[str, float]:
    """Calculate latency summary metrics in milliseconds.

    This is used during training smoke tests and later API latency validation.
    """
    values = np.asarray(latencies_ms, dtype=float)

    if values.size == 0:
        return {
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    return {
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_max_ms": float(np.max(values)),
    }


def build_baseline_stats_payload(
    *,
    model_name: str,
    model_version: str,
    feature_schema_version: str,
    feature_matrix: pd.DataFrame,
    anomaly_scores: np.ndarray,
    predictions: np.ndarray,
    latency_measurements_ms: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the baseline stats payload written beside a trained model artifact.

    The payload is intentionally explicit that quality metrics are proxy metrics
    unless real labels are later introduced through backtesting or delayed truth.
    """
    feature_stats = calculate_feature_baseline_stats(feature_matrix)
    prediction_summary = summarize_anomaly_predictions(
        anomaly_scores=anomaly_scores,
        predictions=predictions,
    )
    latency_summary = calculate_latency_summary(latency_measurements_ms or [])

    return {
        "model_name": model_name,
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "metric_type": "unsupervised_training_baseline",
        "label_availability": "unlabeled_proxy_metrics",
        "baseline_anomaly_rate": prediction_summary.anomaly_rate,
        "prediction_summary": prediction_summary.to_dict(),
        "feature_baselines": feature_stats,
        "latency_summary": latency_summary,
        "notes": (
            "Isolation Forest is trained without ground-truth anomaly labels in this "
            "checkpoint. Precision, recall, false positive rate, and false negative "
            "rate are not claimed here. They require delayed labels, backtesting, "
            "or simulated labels in later evaluation work."
        ),
    }

# FALSE_POSITIVE_FALSE_NEGATIVE_THRESHOLD_HELPERS
def estimate_threshold_from_scores(
    anomaly_scores: list[float],
    *,
    target_anomaly_rate: float,
) -> float:
    """Estimate an anomaly threshold from model scores.

    The project currently uses unsupervised anomaly detection with proxy metrics.
    Because real labels are not available, this helper does not claim precision,
    recall, false-positive rate, or false-negative rate.

    Assumption:
        Lower anomaly scores are treated as more anomalous.

    Args:
        anomaly_scores: Raw anomaly scores produced by the model.
        target_anomaly_rate: Desired fraction of records to flag as anomalous.

    Returns:
        Score threshold where records at or below the threshold are anomalous.

    Raises:
        ValueError: If inputs are empty, non-finite, or outside valid range.
    """
    import math

    if not 0 < target_anomaly_rate < 1:
        raise ValueError("target_anomaly_rate must be between 0 and 1")

    cleaned_scores = [
        float(score)
        for score in anomaly_scores
        if score is not None and math.isfinite(float(score))
    ]

    if not cleaned_scores:
        raise ValueError("anomaly_scores must contain at least one finite score")

    sorted_scores = sorted(cleaned_scores)
    threshold_index = max(
        0,
        min(
            len(sorted_scores) - 1,
            int(len(sorted_scores) * target_anomaly_rate) - 1,
        ),
    )

    return sorted_scores[threshold_index]


def summarize_threshold_tradeoff(
    *,
    threshold: float,
    baseline_anomaly_rate: float,
    current_anomaly_rate: float | None = None,
    label_availability: str = "unlabeled_proxy_metrics",
) -> dict[str, object]:
    """Summarize threshold tradeoffs honestly for an unsupervised model.

    This function intentionally separates real metrics from proxy reasoning.
    Without known-good / known-bad labels, Project 4 can discuss false-positive
    and false-negative risk, but it cannot truthfully compute real FP/FN rates.

    Args:
        threshold: Active anomaly score threshold.
        baseline_anomaly_rate: Training/baseline anomaly rate.
        current_anomaly_rate: Optional observed anomaly rate from production-like
            inference logs.
        label_availability: Label status for the metric interpretation.

    Returns:
        Dictionary suitable for docs, metadata, or operational reporting.
    """
    if not 0 <= baseline_anomaly_rate <= 1:
        raise ValueError("baseline_anomaly_rate must be between 0 and 1")

    if current_anomaly_rate is not None and not 0 <= current_anomaly_rate <= 1:
        raise ValueError("current_anomaly_rate must be between 0 and 1")

    labels_available = label_availability == "real_labels_available"

    summary: dict[str, object] = {
        "threshold": float(threshold),
        "baseline_anomaly_rate": float(baseline_anomaly_rate),
        "current_anomaly_rate": (
            None if current_anomaly_rate is None else float(current_anomaly_rate)
        ),
        "label_availability": label_availability,
        "metrics_are_proxy": not labels_available,
        "real_false_positive_rate_available": labels_available,
        "real_false_negative_rate_available": labels_available,
        "false_positive_definition": (
            "A normal entity/event incorrectly flagged as anomalous."
        ),
        "false_negative_definition": (
            "A real anomalous entity/event not flagged by the model."
        ),
        "lower_threshold_effect": (
            "Fewer records are flagged; alert fatigue may fall, but missed "
            "anomaly risk can rise."
        ),
        "higher_threshold_effect": (
            "More records are flagged; missed anomaly risk may fall, but false "
            "positive risk and alert fatigue can rise."
        ),
        "operational_position": (
            "With unlabeled anomaly data, threshold quality is judged by proxy "
            "signals: anomaly-rate stability, drift status, alert volume, "
            "latency, investigation feedback, and later delayed truth labels."
        ),
    }

    if current_anomaly_rate is not None:
        summary["anomaly_rate_delta"] = float(
            current_anomaly_rate - baseline_anomaly_rate
        )
        summary["anomaly_rate_ratio"] = (
            None
            if baseline_anomaly_rate == 0
            else float(current_anomaly_rate / baseline_anomaly_rate)
        )

    return summary
