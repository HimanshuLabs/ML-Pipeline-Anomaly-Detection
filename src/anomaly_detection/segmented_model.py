"""Segmented anomaly model: one IsolationForest per entity type.

A single model over the unified real-source frame learns the entity type (each
type pads the features it does not have with 0.0), so whole types end up
flagged or never flagged. SegmentedIsolationForest fits one IsolationForest per
entity type on only the features that type's source provides
(source_extract.SEGMENT_FEATURE_COLUMNS) and routes every row by its
entity_type column to its own segment.

The class lives at module level so joblib can pickle and load it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SINGLE_MODEL_KIND = "isolation_forest"
SEGMENTED_MODEL_KIND = "segmented_isolation_forest"
ENTITY_TYPE_COLUMN = "entity_type"


class SegmentedModelError(ValueError):
    """Raised when a segmented model cannot be fitted or cannot score a frame."""


@dataclass
class ModelSegment:
    """One entity type's fitted model and the ordered features it was fitted on."""

    feature_names: list[str]
    model: IsolationForest
    row_count: int


def segment_spec_fingerprint(segment_features: Mapping[str, Sequence[str]]) -> str:
    """sha256 of the segment spec: entity types (sorted) and each type's ordered features."""

    canonical = json.dumps(
        {str(t): [str(name) for name in names] for t, names in segment_features.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entity_types(frame: pd.DataFrame) -> np.ndarray:
    if ENTITY_TYPE_COLUMN not in frame.columns:
        raise SegmentedModelError(f"frame has no {ENTITY_TYPE_COLUMN} column to route rows by")
    values = frame[ENTITY_TYPE_COLUMN]
    if values.isna().any():
        raise SegmentedModelError(f"{ENTITY_TYPE_COLUMN} has {int(values.isna().sum())} null(s)")
    return values.astype(str).to_numpy()


def _segment_matrix(frame: pd.DataFrame, mask: np.ndarray, names: list[str], label: str):
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise SegmentedModelError(f"segment {label!r} is missing feature column(s): {missing}")
    return frame.loc[mask, names].astype("float64")


def check_segment_row_counts(
    entity_types: Sequence[str] | np.ndarray,
    *,
    segment_features: Mapping[str, Sequence[str]],
    min_rows_per_segment: int,
) -> dict[str, int]:
    """Row count per entity type; refuses unknown types and segments below the minimum."""

    counts = pd.Series(np.asarray(entity_types, dtype=str)).value_counts().sort_index()
    unknown = [t for t in counts.index if t not in segment_features]
    if unknown:
        raise SegmentedModelError(f"no segment definition for entity type(s): {unknown}")

    too_small = [f"{t} has {int(n)} rows" for t, n in counts.items() if n < min_rows_per_segment]
    if too_small:
        raise SegmentedModelError(
            f"segment(s) too small to train: {', '.join(too_small)} "
            f"(min_rows_per_segment {min_rows_per_segment})"
        )
    return {str(t): int(n) for t, n in counts.items()}


class SegmentedIsolationForest:
    """Per-entity-type IsolationForests behind the IsolationForest scoring interface."""

    def __init__(self, segments: Mapping[str, ModelSegment]) -> None:
        if not segments:
            raise SegmentedModelError("a segmented model needs at least one segment")
        self.segments: dict[str, ModelSegment] = dict(sorted(segments.items()))

    def feature_spec(self) -> dict[str, list[str]]:
        """Entity type -> ordered feature names, sorted by entity type."""

        return {t: list(segment.feature_names) for t, segment in self.segments.items()}

    def _score(self, frame: pd.DataFrame, method: str) -> np.ndarray:
        entity_types = _entity_types(frame)
        unknown = sorted(set(entity_types) - set(self.segments))
        if unknown:
            raise SegmentedModelError(
                f"unknown entity_type(s) {unknown}; this model has segments "
                f"{sorted(self.segments)}"
            )

        output = np.empty(len(frame), dtype=float)
        for entity_type, segment in self.segments.items():
            mask = entity_types == entity_type
            if mask.any():
                matrix = _segment_matrix(frame, mask, segment.feature_names, entity_type)
                output[mask] = getattr(segment.model, method)(matrix)
        return output

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        """Each row's own segment decision_function (lower is more anomalous)."""

        return self._score(frame, "decision_function")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Each row's own segment prediction: -1 anomaly, 1 normal."""

        return self._score(frame, "predict").astype(int)


def fit_segmented_isolation_forest(
    frame: pd.DataFrame,
    *,
    segment_features: Mapping[str, Sequence[str]],
    build_model: Callable[[], IsolationForest],
    min_rows_per_segment: int,
) -> SegmentedIsolationForest:
    """Fit one model per entity type present in the frame, on that type's features only."""

    entity_types = _entity_types(frame)
    counts = check_segment_row_counts(
        entity_types,
        segment_features=segment_features,
        min_rows_per_segment=min_rows_per_segment,
    )

    segments: dict[str, ModelSegment] = {}
    for entity_type, row_count in counts.items():
        names = [str(name) for name in segment_features[entity_type]]
        matrix = _segment_matrix(frame, entity_types == entity_type, names, entity_type)
        segments[entity_type] = ModelSegment(
            feature_names=names,
            model=build_model().fit(matrix),
            row_count=row_count,
        )
    return SegmentedIsolationForest(segments)


def score_distribution(scores: np.ndarray, is_anomaly: np.ndarray) -> dict[str, Any]:
    """Row/anomaly counts, anomaly rate and p5/p50/p95 of decision_function scores."""

    row_count = int(len(scores))
    if row_count == 0:
        return {
            "row_count": 0,
            "anomaly_count": 0,
            "anomaly_rate": None,
            "score_p5": None,
            "score_p50": None,
            "score_p95": None,
        }

    p5, p50, p95 = np.percentile(scores, [5, 50, 95])
    anomaly_count = int(np.asarray(is_anomaly).sum())
    return {
        "row_count": row_count,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_count / row_count,
        "score_p5": float(p5),
        "score_p50": float(p50),
        "score_p95": float(p95),
    }


# --- feature spec: what a model consumes, for compatibility and labels ---------------


@dataclass(frozen=True)
class FeatureSpec:
    """The features a model consumes: an ordered list, or one ordered list per segment."""

    model_kind: str
    feature_names: tuple[str, ...]
    segments: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def is_segmented(self) -> bool:
        return self.model_kind == SEGMENTED_MODEL_KIND

    def difference(self, other: FeatureSpec) -> str | None:
        """Why two models cannot serve the same contract; None when they can."""

        if self.model_kind != other.model_kind:
            return (
                f"a {self.model_kind} and a {other.model_kind} model are never compatible"
            )
        if self.is_segmented:
            if self.segments != other.segments:
                return "segment spec differs (entity types or per-type ordered features)"
            return None
        if self.feature_names != other.feature_names:
            return (
                f"feature_names differ ({len(self.feature_names)} vs "
                f"{len(other.feature_names)} features)"
            )
        return None


def _names(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(name, str) for name in value)
    ):
        raise ValueError(f"{label} must be a non-empty list of feature names")
    return tuple(value)


def feature_spec_from_metadata(metadata: Mapping[str, Any]) -> FeatureSpec:
    """FeatureSpec from a model metadata.json; models without model_kind are single models."""

    model_kind = metadata.get("model_kind") or SINGLE_MODEL_KIND
    feature_names = _names(metadata.get("feature_names"), "feature_names")

    if model_kind == SINGLE_MODEL_KIND:
        return FeatureSpec(model_kind=model_kind, feature_names=feature_names)

    if model_kind != SEGMENTED_MODEL_KIND:
        raise ValueError(f"unknown model_kind {model_kind!r}")

    segments = metadata.get("segments")
    if not isinstance(segments, Mapping) or not segments:
        raise ValueError("segmented metadata needs a non-empty segments object")
    spec = tuple(
        (
            str(entity_type),
            _names(
                (segment or {}).get("feature_names") if isinstance(segment, Mapping) else None,
                f"segments.{entity_type}.feature_names",
            ),
        )
        for entity_type, segment in sorted(segments.items())
    )
    return FeatureSpec(model_kind=model_kind, feature_names=feature_names, segments=spec)


def feature_spec_for(
    feature_names: Sequence[str],
    segment_features: Mapping[str, Sequence[str]] | None = None,
) -> FeatureSpec:
    """FeatureSpec of a model being trained."""

    names = tuple(str(name) for name in feature_names)
    if segment_features is None:
        return FeatureSpec(model_kind=SINGLE_MODEL_KIND, feature_names=names)
    return FeatureSpec(
        model_kind=SEGMENTED_MODEL_KIND,
        feature_names=names,
        segments=tuple(
            (str(t), tuple(str(n) for n in segment_features[t])) for t in sorted(segment_features)
        ),
    )
