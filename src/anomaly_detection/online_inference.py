"""Online anomaly inference service for Project 4.

The online path keeps the active production model loaded in memory and scores
validated feature payloads with low per-request overhead.

This module intentionally does not own HTTP concerns. FastAPI wiring lives in
api/main.py. Keeping scoring separate makes the service testable without
starting an API server.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from anomaly_detection.batch_inference import (
    DEFAULT_ACTIVE_MODEL_PATH,
    ActiveModelArtifacts,
    BatchInferenceError,
    get_required_feature_names,
    hash_feature_payload,
    load_active_model_artifacts,
    prepare_feature_matrix,
)


class OnlineInferenceError(RuntimeError):
    """Raised when online inference cannot safely score a request."""


@dataclass(frozen=True)
class OnlinePredictionRecord:
    """Serializable online prediction response."""

    prediction_id: str
    model_name: str
    model_version: str
    dataset_snapshot_id: str | None
    training_dataset_id: str | None
    feature_schema_version: str
    entity_type: str
    entity_id: str
    prediction_timestamp: str
    anomaly_score: float
    is_anomaly: bool
    threshold_used: float
    drift_status: str
    feature_payload_hash: str
    latency_ms: float
    prediction_status: str = "success"
    error_message: str | None = None


@dataclass(frozen=True)
class ActiveModelInfo:
    """Serializable active model information for API responses."""

    model_name: str
    model_version: str
    status: str
    artifact_path: str
    feature_schema_version: str
    feature_count: int
    dataset_snapshot_id: str | None
    training_dataset_id: str | None
    baseline_anomaly_rate: float | None
    threshold: float
    source_projects: list[str]
    snapshot_type: str | None
    latency_budget_p95_ms: float | None
    loaded_at: str


def _as_float(value: Any, default: float | None = None) -> float | None:
    """Convert a numeric-like value to float."""

    if value is None:
        return default

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default

    if np.isnan(numeric_value) or np.isinf(numeric_value):
        return default

    return numeric_value


def resolve_threshold(baseline_stats: dict[str, Any]) -> float:
    """Resolve the anomaly threshold used for scoring.

    Isolation Forest's decision_function returns values where negative scores
    are generally more anomalous. Older artifacts may not contain an explicit
    threshold. In that case, use 0.0, which matches the Isolation Forest
    decision boundary.
    """

    candidate_keys = (
        "threshold",
        "threshold_used",
        "anomaly_threshold",
        "decision_threshold",
    )

    for key in candidate_keys:
        value = _as_float(baseline_stats.get(key))
        if value is not None:
            return value

    prediction_summary = baseline_stats.get("prediction_summary")
    if isinstance(prediction_summary, dict):
        for key in candidate_keys:
            value = _as_float(prediction_summary.get(key))
            if value is not None:
                return value

    return 0.0


def _current_timestamp() -> str:
    return datetime.now(UTC).isoformat()


class OnlineInferenceService:
    """In-memory online inference service for active anomaly model."""

    def __init__(
        self,
        active_model_path: Path | None = None,
        artifacts: ActiveModelArtifacts | None = None,
    ) -> None:
        self.loaded_at = _current_timestamp()
        self.active_model_path = active_model_path or DEFAULT_ACTIVE_MODEL_PATH

        if artifacts is not None:
            self.artifacts = artifacts
        else:
            self.artifacts = load_active_model_artifacts(self.active_model_path)

        self.active_model_config = self._load_active_model_config()
        self.feature_names = get_required_feature_names(self.artifacts)
        self.threshold = resolve_threshold(self.artifacts.baseline_stats)

    def _load_active_model_config(self) -> dict[str, Any]:
        """Load active model pointer metadata for API reporting."""

        if not self.active_model_path.exists():
            return {}

        loaded = yaml.safe_load(
            self.active_model_path.read_text(encoding="utf-8")
        )
        if not isinstance(loaded, dict):
            return {}

        return loaded

    def active_model_info(self) -> ActiveModelInfo:
        """Return active model metadata without exposing the model object."""

        baseline_anomaly_rate = _as_float(
            self.artifacts.baseline_stats.get("baseline_anomaly_rate")
        )

        active_model_config = self.active_model_config
        source_projects: list[str] = []
        snapshot_type: str | None = None
        latency_budget_p95_ms: float | None = None
        status = str(active_model_config.get("status") or "production")

        training_snapshot_metadata = self.artifacts.metadata.get(
            "training_snapshot_metadata"
        )
        if isinstance(training_snapshot_metadata, dict):
            snapshot_type_raw = training_snapshot_metadata.get("snapshot_type")
            snapshot_type = (
                str(snapshot_type_raw) if snapshot_type_raw is not None else None
            )

        raw_source_projects = active_model_config.get("source_projects")
        if isinstance(raw_source_projects, list):
            source_projects = [str(project) for project in raw_source_projects]

        raw_latency_budget = active_model_config.get("latency_budget")
        if isinstance(raw_latency_budget, dict):
            latency_budget_p95_ms = _as_float(
                raw_latency_budget.get("online_prediction_p95_ms")
            )

        feature_count = len(self.feature_names)

        return ActiveModelInfo(
            model_name=self.artifacts.model_name,
            model_version=self.artifacts.model_version,
            status=status,
            artifact_path=str(self.artifacts.model_path),
            feature_schema_version=self.artifacts.feature_schema_version,
            feature_count=feature_count,
            dataset_snapshot_id=self.artifacts.dataset_snapshot_id,
            training_dataset_id=self.artifacts.training_dataset_id,
            baseline_anomaly_rate=baseline_anomaly_rate,
            threshold=self.threshold,
            source_projects=source_projects,
            snapshot_type=snapshot_type,
            latency_budget_p95_ms=latency_budget_p95_ms,
            loaded_at=self.loaded_at,
        )

    def predict(
        self,
        feature_payload: dict[str, Any],
        *,
        entity_id: str | None = None,
        entity_type: str = "user",
    ) -> OnlinePredictionRecord:
        """Score a single feature payload."""

        if not isinstance(feature_payload, dict) or not feature_payload:
            raise OnlineInferenceError("feature_payload must be a non-empty object")

        resolved_entity_id = (
            entity_id
            or feature_payload.get("entity_id")
            or feature_payload.get("user_id")
            or f"online_{uuid4().hex[:12]}"
        )

        started_at = time.perf_counter()

        try:
            feature_frame = pd.DataFrame([feature_payload])
            matrix = prepare_feature_matrix(feature_frame, self.feature_names)
            scores = self.artifacts.model.decision_function(matrix)
            anomaly_score = float(scores[0])
            is_anomaly = bool(anomaly_score < self.threshold)
            feature_payload_hash = hash_feature_payload(feature_payload)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 3)

            return OnlinePredictionRecord(
                prediction_id=str(uuid4()),
                model_name=self.artifacts.model_name,
                model_version=self.artifacts.model_version,
                dataset_snapshot_id=self.artifacts.dataset_snapshot_id,
                training_dataset_id=self.artifacts.training_dataset_id,
                feature_schema_version=self.artifacts.feature_schema_version,
                entity_type=str(entity_type),
                entity_id=str(resolved_entity_id),
                prediction_timestamp=_current_timestamp(),
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                threshold_used=self.threshold,
                drift_status="not_evaluated",
                feature_payload_hash=feature_payload_hash,
                latency_ms=latency_ms,
            )
        except BatchInferenceError as exc:
            raise OnlineInferenceError(str(exc)) from exc

    def predict_batch(
        self,
        feature_payloads: list[dict[str, Any]],
        *,
        entity_type: str = "user",
    ) -> list[OnlinePredictionRecord]:
        """Score multiple online payloads using the in-memory model."""

        if not isinstance(feature_payloads, list) or not feature_payloads:
            raise OnlineInferenceError("feature_payloads must be a non-empty list")

        predictions: list[OnlinePredictionRecord] = []
        for payload in feature_payloads:
            if not isinstance(payload, dict):
                raise OnlineInferenceError("each batch item must be an object")

            predictions.append(
                self.predict(
                    payload,
                    entity_id=payload.get("entity_id") or payload.get("user_id"),
                    entity_type=entity_type,
                )
            )

        return predictions


def prediction_to_dict(record: OnlinePredictionRecord) -> dict[str, Any]:
    """Convert a prediction record into a JSON-safe dictionary."""

    return asdict(record)


def model_info_to_dict(info: ActiveModelInfo) -> dict[str, Any]:
    """Convert active model info into a JSON-safe dictionary."""

    return asdict(info)
