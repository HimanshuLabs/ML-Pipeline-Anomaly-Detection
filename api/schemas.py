"""Pydantic schemas for the Project 4 online inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """API health response."""

    status: str
    service: str
    active_model_version: str | None
    model_loaded: bool


class ActiveModelResponse(BaseModel):
    """Active production model metadata response."""

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


class PredictRequest(BaseModel):
    """Single online prediction request."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str | None = Field(default=None)
    entity_type: str = Field(default="user")
    feature_payload: dict[str, Any] = Field(
        ...,
        description="Model-ready feature payload containing all required active model features.",
    )


class BatchPredictRequest(BaseModel):
    """Batch online prediction request."""

    model_config = ConfigDict(extra="forbid")

    entity_type: str = Field(default="user")
    feature_payloads: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of model-ready feature payloads.",
    )


class PredictionResponse(BaseModel):
    """Online prediction response."""

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
    prediction_status: str
    error_message: str | None


class BatchPredictionResponse(BaseModel):
    """Batch online prediction response."""

    predictions: list[PredictionResponse]
    prediction_count: int
    model_version: str


class RollbackStubResponse(BaseModel):
    """Rollback stub response.

    Real rollback is intentionally implemented in a later checkpoint.
    """

    status: str
    action_taken: bool
    reason: str
    active_model_version: str | None
