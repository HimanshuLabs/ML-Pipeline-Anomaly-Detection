"""FastAPI application for Project 4 online anomaly inference."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from anomaly_detection.online_inference import (
    OnlineInferenceError,
    OnlineInferenceService,
    model_info_to_dict,
    prediction_to_dict,
)

from api.schemas import (
    ActiveModelResponse,
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
    RollbackStubResponse,
)


logger = logging.getLogger(__name__)

SERVICE_NAME = "project4-online-anomaly-inference"

PREDICTION_REQUESTS_TOTAL = Counter(
    "project4_prediction_requests_total",
    "Total number of online anomaly prediction requests.",
    ["endpoint", "model_version", "status"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "project4_prediction_errors_total",
    "Total number of failed online anomaly prediction requests.",
    ["endpoint", "model_version"],
)

ANOMALIES_DETECTED_TOTAL = Counter(
    "project4_anomalies_detected_total",
    "Total number of anomalies detected by online inference.",
    ["endpoint", "model_version"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "project4_prediction_latency_seconds",
    "Online anomaly prediction latency in seconds.",
    ["endpoint", "model_version"],
)


class AppState:
    """Mutable application state."""

    inference_service: OnlineInferenceService | None = None


state = AppState()


def get_inference_service() -> OnlineInferenceService:
    """Return the loaded online inference service."""

    if state.inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Online inference service is not loaded.",
        )

    return state.inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load active production model once during API startup."""

    try:
        state.inference_service = OnlineInferenceService()
        info = state.inference_service.active_model_info()
        logger.info(
            "Loaded active anomaly model",
            extra={
                "model_name": info.model_name,
                "model_version": info.model_version,
                "feature_count": info.feature_count,
            },
        )
    except Exception as exc:  # pragma: no cover - covered by runtime smoke checks
        logger.exception("Failed to load active anomaly model")
        state.inference_service = None

    yield


app = FastAPI(
    title="Project 4 Online Anomaly Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health and model-load status."""

    if state.inference_service is None:
        return HealthResponse(
            status="degraded",
            service=SERVICE_NAME,
            active_model_version=None,
            model_loaded=False,
        )

    info = state.inference_service.active_model_info()
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        active_model_version=info.model_version,
        model_loaded=True,
    )


@app.get("/model/active", response_model=ActiveModelResponse)
def active_model() -> ActiveModelResponse:
    """Return active production model metadata."""

    service = get_inference_service()
    info = service.active_model_info()
    return ActiveModelResponse(**model_info_to_dict(info))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    """Score one model-ready feature payload."""

    service = get_inference_service()
    model_version = service.active_model_info().model_version
    started_at = time.perf_counter()

    try:
        prediction = service.predict(
            request.feature_payload,
            entity_id=request.entity_id,
            entity_type=request.entity_type,
        )

        if prediction.is_anomaly:
            ANOMALIES_DETECTED_TOTAL.labels(
                endpoint="/predict",
                model_version=prediction.model_version,
            ).inc()

        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict",
            model_version=prediction.model_version,
            status="success",
        ).inc()

        PREDICTION_LATENCY_SECONDS.labels(
            endpoint="/predict",
            model_version=prediction.model_version,
        ).observe(time.perf_counter() - started_at)

        return PredictionResponse(**prediction_to_dict(prediction))

    except OnlineInferenceError as exc:
        PREDICTION_ERRORS_TOTAL.labels(
            endpoint="/predict",
            model_version=model_version,
        ).inc()
        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict",
            model_version=model_version,
            status="error",
        ).inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    """Score multiple model-ready feature payloads."""

    service = get_inference_service()
    model_version = service.active_model_info().model_version
    started_at = time.perf_counter()

    try:
        predictions = service.predict_batch(
            request.feature_payloads,
            entity_type=request.entity_type,
        )

        for prediction in predictions:
            if prediction.is_anomaly:
                ANOMALIES_DETECTED_TOTAL.labels(
                    endpoint="/predict/batch",
                    model_version=prediction.model_version,
                ).inc()

        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict/batch",
            model_version=model_version,
            status="success",
        ).inc()

        PREDICTION_LATENCY_SECONDS.labels(
            endpoint="/predict/batch",
            model_version=model_version,
        ).observe(time.perf_counter() - started_at)

        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(**prediction_to_dict(prediction))
                for prediction in predictions
            ],
            prediction_count=len(predictions),
            model_version=model_version,
        )

    except OnlineInferenceError as exc:
        PREDICTION_ERRORS_TOTAL.labels(
            endpoint="/predict/batch",
            model_version=model_version,
        ).inc()
        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict/batch",
            model_version=model_version,
            status="error",
        ).inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in text format."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/admin/rollback", response_model=RollbackStubResponse)
def rollback_stub() -> RollbackStubResponse:
    """Return rollback stub status without mutating active model config."""

    active_version: str | None = None
    if state.inference_service is not None:
        active_version = state.inference_service.active_model_info().model_version

    return RollbackStubResponse(
        status="planned",
        action_taken=False,
        reason=(
            "Rollback endpoint is intentionally stubbed in Checkpoint 10. "
            "Actual rollback controls are implemented in the rollback checkpoint."
        ),
        active_model_version=active_version,
    )
