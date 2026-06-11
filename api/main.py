"""FastAPI application for Project 4 online anomaly inference."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from anomaly_detection.metrics import (
    observe_prediction_latency_ms,
    prometheus_content_type,
    record_anomaly_detected,
    record_prediction_error,
    record_prediction_request,
    render_prometheus_metrics,
    set_active_model_version,
)
from anomaly_detection.prediction_logging import write_online_predictions_jsonl

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



class AppState:
    """Mutable application state."""

    inference_service: OnlineInferenceService | None = None


state = AppState()


def persist_online_prediction_evidence(
    prediction_records: list[dict[str, Any]],
) -> None:
    """Persist online prediction evidence without breaking the API response path."""

    try:
        write_online_predictions_jsonl(prediction_records)
    except Exception:
        logger.exception("Failed to persist online prediction evidence")


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
        set_active_model_version(
            model_name=info.model_name,
            model_version=info.model_version,
        )
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
            record_anomaly_detected(
                endpoint="/predict",
                model_version=prediction.model_version,
            )

        record_prediction_request(
            endpoint="/predict",
            model_version=prediction.model_version,
            status="success",
        )

        observe_prediction_latency_ms(
            endpoint="/predict",
            model_version=prediction.model_version,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )

        prediction_payload = prediction_to_dict(prediction)
        persist_online_prediction_evidence([prediction_payload])

        return PredictionResponse(**prediction_payload)

    except OnlineInferenceError as exc:
        record_prediction_error(
            endpoint="/predict",
            model_version=model_version,
        )
        record_prediction_request(
            endpoint="/predict",
            model_version=model_version,
            status="error",
        )
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
                record_anomaly_detected(
                    endpoint="/predict/batch",
                    model_version=prediction.model_version,
                )

        record_prediction_request(
            endpoint="/predict/batch",
            model_version=model_version,
            status="success",
        )

        observe_prediction_latency_ms(
            endpoint="/predict/batch",
            model_version=model_version,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )

        prediction_payloads = [
            prediction_to_dict(prediction)
            for prediction in predictions
        ]
        persist_online_prediction_evidence(prediction_payloads)

        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(**prediction_payload)
                for prediction_payload in prediction_payloads
            ],
            prediction_count=len(predictions),
            model_version=model_version,
        )

    except OnlineInferenceError as exc:
        record_prediction_error(
            endpoint="/predict/batch",
            model_version=model_version,
        )
        record_prediction_request(
            endpoint="/predict/batch",
            model_version=model_version,
            status="error",
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in text format."""

    return Response(
        content=render_prometheus_metrics(),
        media_type=prometheus_content_type(),
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
