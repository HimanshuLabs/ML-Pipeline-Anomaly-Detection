"""FastAPI application for Project 4 online anomaly inference."""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Response,
)

from anomaly_detection import drift
from anomaly_detection.database import insert_online_predictions
from anomaly_detection.logging_config import configure_logging
from anomaly_detection.metrics import (
    observe_prediction_latency_ms,
    prometheus_content_type,
    publish_latest_drift,
    record_anomaly_detected,
    record_model_reload,
    record_model_rollback,
    record_pointer_poll_failure,
    record_prediction_error,
    record_prediction_persist_failure,
    record_prediction_request,
    record_predictions_scored,
    render_prometheus_metrics,
    replace_active_model_version,
)
from anomaly_detection.online_inference import (
    OnlineInferenceError,
    OnlineInferenceService,
    OnlinePredictionRecord,
    model_info_to_dict,
    prediction_to_dict,
)
from anomaly_detection.prediction_logging import write_online_predictions_jsonl
from anomaly_detection.registry import (
    POINTER_SOURCE_FILE,
    POINTER_SOURCE_POSTGRES,
    RegistryError,
    database_enabled,
    materialize_active_pointer,
    read_shared_pointer_version,
    resolve_project_path,
)
from anomaly_detection.rollback import (
    RollbackError,
    RollbackServingError,
    rollback_active_model,
)
from api.schemas import (
    ActiveModelResponse,
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
    RollbackRequest,
    RollbackResponse,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "project4-online-anomaly-inference"

# Serializes /admin/rollback and pointer-poller swaps so they can never interleave.
ROLLBACK_LOCK = threading.Lock()

POINTER_POLL_SECONDS_ENV = "PROJECT4_POINTER_POLL_SECONDS"
DEFAULT_POINTER_POLL_SECONDS = 30.0

ADMIN_TOKEN_ENV = "ADMIN_TOKEN"
MIN_ADMIN_TOKEN_LENGTH = 16


def _configured_admin_token() -> str | None:
    """The server's admin token, or None when admin endpoints are disabled."""

    token = os.environ.get(ADMIN_TOKEN_ENV, "")
    return token if len(token) >= MIN_ADMIN_TOKEN_LENGTH else None


def require_admin_token(
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    """Authorize /admin/* requests by the X-Admin-Token header.

    The token is read from the environment on every request and is never logged
    or returned.
    """

    configured_token = _configured_admin_token()
    if configured_token is None:
        raise HTTPException(status_code=503, detail="admin disabled")

    if x_admin_token is None or not hmac.compare_digest(
        x_admin_token.encode("utf-8"),
        configured_token.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="invalid admin token")


admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_token)])


class DriftStatusCache:
    """Latest drift run per segment of the served model version, held in memory.

    Refreshed when a model is swapped in and on every pointer poll (database
    mode), never per request. Postgres when the database is configured, else the
    JSONL drift events. A failed refresh keeps the previous result for the same
    model version.
    """

    def __init__(self) -> None:
        self.model_name: str | None = None
        self.model_version: str | None = None
        self.summaries: dict[str | None, drift.SegmentDriftSummary] = {}
        self._refresh_failing = False

    def status_for(self, entity_type: str | None) -> str:
        """normal | warning | critical of the request's segment, else not_evaluated."""

        summaries = self.summaries
        summary = summaries.get(entity_type) or summaries.get(None)
        return summary.drift_status if summary is not None else drift.NOT_EVALUATED

    def refresh(self, *, model_name: str, model_version: str) -> None:
        try:
            if database_enabled():
                summaries = drift.read_latest_drift_database(
                    model_name=model_name, model_version=model_version
                )
            else:
                summaries = drift.read_latest_drift_jsonl(
                    drift.DEFAULT_DRIFT_EVENTS_PATH,
                    model_name=model_name,
                    model_version=model_version,
                )
        except Exception:  # noqa: BLE001 - drift status must never affect serving
            if not self._refresh_failing:
                logger.warning(
                    "Could not read the latest drift result for %s %s; keeping the cached one",
                    model_name,
                    model_version,
                    exc_info=True,
                )
                self._refresh_failing = True
            if (self.model_name, self.model_version) != (model_name, model_version):
                self.model_name, self.model_version, self.summaries = model_name, model_version, {}
            return
        self._refresh_failing = False

        same_model = (self.model_name, self.model_version) == (model_name, model_version)
        previous = self.summaries if same_model else {}
        new_runs = {
            segment
            for segment, summary in summaries.items()
            if segment not in previous or previous[segment].detected_at != summary.detected_at
        }
        self.model_name, self.model_version, self.summaries = model_name, model_version, summaries
        publish_latest_drift(model_version=model_version, summaries=summaries, new_runs=new_runs)
        for segment in sorted(new_runs, key=str):
            logger.info(
                "Loaded drift run of %s %s segment=%s: %s",
                model_name,
                model_version,
                segment or "all",
                summaries[segment].drift_status,
            )


class AppState:
    """Mutable serving state of one API process."""

    def __init__(self) -> None:
        self.inference_service: OnlineInferenceService | None = None
        self.pointer_source: str = POINTER_SOURCE_FILE
        self.pointer_version: int | None = None
        self.poller: PointerPoller | None = None
        self.drift_cache = DriftStatusCache()


def refresh_drift_status(holder: AppState) -> None:
    """Reload the latest drift result of the served model version into the cache."""

    service = holder.inference_service
    if service is None:
        return
    info = service.active_model_info()
    holder.drift_cache.refresh(model_name=info.model_name, model_version=info.model_version)


def swap_inference_service(
    holder: AppState,
    new_service: OnlineInferenceService,
    *,
    pointer_version: int | None,
    pointer_source: str,
) -> None:
    """Serve new_service from the next request on (one assignment) and update metrics."""

    holder.inference_service = new_service
    holder.pointer_version = pointer_version
    holder.pointer_source = pointer_source

    new_info = new_service.active_model_info()
    replace_active_model_version(
        model_name=new_info.model_name, model_version=new_info.model_version
    )
    refresh_drift_status(holder)


def load_service_from_shared_pointer() -> tuple[OnlineInferenceService, int]:
    """Build a serving model from the database pointer; returns it and its pointer_version."""

    with TemporaryDirectory(prefix="project4_shared_pointer_") as pointer_dir:
        pointer_path = Path(pointer_dir) / "active_model.yaml"
        pointer_version = materialize_active_pointer(pointer_path)
        return OnlineInferenceService(active_model_path=pointer_path), pointer_version


def pointer_poll_interval_seconds() -> float:
    """PROJECT4_POINTER_POLL_SECONDS (default 30)."""

    raw_value = os.environ.get(POINTER_POLL_SECONDS_ENV, "")
    if not raw_value:
        return DEFAULT_POINTER_POLL_SECONDS
    try:
        interval = float(raw_value)
    except ValueError:
        logger.warning("%s is not a number; using %s", POINTER_POLL_SECONDS_ENV, 30)
        return DEFAULT_POINTER_POLL_SECONDS
    return interval if interval > 0 else DEFAULT_POINTER_POLL_SECONDS


class PointerPoller:
    """Polls the shared pointer_version and swaps in the new model when it changes.

    The old model keeps serving if the new one fails to load or the database is
    unreachable (a WARNING once per outage, an ERROR once per failed version).
    """

    def __init__(
        self,
        holder: AppState,
        *,
        interval_seconds: float,
        read_version: Callable[[], int | None] = read_shared_pointer_version,
        load_service: Callable[
            [], tuple[OnlineInferenceService, int]
        ] = load_service_from_shared_pointer,
        lock: threading.Lock = ROLLBACK_LOCK,
    ) -> None:
        self.holder = holder
        self.interval_seconds = interval_seconds
        self._read_version = read_version
        self._load_service = load_service
        self._lock = lock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_outage = False
        self._failed_version: int | None = None

    def poll_once(self) -> bool:
        """Check the pointer once; True when a new model was swapped in."""

        try:
            version = self._read_version()
        except Exception:  # noqa: BLE001 - keep serving whatever the failure
            record_pointer_poll_failure()
            if not self._in_outage:
                logger.warning(
                    "Shared model pointer is unreachable; keeping the current model"
                )
                self._in_outage = True
            return False

        if self._in_outage:
            logger.info("Shared model pointer is reachable again")
            self._in_outage = False

        if version is None or version == self.holder.pointer_version:
            # Same model: pick up drift runs written since the last poll.
            refresh_drift_status(self.holder)
            return False

        try:
            new_service, loaded_version = self._load_service()
        except Exception:  # noqa: BLE001 - keep the old model
            record_model_reload(result="failed")
            if self._failed_version != version:
                logger.error(
                    "Could not load the model for shared pointer version %s; "
                    "keeping the current model",
                    version,
                    exc_info=True,
                )
                self._failed_version = version
            return False

        with self._lock:
            swap_inference_service(
                self.holder,
                new_service,
                pointer_version=loaded_version,
                pointer_source=POINTER_SOURCE_POSTGRES,
            )
        self._failed_version = None
        record_model_reload(result="success")
        logger.info(
            "Swapped in model %s from shared pointer version %s",
            new_service.active_model_info().model_version,
            loaded_version,
        )
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.poll_once()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pointer-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5)
            self._thread = None


state = AppState()


# Per sink: True while writes keep failing, so one ERROR is logged per failure burst.
PERSIST_FAILING: dict[str, bool] = {"jsonl": False, "postgres": False}


def _persist_to_sink(sink: str, write: Callable[[], object]) -> None:
    try:
        write()
    except Exception:  # noqa: BLE001 - evidence persistence must never affect responses
        record_prediction_persist_failure(sink=sink)
        if not PERSIST_FAILING[sink]:
            logger.exception("Failed to persist online prediction evidence to %s", sink)
            PERSIST_FAILING[sink] = True
        return

    if PERSIST_FAILING[sink]:
        logger.info("Online prediction evidence persists to %s again", sink)
        PERSIST_FAILING[sink] = False


def persist_online_prediction_evidence(
    prediction_records: list[dict[str, Any]],
) -> None:
    """Persist online prediction evidence: JSONL always, Postgres in database mode.

    Runs as a background task after the response is sent; failures are counted
    and logged but never reach the client.
    """

    _persist_to_sink("jsonl", lambda: write_online_predictions_jsonl(prediction_records))

    if database_enabled():
        _persist_to_sink("postgres", lambda: insert_online_predictions(prediction_records))


def with_cached_drift_status(prediction: OnlinePredictionRecord) -> OnlinePredictionRecord:
    """The prediction with its segment's drift status from the in-memory cache."""

    return replace(prediction, drift_status=state.drift_cache.status_for(prediction.entity_type))


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

    configure_logging()

    if _configured_admin_token() is None:
        logger.warning(
            "%s is not set or is too short; /admin endpoints are disabled",
            ADMIN_TOKEN_ENV,
        )

    use_database = database_enabled()

    try:
        service: OnlineInferenceService | None = None
        pointer_version: int | None = None
        pointer_source = POINTER_SOURCE_FILE
        if use_database:
            try:
                service, pointer_version = load_service_from_shared_pointer()
                pointer_source = POINTER_SOURCE_POSTGRES
            except Exception:  # noqa: BLE001 - fall back to the file mirror
                logger.warning(
                    "Could not load the model from the shared database pointer; "
                    "starting from the file pointer",
                    exc_info=True,
                )
        if service is None:
            service = OnlineInferenceService()

        swap_inference_service(
            state,
            service,
            pointer_version=pointer_version,
            pointer_source=pointer_source,
        )
        info = service.active_model_info()
        logger.info(
            "Loaded active anomaly model",
            extra={
                "model_name": info.model_name,
                "model_version": info.model_version,
                "feature_count": info.feature_count,
                "pointer_source": pointer_source,
            },
        )
    except Exception:  # pragma: no cover - covered by runtime smoke checks
        logger.exception("Failed to load active anomaly model")
        state.inference_service = None

    poller: PointerPoller | None = None
    if use_database:
        poller = PointerPoller(state, interval_seconds=pointer_poll_interval_seconds())
        poller.start()
        state.poller = poller

    yield

    if poller is not None:
        poller.stop()
        state.poller = None


app = FastAPI(
    title="Project 4 Online Anomaly Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/livez")
def livez() -> dict[str, str]:
    """Liveness: the process is up. Never touches the model."""

    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Readiness: 200 when a model is loaded, 503 when it is not."""

    if state.inference_service is None:
        response.status_code = 503
        return HealthResponse(
            status="degraded",
            service=SERVICE_NAME,
            active_model_version=None,
            model_loaded=False,
            pointer_source=state.pointer_source,
        )

    info = state.inference_service.active_model_info()
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        active_model_version=info.model_version,
        model_loaded=True,
        pointer_source=state.pointer_source,
    )


@app.get("/model/active", response_model=ActiveModelResponse)
def active_model() -> ActiveModelResponse:
    """Return active production model metadata."""

    service = get_inference_service()
    info = service.active_model_info()
    return ActiveModelResponse(**model_info_to_dict(info), pointer_source=state.pointer_source)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Score one model-ready feature payload; evidence is persisted after the response."""

    service = get_inference_service()
    model_version = service.active_model_info().model_version
    started_at = time.perf_counter()

    try:
        prediction = with_cached_drift_status(
            service.predict(
                request.feature_payload,
                entity_id=request.entity_id,
                entity_type=request.entity_type,
            )
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
        record_predictions_scored(
            endpoint="/predict",
            model_version=prediction.model_version,
            count=1,
        )

        observe_prediction_latency_ms(
            endpoint="/predict",
            model_version=prediction.model_version,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )

        prediction_payload = prediction_to_dict(prediction)
        background_tasks.add_task(persist_online_prediction_evidence, [prediction_payload])

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
def predict_batch(
    request: BatchPredictRequest,
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    """Score multiple payloads; evidence is persisted after the response (one transaction)."""

    service = get_inference_service()
    model_version = service.active_model_info().model_version
    started_at = time.perf_counter()

    try:
        predictions = [
            with_cached_drift_status(prediction)
            for prediction in service.predict_batch(
                request.feature_payloads,
                entity_type=request.entity_type,
            )
        ]

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
        record_predictions_scored(
            endpoint="/predict/batch",
            model_version=model_version,
            count=len(predictions),
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
        background_tasks.add_task(persist_online_prediction_evidence, prediction_payloads)

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


@admin_router.post("/rollback", response_model=RollbackResponse)
def rollback_model(request: RollbackRequest | None = None) -> dict[str, Any]:
    """Validate or apply rollback to the previous stable approved model.

    An applied rollback commits only after the target has loaded as a full
    inference service, and then swaps the served model in one assignment.
    """

    rollback_request = request or RollbackRequest()

    with ROLLBACK_LOCK:
        loaded_services: list[OnlineInferenceService] = []

        def _load_service(pointer_path: Path) -> OnlineInferenceService:
            service = OnlineInferenceService(active_model_path=pointer_path)
            loaded_services.append(service)
            return service

        try:
            rollback_event = rollback_active_model(
                rollback_reason=rollback_request.rollback_reason,
                triggered_by=rollback_request.triggered_by,
                dry_run=rollback_request.dry_run,
                verify_service=_load_service,
            )

            if rollback_request.dry_run:
                # A dry run changes nothing, so the active version is the one it started from.
                active_model_version = rollback_event.from_model_version
            else:
                # The last service loaded is the one built from the committed pointer.
                new_service = loaded_services[-1]
                pointer_source = POINTER_SOURCE_FILE
                pointer_version: int | None = None
                if database_enabled():
                    pointer_source = POINTER_SOURCE_POSTGRES
                    try:
                        pointer_version = read_shared_pointer_version(rollback_event.model_name)
                    except Exception:  # noqa: BLE001 - the poller re-syncs if this is missed
                        logger.warning("Could not read the new shared pointer_version")
                swap_inference_service(
                    state,
                    new_service,
                    pointer_version=pointer_version,
                    pointer_source=pointer_source,
                )
                new_info = new_service.active_model_info()
                record_model_rollback(
                    model_name=rollback_event.model_name,
                    from_model_version=rollback_event.from_model_version,
                    to_model_version=rollback_event.to_model_version,
                    triggered_by=rollback_event.triggered_by,
                )
                active_model_version = new_info.model_version

            return {
                "status": "validated" if rollback_request.dry_run else "applied",
                "action_taken": not rollback_request.dry_run,
                "rollback_id": rollback_event.rollback_id,
                "model_name": rollback_event.model_name,
                "from_model_version": rollback_event.from_model_version,
                "to_model_version": rollback_event.to_model_version,
                "rollback_reason": rollback_event.rollback_reason,
                "triggered_by": rollback_event.triggered_by,
                "validation_status": rollback_event.validation_status,
                "dry_run": rollback_event.dry_run,
                "active_model_version": active_model_version,
                # Stored repo-relative since 14c; the response keeps the absolute path.
                "rollback_event_path": (
                    str(resolve_project_path(rollback_event.rollback_event_path))
                    if rollback_event.rollback_event_path
                    else None
                ),
            }

        except (RollbackError, RegistryError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except RollbackServingError as exc:
            logger.error("Rollback refused or undone: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(admin_router)
