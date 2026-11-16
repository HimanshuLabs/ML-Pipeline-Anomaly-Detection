"""MLflow experiment tracking for Project 4 training runs.

A tracking run is a record of how a model version was trained (params, per-segment
metrics, the artifact bundle). It is never the source of truth: the Postgres model
registry and the active-model pointer decide promotion, rollback and serving, and
MLflow's model registry is not used.

Tracking is on when MLFLOW_TRACKING_URI is set; otherwise mlflow is never imported.
PROJECT4_MLFLOW_REQUIRED=true refuses training (before any file is written) when the
tracking server cannot be reached; by default a warning is logged and training goes
on with mlflow_run_id null.

Usage:
    python -m anomaly_detection.tracking backfill --model-version v002 --model-version v004
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
REQUIRED_ENV = "PROJECT4_MLFLOW_REQUIRED"
EXPERIMENT_NAME = "project4-isolation-forest"
DEFAULT_MODEL_NAME = "isolation_forest"
REGISTRY_STATUS_AT_TRAINING = "candidate"
BUNDLE_ARTIFACT_PATH = "bundle"
BUNDLE_FILES = (
    "metadata.json",
    "feature_schema.json",
    "baseline_stats.json",
    "evaluation_report.json",
    "model.joblib",
)
RUN_DESCRIPTION = (
    "Record of a Project 4 training run. The Postgres model registry and the "
    "active-model pointer are authoritative for promotion, rollback and serving; "
    "this run never decides which model is served."
)

# Set before mlflow is imported: no usage telemetry, no GitPython warnings (no git in the image).
MLFLOW_PROCESS_ENV = {"MLFLOW_DISABLE_TELEMETRY": "true", "GIT_PYTHON_REFRESH": "quiet"}
# Applied only while checking that the tracking server answers.
PREFLIGHT_ENV = {"MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "10"}
TRUE_VALUES = {"1", "true", "yes", "on"}


class TrackingError(RuntimeError):
    """MLflow tracking failed while PROJECT4_MLFLOW_REQUIRED is true (or for backfill)."""


def tracking_uri() -> str | None:
    """MLFLOW_TRACKING_URI, or None when tracking is off."""

    value = os.environ.get(TRACKING_URI_ENV, "").strip()
    return value or None


def tracking_required() -> bool:
    return os.environ.get(REQUIRED_ENV, "").strip().lower() in TRUE_VALUES


def display_uri(uri: str) -> str:
    """The URI without any user:password part, safe to log."""

    parts = urllib.parse.urlsplit(uri)
    if parts.username is None and parts.password is None:
        return uri
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


def _mlflow_client(uri: str) -> Any:
    os.environ.update(MLFLOW_PROCESS_ENV)
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


@contextmanager
def _preflight_env() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in PREFLIGHT_ENV}
    os.environ.update(PREFLIGHT_ENV)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _local_artifact_location(uri: str) -> str | None:
    """sqlite stores keep artifacts next to the database; servers choose their own."""

    if not uri.startswith("sqlite:///"):
        return None
    database = Path(uri.removeprefix("sqlite:///"))
    return (database.parent / "artifacts").resolve().as_uri()


def _experiment_id(client: Any, uri: str) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return str(experiment.experiment_id)
    return str(
        client.create_experiment(
            EXPERIMENT_NAME, artifact_location=_local_artifact_location(uri)
        )
    )


def _connect(uri: str) -> tuple[Any, str]:
    """Client and experiment id, after a quick check that the store answers."""

    client = _mlflow_client(uri)
    with _preflight_env():
        client.search_experiments(max_results=1)
        experiment_id = _experiment_id(client, uri)
    return client, experiment_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def training_params(
    metadata: Mapping[str, Any], extra: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Params of a model version from its metadata.json."""

    snapshot = metadata.get("training_snapshot_metadata") or {}
    params: dict[str, Any] = {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "algorithm": metadata.get("algorithm"),
        "model_kind": metadata.get("model_kind"),
        "feature_schema_version": metadata.get("feature_schema_version"),
        "feature_set_fingerprint": metadata.get("feature_set_fingerprint"),
        "feature_count": len(metadata.get("feature_names") or []),
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_type": snapshot.get("snapshot_type"),
    }
    for name, value in (metadata.get("model_parameters") or {}).items():
        params[f"model.{name}"] = value
    for name, value in (metadata.get("runtime_versions") or {}).items():
        params[f"runtime.{name}"] = value
    segments = metadata.get("segments") or {}
    if segments:
        params["segments"] = ",".join(sorted(segments))
    params.update(extra or {})
    return {name: str(value) for name, value in params.items() if value is not None}


def training_metrics(
    metadata: Mapping[str, Any], baseline_stats: Mapping[str, Any] | None
) -> dict[str, float]:
    """Training-set metrics overall and per segment."""

    metrics: dict[str, Any] = {
        "train.rows": metadata.get("row_count"),
        "train.anomaly_rate": metadata.get("baseline_anomaly_rate"),
        "train.feature_count": len(metadata.get("feature_names") or []),
    }
    for entity_type, segment in (metadata.get("segments") or {}).items():
        metrics[f"segment.{entity_type}.rows"] = segment.get("row_count")
        metrics[f"segment.{entity_type}.feature_count"] = len(segment.get("feature_names") or [])
    for entity_type, stats in ((baseline_stats or {}).get("segments") or {}).items():
        for name in ("anomaly_rate", "anomaly_count", "score_p5", "score_p50", "score_p95"):
            metrics[f"segment.{entity_type}.{name}"] = stats.get(name)
    return {name: float(value) for name, value in metrics.items() if _is_number(value)}


def evaluation_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """evaluation_report.json values, overall and per entity type."""

    groups = {"overall": report.get("overall") or {}, **(report.get("entity_types") or {})}
    return {
        f"eval.{group}.{name}": float(value)
        for group, stats in groups.items()
        for name, value in stats.items()
        if _is_number(value)
    }


def _tags(**values: Any) -> dict[str, str]:
    return {f"p4.{name}": str(value) for name, value in values.items() if value is not None}


@dataclass
class TrainingRun:
    """One MLflow run, or an inactive stand-in when tracking is off or unavailable."""

    run_id: str | None = None
    experiment_id: str | None = None
    required: bool = False
    client: Any = None

    @property
    def active(self) -> bool:
        return self.run_id is not None

    def _record(self, action: str, record: Any) -> None:
        if not self.active:
            return
        try:
            record()
        except Exception as exc:
            if self.required:
                raise TrackingError(f"MLflow tracking failed to {action}: {exc}") from exc
            LOGGER.warning("MLflow tracking failed to %s; training goes on: %s", action, exc)

    def _log(
        self,
        params: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        from mlflow.entities import Metric, Param, RunTag

        now = int(time.time() * 1000)
        param_items = [Param(key, value) for key, value in (params or {}).items()]
        for start in range(0, len(param_items), 100):
            self.client.log_batch(self.run_id, params=param_items[start:start + 100])
        metric_items = [Metric(key, value, now, 0) for key, value in (metrics or {}).items()]
        for start in range(0, len(metric_items), 1000):
            self.client.log_batch(self.run_id, metrics=metric_items[start:start + 1000])
        if tags:
            self.client.log_batch(
                self.run_id, tags=[RunTag(key, value) for key, value in tags.items()]
            )

    def log_bundle(
        self,
        *,
        metadata: Mapping[str, Any],
        artifact_dir: Path,
        baseline_stats: Mapping[str, Any] | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Params, training metrics, lineage tags and the bundle files of a model version."""

        def record() -> None:
            snapshot = metadata.get("training_snapshot_metadata") or {}
            model_path = artifact_dir / "model.joblib"
            self._log(
                params=training_params(metadata, extra_params),
                metrics=training_metrics(metadata, baseline_stats),
                tags=_tags(
                    snapshot_id=snapshot.get("snapshot_id") or "unknown",
                    feature_set_fingerprint=metadata.get("feature_set_fingerprint") or "unknown",
                    model_sha256=sha256_file(model_path) if model_path.is_file() else None,
                ),
            )
            for name in BUNDLE_FILES:
                path = artifact_dir / name
                if path.is_file():
                    self.client.log_artifact(self.run_id, str(path), BUNDLE_ARTIFACT_PATH)

        self._record("log the model bundle", record)

    def log_evaluation(
        self, *, report: Mapping[str, Any], report_path: Path | None = None
    ) -> None:
        """evaluation_report.json metrics, and the file itself when report_path is given."""

        def record() -> None:
            self._log(metrics=evaluation_metrics(report))
            if report_path is not None:
                self.client.log_artifact(self.run_id, str(report_path), BUNDLE_ARTIFACT_PATH)

        self._record("log the evaluation report", record)

    def end(self, status: str) -> None:
        self._record("end the run", lambda: self.client.set_terminated(self.run_id, status))

    def end_quietly(self, status: str) -> None:
        """End the run on an error path: never raises, even when tracking is required,
        so the error that ended the run is the one that propagates."""

        if not self.active:
            return
        try:
            self.client.set_terminated(self.run_id, status)
        except Exception as exc:
            LOGGER.warning(
                "MLflow tracking failed to end run %s as %s: %s", self.run_id, status, exc
            )


def _create_run(
    client: Any,
    experiment_id: str,
    *,
    entrypoint: str,
    model_name: str,
    model_version: str,
    extra_tags: Mapping[str, str] | None = None,
) -> str:
    tags = {
        **_tags(
            model_name=model_name,
            model_version=model_version,
            entrypoint=entrypoint,
            registry_status_at_training=REGISTRY_STATUS_AT_TRAINING,
        ),
        "mlflow.note.content": RUN_DESCRIPTION,
        **(extra_tags or {}),
    }
    run = client.create_run(experiment_id, run_name=f"{model_version}-{entrypoint}", tags=tags)
    return str(run.info.run_id)


def start_training_run(
    *, entrypoint: str, model_version: str, model_name: str = DEFAULT_MODEL_NAME
) -> TrainingRun:
    """Start a run; inactive when tracking is off, or unavailable and not required."""

    uri = tracking_uri()
    required = tracking_required()
    if uri is None:
        LOGGER.info("MLflow tracking is off (%s is not set)", TRACKING_URI_ENV)
        return TrainingRun(required=required)

    try:
        client, experiment_id = _connect(uri)
        run_id = _create_run(
            client,
            experiment_id,
            entrypoint=entrypoint,
            model_name=model_name,
            model_version=model_version,
        )
    except Exception as exc:
        message = f"MLflow tracking is unavailable at {display_uri(uri)}: {exc}"
        if required:
            raise TrackingError(f"{message} ({REQUIRED_ENV} is true)") from exc
        LOGGER.warning("%s; training goes on with mlflow_run_id null", message)
        return TrainingRun(required=required)

    LOGGER.info("MLflow run %s started in experiment %s", run_id, EXPERIMENT_NAME)
    return TrainingRun(
        run_id=run_id, experiment_id=experiment_id, required=required, client=client
    )


@contextmanager
def training_run(
    *, entrypoint: str, model_version: str, model_name: str = DEFAULT_MODEL_NAME
) -> Iterator[TrainingRun]:
    """A training run, ended FINISHED or (when the body raises) FAILED."""

    run = start_training_run(
        entrypoint=entrypoint, model_version=model_version, model_name=model_name
    )
    try:
        yield run
    except BaseException:
        run.end_quietly("FAILED")
        raise
    run.end("FINISHED")


# --- backfill --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_backfill(client: Any, experiment_id: str, model_version: str) -> str | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=(
            "tags.`p4.backfilled` = 'true' "
            f"and tags.`p4.model_version` = '{model_version}' "
            "and attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    return str(runs[0].info.run_id) if runs else None


def backfill(
    model_versions: list[str],
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, tuple[str, bool]]:
    """Record existing model versions from their bundle files (read only).

    Idempotent: a version with a finished backfilled run is skipped. Returns
    {model_version: (run_id, created)}.
    """

    from anomaly_detection.training import load_training_config

    uri = tracking_uri()
    if uri is None:
        raise TrackingError(f"{TRACKING_URI_ENV} is not set; nothing to backfill into.")
    config = load_training_config(artifact_root_override=artifact_root)
    try:
        client, experiment_id = _connect(uri)
    except Exception as exc:
        raise TrackingError(f"MLflow tracking is unavailable at {display_uri(uri)}: {exc}") from exc

    results: dict[str, tuple[str, bool]] = {}
    for model_version in model_versions:
        bundle = config.artifact_root / config.model_name / f"model_version={model_version}"
        metadata = _read_json(bundle / "metadata.json")
        if metadata is None:
            raise TrackingError(f"No model bundle with metadata.json at {bundle}")

        existing = _existing_backfill(client, experiment_id, model_version)
        if existing is not None:
            results[model_version] = (existing, False)
            continue

        run = TrainingRun(
            run_id=_create_run(
                client,
                experiment_id,
                entrypoint="backfill",
                model_name=str(metadata.get("model_name") or config.model_name),
                model_version=model_version,
                extra_tags={"p4.backfilled": "true"},
            ),
            experiment_id=experiment_id,
            required=True,
            client=client,
        )
        try:
            run.log_bundle(
                metadata=metadata,
                artifact_dir=bundle,
                baseline_stats=_read_json(bundle / "baseline_stats.json"),
            )
            report = _read_json(bundle / "evaluation_report.json")
            if report is not None:  # the file itself is already part of the bundle
                run.log_evaluation(report=report)
        except BaseException:
            run.end_quietly("FAILED")
            raise
        run.end("FINISHED")
        results[model_version] = (str(run.run_id), True)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLflow tracking utilities (MLflow is a record, not the source of truth)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    backfill_parser = commands.add_parser(
        "backfill",
        help="Record existing model versions as runs tagged p4.backfilled=true (idempotent).",
    )
    backfill_parser.add_argument(
        "--model-version", action="append", required=True, dest="model_versions",
        help="Model version to backfill; repeat for several (e.g. v002 and v004).",
    )
    backfill_parser.add_argument(
        "--artifact-root", default=None, help="Optional artifact root override."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from anomaly_detection.logging_config import configure_logging

    configure_logging()
    args = parse_args(argv)
    try:
        results = backfill(args.model_versions, artifact_root=args.artifact_root)
    except TrackingError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    for model_version, (run_id, created) in results.items():
        if created:
            print(f"OK: backfilled {model_version} run_id={run_id}")
        else:
            print(f"SKIP: {model_version} already backfilled run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
