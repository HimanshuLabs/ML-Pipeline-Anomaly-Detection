"""MLflow experiment tracking: a record of training runs, never the source of truth.

Tests track to a sqlite store under tmp_path (fixture mlflow_tracking); no server
and no network. Without the fixture MLFLOW_TRACKING_URI is unset (conftest).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from anomaly_detection import registry
from tests.conftest import REPO_ROOT, _sample_feature_frame
from tests.test_retrain import _unregistered_version, _write_real_snapshot

BUNDLE_FILES = ("baseline_stats.json", "feature_schema.json", "metadata.json", "model.joblib")
UNREACHABLE_URI = "http://127.0.0.1:9"  # discard port: connection refused at once


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _client(uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def _runs(uri: str) -> list:
    from anomaly_detection import tracking

    experiment = _client(uri).get_experiment_by_name(tracking.EXPERIMENT_NAME)
    return [] if experiment is None else _client(uri).search_runs([experiment.experiment_id])


def _bundle_artifacts(uri: str, run_id: str) -> list[str]:
    return sorted(Path(item.path).name for item in _client(uri).list_artifacts(run_id, "bundle"))


def _python(code: str) -> str:
    """Run code in a fresh interpreter without MLFLOW_TRACKING_URI."""

    env = {key: value for key, value in os.environ.items() if key != "MLFLOW_TRACKING_URI"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    ).stdout


def _demo_snapshot(tmp_path: Path):
    from anomaly_detection.training_snapshot import TrainingSnapshotConfig, create_training_snapshot

    return create_training_snapshot(
        _sample_feature_frame(row_count=48),
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path / "snapshots"),
        source_tables=["source_table_tracking"],
    )


# --- tracking off ----------------------------------------------------------------------


def test_tracking_off_without_uri_never_imports_mlflow() -> None:
    out = _python(
        "import sys\n"
        "from anomaly_detection import retrain, tracking, training\n"
        "with tracking.training_run(entrypoint='train', model_version='v900') as run:\n"
        "    assert run.run_id is None and run.experiment_id is None\n"
        "print('mlflow' in sys.modules)\n"
    )
    assert out.strip() == "False"


def test_importing_the_api_does_not_import_mlflow() -> None:
    out = _python(
        "import sys\n"
        "import api.main\n"
        "print(sorted(name for name in sys.modules if name.split('.')[0] == 'mlflow'))\n"
    )
    assert out.strip() == "[]"


# --- tracked training ------------------------------------------------------------------


def test_training_cli_records_a_run_that_metadata_links_to(
    tmp_path: Path, mlflow_tracking: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anomaly_detection import training

    snapshot = _demo_snapshot(tmp_path)
    monkeypatch.setitem(
        training.train_isolation_forest_from_snapshot.__kwdefaults__,
        "registry_path",
        tmp_path / "empty_registry.json",
    )
    monkeypatch.setattr(sys, "argv", [
        "training", "--snapshot", str(snapshot.features_path),
        "--model-version", "v900", "--artifact-root", str(tmp_path / "artifacts"),
    ])

    training.main()

    artifact_dir = tmp_path / "artifacts" / "isolation_forest" / "model_version=v900"
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    (run,) = _runs(mlflow_tracking)
    assert metadata["mlflow_run_id"] == run.info.run_id
    assert metadata["mlflow_experiment_id"] == run.info.experiment_id
    assert run.info.run_name == "v900-train"
    assert run.info.status == "FINISHED"

    tags = run.data.tags
    assert {key: tags[key] for key in (
        "p4.model_name", "p4.model_version", "p4.entrypoint", "p4.registry_status_at_training"
    )} == {
        "p4.model_name": "isolation_forest",
        "p4.model_version": "v900",
        "p4.entrypoint": "train",
        "p4.registry_status_at_training": "candidate",
    }
    assert tags["p4.snapshot_id"] == metadata["training_snapshot_metadata"]["snapshot_id"]
    assert tags["p4.feature_set_fingerprint"] == metadata["feature_set_fingerprint"]
    assert tags["p4.model_sha256"] == _sha256(artifact_dir / "model.joblib")
    assert "authoritative" in tags["mlflow.note.content"]
    assert "p4.backfilled" not in tags

    params = run.data.params
    assert params["model.n_estimators"] == str(metadata["model_parameters"]["n_estimators"])
    assert params["model.contamination"] == str(metadata["model_parameters"]["contamination"])
    assert params["feature_schema_version"] == metadata["feature_schema_version"]
    assert params["runtime.numpy"] == metadata["runtime_versions"]["numpy"]
    metrics = run.data.metrics
    assert metrics["train.rows"] == metadata["row_count"]
    assert metrics["train.anomaly_rate"] == pytest.approx(metadata["baseline_anomaly_rate"])
    assert _bundle_artifacts(mlflow_tracking, run.info.run_id) == sorted(BUNDLE_FILES)


def test_retrain_records_segment_and_evaluation_metrics(
    tmp_path: Path, mlflow_tracking: str
) -> None:
    from anomaly_detection.retrain import retrain

    snapshot = _write_real_snapshot(tmp_path / "snapshots")
    version = _unregistered_version(registry.DEFAULT_REGISTRY_PATH)

    result = retrain(
        snapshot=str(snapshot.snapshot_path),
        model_version=version,
        artifact_root=tmp_path / "artifacts",
    )

    (run,) = _runs(mlflow_tracking)
    metadata = json.loads(Path(result.artifact.metadata_path).read_text(encoding="utf-8"))
    assert metadata["mlflow_run_id"] == run.info.run_id
    assert run.info.run_name == f"{version}-retrain"
    assert run.info.status == "FINISHED"
    assert run.data.tags["p4.entrypoint"] == "retrain"
    assert run.data.params["segments"] == ",".join(sorted(metadata["segments"]))

    metrics = run.data.metrics
    report = result.evaluation_report
    for entity_type in metadata["segments"]:
        assert f"segment.{entity_type}.anomaly_rate" in metrics
        assert f"segment.{entity_type}.score_p95" in metrics
        assert metrics[f"segment.{entity_type}.rows"] == metadata["segments"][entity_type][
            "row_count"
        ]
    assert metrics["eval.overall.anomaly_rate"] == pytest.approx(
        report["overall"]["anomaly_rate"]
    )
    assert metrics["eval.campaign.score_p50"] == pytest.approx(
        report["entity_types"]["campaign"]["score_p50"]
    )
    assert _bundle_artifacts(mlflow_tracking, run.info.run_id) == sorted(
        (*BUNDLE_FILES, "evaluation_report.json")
    )


def test_a_failed_training_run_is_marked_failed(mlflow_tracking: str) -> None:
    from anomaly_detection import tracking

    with (
        pytest.raises(RuntimeError, match="boom"),
        tracking.training_run(entrypoint="train", model_version="v901"),
    ):
        raise RuntimeError("boom")

    (run,) = _runs(mlflow_tracking)
    assert run.info.status == "FAILED"


def test_ending_a_failed_run_never_hides_the_original_error(
    mlflow_tracking: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from mlflow.tracking import MlflowClient

    from anomaly_detection import tracking

    def _store_down(*args, **kwargs):
        raise ConnectionError("tracking store down")

    monkeypatch.setenv("PROJECT4_MLFLOW_REQUIRED", "true")  # strict must not matter here
    monkeypatch.setattr(MlflowClient, "set_terminated", _store_down)

    with (
        pytest.raises(RuntimeError, match="boom") as raised,
        tracking.training_run(entrypoint="train", model_version="v902"),
    ):
        raise RuntimeError("boom")

    assert raised.type is RuntimeError  # not TrackingError
    assert any("tracking store down" in r.getMessage() for r in caplog.records)


# --- failure semantics -----------------------------------------------------------------


def test_unreachable_tracking_fails_open_and_training_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from anomaly_detection.retrain import retrain

    monkeypatch.setenv("MLFLOW_TRACKING_URI", UNREACHABLE_URI)
    for name in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    snapshot = _write_real_snapshot(tmp_path / "snapshots")
    version = _unregistered_version(registry.DEFAULT_REGISTRY_PATH)

    started = time.monotonic()
    result = retrain(
        snapshot=str(snapshot.snapshot_path),
        model_version=version,
        artifact_root=tmp_path / "artifacts",
    )

    assert time.monotonic() - started < 60
    metadata = json.loads(Path(result.artifact.metadata_path).read_text(encoding="utf-8"))
    assert metadata["mlflow_run_id"] is None
    assert metadata["mlflow_experiment_id"] is None
    assert result.registry_entry.status == "candidate"
    assert any("MLflow tracking is unavailable" in r.getMessage() for r in caplog.records)
    # The short preflight settings apply to the preflight only.
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ
    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ


def test_strict_tracking_refuses_before_any_file_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from anomaly_detection import retrain as retrain_module
    from anomaly_detection.tracking import TrackingError

    monkeypatch.setenv("MLFLOW_TRACKING_URI", UNREACHABLE_URI)
    monkeypatch.setenv("PROJECT4_MLFLOW_REQUIRED", "true")
    snapshot = _write_real_snapshot(tmp_path / "snapshots")
    registry_before = registry.DEFAULT_REGISTRY_PATH.read_bytes()
    version = _unregistered_version(registry.DEFAULT_REGISTRY_PATH)

    with pytest.raises(TrackingError):
        retrain_module.retrain(
            snapshot=str(snapshot.snapshot_path),
            model_version=version,
            artifact_root=tmp_path / "artifacts",
        )
    exit_code = retrain_module.main([
        "--snapshot", str(snapshot.snapshot_path), "--model-version", version,
        "--artifact-root", str(tmp_path / "artifacts"),
    ])

    assert exit_code == 1
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()
    assert registry.DEFAULT_REGISTRY_PATH.read_bytes() == registry_before


# --- backfill --------------------------------------------------------------------------


def test_backfill_records_v002_and_v004_once_and_leaves_their_bundles_byte_identical(
    mlflow_tracking: str, capsys: pytest.CaptureFixture
) -> None:
    from anomaly_detection import tracking

    artifact_root = REPO_ROOT / "artifacts" / "models"
    bundles = {
        version: artifact_root / "isolation_forest" / f"model_version={version}"
        for version in ("v002", "v004")
    }
    files = sorted(path for bundle in bundles.values() for path in bundle.iterdir())
    before = {path: _sha256(path) for path in files}
    argv = [
        "backfill", "--model-version", "v002", "--model-version", "v004",
        "--artifact-root", str(artifact_root),
    ]

    assert tracking.main(argv) == 0
    first_runs = {run.info.run_name: run.info.run_id for run in _runs(mlflow_tracking)}
    assert tracking.main(argv) == 0  # idempotent: found by tag, nothing new

    assert {path: _sha256(path) for path in files} == before
    runs = {run.info.run_name: run for run in _runs(mlflow_tracking)}
    assert sorted(runs) == ["v002-backfill", "v004-backfill"]
    assert {name: run.info.run_id for name, run in runs.items()} == first_runs
    assert capsys.readouterr().out.count("SKIP") == 2

    for version, bundle in bundles.items():
        run = runs[f"{version}-backfill"]
        assert run.info.status == "FINISHED"
        assert run.data.tags["p4.backfilled"] == "true"
        assert run.data.tags["p4.entrypoint"] == "backfill"
        assert run.data.tags["p4.model_version"] == version
        assert run.data.tags["p4.model_sha256"] == _sha256(bundle / "model.joblib")
        expected = sorted(path.name for path in bundle.iterdir())
        assert _bundle_artifacts(mlflow_tracking, run.info.run_id) == expected
    v004 = runs["v004-backfill"].data.metrics
    assert "eval.overall.anomaly_rate" in v004 and "segment.campaign.anomaly_rate" in v004


def test_backfill_needs_a_tracking_uri(capsys: pytest.CaptureFixture) -> None:
    from anomaly_detection import tracking

    assert tracking.main(["backfill", "--model-version", "v004"]) == 1
    assert "MLFLOW_TRACKING_URI" in capsys.readouterr().err


# --- static guards ---------------------------------------------------------------------

MLFLOW_MODEL_REGISTRY_API = re.compile(
    r"\b(register_model|registered_model\w*|create_model_version|log_model"
    r"|transition_model_version_stage|\w*model_version_alias|\w*registered_model_alias)\b"
)


def test_mlflow_model_registry_is_never_used() -> None:
    tracking_source = (REPO_ROOT / "src" / "anomaly_detection" / "tracking.py").read_text(
        encoding="utf-8"
    )
    assert "from mlflow" in tracking_source or "import mlflow" in tracking_source

    offenders = []
    for root in ("src", "api", "scripts"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "mlflow" not in text:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if MLFLOW_MODEL_REGISTRY_API.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders


def test_compose_mlflow_service_uses_its_own_postgres_database_and_stays_local() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["mlflow"]
    command = " ".join(service["entrypoint"])
    environment = service["environment"]

    assert service["image"] == "project4-anomaly-tools:local"  # runtime image has no mlflow
    assert "build" not in service and "restart" not in service
    assert service["ports"] == ["127.0.0.1:5004:5000"]
    assert service["depends_on"] == {"anomaly-postgres": {"condition": "service_healthy"}}
    # The password only ever travels through the environment, never the command line.
    assert environment["MLFLOW_BACKEND_STORE_URI"] == (
        "postgresql+psycopg2://project4_mlflow:${PROJECT4_MLFLOW_DB_PASSWORD:-}"
        "@anomaly-postgres:5432/project4_mlflow"
    )
    assert environment["PROJECT4_MLFLOW_DB_PASSWORD"] == "${PROJECT4_MLFLOW_DB_PASSWORD:-}"
    assert 'if [ -z "$$PROJECT4_MLFLOW_DB_PASSWORD" ]' in command  # refuses without it
    assert "exec mlflow server" in command
    assert "--backend-store-uri" not in command and "sqlite" not in command
    assert "--workers" not in command  # MLflow's default
    # The existing volume (it already holds mlflow-artifacts/), at the tools image's own
    # app-owned path; recorded runs keep resolving under the same subfolder.
    assert "--serve-artifacts --artifacts-destination /var/lib/mlflow/mlflow-artifacts" in command
    assert service["volumes"] == ["mlflow_artifacts:/var/lib/mlflow"]
    assert "/app/logs" not in command
    assert "mlflow_artifacts" in compose["volumes"]
    assert "/health" in " ".join(service["healthcheck"]["test"])
    assert environment["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert environment["GIT_PYTHON_REFRESH"] == "quiet"


def test_configs_drop_the_file_store_and_document_the_tracking_settings() -> None:
    env_lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    model_config = yaml.safe_load((REPO_ROOT / "configs" / "model_config.yaml").read_text())

    assert "mlflow_tracking_uri" not in model_config["training"]
    assert not any("file:./mlruns" in line for line in env_lines)
    for line in (
        "MLFLOW_TRACKING_URI=http://127.0.0.1:5004",
        "PROJECT4_MLFLOW_REQUIRED=false",
        "PROJECT4_MLFLOW_DB_PASSWORD=",
        "MLFLOW_DISABLE_TELEMETRY=true",
        "GIT_PYTHON_REFRESH=quiet",
    ):
        assert line in env_lines, line
    for path in (
        REPO_ROOT / "docker-compose.yml",
        REPO_ROOT / ".env.example",
        REPO_ROOT / "configs" / "model_config.yaml",
        *sorted((REPO_ROOT / "src").rglob("*.py")),
    ):
        assert "MLFLOW_ALLOW_FILE_STORE" not in path.read_text(encoding="utf-8"), path


SETUP_COMMAND = (
    "PROJECT4_MLFLOW_DB_PASSWORD=\"$(grep '^PROJECT4_MLFLOW_DB_PASSWORD=' .env | cut -d= -f2-)\""
    " \\\n"
    "  docker compose exec -T -e PROJECT4_MLFLOW_DB_PASSWORD anomaly-postgres \\\n"
    "  psql -v ON_ERROR_STOP=1 -U manshu -d postgres < sql/create_mlflow_database.sql"
)


def test_mlflow_database_script_creates_a_dedicated_role_and_database_only() -> None:
    sql = (REPO_ROOT / "sql" / "create_mlflow_database.sql").read_text(encoding="utf-8")

    assert "\\getenv mlflow_password PROJECT4_MLFLOW_DB_PASSWORD" in sql
    assert "refusing" in sql  # an empty password stops the script
    assert "CREATE ROLE project4_mlflow LOGIN" in sql
    assert "NOSUPERUSER NOCREATEDB NOCREATEROLE" in sql
    assert "PASSWORD :'mlflow_password'" in sql
    assert not re.search(r"PASSWORD\s+'", sql)  # never a literal password
    assert "CREATE DATABASE project4_mlflow OWNER project4_mlflow" in sql
    assert "\\gexec" in sql  # role and database only when absent
    assert "REVOKE ALL ON DATABASE project4_mlflow FROM PUBLIC;" in sql
    assert "project4_ml " not in sql and "project4_ml;" not in sql
    # Run through compose against the maintenance database; no container name.
    assert "-- " + SETUP_COMMAND.replace("\n", "\n-- ") in sql
    assert "project4_postgres" not in sql
    assert "set -a" not in sql  # only the one variable is read from .env


def test_readme_documents_the_generated_password_and_the_setup_command() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    unindented = "\n".join(line.strip() for line in readme.splitlines())

    assert "openssl rand -hex 24" in readme
    assert "\n".join(line.strip() for line in SETUP_COMMAND.splitlines()) in unindented
    assert "MLflow is not the source of truth" in readme
    assert "project4_postgres psql" not in readme
    assert "set -a; . ./.env; set +a" not in readme  # only the one variable is read
    assert "stays on disk unregistered" in readme  # strict-mode leftover bundle
