"""Tests for the Project 4 online inference FastAPI service."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from anomaly_detection.online_inference import OnlineInferenceService
from api.main import app, state

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "_registry" / "model_registry.json"


@pytest.fixture()
def inference_service() -> OnlineInferenceService:
    """Load the active online inference service once per test."""

    return OnlineInferenceService()


@pytest.fixture()
def client(inference_service: OnlineInferenceService) -> Generator[TestClient, None, None]:
    """Return a TestClient with the inference service preloaded."""

    state.inference_service = inference_service
    with TestClient(app) as test_client:
        yield test_client
    state.inference_service = None


@pytest.fixture()
def valid_feature_payload(inference_service: OnlineInferenceService) -> dict[str, float | str]:
    """Build a complete model-ready payload for the active model."""

    payload: dict[str, float | str] = {
        feature_name: 1.0
        for feature_name in inference_service.feature_names
    }
    payload["entity_id"] = "test_user_001"
    return payload


def test_health_returns_loaded_active_model(
    client: TestClient,
    active_model_version: str,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "project4-online-anomaly-inference"
    assert body["active_model_version"] == active_model_version
    assert body["model_loaded"] is True


def test_active_model_endpoint_returns_active_model_metadata(
    client: TestClient,
    active_model_pointer: dict,
    active_model_version: str,
) -> None:
    artifact_metadata_path = (
        REPO_ROOT / active_model_pointer["artifact_path"]
    ).parent / "metadata.json"
    expected_feature_count = len(
        json.loads(artifact_metadata_path.read_text(encoding="utf-8"))["feature_names"]
    )

    response = client.get("/model/active")

    assert response.status_code == 200
    body = response.json()

    assert body["model_name"] == "isolation_forest"
    assert body["model_version"] == active_model_version
    assert body["status"] == "production"
    assert body["feature_schema_version"] == active_model_pointer["feature_schema_version"]
    assert body["feature_count"] == expected_feature_count
    assert body["baseline_anomaly_rate"] is not None
    assert body["threshold"] == 0.0
    assert body["source_projects"] == ["project_1", "project_2_3"]
    assert body["snapshot_type"] == "real_source_extract"
    assert body["latency_budget_p95_ms"] == 200.0


def test_predict_scores_single_payload_with_active_model(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    active_model_version: str,
) -> None:
    response = client.post(
        "/predict",
        json={
            "entity_id": "test_user_001",
            "entity_type": "user",
            "feature_payload": valid_feature_payload,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["model_name"] == "isolation_forest"
    assert body["model_version"] == active_model_version
    assert body["entity_type"] == "user"
    assert body["entity_id"] == "test_user_001"
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["is_anomaly"], bool)
    assert body["threshold_used"] == 0.0
    assert body["drift_status"] == "not_evaluated"
    assert body["prediction_status"] == "success"
    assert body["error_message"] is None
    assert body["latency_ms"] >= 0


def test_predict_batch_scores_multiple_payloads(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    active_model_version: str,
) -> None:
    second_payload = dict(valid_feature_payload)
    second_payload["entity_id"] = "test_user_002"

    response = client.post(
        "/predict/batch",
        json={
            "entity_type": "user",
            "feature_payloads": [valid_feature_payload, second_payload],
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["prediction_count"] == 2
    assert body["model_version"] == active_model_version
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["model_version"] == active_model_version
    assert body["predictions"][1]["entity_id"] == "test_user_002"


def test_metrics_endpoint_exposes_prometheus_metrics(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
) -> None:
    prediction_response = client.post(
        "/predict",
        json={
            "entity_id": "metrics_user_001",
            "entity_type": "user",
            "feature_payload": valid_feature_payload,
        },
    )
    assert prediction_response.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "prediction_requests_total" in response.text
    assert "prediction_latency_ms" in response.text
    assert "active_model_version" in response.text


def test_rollback_endpoint_dry_run_rejects_incompatible_targets_without_mutation(
    client: TestClient,
    active_model_version: str,
    isolate_repo_state: Path,
) -> None:
    # The real registry has no compatible rollback target for the active model
    # (v001 is retired and has different feature_names), so the dry run must refuse.
    registry_models = json.loads(REAL_REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
    other_versions = [
        model["model_version"]
        for model in registry_models
        if model["model_version"] != active_model_version
    ]
    pointer_path = isolate_repo_state / "configs" / "active_model.yaml"
    registry_path = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    pointer_before = pointer_path.read_bytes()
    registry_before = registry_path.read_bytes()

    response = client.post(
        "/admin/rollback",
        json={
            "rollback_reason": "pytest dry-run validation",
            "triggered_by": "pytest",
            "dry_run": True,
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail.startswith("No compatible rollback target for isolation_forest")
    for version in other_versions:
        assert version in detail
    assert pointer_path.read_bytes() == pointer_before
    assert registry_path.read_bytes() == registry_before
    assert client.get("/model/active").json()["model_version"] == active_model_version


def test_rollback_endpoint_returns_400_for_registry_errors(
    client: TestClient,
    isolate_repo_state: Path,
) -> None:
    registry_path = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload["models"][0]["unexpected_field"] = "not in the registry schema"
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")

    response = TestClient(app, raise_server_exceptions=False).post(
        "/admin/rollback",
        json={"dry_run": True},
    )

    assert response.status_code == 400
    assert "does not match the registry schema" in response.json()["detail"]
    assert "unexpected_field" in response.json()["detail"]


def test_rollback_and_promotion_do_not_depend_on_working_directory(
    client: TestClient,
    active_model_version: str,
    isolate_repo_state: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.registry import promote_model_version

    pointer_path = isolate_repo_state / "configs" / "active_model.yaml"
    registry_path = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    unrelated_directory = tmp_path_factory.mktemp("unrelated_cwd")
    monkeypatch.chdir(unrelated_directory)

    response = client.post("/admin/rollback", json={"dry_run": True})

    # The dry run reached target selection, so it read the project-root pointer
    # and registry, not configs/ under the unrelated working directory.
    assert response.status_code == 400
    assert "No compatible rollback target" in response.json()["detail"]

    pointer_before = pointer_path.read_bytes()
    registry_before = registry_path.read_bytes()

    promote_model_version(model_name="isolation_forest", model_version=active_model_version)

    assert pointer_path.read_bytes() != pointer_before
    assert registry_path.read_bytes() != registry_before
    assert list(unrelated_directory.iterdir()) == []


def test_predict_rejects_missing_required_model_features(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "entity_id": "bad_user_001",
            "entity_type": "user",
            "feature_payload": {"one_feature_only": 1.0},
        },
    )

    assert response.status_code == 422
    assert "missing required model features" in response.json()["detail"]


@pytest.mark.parametrize("value", ["inf", "-inf"])
def test_predict_rejects_non_finite_feature(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    inference_service: OnlineInferenceService,
    value: str,
) -> None:
    feature_name = inference_service.feature_names[0]
    valid_feature_payload[feature_name] = value

    response = client.post(
        "/predict",
        json={"feature_payload": valid_feature_payload},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Input batch contains non-finite (inf/-inf) model feature values: "
        f"{feature_name}=1"
    )


def test_batch_predict_rejects_empty_payload_list(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json={
            "entity_type": "user",
            "feature_payloads": [],
        },
    )

    assert response.status_code == 422


def test_predict_persists_online_prediction_evidence(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    active_model_version: str,
    tmp_path,
    monkeypatch,
) -> None:
    """Single online prediction should append one durable evidence record."""

    from api import main as api_main

    output_path = tmp_path / "online_predictions.jsonl"

    def _write_to_tmp(records):
        from anomaly_detection.prediction_logging import write_online_predictions_jsonl

        return write_online_predictions_jsonl(records, output_path)

    monkeypatch.setattr(
        api_main,
        "write_online_predictions_jsonl",
        _write_to_tmp,
    )

    response = client.post(
        "/predict",
        json={
            "entity_id": "logged_user_001",
            "entity_type": "user",
            "feature_payload": valid_feature_payload,
        },
    )

    assert response.status_code == 200

    records = output_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert '"prediction_source":"online"' in records[0]
    assert '"entity_id":"logged_user_001"' in records[0]
    assert f'"model_version":"{active_model_version}"' in records[0]


def test_predict_batch_persists_online_prediction_evidence(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    tmp_path,
    monkeypatch,
) -> None:
    """Batch online prediction should append all prediction evidence records."""

    from api import main as api_main

    output_path = tmp_path / "online_predictions.jsonl"

    def _write_to_tmp(records):
        from anomaly_detection.prediction_logging import write_online_predictions_jsonl

        return write_online_predictions_jsonl(records, output_path)

    monkeypatch.setattr(
        api_main,
        "write_online_predictions_jsonl",
        _write_to_tmp,
    )

    second_payload = dict(valid_feature_payload)
    second_payload["entity_id"] = "logged_user_002"

    response = client.post(
        "/predict/batch",
        json={
            "entity_type": "user",
            "feature_payloads": [valid_feature_payload, second_payload],
        },
    )

    assert response.status_code == 200

    records = output_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    assert all('"prediction_source":"online"' in record for record in records)
    assert '"entity_id":"test_user_001"' in records[0]
    assert '"entity_id":"logged_user_002"' in records[1]

def test_predict_reports_latency_under_configured_budget(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    active_model_version: str,
) -> None:
    """Verify /predict reports latency below the configured local budget."""

    active_response = client.get("/model/active")
    assert active_response.status_code == 200

    latency_budget_ms = active_response.json()["latency_budget_p95_ms"]

    assert latency_budget_ms == 200.0

    response = client.post(
        "/predict",
        json={
            "entity_id": "latency_budget_user_001",
            "feature_payload": valid_feature_payload,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == active_model_version
    assert body["latency_ms"] >= 0
    assert body["latency_ms"] < latency_budget_ms


# --- health and liveness -----------------------------------------------------


def test_health_returns_503_with_same_fields_when_no_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state, "inference_service", None)

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "service": "project4-online-anomaly-inference",
        "active_model_version": None,
        "model_loaded": False,
    }


@pytest.mark.parametrize("model_loaded", [True, False], ids=["with_model", "without_model"])
def test_livez_is_200_with_and_without_a_model(
    monkeypatch: pytest.MonkeyPatch,
    inference_service: OnlineInferenceService,
    model_loaded: bool,
) -> None:
    monkeypatch.setattr(state, "inference_service", inference_service if model_loaded else None)

    response = TestClient(app).get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


# --- applied rollback swaps the served model ---------------------------------


def _install_two_model_registry(
    mirror_root: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    *,
    break_target_baseline: bool = False,
) -> tuple[Path, Path, list[str]]:
    """Replace the mirror registry with v101 (archived) and v102 (active).

    Each version gets its own copy of the trained artifact directory so a test
    can damage one without affecting the shared trained models.
    """
    import shutil

    from anomaly_detection.registry import promote_model_version, register_model

    registry_path = mirror_root / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    pointer_path = mirror_root / "configs" / "active_model.yaml"
    registry_path.unlink()

    feature_names: list[str] = []
    for version in ("v101", "v102"):
        source_dir = trained_metadata_paths[version].parent
        artifact_dir = tmp_path / "models" / f"model_version={version}"
        shutil.copytree(source_dir, artifact_dir)
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        feature_names = metadata["feature_names"]

        register_model(
            model_name="isolation_forest",
            model_version=version,
            artifact_path=str(artifact_dir / "model.joblib"),
            dataset_snapshot_id=f"snapshot_{version}",
            feature_schema_version=metadata["feature_schema_version"],
            training_timestamp=metadata["training_finished_at_utc"],
        )
        promote_model_version(model_name="isolation_forest", model_version=version)

    if break_target_baseline:
        (tmp_path / "models" / "model_version=v101" / "baseline_stats.json").unlink()

    return registry_path, pointer_path, feature_names


def _predict_version(client: TestClient, feature_names: list[str]) -> str:
    response = client.post(
        "/predict",
        json={"feature_payload": {name: 1.0 for name in feature_names}},
    )
    assert response.status_code == 200
    return response.json()["model_version"]


def _apply_rollback(client: TestClient):
    return client.post(
        "/admin/rollback",
        json={"rollback_reason": "pytest", "triggered_by": "pytest", "dry_run": False},
    )


def test_applied_rollback_serves_target_version_immediately(
    isolate_repo_state: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.registry import load_model_registry

    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths
    )

    with TestClient(app) as client:
        assert _predict_version(client, feature_names) == "v102"

        response = _apply_rollback(client)

        assert response.status_code == 200
        assert response.json()["active_model_version"] == "v101"
        assert response.json()["to_model_version"] == "v101"
        assert client.get("/model/active").json()["model_version"] == "v101"
        assert client.get("/health").json()["active_model_version"] == "v101"
        assert _predict_version(client, feature_names) == "v101"
        assert 'active_model_version{model_name="isolation_forest",model_version="v101"} 1.0' in (
            client.get("/metrics").text
        )

    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    assert pointer["active_model_version"] == "v101"
    assert {entry.model_version: entry.status for entry in load_model_registry(registry_path)} == {
        "v101": "production",
        "v102": "rolled_back",
    }


def test_rollback_target_that_cannot_serve_is_refused_before_commit(
    isolate_repo_state: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths, break_target_baseline=True
    )
    registry_before = registry_path.read_bytes()
    pointer_before = pointer_path.read_bytes()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = _apply_rollback(client)

        assert response.status_code == 500
        assert "v101" in response.json()["detail"]
        assert registry_path.read_bytes() == registry_before
        assert pointer_path.read_bytes() == pointer_before
        assert _predict_version(client, feature_names) == "v102"

    assert not (isolate_repo_state / "logs" / "alerts" / "rollback_events.jsonl").exists()


def test_rollback_failing_after_commit_restores_registry_and_pointer(
    isolate_repo_state: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import main as api_main

    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths
    )
    registry_before = registry_path.read_bytes()
    pointer_before = pointer_path.read_bytes()
    real_service_class = api_main.OnlineInferenceService

    def _service_failing_on_committed_pointer(active_model_path=None, **kwargs):
        if active_model_path is not None and Path(active_model_path) == pointer_path:
            raise RuntimeError("simulated load failure of the committed pointer")
        return real_service_class(active_model_path=active_model_path, **kwargs)

    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(
            api_main, "OnlineInferenceService", _service_failing_on_committed_pointer
        )

        response = _apply_rollback(client)

        assert response.status_code == 500
        assert "restored" in response.json()["detail"]
        assert registry_path.read_bytes() == registry_before
        assert pointer_path.read_bytes() == pointer_before
        assert _predict_version(client, feature_names) == "v102"
        assert client.get("/model/active").json()["model_version"] == "v102"

    assert not (isolate_repo_state / "logs" / "alerts" / "rollback_events.jsonl").exists()


def test_rollback_dry_run_with_compatible_target_writes_nothing(
    isolate_repo_state: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths
    )
    registry_before = registry_path.read_bytes()
    pointer_before = pointer_path.read_bytes()

    with TestClient(app) as client:
        response = client.post("/admin/rollback", json={"dry_run": True})

        assert response.status_code == 200
        assert response.json()["status"] == "validated"
        assert response.json()["to_model_version"] == "v101"
        assert response.json()["active_model_version"] == "v102"
        assert _predict_version(client, feature_names) == "v102"

    assert registry_path.read_bytes() == registry_before
    assert pointer_path.read_bytes() == pointer_before
    assert not (isolate_repo_state / "logs" / "alerts" / "rollback_events.jsonl").exists()
