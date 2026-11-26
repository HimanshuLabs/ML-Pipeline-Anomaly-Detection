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
from tests.conftest import FIXTURE_REGISTRY_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
# Used only to locate the registry inside the per-test mirror (whose content is the fixture).
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
    admin_headers: dict[str, str],
) -> None:
    # The fixture registry has no compatible rollback target for the active v004 (v001 and
    # v003 are retired, v005 is rolled_back, v002 is a single model), so the dry run must
    # refuse.
    registry_models = json.loads(FIXTURE_REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
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
        headers=admin_headers,
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
    admin_headers: dict[str, str],
) -> None:
    registry_path = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload["models"][0]["unexpected_field"] = "not in the registry schema"
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")

    response = TestClient(app, raise_server_exceptions=False).post(
        "/admin/rollback",
        json={"dry_run": True},
        headers=admin_headers,
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
    admin_headers: dict[str, str],
) -> None:
    from anomaly_detection.registry import promote_model_version

    pointer_path = isolate_repo_state / "configs" / "active_model.yaml"
    registry_path = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    unrelated_directory = tmp_path_factory.mktemp("unrelated_cwd")
    monkeypatch.chdir(unrelated_directory)

    response = client.post("/admin/rollback", json={"dry_run": True}, headers=admin_headers)

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
        "pointer_source": "file",
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


def _apply_rollback(client: TestClient, admin_headers: dict[str, str]):
    return client.post(
        "/admin/rollback",
        json={"rollback_reason": "pytest", "triggered_by": "pytest", "dry_run": False},
        headers=admin_headers,
    )


def test_applied_rollback_serves_target_version_immediately(
    isolate_repo_state: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    admin_headers: dict[str, str],
) -> None:
    from anomaly_detection.registry import load_model_registry

    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths
    )

    with TestClient(app) as client:
        assert _predict_version(client, feature_names) == "v102"

        response = _apply_rollback(client, admin_headers)

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
    admin_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths, break_target_baseline=True
    )
    registry_before = registry_path.read_bytes()
    pointer_before = pointer_path.read_bytes()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = _apply_rollback(client, admin_headers)

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
    admin_headers: dict[str, str],
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

        response = _apply_rollback(client, admin_headers)

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
    admin_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(state, "inference_service", None)
    registry_path, pointer_path, feature_names = _install_two_model_registry(
        isolate_repo_state, tmp_path, trained_metadata_paths
    )
    registry_before = registry_path.read_bytes()
    pointer_before = pointer_path.read_bytes()

    with TestClient(app) as client:
        response = client.post("/admin/rollback", json={"dry_run": True}, headers=admin_headers)

        assert response.status_code == 200
        assert response.json()["status"] == "validated"
        assert response.json()["to_model_version"] == "v101"
        assert response.json()["active_model_version"] == "v102"
        assert _predict_version(client, feature_names) == "v102"

    assert registry_path.read_bytes() == registry_before
    assert pointer_path.read_bytes() == pointer_before
    assert not (isolate_repo_state / "logs" / "alerts" / "rollback_events.jsonl").exists()


# --- admin authorization -----------------------------------------------------


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "apply"])
def test_admin_disabled_when_admin_token_is_unset(client: TestClient, dry_run: bool) -> None:
    response = client.post(
        "/admin/rollback",
        json={"dry_run": dry_run},
        headers={"X-Admin-Token": "any-value-at-all-0123456789"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "admin disabled"}


def test_admin_disabled_when_admin_token_is_too_short(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_token = "fifteen-chars-x"
    assert len(short_token) == 15
    monkeypatch.setenv("ADMIN_TOKEN", short_token)

    response = client.post(
        "/admin/rollback",
        json={"dry_run": True},
        headers={"X-Admin-Token": short_token},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "admin disabled"}


@pytest.mark.parametrize(
    "request_headers",
    [
        pytest.param({}, id="missing_header"),
        pytest.param({"X-Admin-Token": "wrong-token-0123456789abcdefghij"}, id="wrong_token"),
        pytest.param({"X-Admin-Token": "short"}, id="different_length"),
        pytest.param({"X-Admin-Token": ""}, id="empty_header"),
    ],
)
def test_admin_rejects_missing_or_wrong_token(
    client: TestClient,
    admin_headers: dict[str, str],
    isolate_repo_state: Path,
    request_headers: dict[str, str],
) -> None:
    pointer_path = isolate_repo_state / "configs" / "active_model.yaml"
    pointer_before = pointer_path.read_bytes()

    response = client.post(
        "/admin/rollback",
        json={"dry_run": False},
        headers=request_headers,
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid admin token"}
    assert pointer_path.read_bytes() == pointer_before


def test_admin_token_never_appears_in_responses_or_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    admin_headers: dict[str, str],
) -> None:
    import logging

    token = admin_headers["X-Admin-Token"]
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(state, "inference_service", None)

    with TestClient(app) as client:
        responses = [
            client.post("/admin/rollback", json={"dry_run": True}, headers=admin_headers),
            client.post(
                "/admin/rollback",
                json={"dry_run": True},
                headers={"X-Admin-Token": token[:-1] + "x"},
            ),
            client.post("/admin/rollback", json={"dry_run": True}),
        ]

    assert [response.status_code for response in responses] == [400, 401, 401]
    for response in responses:
        assert token not in response.text
        assert token not in str(response.headers)

    for record in caplog.records:
        assert token not in record.getMessage()
        assert token not in str(record.args)


def test_admin_disabled_logs_one_startup_warning_without_the_token_length(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    short_token = "too-short-token"
    monkeypatch.setenv("ADMIN_TOKEN", short_token)
    monkeypatch.setattr(state, "inference_service", None)
    caplog.set_level(logging.WARNING, logger="api.main")

    with TestClient(app):
        pass

    warnings = [
        record
        for record in caplog.records
        if record.name == "api.main" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "admin" in warnings[0].getMessage().lower()
    assert short_token not in warnings[0].getMessage()
    assert str(len(short_token)) not in warnings[0].getMessage()


def test_non_admin_endpoints_need_no_token(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    admin_headers: dict[str, str],
) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/livez").status_code == 200
    assert client.get("/model/active").status_code == 200
    predict_response = client.post("/predict", json={"feature_payload": valid_feature_payload})
    assert predict_response.status_code == 200
    assert client.get("/metrics").status_code == 200


# --- shared pointer: pointer_source and the poller (no database needed) ---------------


def test_file_mode_reports_pointer_source_file_and_starts_no_poller(
    client: TestClient,
) -> None:
    assert client.get("/health").json()["pointer_source"] == "file"
    assert client.get("/model/active").json()["pointer_source"] == "file"
    assert state.poller is None


class _FakeInfo:
    def __init__(self, model_version: str) -> None:
        self.model_name = "isolation_forest"
        self.model_version = model_version


class _FakeService:
    def __init__(self, model_version: str) -> None:
        self._info = _FakeInfo(model_version)

    def active_model_info(self) -> _FakeInfo:
        return self._info


def _holder(model_version: str, pointer_version: int):
    from api.main import AppState, swap_inference_service

    holder = AppState()
    swap_inference_service(
        holder,
        _FakeService(model_version),
        pointer_version=pointer_version,
        pointer_source="postgres",
    )
    return holder


def _metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_poller_swaps_in_the_new_model_when_pointer_version_changes() -> None:
    from api.main import PointerPoller

    holder = _holder("v101", pointer_version=1)
    loads: list[int] = []

    def _load():
        loads.append(1)
        return _FakeService("v102"), 2

    poller = PointerPoller(
        holder, interval_seconds=30, read_version=lambda: 2, load_service=_load
    )
    successes_before = _metric_value("model_reload_total", {"result": "success"})

    assert poller.poll_once() is True
    assert holder.inference_service.active_model_info().model_version == "v102"
    assert holder.pointer_version == 2
    assert _metric_value("model_reload_total", {"result": "success"}) == successes_before + 1

    assert poller.poll_once() is False
    assert loads == [1]


def test_poller_keeps_the_old_model_when_the_new_one_fails_to_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from api.main import PointerPoller

    caplog.set_level(logging.ERROR, logger="api.main")
    holder = _holder("v101", pointer_version=1)

    def _broken_load():
        raise RuntimeError("artifact missing")

    poller = PointerPoller(
        holder, interval_seconds=30, read_version=lambda: 2, load_service=_broken_load
    )
    failures_before = _metric_value("model_reload_total", {"result": "failed"})

    assert poller.poll_once() is False
    assert poller.poll_once() is False

    assert holder.inference_service.active_model_info().model_version == "v101"
    assert holder.pointer_version == 1
    assert _metric_value("model_reload_total", {"result": "failed"}) == failures_before + 2
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1


def test_poller_keeps_serving_through_a_database_outage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from api.main import PointerPoller

    caplog.set_level(logging.INFO, logger="api.main")
    holder = _holder("v101", pointer_version=1)
    database_up = {"value": False}

    def _read_version():
        if not database_up["value"]:
            raise ConnectionError("database unreachable")
        return 1

    poller = PointerPoller(
        holder,
        interval_seconds=30,
        read_version=_read_version,
        load_service=lambda: pytest.fail("must not reload during an outage"),
    )
    poll_failures_before = _metric_value("active_pointer_poll_failures_total")

    for _ in range(3):
        assert poller.poll_once() is False
    database_up["value"] = True
    assert poller.poll_once() is False

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert any("reachable again" in record.getMessage() for record in caplog.records)
    assert _metric_value("active_pointer_poll_failures_total") == poll_failures_before + 3
    assert holder.inference_service.active_model_info().model_version == "v101"


def test_poller_thread_starts_and_stops() -> None:
    import time as time_module

    from api.main import PointerPoller

    holder = _holder("v101", pointer_version=1)
    poller = PointerPoller(
        holder,
        interval_seconds=0.01,
        read_version=lambda: 2,
        load_service=lambda: (_FakeService("v102"), 2),
    )

    poller.start()
    deadline = time_module.monotonic() + 5
    while holder.pointer_version != 2 and time_module.monotonic() < deadline:
        time_module.sleep(0.01)
    poller.stop()

    assert holder.inference_service.active_model_info().model_version == "v102"
    assert poller._thread is None


# --- prediction evidence persistence (background tasks) -----------------------------


def test_predict_persists_evidence_in_a_background_task(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import main as api_main

    persisted: list[list[dict]] = []
    monkeypatch.setattr(
        api_main,
        "persist_online_prediction_evidence",
        lambda records: persisted.append(records),
    )

    single = client.post("/predict", json={"feature_payload": valid_feature_payload})
    batch = client.post(
        "/predict/batch",
        json={"feature_payloads": [valid_feature_payload, valid_feature_payload]},
    )

    assert single.status_code == 200 and batch.status_code == 200
    assert [len(records) for records in persisted] == [1, 2]
    assert persisted[0][0]["prediction_id"] == single.json()["prediction_id"]
    assert [record["prediction_id"] for record in persisted[1]] == [
        prediction["prediction_id"] for prediction in batch.json()["predictions"]
    ]


@pytest.mark.expect_persist_failure
def test_jsonl_persist_failures_are_counted_logged_once_per_burst_and_never_returned(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from api import main as api_main

    caplog.set_level(logging.INFO, logger="api.main")
    sink_up = {"value": False}
    real_writer = api_main.write_online_predictions_jsonl

    def _flaky_writer(records):
        if not sink_up["value"]:
            raise OSError("disk full")
        return real_writer(records)

    monkeypatch.setattr(api_main, "write_online_predictions_jsonl", _flaky_writer)
    failures_before = _metric_value("prediction_persist_failures_total", {"sink": "jsonl"})
    postgres_before = _metric_value("prediction_persist_failures_total", {"sink": "postgres"})

    def _predict() -> int:
        return client.post("/predict", json={"feature_payload": valid_feature_payload}).status_code

    statuses = [_predict(), _predict()]
    sink_up["value"] = True
    statuses.append(_predict())
    sink_up["value"] = False
    statuses.append(_predict())

    errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and "Failed to persist" in record.getMessage()
    ]
    assert statuses == [200, 200, 200, 200]
    assert _metric_value("prediction_persist_failures_total", {"sink": "jsonl"}) == (
        failures_before + 3
    )
    assert len(errors) == 2
    assert any("persists to jsonl again" in record.getMessage() for record in caplog.records)
    # File mode never touches the postgres sink.
    assert _metric_value("prediction_persist_failures_total", {"sink": "postgres"}) == (
        postgres_before
    )


@pytest.mark.expect_persist_failure
def test_postgres_persist_failure_is_counted_and_never_returned(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import main as api_main

    def _unreachable(_records):
        raise ConnectionError("database unreachable")

    monkeypatch.setattr(api_main, "database_enabled", lambda: True)
    monkeypatch.setattr(api_main, "insert_online_predictions", _unreachable)
    failures_before = _metric_value("prediction_persist_failures_total", {"sink": "postgres"})

    response = client.post("/predict", json={"feature_payload": valid_feature_payload})

    assert response.status_code == 200
    assert _metric_value("prediction_persist_failures_total", {"sink": "postgres"}) == (
        failures_before + 1
    )


# --- clean-up: /health names a failed reload in database-pointer mode -----------------


def test_health_reports_a_failed_reload_and_stays_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.main
    from api.main import PointerPoller

    holder = _holder("v101", pointer_version=1)

    def _broken_load():
        raise RuntimeError("artifact missing at /some/path/model.joblib")

    failing = PointerPoller(
        holder,
        interval_seconds=30,
        read_version=lambda: 2,
        load_service=_broken_load,
        read_model_version=lambda: "v102",
    )
    assert failing.poll_once() is False
    monkeypatch.setattr(api.main, "state", holder)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "project4-online-anomaly-inference",
        "active_model_version": "v101",
        "model_loaded": True,
        "pointer_source": "postgres",
        "pointer_model_version": "v102",
        "model_reload_status": "failed: pointer_version 2 could not be loaded (RuntimeError)",
    }

    working = PointerPoller(
        holder,
        interval_seconds=30,
        read_version=lambda: 2,
        load_service=lambda: (_FakeService("v102"), 2),
        read_model_version=lambda: pytest.fail("a successful load reads no pointer name"),
    )
    assert working.poll_once() is True

    body = TestClient(app).get("/health").json()
    assert (body["active_model_version"], body["pointer_model_version"]) == ("v102", "v102")
    assert body["model_reload_status"] == "current"


def test_health_failed_reload_shows_null_when_the_pointer_name_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import api.main
    from api.main import PointerPoller

    caplog.set_level(logging.ERROR, logger="api.main")
    holder = _holder("v101", pointer_version=1)

    def _broken_load():
        raise RuntimeError("artifact missing")

    def _unreadable_pointer():
        raise ConnectionError("database unreachable")

    poller = PointerPoller(
        holder,
        interval_seconds=30,
        read_version=lambda: 2,
        load_service=_broken_load,
        read_model_version=_unreadable_pointer,
    )
    assert poller.poll_once() is False
    assert poller.poll_once() is False
    monkeypatch.setattr(api.main, "state", holder)

    body = TestClient(app).get("/health").json()

    assert body["status"] == "ok" and body["active_model_version"] == "v101"
    assert body["pointer_model_version"] is None
    assert body["model_reload_status"].startswith("failed: pointer_version 2 ")
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1


def test_file_mode_health_has_no_reload_fields_after_a_database_mode_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state, "pointer_model_version", "v999")
    monkeypatch.setattr(state, "model_reload_status", "current")
    monkeypatch.setattr(state, "inference_service", None)

    with TestClient(app) as client:
        health = client.get("/health").json()

    assert set(health) == {
        "status",
        "service",
        "active_model_version",
        "model_loaded",
        "pointer_source",
    }
    assert health["pointer_source"] == "file"
