"""Tests for the Project 4 online inference FastAPI service."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anomaly_detection.online_inference import OnlineInferenceService
from api.main import app, state


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


def test_health_returns_loaded_active_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "project4-online-anomaly-inference"
    assert body["active_model_version"] == "v002"
    assert body["model_loaded"] is True


def test_active_model_endpoint_returns_v002_metadata(client: TestClient) -> None:
    response = client.get("/model/active")

    assert response.status_code == 200
    body = response.json()

    assert body["model_name"] == "isolation_forest"
    assert body["model_version"] == "v002"
    assert body["status"] == "production"
    assert body["feature_schema_version"] == "feature_schema_v001"
    assert body["feature_count"] == 51
    assert body["baseline_anomaly_rate"] is not None
    assert body["threshold"] == 0.0
    assert body["source_projects"] == ["project_1", "project_2_3"]
    assert body["snapshot_type"] == "real_source_extract"
    assert body["latency_budget_p95_ms"] == 200.0


def test_predict_scores_single_payload_with_active_model(
    client: TestClient,
    valid_feature_payload: dict[str, float | str],
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
    assert body["model_version"] == "v002"
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
    assert body["model_version"] == "v002"
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["model_version"] == "v002"
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


def test_rollback_endpoint_validates_previous_stable_model_without_mutation(
    client: TestClient,
) -> None:
    active_model_path = Path("configs/active_model.yaml")
    before_payload = active_model_path.read_text(encoding="utf-8")

    response = client.post(
        "/admin/rollback",
        json={
            "rollback_reason": "pytest dry-run validation",
            "triggered_by": "pytest",
            "dry_run": True,
        },
    )

    after_payload = active_model_path.read_text(encoding="utf-8")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "validated"
    assert body["action_taken"] is False
    assert body["from_model_version"] == "v002"
    assert body["to_model_version"] == "v001"
    assert body["validation_status"] == "dry_run_validated"
    assert body["dry_run"] is True
    assert body["active_model_version"] == "v002"
    assert before_payload == after_payload


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
    assert '"model_version":"v002"' in records[0]


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

    assert body["model_version"] == "v002"
    assert body["latency_ms"] >= 0
    assert body["latency_ms"] < latency_budget_ms
