"""Serving the segmented model: API, batch inference, promotion CLI, rollback, build check."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from anomaly_detection import registry
from api.main import app, state
from tests.conftest import SEGMENTED_TEST_TYPES

MODEL_NAME = "isolation_forest"


# --- helpers --------------------------------------------------------------------------


def _register_copy(tmp_path: Path, metadata_path: Path, version: str) -> Path:
    """Register a private copy of a trained bundle (so a test may damage it)."""

    artifact_dir = tmp_path / "models" / f"model_version={version}"
    shutil.copytree(metadata_path.parent, artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    registry.register_model(
        model_name=MODEL_NAME,
        model_version=version,
        artifact_path=str(artifact_dir / "model.joblib"),
        dataset_snapshot_id=str(metadata["training_snapshot_metadata"]["snapshot_id"]),
        feature_schema_version=metadata["feature_schema_version"],
        training_timestamp=metadata["training_finished_at_utc"],
    )
    return artifact_dir


def _promote(version: str) -> None:
    registry.promote_model_version(model_name=MODEL_NAME, model_version=version)


def _segments(metadata_path: Path) -> dict[str, list[str]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {t: s["feature_names"] for t, s in metadata["segments"].items()}


def _expected_score(model_path: Path, entity_type: str, values: dict[str, float]) -> float:
    segment = joblib.load(model_path).segments[entity_type]
    frame = pd.DataFrame([{name: values[name] for name in segment.feature_names}])
    return float(segment.model.decision_function(frame)[0])


def _payload(features: list[str], value: float = 0.5) -> dict[str, float]:
    return {name: value + 0.01 * index for index, name in enumerate(features)}


@pytest.fixture()
def segmented_client(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    """Mirror registry with segmented v202 promoted; the API started from the pointer."""

    artifact_dir = _register_copy(tmp_path, segmented_metadata_paths["v202"], "v202")
    _promote("v202")
    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        yield client, artifact_dir


# --- online serving -------------------------------------------------------------------


def test_predict_scores_each_entity_type_with_its_own_segment(segmented_client) -> None:
    client, artifact_dir = segmented_client
    segments = _segments(artifact_dir / "metadata.json")
    all_features = sorted({name for names in segments.values() for name in names})

    for entity_type, features in segments.items():
        values = _payload(features)
        # Other segments' features are present but unusable; they must be ignored.
        payload = {name: "not-a-number" for name in all_features if name not in features}
        payload.update(values)

        response = client.post(
            "/predict",
            json={"entity_type": entity_type, "entity_id": "e1", "feature_payload": payload},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model_version"] == "v202"
        assert body["entity_type"] == entity_type
        expected = _expected_score(artifact_dir / "model.joblib", entity_type, values)
        assert body["anomaly_score"] == pytest.approx(expected)
        assert body["is_anomaly"] == (expected < 0.0)
        assert body["threshold_used"] == 0.0


def test_predict_unknown_entity_type_is_422_naming_the_allowed_types(segmented_client) -> None:
    client, artifact_dir = segmented_client
    features = _segments(artifact_dir / "metadata.json")["user"]

    response = client.post(
        "/predict",
        json={"entity_type": "web_customer", "feature_payload": _payload(features)},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "web_customer" in detail
    for entity_type in SEGMENTED_TEST_TYPES:
        assert entity_type in detail


def test_predict_missing_segment_feature_is_422_naming_it(segmented_client) -> None:
    client, artifact_dir = segmented_client
    segments = _segments(artifact_dir / "metadata.json")
    features = segments["campaign"]
    payload = _payload(features)
    del payload[features[0]]

    response = client.post(
        "/predict", json={"entity_type": "campaign", "feature_payload": payload}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    missing = detail.split("missing required model features:")[1]
    # Only this segment's missing feature is required; other segments' are not.
    assert [name.strip() for name in missing.split(",")] == [features[0]]


def test_predict_non_finite_segment_feature_is_422(segmented_client) -> None:
    client, artifact_dir = segmented_client
    features = _segments(artifact_dir / "metadata.json")["product"]
    payload = _payload(features)
    payload[features[1]] = "inf"

    response = client.post("/predict", json={"entity_type": "product", "feature_payload": payload})

    assert response.status_code == 422
    assert "non-finite" in response.json()["detail"]


def test_predict_batch_routes_mixed_per_payload_entity_types(segmented_client) -> None:
    client, artifact_dir = segmented_client
    segments = _segments(artifact_dir / "metadata.json")
    user_values = _payload(segments["user"], 0.2)
    campaign_values = _payload(segments["campaign"], 0.4)
    product_values = _payload(segments["product"], 0.6)

    response = client.post(
        "/predict/batch",
        json={
            "entity_type": "user",  # request level; the first payload uses it
            "feature_payloads": [
                {**user_values, "entity_id": "u1"},
                {**campaign_values, "entity_type": "campaign", "entity_id": "c1"},
                {**product_values, "entity_type": "product", "entity_id": "p1"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    model_path = artifact_dir / "model.joblib"
    for prediction, entity_type, values in zip(
        predictions,
        ("user", "campaign", "product"),
        (user_values, campaign_values, product_values),
        strict=True,
    ):
        assert prediction["entity_type"] == entity_type
        assert prediction["model_version"] == "v202"
        assert prediction["anomaly_score"] == pytest.approx(
            _expected_score(model_path, entity_type, values)
        )


def test_model_active_reports_model_kind_and_segments(segmented_client) -> None:
    client, artifact_dir = segmented_client
    segments = _segments(artifact_dir / "metadata.json")

    body = client.get("/model/active").json()

    assert body["model_version"] == "v202"
    assert body["model_kind"] == "segmented_isolation_forest"
    assert body["segments"] == {t: len(names) for t, names in segments.items()}
    assert body["feature_schema_version"] == "feature_schema_v003"
    health = client.get("/health").json()
    assert set(health) == {
        "status",
        "service",
        "active_model_version",
        "model_loaded",
        "pointer_source",
    }
    assert health["active_model_version"] == "v202"


def test_single_model_reports_its_kind_and_no_segments(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_copy(tmp_path, trained_metadata_paths["v101"], "v101")  # a single model
    _promote("v101")

    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        body = client.get("/model/active").json()

    assert body["model_version"] == "v101"
    assert body["model_kind"] == "isolation_forest"
    assert body["segments"] is None


# --- batch inference ------------------------------------------------------------------


def test_batch_inference_routes_rows_by_their_entity_type_column(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.batch_inference import BatchInferenceError, run_batch_inference

    artifact_dir = _register_copy(tmp_path, segmented_metadata_paths["v202"], "v202")
    _promote("v202")
    segments = _segments(artifact_dir / "metadata.json")
    all_features = sorted({name for names in segments.values() for name in names})

    rows, expected = [], []
    for index, entity_type in enumerate(["product", "user", "campaign", "user"]):
        values = _payload(segments[entity_type], 0.1 * (index + 1))
        row = {name: np.nan for name in all_features}  # other segments: missing, ignored
        row.update(values)
        row.update({"entity_type": entity_type, "entity_id": f"row{index}"})
        rows.append(row)
        expected.append(
            (entity_type, _expected_score(artifact_dir / "model.joblib", entity_type, values))
        )

    records = run_batch_inference(pd.DataFrame(rows), persist_jsonl=False)

    assert [(r.entity_type, r.entity_id) for r in records] == [
        (t, f"row{i}") for i, (t, _) in enumerate(expected)
    ]
    for record, (entity_type, score) in zip(records, expected, strict=True):
        assert record.model_version == "v202"
        assert record.anomaly_score == pytest.approx(score)
        assert record.is_anomaly == (score < 0.0)
        assert set(record.feature_payload) == set(segments[entity_type])

    # Without an entity_type column every row is the entity_type argument.
    campaign_only = pd.DataFrame([_payload(segments["campaign"])])
    (record,) = run_batch_inference(campaign_only, entity_type="campaign", persist_jsonl=False)
    assert record.entity_type == "campaign"

    unknown = pd.DataFrame([{**_payload(segments["user"]), "entity_type": "funnel_day"}])
    with pytest.raises(BatchInferenceError, match="funnel_day.*campaign.*product.*user"):
        run_batch_inference(unknown, persist_jsonl=False)


# --- rollback -------------------------------------------------------------------------


def _apply_rollback(client: TestClient, admin_headers: dict[str, str]):
    return client.post(
        "/admin/rollback",
        json={"rollback_reason": "pytest", "triggered_by": "pytest", "dry_run": False},
        headers=admin_headers,
    )


def test_rollback_from_segmented_to_a_single_model_is_refused(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    admin_headers: dict[str, str],
) -> None:
    _register_copy(tmp_path, trained_metadata_paths["v101"], "v101")
    _promote("v101")  # single model; archived by the next promotion
    _register_copy(tmp_path, segmented_metadata_paths["v202"], "v202")
    _promote("v202")
    pointer_before = registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes()

    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        response = _apply_rollback(client, admin_headers)
        assert response.status_code == 400
        assert "v101" in response.json()["detail"]
        assert "never compatible" in response.json()["detail"]
        assert client.get("/model/active").json()["model_version"] == "v202"

    assert registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes() == pointer_before


def test_rollback_between_compatible_segmented_models_swaps_and_serves(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    admin_headers: dict[str, str],
) -> None:
    v201_dir = _register_copy(tmp_path, segmented_metadata_paths["v201"], "v201")
    _promote("v201")
    _register_copy(tmp_path, segmented_metadata_paths["v202"], "v202")
    _promote("v202")
    features = _segments(v201_dir / "metadata.json")["campaign"]
    values = _payload(features)
    request = {"entity_type": "campaign", "feature_payload": values}

    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        assert client.post("/predict", json=request).json()["model_version"] == "v202"

        response = _apply_rollback(client, admin_headers)

        assert response.status_code == 200, response.text
        assert response.json()["to_model_version"] == "v201"
        body = client.post("/predict", json=request).json()
        assert body["model_version"] == "v201"
        assert body["anomaly_score"] == pytest.approx(
            _expected_score(v201_dir / "model.joblib", "campaign", values)
        )

    pointer = yaml.safe_load(registry.DEFAULT_ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    assert pointer["active_model_version"] == "v201"


# --- registry CLI promote -------------------------------------------------------------


def _real_segmented_entry() -> registry.ModelRegistryEntry | None:
    """The newest real registry version whose bundle is a segmented model, if any."""

    segmented = []
    for entry in registry.load_model_registry(registry.DEFAULT_REGISTRY_PATH):
        metadata_path = registry.PROJECT_ROOT / entry.metadata_path
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_kind") == "segmented_isolation_forest":
            segmented.append(entry)
    return max(segmented, key=lambda entry: int(entry.model_version[1:]), default=None)


def test_cli_promotes_a_real_segmented_bundle_and_the_api_serves_every_segment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Whether or not it is already the live model, the real segmented bundle must
    # pass the CLI's serving check, become active, and serve every segment.
    entry = _real_segmented_entry()
    if entry is None:
        pytest.skip("the real registry has no segmented model")
    version = entry.model_version
    segments = _segments(registry.PROJECT_ROOT / entry.metadata_path)

    assert registry.main(["promote", "--model-version", version]) == 0
    assert version in capsys.readouterr().out
    pointer = yaml.safe_load(registry.DEFAULT_ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    assert pointer["active_model_version"] == version
    assert pointer["feature_schema_version"] == entry.feature_schema_version

    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        active = client.get("/model/active").json()
        assert active["model_kind"] == "segmented_isolation_forest"
        assert active["segments"] == {t: len(names) for t, names in segments.items()}
        for entity_type, features in segments.items():
            response = client.post(
                "/predict",
                json={"entity_type": entity_type, "feature_payload": _payload(features)},
            )
            assert response.status_code == 200, (entity_type, response.text)
            assert response.json()["model_version"] == version


def _corrupt_model(artifact_dir: Path) -> None:
    (artifact_dir / "model.joblib").write_text("not a joblib file", encoding="utf-8")


def _unscorable_single_model(artifact_dir: Path) -> None:
    """A loadable artifact the serving code cannot score, labelled as a single model."""
    joblib.dump({"not": "a model"}, artifact_dir / "model.joblib")
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata.pop("segments")
    metadata["model_kind"] = "isolation_forest"
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        pytest.param(_corrupt_model, "does not load", id="unloadable"),
        pytest.param(_unscorable_single_model, "cannot be served", id="unscorable_kind"),
    ],
)
def test_cli_promote_refuses_a_candidate_serving_cannot_score(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    damage,
    expected: str,
) -> None:
    artifact_dir = _register_copy(tmp_path, segmented_metadata_paths["v201"], "v201")
    damage(artifact_dir)
    pointer_before = registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes()
    registry_before = registry.DEFAULT_REGISTRY_PATH.read_bytes()

    assert registry.main(["promote", "--model-version", "v201"]) == 1

    assert expected in capsys.readouterr().err
    assert registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes() == pointer_before
    assert registry.DEFAULT_REGISTRY_PATH.read_bytes() == registry_before


# --- image build check ----------------------------------------------------------------


def _pointer_bundle_dir(pointer_path: Path) -> Path:
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    artifact_path = Path(pointer["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = registry.PROJECT_ROOT / artifact_path
    return artifact_path.parent


def test_bundle_check_passes_for_the_mirror_pointer(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.bundle_check import check_active_bundle, main

    pointer_path = registry.DEFAULT_ACTIVE_MODEL_PATH  # per-test mirror (conftest)

    # The pointer as copied from the repository, whichever model it activates ...
    assert check_active_bundle(pointer_path) == _pointer_bundle_dir(pointer_path)
    assert main(["--active-model", str(pointer_path)]) == 0

    # ... and the mirror pointer on a single and on a segmented model of its own.
    for version, metadata_paths in (
        ("v101", trained_metadata_paths),
        ("v202", segmented_metadata_paths),
    ):
        artifact_dir = _register_copy(tmp_path, metadata_paths[version], version)
        _promote(version)
        assert check_active_bundle(pointer_path) == artifact_dir
        assert main(["--active-model", str(pointer_path)]) == 0


def test_bundle_check_fails_for_a_pointer_whose_bundle_is_missing(tmp_path: Path) -> None:
    from anomaly_detection.bundle_check import BundleCheckError, check_active_bundle, main

    pointer = yaml.safe_load(registry.DEFAULT_ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    pointer["active_model_version"] = "v999"
    pointer["artifact_path"] = "artifacts/models/isolation_forest/model_version=v999/model.joblib"
    pointer_path = tmp_path / "active_model.yaml"
    pointer_path.write_text(yaml.safe_dump(pointer), encoding="utf-8")

    with pytest.raises(BundleCheckError, match=r"model_version=v999/model\.joblib"):
        check_active_bundle(pointer_path)
    assert main(["--active-model", str(pointer_path)]) == 1


# --- latency benchmark ----------------------------------------------------------------


def test_benchmark_sends_entity_type_and_that_segments_features(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.online_inference import OnlineInferenceService
    from scripts.benchmark_latency import benchmark_entity_types, build_request_body

    _register_copy(tmp_path, trained_metadata_paths["v101"], "v101")  # a single model
    _promote("v101")
    single = OnlineInferenceService()
    assert benchmark_entity_types(single, None) == ["user"]
    body = build_request_body(single, entity_id="b0", entity_type="user")
    assert set(body["feature_payload"]) - {"entity_id"} == set(single.feature_names)

    artifact_dir = _register_copy(tmp_path, segmented_metadata_paths["v202"], "v202")
    _promote("v202")
    service = OnlineInferenceService()
    segments = _segments(artifact_dir / "metadata.json")

    assert benchmark_entity_types(service, None) == sorted(segments)
    body = build_request_body(service, entity_id="b1", entity_type="product")
    assert body["entity_type"] == "product"
    assert set(body["feature_payload"]) - {"entity_id"} == set(segments["product"])
    service.predict(body["feature_payload"], entity_type=body["entity_type"])
