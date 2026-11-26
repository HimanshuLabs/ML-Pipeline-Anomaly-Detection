from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from anomaly_detection.batch_inference import (
    BatchInferenceError,
    BatchPredictionRecord,
    hash_feature_payload,
    load_active_model_artifacts,
    prepare_feature_matrix,
    score_feature_batch,
    write_predictions_jsonl,
)


class DummyAnomalyModel:
    """Small deterministic model used only for batch inference unit tests."""

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([0.25, -0.40], dtype=float)[: len(features)]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([1, -1], dtype=int)[: len(features)]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_artifact_bundle(tmp_path: Path, extra_metadata: dict | None = None) -> Path:
    model_dir = tmp_path / "artifacts" / "models" / "isolation_forest" / "model_version=vtest"
    model_dir.mkdir(parents=True)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    baseline_path = model_dir / "baseline_stats.json"
    schema_path = model_dir / "feature_schema.json"
    active_path = tmp_path / "active_model.yaml"

    joblib.dump(DummyAnomalyModel(), model_path)

    _write_json(
        metadata_path,
        {
            "model_name": "isolation_forest",
            "model_version": "vtest",
            "feature_schema_version": "feature_schema_test",
            "feature_names": ["feature_a", "feature_b"],
            "baseline_stats_path": str(baseline_path),
            "feature_schema_path": str(schema_path),
            "model_path": str(model_path),
            **(extra_metadata or {}),
        },
    )

    _write_json(
        baseline_path,
        {
            "model_name": "isolation_forest",
            "model_version": "vtest",
            "feature_schema_version": "feature_schema_test",
            "baseline_anomaly_rate": 0.50,
            "label_availability": "unlabeled_proxy_metrics",
            "metric_type": "unit_test_proxy",
            "prediction_summary": {
                "threshold_used": 0.0,
            },
            "feature_baselines": {
                "feature_a": {"mean": 1.0, "variance": 0.1},
                "feature_b": {"mean": 2.0, "variance": 0.2},
            },
        },
    )

    _write_json(
        schema_path,
        {
            "feature_schema_version": "feature_schema_test",
            "feature_count": 2,
            "feature_names": ["feature_a", "feature_b"],
            "entity_key": "entity_id",
            "source": "unit_test",
            "timestamp_column": "event_timestamp",
        },
    )

    active_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "isolation_forest",
                "active_model_version": "vtest",
                "artifact_path": str(model_path),
                "feature_schema_version": "feature_schema_test",
                "status": "production",
                "dataset_snapshot_id": "snapshot_test",
                "training_dataset_id": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return active_path


def test_load_active_model_artifacts_from_active_pointer(tmp_path: Path) -> None:
    active_path = _build_artifact_bundle(tmp_path)

    artifacts = load_active_model_artifacts(active_path)

    assert artifacts.model_name == "isolation_forest"
    assert artifacts.model_version == "vtest"
    assert artifacts.feature_schema_version == "feature_schema_test"
    assert artifacts.dataset_snapshot_id == "snapshot_test"
    assert artifacts.training_dataset_id is None
    assert artifacts.metadata["feature_names"] == ["feature_a", "feature_b"]


def test_prepare_feature_matrix_selects_ordered_numeric_features() -> None:
    features = pd.DataFrame(
        [
            {
                "entity_id": "customer_001",
                "feature_b": "2.5",
                "feature_a": 1,
                "ignored_column": "not_model_input",
            }
        ]
    )

    matrix = prepare_feature_matrix(features, ["feature_a", "feature_b"])

    assert list(matrix.columns) == ["feature_a", "feature_b"]
    assert matrix.iloc[0].to_dict() == {"feature_a": 1.0, "feature_b": 2.5}


def test_prepare_feature_matrix_fails_for_missing_feature() -> None:
    features = pd.DataFrame([{"feature_a": 1.0}])

    with pytest.raises(BatchInferenceError, match="missing required model features"):
        prepare_feature_matrix(features, ["feature_a", "feature_b"])


def test_prepare_feature_matrix_fails_for_non_numeric_feature() -> None:
    features = pd.DataFrame(
        [
            {
                "feature_a": 1.0,
                "feature_b": "bad_value",
            }
        ]
    )

    with pytest.raises(BatchInferenceError, match="null/non-numeric"):
        prepare_feature_matrix(features, ["feature_a", "feature_b"])


@pytest.mark.parametrize("value", ["inf", "-inf", np.inf, -np.inf])
def test_prepare_feature_matrix_rejects_non_finite_values(value: float | str) -> None:
    features = pd.DataFrame(
        {"feature_a": [value, value], "feature_b": [1.0, value]}
    )

    with pytest.raises(BatchInferenceError) as exc_info:
        prepare_feature_matrix(features, ["feature_a", "feature_b"])

    assert str(exc_info.value) == (
        "Input batch contains non-finite (inf/-inf) model feature values: "
        "feature_a=2, feature_b=1"
    )


def test_hash_feature_payload_is_stable_for_key_order() -> None:
    first_hash = hash_feature_payload({"feature_b": 2.0, "feature_a": 1.0})
    second_hash = hash_feature_payload({"feature_a": 1.0, "feature_b": 2.0})

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_score_feature_batch_returns_prediction_contract(tmp_path: Path) -> None:
    active_path = _build_artifact_bundle(tmp_path)
    artifacts = load_active_model_artifacts(active_path)

    features = pd.DataFrame(
        [
            {
                "entity_id": "customer_001",
                "source_record_id": "source_001",
                "feature_a": 1.0,
                "feature_b": 2.0,
            },
            {
                "entity_id": "customer_002",
                "source_record_id": "source_002",
                "feature_a": 10.0,
                "feature_b": 20.0,
            },
        ]
    )

    predictions = score_feature_batch(
        features,
        artifacts,
        entity_type="customer",
        source_project="unit_test_project",
        source_table="unit_test_table",
        batch_run_id="batch_test_001",
    )

    assert len(predictions) == 2
    assert all(isinstance(prediction, BatchPredictionRecord) for prediction in predictions)

    first, second = predictions

    assert first.batch_run_id == "batch_test_001"
    assert first.model_name == "isolation_forest"
    assert first.model_version == "vtest"
    assert first.entity_id == "customer_001"
    assert first.source_record_id == "source_001"
    assert first.anomaly_score == 0.25
    assert first.is_anomaly is False
    assert first.threshold_used == 0.0
    assert first.prediction_status == "success"
    assert len(first.feature_payload_hash) == 64

    assert second.entity_id == "customer_002"
    assert second.source_record_id == "source_002"
    assert second.anomaly_score == -0.40
    assert second.is_anomaly is True


def test_write_predictions_jsonl_appends_records(tmp_path: Path) -> None:
    active_path = _build_artifact_bundle(tmp_path)
    artifacts = load_active_model_artifacts(active_path)

    features = pd.DataFrame(
        [
            {
                "entity_id": "customer_001",
                "feature_a": 1.0,
                "feature_b": 2.0,
            }
        ]
    )

    predictions = score_feature_batch(features, artifacts)
    output_path = tmp_path / "batch_predictions.jsonl"

    written_path = write_predictions_jsonl(predictions, output_path)

    assert written_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["model_version"] == "vtest"
    assert record["entity_id"] == "customer_001"
    assert record["prediction_status"] == "success"
    assert record["threshold_used"] == 0.0
    assert len(record["feature_payload_hash"]) == 64


# --- training/runtime library version checks ----------------------------------


def _runtime_version_records(caplog: pytest.LogCaptureFixture, level: int) -> list:
    return [
        record
        for record in caplog.records
        if record.name == "anomaly_detection.batch_inference" and record.levelno == level
    ]


def _recorded_versions(scikit_learn_version: str) -> dict[str, str]:
    return {
        "python": "3.12.0",
        "scikit-learn": scikit_learn_version,
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "joblib": "1.4.0",
    }


def test_loading_model_trained_with_other_sklearn_warns_once_and_loads(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import sklearn

    caplog.set_level(logging.INFO, logger="anomaly_detection.batch_inference")
    active_path = _build_artifact_bundle(
        tmp_path,
        {"runtime_versions": _recorded_versions("0.0.1")},
    )

    artifacts = load_active_model_artifacts(active_path)

    warnings = _runtime_version_records(caplog, logging.WARNING)
    assert artifacts.model_version == "vtest"
    assert len(warnings) == 1
    assert "0.0.1" in warnings[0].getMessage()
    assert sklearn.__version__ in warnings[0].getMessage()


def test_loading_model_trained_with_same_sklearn_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import sklearn

    caplog.set_level(logging.INFO, logger="anomaly_detection.batch_inference")
    active_path = _build_artifact_bundle(
        tmp_path,
        {"runtime_versions": _recorded_versions(sklearn.__version__)},
    )

    load_active_model_artifacts(active_path)

    assert _runtime_version_records(caplog, logging.WARNING) == []


def test_loading_model_without_recorded_versions_logs_info_and_loads(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="anomaly_detection.batch_inference")
    active_path = _build_artifact_bundle(tmp_path)

    artifacts = load_active_model_artifacts(active_path)

    infos = [
        record
        for record in _runtime_version_records(caplog, logging.INFO)
        if "not recorded" in record.getMessage()
    ]
    assert artifacts.model_version == "vtest"
    assert len(infos) == 1
    assert _runtime_version_records(caplog, logging.WARNING) == []


def test_batch_run_fails_naming_the_postgres_sink_after_jsonl_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import database
    from anomaly_detection.batch_inference import run_batch_inference

    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"

    def _broken_insert(_records):
        raise ConnectionError("database unreachable")

    monkeypatch.setattr(database, "is_configured", lambda: True)
    monkeypatch.setattr(database, "insert_batch_predictions", _broken_insert)

    features = pd.DataFrame(
        [
            {"entity_id": "customer_001", "feature_a": 1.0, "feature_b": 2.0},
            {"entity_id": "customer_002", "feature_a": 3.0, "feature_b": 4.0},
        ]
    )

    with pytest.raises(BatchInferenceError) as error:
        run_batch_inference(features, active_model_path=active_path, output_path=output_path)

    message = str(error.value)
    assert "postgres sink (ml.batch_predictions) failed" in message
    assert str(output_path) in message
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


# --- clean-up: a repeat batch run is refused before anything is written -------------------


def _two_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"entity_id": "customer_001", "feature_a": 1.0, "feature_b": 2.0},
            {"entity_id": "customer_002", "feature_a": 3.0, "feature_b": 4.0},
        ]
    )


def _bundle_of_version(tmp_path: Path, version: str) -> Path:
    """The dummy bundle under tmp_path/version, relabelled as model version `version`."""
    active_path = _build_artifact_bundle(tmp_path / version)
    pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    model_dir = Path(pointer["artifact_path"]).parent
    for name in ("metadata.json", "baseline_stats.json"):
        payload = json.loads((model_dir / name).read_text(encoding="utf-8"))
        _write_json(model_dir / name, {**payload, "model_version": version})
    pointer["active_model_version"] = version
    active_path.write_text(yaml.safe_dump(pointer, sort_keys=True), encoding="utf-8")
    return active_path


def _run_ids(output_path: Path) -> list[str]:
    lines = output_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["batch_run_id"] for line in lines]


def test_a_repeat_run_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import BatchRepeatError, run_batch_inference

    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"
    first = run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)
    before = output_path.read_bytes()

    with pytest.raises(BatchRepeatError) as error:
        run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)

    message = str(error.value)
    assert first[0].batch_run_id in message
    assert "Nothing was written" in message
    assert "--allow-repeat" in message
    assert output_path.read_bytes() == before


def test_allow_repeat_scores_again_under_a_new_run_id(tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import run_batch_inference

    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"
    first = run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)

    second = run_batch_inference(
        _two_rows(), active_model_path=active_path, output_path=output_path, allow_repeat=True
    )

    assert second[0].batch_run_id != first[0].batch_run_id
    assert [p.anomaly_score for p in second] == [p.anomaly_score for p in first]
    assert _run_ids(output_path) == [first[0].batch_run_id] * 2 + [second[0].batch_run_id] * 2


def test_a_changed_row_or_another_model_is_not_a_repeat(tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import run_batch_inference

    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"
    run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)

    changed = _two_rows()
    changed.loc[1, "feature_b"] = 4.5
    run_batch_inference(changed, active_model_path=active_path, output_path=output_path)
    other_model = _bundle_of_version(tmp_path, "vtest2")
    records = run_batch_inference(
        _two_rows(), active_model_path=other_model, output_path=output_path
    )

    assert {record.model_version for record in records} == {"vtest2"}
    assert len(set(_run_ids(output_path))) == 3


def test_an_unreadable_jsonl_line_stops_the_run(tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import run_batch_inference

    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"
    output_path.write_text('{"model_name": "isolation_forest", "model_ver\n', encoding="utf-8")
    before = output_path.read_bytes()

    with pytest.raises(BatchInferenceError) as error:
        run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)

    message = str(error.value)
    assert "unreadable line 1" in message
    assert "Nothing was written" in message
    assert "killed mid-write" in message
    assert output_path.read_bytes() == before


def test_an_unreadable_database_does_not_stop_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from sqlalchemy.exc import OperationalError

    from anomaly_detection import database
    from anomaly_detection.batch_inference import run_batch_inference

    def _unreadable(*_args):
        raise OperationalError("SELECT", {}, Exception("simulated"))

    inserted: list[int] = []
    monkeypatch.setattr(database, "is_configured", lambda: True)
    monkeypatch.setattr(database, "batch_run_fingerprints", _unreadable)
    monkeypatch.setattr(
        database, "insert_batch_predictions", lambda records: inserted.append(len(records))
    )
    caplog.set_level(logging.WARNING, logger="anomaly_detection.batch_inference")
    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"

    run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)

    assert inserted == [2]
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert any(
        "could not read ml.batch_predictions (OperationalError)" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_the_second_run_check_logs_lines_read_and_elapsed_seconds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from anomaly_detection.batch_inference import run_batch_inference

    caplog.set_level(logging.INFO, logger="anomaly_detection.batch_inference")
    active_path = _build_artifact_bundle(tmp_path)
    output_path = tmp_path / "batch_predictions.jsonl"
    run_batch_inference(_two_rows(), active_model_path=active_path, output_path=output_path)
    run_batch_inference(
        _two_rows(), active_model_path=_bundle_of_version(tmp_path, "vtest2"),
        output_path=output_path,
    )

    [line] = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Second-run check read")
    ]
    assert line.startswith(f"Second-run check read 2 JSONL lines of {output_path}, 0 of them")
    assert line.endswith(" s")


def test_cli_allow_repeat_reaches_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import batch_inference

    active_path = _build_artifact_bundle(tmp_path)
    input_path = tmp_path / "features.csv"
    _two_rows().to_csv(input_path, index=False)
    output_path = tmp_path / "out.jsonl"
    monkeypatch.setattr("sys.argv", [
        "batch_inference", "--input-path", str(input_path), "--output-path", str(output_path),
        "--active-model-path", str(active_path), "--allow-repeat"])

    batch_inference.main()
    batch_inference.main()

    assert len(set(_run_ids(output_path))) == 2
    assert len(_run_ids(output_path)) == 4


def test_cli_resume_and_input_path_are_exclusive(tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import _build_parser

    run_id = "3f2b8f0e-6c1a-4d55-9a57-0f1e2d3c4b5a"
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--input-path", str(tmp_path / "f.csv"), "--resume-run", run_id]
        )
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])

    args = _build_parser().parse_args(["--resume-run", run_id])
    assert (args.resume_run, args.input_path, args.allow_repeat) == (run_id, None, False)


def test_resume_refuses_file_mode_and_a_malformed_run_id(tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import resume_batch_run

    output_path = tmp_path / "batch_predictions.jsonl"

    with pytest.raises(BatchInferenceError, match="Not a batch_run_id"):
        resume_batch_run("not-a-run", output_path=output_path)
    with pytest.raises(BatchInferenceError) as error:
        resume_batch_run("3f2b8f0e-6c1a-4d55-9a57-0f1e2d3c4b5a", output_path=output_path)

    assert "not configured" in str(error.value)
    assert "Nothing was written" in str(error.value)
    assert not output_path.exists()
