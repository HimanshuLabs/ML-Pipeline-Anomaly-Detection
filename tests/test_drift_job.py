"""Drift job: the active model's baseline, per segment for segmented models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection import drift
from tests.conftest import real_source_test_frame
from tests.test_database import (  # noqa: F401 - pytest fixtures
    _count,
    database_mode,
    initialized_database,
    postgres_engine,
)
from tests.test_segmented_serving import _promote, _register_copy, _segments

# segmented_metadata_paths trains v202 on real_source_test_frame(seed=8).
V202_TRAINING_SEED = 8
SHIFTED_TYPE = "campaign"


def _serve_segmented(tmp_path: Path, segmented_metadata_paths: dict[str, Path]) -> Path:
    """Promote this test's own segmented v202 in the mirror; return its artifact dir."""
    artifact_dir = _register_copy(tmp_path, segmented_metadata_paths["v202"], "v202")
    _promote("v202")
    return artifact_dir


def _current_frame(artifact_dir: Path, *, shift: float = 5.0) -> pd.DataFrame:
    """The rows v202 was trained on, with the campaign segment's features shifted."""
    frame = real_source_test_frame(seed=V202_TRAINING_SEED)
    campaign = frame["entity_type"] == SHIFTED_TYPE
    for name in _segments(artifact_dir / "metadata.json")[SHIFTED_TYPE]:
        frame.loc[campaign, name] = frame.loc[campaign, name] + shift
    return frame


def test_segmented_model_is_evaluated_per_segment_against_its_own_baseline(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.drift_job import run_drift_job

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    segments = _segments(artifact_dir / "metadata.json")
    input_path = tmp_path / "current.parquet"
    _current_frame(artifact_dir).to_parquet(input_path, index=False)

    result = run_drift_job(source="input", input_path=input_path, min_rows=30)

    assert (result.model_name, result.model_version) == ("isolation_forest", "v202")
    assert set(result.segments) == set(segments)
    for entity_type, outcome in result.segments.items():
        # Each segment is evaluated on exactly its own features.
        assert outcome.feature_count == len(segments[entity_type])
        assert outcome.row_count == 40
    assert result.segments[SHIFTED_TYPE].status == "critical"
    for entity_type in set(segments) - {SHIFTED_TYPE}:
        assert result.segments[entity_type].status == "normal"

    records = drift.read_drift_events_jsonl(drift.DEFAULT_DRIFT_EVENTS_PATH)
    assert {record["entity_type"] for record in records} == set(segments)
    assert all(record["model_version"] == "v202" for record in records)
    assert result.alert_count == result.segments[SHIFTED_TYPE].critical_count > 0


def test_too_few_rows_in_a_segment_is_refused(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.drift_job import DriftJobError, run_drift_job

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    frame = _current_frame(artifact_dir)
    campaign_rows = frame.index[frame["entity_type"] == SHIFTED_TYPE][5:]
    input_path = tmp_path / "current.csv"
    frame.drop(campaign_rows).to_csv(input_path, index=False)

    with pytest.raises(DriftJobError, match=r"campaign has 5 rows.*minimum 30"):
        run_drift_job(source="input", input_path=input_path, min_rows=30)

    assert not drift.DEFAULT_DRIFT_EVENTS_PATH.exists()


def test_segmented_input_without_entity_type_is_refused(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.drift_job import DriftJobError, run_drift_job

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    input_path = tmp_path / "current.parquet"
    _current_frame(artifact_dir).drop(columns=["entity_type"]).to_parquet(input_path)

    with pytest.raises(DriftJobError, match="entity_type"):
        run_drift_job(source="input", input_path=input_path, min_rows=30)


def test_single_model_is_evaluated_as_one_set_of_features(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.drift_job import run_drift_job

    artifact_dir = _register_copy(tmp_path, trained_metadata_paths["v101"], "v101")
    _promote("v101")
    feature_names = json.loads((artifact_dir / "metadata.json").read_text())["feature_names"]
    input_path = tmp_path / "current.csv"
    pd.DataFrame([{name: 1.0 + 0.01 * row for name in feature_names} for row in range(40)]).to_csv(
        input_path, index=False
    )

    result = run_drift_job(source="input", input_path=input_path, min_rows=30)

    assert list(result.segments) == [None]
    assert result.segments[None].feature_count == len(feature_names)


def test_cli_prints_a_per_segment_summary_and_exits_zero_on_drift(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from anomaly_detection.drift_job import main

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    input_path = tmp_path / "current.parquet"
    _current_frame(artifact_dir).to_parquet(input_path, index=False)

    assert main(["--input", str(input_path), "--min-rows", "30"]) == 0
    out = capsys.readouterr().out
    assert "model=isolation_forest v202" in out
    assert "campaign" in out and "critical" in out

    assert main(["--input", str(tmp_path / "missing.parquet")]) == 1


def test_batch_predictions_source_needs_the_database(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.drift_job import DriftJobError, run_drift_job

    _serve_segmented(tmp_path, segmented_metadata_paths)

    with pytest.raises(DriftJobError, match="PROJECT4_POSTGRES"):
        run_drift_job(source="batch-predictions", hours=24)


@pytest.mark.postgres
def test_drift_job_from_batch_predictions_writes_jsonl_and_postgres(
    initialized_database,  # noqa: F811 - pytest fixture
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.batch_inference import run_batch_inference
    from anomaly_detection.drift_job import run_drift_job
    from anomaly_detection.registry import promote_model_version, register_model_from_artifacts

    register_model_from_artifacts(metadata_path=segmented_metadata_paths["v202"])
    promote_model_version(model_name="isolation_forest", model_version="v202")
    artifact_dir = segmented_metadata_paths["v202"].parent
    run_batch_inference(_current_frame(artifact_dir), output_path=tmp_path / "batch.jsonl")

    result = run_drift_job(source="batch-predictions", hours=24, min_rows=30)

    assert result.model_version == "v202"
    assert result.segments[SHIFTED_TYPE].status == "critical"
    assert result.segments["user"].status == "normal"
    assert _count(
        initialized_database,
        "SELECT COUNT(*) FROM monitoring.drift_events "
        "WHERE model_version = 'v202' AND entity_type = :t",
        {"t": SHIFTED_TYPE},
    ) == result.segments[SHIFTED_TYPE].feature_count
    assert _count(
        initialized_database,
        "SELECT COUNT(*) FROM monitoring.alert_events WHERE model_version = 'v202'",
    ) == result.alert_count > 0
    jsonl = drift.read_drift_events_jsonl(drift.DEFAULT_DRIFT_EVENTS_PATH)
    assert len(jsonl) == result.drift_event_count
