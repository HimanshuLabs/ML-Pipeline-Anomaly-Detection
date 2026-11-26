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


def test_no_current_rows_is_not_evaluated_exits_zero_and_records_nothing(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from anomaly_detection import drift_job

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    segments = _segments(artifact_dir / "metadata.json")
    # An empty batch-predictions result is a frame with no columns at all.
    monkeypatch.setattr(drift_job, "load_batch_prediction_features", lambda **_: pd.DataFrame())
    monkeypatch.setattr(drift_job, "last_batch_prediction", lambda **_: None)  # never scored

    result = drift_job.run_drift_job(source="batch-predictions", hours=24)

    assert result.status == drift.NOT_EVALUATED
    assert result.segments == {} and result.drift_event_count == result.alert_count == 0
    assert result.skipped == {entity_type: "no rows" for entity_type in sorted(segments)}

    # The CLI, with an input file that has the columns but no rows.
    empty_input = tmp_path / "empty.parquet"
    _current_frame(artifact_dir).head(0).to_parquet(empty_input, index=False)
    assert drift_job.main(["--input", str(empty_input)]) == 0
    out = capsys.readouterr().out
    assert "not_evaluated" in out and "model=isolation_forest v202" in out
    assert not drift.DEFAULT_DRIFT_EVENTS_PATH.exists()  # nothing recorded

    # A single (unsegmented) model with no rows is not evaluated either.
    _register_copy(tmp_path, trained_metadata_paths["v101"], "v101")
    _promote("v101")
    single = drift_job.run_drift_job(source="batch-predictions", hours=24)
    assert single.status == drift.NOT_EVALUATED and single.skipped == {"all": "no rows"}


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


# --- clean-up: the batch window is read in chunks, with the same drift output -----------


def _payload_rows(
    artifact_dir: Path, *, int_feature: str | None = None, holes: bool = False
) -> list[dict]:
    """Batch-prediction rows as the database returns them, segments interleaved."""
    import numpy as np

    segments = _segments(artifact_dir / "metadata.json")
    frame = _current_frame(artifact_dir)
    rows = []
    for position in np.random.default_rng(3).permutation(len(frame)):
        record = frame.iloc[int(position)]
        entity_type = str(record["entity_type"])
        payload = {name: float(record[name]) for name in segments[entity_type]}
        if int_feature is not None:
            payload[int_feature] = int(round(payload[int_feature]))
        rows.append({"entity_type": entity_type, "feature_payload": payload})
    if holes:
        first = rows[0]["feature_payload"]
        first[next(iter(first))] = None
        fifth = rows[5]["feature_payload"]
        fifth.pop(next(iter(fifth)))
    return rows


def _serve_payload_rows(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict], *, chunk_rows: int = 7
) -> list[int]:
    """The database stub: rows come chunk_rows at a time; returns the chunk sizes served."""
    import copy

    from anomaly_detection import database, drift_job

    served: list[int] = []

    def _iter(**kwargs):
        assert kwargs["chunk_rows"] == chunk_rows
        for start in range(0, len(rows), chunk_rows):
            chunk = copy.deepcopy(rows[start : start + chunk_rows])
            served.append(len(chunk))
            yield chunk

    monkeypatch.setattr(database, "is_configured", lambda *args, **kwargs: True)
    monkeypatch.setattr(database, "iter_batch_feature_payloads", _iter)
    monkeypatch.setattr(drift_job, "PAYLOAD_CHUNK_ROWS", chunk_rows)
    return served


def _must_not_read_in_one_piece(**kwargs):
    raise AssertionError("the batch window was read in one piece")


def _load_current(**overrides):
    from datetime import UTC, datetime

    from anomaly_detection.drift_job import load_batch_prediction_features

    arguments = {
        "model_name": "isolation_forest",
        "model_version": "v202",
        "since": datetime(2026, 11, 25, tzinfo=UTC),
        "by_entity_type": True,
    }
    arguments.update(overrides)
    return load_batch_prediction_features(**arguments)


def _outcomes(current, *, baseline_path: Path, min_rows: int = 30):
    from datetime import UTC, datetime, timedelta

    from anomaly_detection.drift_job import evaluate_active_model_drift

    end = datetime(2026, 11, 26, 12, tzinfo=UTC)
    return evaluate_active_model_drift(
        current,
        baseline=drift.load_baseline_stats(baseline_path),
        threshold_config=drift.load_drift_thresholds(drift.DEFAULT_DRIFT_THRESHOLDS_PATH),
        min_rows=min_rows,
        window=(end - timedelta(hours=24), end),
        detected_at=end.isoformat(),
    )


def _without(record: dict, *names: str) -> dict:
    return {key: value for key, value in record.items() if key not in names}


def _comparable(result) -> tuple:
    """Every outcome, drift event and alert field except ids and timestamps."""
    from dataclasses import asdict

    outcomes, skipped = result
    return skipped, {
        entity_type: (
            outcome.row_count,
            outcome.feature_count,
            outcome.status,
            outcome.normal_count,
            outcome.warning_count,
            outcome.critical_count,
            [
                _without(asdict(event), "drift_event_id", "detected_at")
                for event in outcome.result.drift_events
            ],
            [
                _without(asdict(alert), "alert_event_id", "drift_event_id", "triggered_at")
                for alert in drift.create_alerts_for_drift_result(outcome.result)
            ],
        )
        for entity_type, outcome in outcomes.items()
    }


def _comparable_or_error(current, *, baseline_path: Path) -> tuple:
    from anomaly_detection.drift_job import DriftJobError

    try:
        return ("ok", _comparable(_outcomes(current, baseline_path=baseline_path)))
    except DriftJobError as exc:
        return ("error", str(exc))


@pytest.mark.parametrize("variant", ["int_feature", "holes"])
def test_chunked_payloads_give_the_same_drift_events_as_one_frame(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    from anomaly_detection import database
    from anomaly_detection.drift_job import CurrentPayloads, payload_frame

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    segments = _segments(artifact_dir / "metadata.json")
    assert all("session_count" in names for names in segments.values())
    rows = _payload_rows(
        artifact_dir,
        int_feature="session_count" if variant == "int_feature" else None,
        holes=variant == "holes",
    )
    served = _serve_payload_rows(monkeypatch, rows)
    monkeypatch.setattr(database, "read_batch_feature_payloads", _must_not_read_in_one_piece)

    current = _load_current()

    assert isinstance(current, CurrentPayloads)
    assert len(served) > 1 and sum(served) == len(current) == len(rows)
    baseline_path = segmented_metadata_paths["v202"].parent / "baseline_stats.json"
    chunked = _comparable(_outcomes(current, baseline_path=baseline_path))
    assert chunked == _comparable(_outcomes(payload_frame(rows), baseline_path=baseline_path))

    _, by_segment = chunked
    dtypes = {
        (entity_type, event["feature_name"]): event["feature_dtype"]
        for entity_type, outcome in by_segment.items()
        for event in outcome[6]
    }
    for (_, name), dtype in dtypes.items():
        expected = "int64" if variant == "int_feature" and name == "session_count" else "float64"
        assert dtype == expected, (name, dtype)
    assert by_segment[SHIFTED_TYPE][2] == "critical" and by_segment[SHIFTED_TYPE][7]


@pytest.mark.parametrize(
    "damage", ["null_entity_type", "unknown_entity_type", "too_few_rows", "feature_never_sent"]
)
def test_chunked_read_refuses_like_the_one_frame(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    from anomaly_detection import database
    from anomaly_detection.drift_job import payload_frame

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    rows = _payload_rows(artifact_dir)
    if damage == "null_entity_type":
        rows[3]["entity_type"] = None
    elif damage == "unknown_entity_type":
        rows[3]["entity_type"] = "planet"
    elif damage == "too_few_rows":
        campaign = [i for i, row in enumerate(rows) if row["entity_type"] == SHIFTED_TYPE]
        dropped = set(campaign[5:])
        rows = [row for i, row in enumerate(rows) if i not in dropped]
    else:
        name = _segments(artifact_dir / "metadata.json")["user"][0]
        for row in rows:
            row["feature_payload"].pop(name, None)
    _serve_payload_rows(monkeypatch, rows)
    monkeypatch.setattr(database, "read_batch_feature_payloads", _must_not_read_in_one_piece)

    baseline_path = segmented_metadata_paths["v202"].parent / "baseline_stats.json"
    chunked = _comparable_or_error(_load_current(), baseline_path=baseline_path)

    assert chunked[0] == "error"
    assert chunked == _comparable_or_error(payload_frame(rows), baseline_path=baseline_path)


@pytest.mark.parametrize("value", [True, "7", 2**60], ids=["bool", "text", "huge_int"])
def test_a_payload_value_the_columns_cannot_carry_is_read_in_one_piece(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: object,
) -> None:
    import logging

    from anomaly_detection import database
    from anomaly_detection.drift_job import payload_frame

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    rows = _payload_rows(artifact_dir)
    payload = rows[4]["feature_payload"]
    payload[next(iter(payload))] = value
    _serve_payload_rows(monkeypatch, rows)
    one_piece_reads: list[dict] = []

    def _read_in_one_piece(**kwargs):
        one_piece_reads.append(kwargs)
        return rows

    monkeypatch.setattr(database, "read_batch_feature_payloads", _read_in_one_piece)

    with caplog.at_level(logging.WARNING, logger="anomaly_detection.drift_job"):
        current = _load_current()

    assert isinstance(current, pd.DataFrame)
    assert len(one_piece_reads) == 1
    warnings = [r for r in caplog.records if r.name == "anomaly_detection.drift_job"]
    assert len(warnings) == 1 and "one piece" in warnings[0].getMessage()
    baseline_path = segmented_metadata_paths["v202"].parent / "baseline_stats.json"
    assert _comparable_or_error(current, baseline_path=baseline_path) == _comparable_or_error(
        payload_frame(rows), baseline_path=baseline_path
    )


def test_batch_window_is_read_in_chunks_not_in_one_piece(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import database, drift_job

    assert drift_job.PAYLOAD_CHUNK_ROWS == 10_000
    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    rows = _payload_rows(artifact_dir)
    served = _serve_payload_rows(monkeypatch, rows, chunk_rows=drift_job.PAYLOAD_CHUNK_ROWS)
    monkeypatch.setattr(database, "read_batch_feature_payloads", _must_not_read_in_one_piece)

    current = _load_current()

    assert isinstance(current, drift_job.CurrentPayloads)
    assert served == [len(rows)]


@pytest.mark.postgres
def test_chunked_drift_matches_the_one_piece_read_on_a_segmented_run(
    initialized_database,  # noqa: F811 - pytest fixture
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime, timedelta

    from anomaly_detection import database, drift_job
    from anomaly_detection.batch_inference import run_batch_inference
    from anomaly_detection.registry import promote_model_version, register_model_from_artifacts

    register_model_from_artifacts(metadata_path=segmented_metadata_paths["v202"])
    promote_model_version(model_name="isolation_forest", model_version="v202")
    artifact_dir = segmented_metadata_paths["v202"].parent
    run_batch_inference(_current_frame(artifact_dir), output_path=tmp_path / "batch.jsonl")

    monkeypatch.setattr(drift_job, "PAYLOAD_CHUNK_ROWS", 7)
    served: list[int] = []
    real_iter = database.iter_batch_feature_payloads

    def _counting(**kwargs):
        for chunk in real_iter(**kwargs):
            served.append(len(chunk))
            yield chunk

    monkeypatch.setattr(database, "iter_batch_feature_payloads", _counting)
    since = datetime.now(UTC) - timedelta(hours=24)

    current = _load_current(since=since)
    one_piece = drift_job.payload_frame(
        database.read_batch_feature_payloads(
            model_name="isolation_forest", model_version="v202", since=since
        )
    )

    assert isinstance(current, drift_job.CurrentPayloads)
    assert len(served) >= 2 and max(served) == 7
    assert sum(served) == len(current) == len(one_piece) > 0
    baseline_path = artifact_dir / "baseline_stats.json"
    assert _comparable(_outcomes(current, baseline_path=baseline_path)) == _comparable(
        _outcomes(one_piece, baseline_path=baseline_path)
    )
