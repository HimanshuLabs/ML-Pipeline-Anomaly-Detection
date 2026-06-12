"""Tests for shared prediction evidence logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from anomaly_detection.prediction_logging import (
    PredictionLoggingError,
    json_safe,
    normalize_prediction_record,
    read_jsonl_records,
    write_batch_predictions_jsonl,
    write_online_predictions_jsonl,
)


@dataclass(frozen=True)
class DummyPrediction:
    prediction_id: str
    model_name: str
    model_version: str
    entity_type: str
    entity_id: str
    anomaly_score: float
    is_anomaly: bool
    threshold_used: float
    prediction_status: str
    prediction_timestamp: str
    feature_payload_hash: str
    latency_ms: float


def _dummy_prediction(entity_id: str = "entity_001") -> DummyPrediction:
    return DummyPrediction(
        prediction_id="prediction_001",
        model_name="isolation_forest",
        model_version="vtest",
        entity_type="user",
        entity_id=entity_id,
        anomaly_score=-0.25,
        is_anomaly=True,
        threshold_used=0.0,
        prediction_status="success",
        prediction_timestamp=datetime.now(UTC).isoformat(),
        feature_payload_hash="a" * 64,
        latency_ms=12.5,
    )


def test_json_safe_converts_numpy_and_datetime_values() -> None:
    payload = {
        "np_int": np.int64(7),
        "np_float": np.float64(1.25),
        "np_bool": np.bool_(True),
        "np_nan": np.float64("nan"),
        "timestamp": datetime(2026, 6, 11, tzinfo=UTC),
    }

    safe_payload = json_safe(payload)

    assert safe_payload["np_int"] == 7
    assert safe_payload["np_float"] == 1.25
    assert safe_payload["np_bool"] is True
    assert safe_payload["np_nan"] is None
    assert safe_payload["timestamp"] == "2026-06-11T00:00:00+00:00"


def test_normalize_prediction_record_adds_source_and_logged_at() -> None:
    record = normalize_prediction_record(
        _dummy_prediction(),
        prediction_source="online",
    )

    assert record["prediction_source"] == "online"
    assert record["logged_at"]
    assert record["prediction_id"] == "prediction_001"
    assert record["model_name"] == "isolation_forest"
    assert record["model_version"] == "vtest"
    assert record["entity_id"] == "entity_001"
    assert record["prediction_status"] == "success"


def test_normalize_prediction_record_rejects_invalid_source() -> None:
    with pytest.raises(PredictionLoggingError, match="prediction_source"):
        normalize_prediction_record(
            _dummy_prediction(),
            prediction_source="bad_source",
        )


def test_normalize_prediction_record_rejects_missing_required_fields() -> None:
    with pytest.raises(PredictionLoggingError, match="missing required fields"):
        normalize_prediction_record(
            {
                "prediction_id": "prediction_missing_fields",
                "model_version": "vtest",
            },
            prediction_source="online",
        )


def test_write_online_predictions_jsonl_appends_records(tmp_path: Path) -> None:
    output_path = tmp_path / "online_predictions.jsonl"

    write_online_predictions_jsonl(
        [_dummy_prediction("user_001")],
        output_path,
    )
    write_online_predictions_jsonl(
        [_dummy_prediction("user_002")],
        output_path,
    )

    records = read_jsonl_records(output_path)

    assert len(records) == 2
    assert records[0]["prediction_source"] == "online"
    assert records[0]["entity_id"] == "user_001"
    assert records[1]["entity_id"] == "user_002"


def test_write_batch_predictions_jsonl_uses_batch_source(tmp_path: Path) -> None:
    output_path = tmp_path / "batch_predictions.jsonl"

    written_path = write_batch_predictions_jsonl(
        [_dummy_prediction("customer_001")],
        output_path,
    )

    records = read_jsonl_records(written_path)

    assert written_path == output_path
    assert len(records) == 1
    assert records[0]["prediction_source"] == "batch"
    assert records[0]["entity_type"] == "user"
    assert records[0]["entity_id"] == "customer_001"
    assert records[0]["threshold_used"] == 0.0


def test_read_jsonl_records_fails_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PredictionLoggingError, match="does not exist"):
        read_jsonl_records(tmp_path / "missing.jsonl")
