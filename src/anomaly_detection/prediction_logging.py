"""Prediction evidence logging for Project 4.

This module provides the shared local-first persistence contract for anomaly
prediction evidence.

The SQL schema already defines ml.batch_predictions and ml.online_predictions.
This checkpoint writes durable JSONL records under logs/predictions/ first so
batch and online inference can share one stable evidence format before database
persistence is wired.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BATCH_PREDICTION_LOG_PATH = (
    PROJECT_ROOT / "logs" / "predictions" / "batch_predictions.jsonl"
)

DEFAULT_ONLINE_PREDICTION_LOG_PATH = (
    PROJECT_ROOT / "logs" / "predictions" / "online_predictions.jsonl"
)


class PredictionLoggingError(RuntimeError):
    """Raised when prediction evidence cannot be persisted safely."""


def json_safe(value: Any) -> Any:
    """Convert common Python, numpy, pandas, and dataclass values to JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()

    if value is None:
        return None

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            return None
        return numeric_value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def normalize_prediction_record(
    record: Any,
    *,
    prediction_source: str,
) -> dict[str, Any]:
    """Normalize a batch or online prediction record for JSONL persistence."""

    if prediction_source not in {"batch", "online"}:
        raise PredictionLoggingError(
            "prediction_source must be one of: batch, online"
        )

    safe_record = json_safe(record)

    if not isinstance(safe_record, dict):
        raise PredictionLoggingError(
            "prediction record must normalize to a JSON object"
        )

    normalized = {
        "prediction_source": prediction_source,
        "logged_at": datetime.now(UTC).isoformat(),
        **safe_record,
    }

    required_fields = [
        "prediction_id",
        "model_name",
        "model_version",
        "entity_type",
        "entity_id",
        "anomaly_score",
        "is_anomaly",
        "threshold_used",
        "prediction_status",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in normalized
    ]

    if missing_fields:
        raise PredictionLoggingError(
            "prediction record missing required fields: "
            + ", ".join(missing_fields)
        )

    return normalized


def append_jsonl_records(
    records: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Append normalized prediction records to a JSONL file."""

    if not records:
        raise PredictionLoggingError("no prediction records supplied")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            safe_record = json_safe(record)
            if not isinstance(safe_record, dict):
                raise PredictionLoggingError(
                    "each JSONL record must be a JSON object"
                )

            handle.write(
                json.dumps(
                    safe_record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return output_path


def write_prediction_records_jsonl(
    records: list[Any],
    output_path: Path,
    *,
    prediction_source: str,
) -> Path:
    """Normalize and append prediction records to a JSONL file."""

    normalized_records = [
        normalize_prediction_record(
            record,
            prediction_source=prediction_source,
        )
        for record in records
    ]

    return append_jsonl_records(normalized_records, output_path)


def write_batch_predictions_jsonl(
    records: list[Any],
    output_path: Path = DEFAULT_BATCH_PREDICTION_LOG_PATH,
) -> Path:
    """Persist batch prediction evidence to JSONL."""

    return write_prediction_records_jsonl(
        records,
        output_path,
        prediction_source="batch",
    )


def write_online_predictions_jsonl(
    records: list[Any],
    output_path: Path = DEFAULT_ONLINE_PREDICTION_LOG_PATH,
) -> Path:
    """Persist online prediction evidence to JSONL."""

    return write_prediction_records_jsonl(
        records,
        output_path,
        prediction_source="online",
    )


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk for tests and smoke validation."""

    if not path.exists():
        raise PredictionLoggingError(f"JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise PredictionLoggingError(
                f"line {line_number} is not a JSON object: {path}"
            )

        records.append(loaded)

    return records
