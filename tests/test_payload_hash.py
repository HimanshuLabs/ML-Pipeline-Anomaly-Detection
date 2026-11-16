"""feature_payload_hash: one canonical hash for online and batch scoring.

Only the features the model scores for the record enter the hash (the entity type's
segment features for a segmented model, the model's features for a single model),
cast to float, in sorted-key canonical JSON. Extra request keys never do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.test_segmented_serving import _promote, _register_copy, _segments

EXTRA_KEYS = {"entity_id": "row-1", "entity_type": "ignored", "unknown_key": "x"}


def _serve(tmp_path: Path, metadata_path: Path, version: str) -> Path:
    artifact_dir = _register_copy(tmp_path, metadata_path, version)
    _promote(version)
    return artifact_dir


def _int_values(names: list[str], start: int = 1) -> dict[str, int]:
    return {name: start + index for index, name in enumerate(names)}


def _online_hash(payload: dict, entity_type: str) -> str:
    from anomaly_detection.online_inference import OnlineInferenceService

    return OnlineInferenceService().predict(payload, entity_type=entity_type).feature_payload_hash


def _batch_hash(row: dict, tmp_path: Path, **kwargs) -> str:
    from anomaly_detection.batch_inference import run_batch_inference

    [record] = run_batch_inference(
        pd.DataFrame([row]), output_path=tmp_path / "batch.jsonl", **kwargs
    )
    return record.feature_payload_hash


def test_single_model_online_and_batch_hash_the_same_row_identically(
    tmp_path: Path, trained_metadata_paths: dict[str, Path]
) -> None:
    artifact_dir = _serve(tmp_path, trained_metadata_paths["v101"], "v101")
    names = json.loads((artifact_dir / "metadata.json").read_text("utf-8"))["feature_names"]
    values = _int_values(names)

    online = _online_hash({**values, **EXTRA_KEYS}, "user")  # ints and extra keys
    batch = _batch_hash(
        {"entity_id": "row-1", **{n: float(v) for n, v in values.items()}}, tmp_path
    )

    assert online == batch


def test_segmented_model_online_and_batch_hash_the_same_row_identically(
    tmp_path: Path, segmented_metadata_paths: dict[str, Path]
) -> None:
    artifact_dir = _serve(tmp_path, segmented_metadata_paths["v202"], "v202")
    segments = _segments(artifact_dir / "metadata.json")
    every_feature = sorted({name for names in segments.values() for name in names})
    values = _int_values(every_feature)  # also the other segments' features

    online = _online_hash({**values, **EXTRA_KEYS}, "campaign")
    batch_row = {"entity_type": "campaign", "entity_id": "row-1", **values}
    batch = _batch_hash(batch_row, tmp_path)
    only_campaign = _online_hash(
        {name: float(values[name]) for name in segments["campaign"]}, "campaign"
    )

    assert online == batch == only_campaign


def test_hash_changes_with_a_value_but_not_with_key_order(
    tmp_path: Path, segmented_metadata_paths: dict[str, Path]
) -> None:
    artifact_dir = _serve(tmp_path, segmented_metadata_paths["v202"], "v202")
    names = _segments(artifact_dir / "metadata.json")["user"]
    values = {name: 0.5 + index for index, name in enumerate(names)}

    original = _online_hash(values, "user")
    reordered = _online_hash(dict(reversed(list(values.items()))), "user")
    changed = _online_hash({**values, names[0]: values[names[0]] + 1.0}, "user")

    assert original == reordered
    assert changed != original


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_hash_like_null(bad: float) -> None:
    from anomaly_detection.batch_inference import canonical_feature_payload_hash

    names = ["a", "b"]
    assert canonical_feature_payload_hash({"a": bad, "b": 1}, names) == (
        canonical_feature_payload_hash({"a": None, "b": 1.0}, names)
    )
