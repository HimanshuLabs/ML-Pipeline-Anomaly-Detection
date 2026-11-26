"""Fault injection: a deterministic report of synthetic faults flagged, and no other output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anomaly_detection import batch_inference, database
from tests.conftest import SEGMENTED_TEST_TYPES, real_source_test_frame

# segmented_metadata_paths trains v202 on real_source_test_frame(seed=8).
V202_TRAINING_SEED = 8
REPORT_STEM = "fault_injection_v202_seed7"


def _must_not_write(*args, **kwargs):
    raise AssertionError("fault injection wrote to a sink or the database")


def _v202_pointer(tmp_path: Path, metadata_path: Path) -> Path:
    """A pointer FILE naming the test's segmented v202 bundle."""
    artifact_dir = metadata_path.parent
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema = json.loads((artifact_dir / "feature_schema.json").read_text(encoding="utf-8"))
    pointer = tmp_path / "active_model.yaml"
    pointer.write_text(
        yaml.safe_dump(
            {
                "model_name": metadata["model_name"],
                "active_model_version": "v202",
                "artifact_path": str(artifact_dir / "model.joblib"),
                "feature_schema_version": schema["feature_schema_version"],
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_fault_injection_report_is_deterministic_and_touches_no_sink(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import fault_injection

    for module, name in (
        (database, "get_engine"),
        (database, "get_shared_engine"),
        (batch_inference, "write_predictions_jsonl"),
        (batch_inference, "_write_batch_predictions_to_database"),
    ):
        monkeypatch.setattr(module, name, _must_not_write)
    pointer = _v202_pointer(tmp_path, segmented_metadata_paths["v202"])
    snapshot = tmp_path / "features.parquet"
    real_source_test_frame(seed=V202_TRAINING_SEED).to_parquet(snapshot, index=False)
    first, second = tmp_path / "first", tmp_path / "second"

    for output_dir in (first, second):
        assert fault_injection.main(
            [
                "--active-model-path",
                str(pointer),
                "--snapshot",
                str(snapshot),
                "--seed",
                "7",
                "--output-dir",
                str(output_dir),
            ]
        ) == 0

    names = sorted(path.name for path in first.iterdir())
    assert names == [f"{REPORT_STEM}.json", f"{REPORT_STEM}.txt"]
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    report = json.loads((first / f"{REPORT_STEM}.json").read_text(encoding="utf-8"))
    assert report["measure"] == "share of synthetic faults flagged"
    assert (report["model_version"], report["snapshot"], report["seed"]) == (
        "v202",
        "features.parquet",
        7,
    )
    assert set(report["segments"]) == set(SEGMENTED_TEST_TYPES)
    for segment in report["segments"].values():
        assert segment["rows"] == 40
        assert segment["faulted_rows"] == 5
        for fault in segment["faults"].values():
            assert 0 <= fault["flagged"] <= 5
            assert fault["share_of_synthetic_faults_flagged"] == fault["flagged"] / 5
    assert {t: set(s["faults"]) for t, s in report["segments"].items()} == {
        "user": {"money_x10", "purchases_without_events"},
        "campaign": {"money_x10"},
        "product": {"money_x10"},
    }

    texts = [
        (first / f"{REPORT_STEM}.txt").read_text(encoding="utf-8").lower(),
        Path(fault_injection.__file__).read_text(encoding="utf-8").lower(),
    ]
    assert "share of synthetic faults flagged" in texts[0]
    for text in texts:
        for word in ("recall", "accuracy", "detection rate", "catches"):
            assert word not in text

    before = {name: (first / name).read_bytes() for name in names}
    assert fault_injection.main(
        [
            "--active-model-path",
            str(pointer),
            "--snapshot",
            str(snapshot),
            "--seed",
            "7",
            "--output-dir",
            str(first),
        ]
    ) == 1
    assert {name: (first / name).read_bytes() for name in names} == before
