"""Tests for the end-to-end real-source retrain (snapshot -> candidate)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly_detection import registry
from anomaly_detection.source_extract import MODEL_FEATURE_COLUMNS, write_snapshot

ENTITY_TYPES = ("user", "customer", "campaign", "funnel_day", "product", "web_customer")
ROWS_PER_ENTITY_TYPE = 12


def _real_source_frame() -> pd.DataFrame:
    """Real-source-shaped rows that pass validate_real_source_features."""
    rng = np.random.default_rng(7)
    rows = []
    for entity_type in ENTITY_TYPES:
        for index in range(ROWS_PER_ENTITY_TYPE):
            row = {
                "source_project": "project_1" if entity_type == "user" else "project_2_3",
                "source_table": f"test.{entity_type}",
                "entity_type": entity_type,
                "entity_id": f"{entity_type}_{index:03d}",
                "source_event_timestamp": "2026-11-01T00:00:00+00:00",
            }
            row.update({column: float(rng.random()) for column in MODEL_FEATURE_COLUMNS})
            row["repeat_purchase_signal"] = float(index % 2)
            row["never_ordered"] = float(index % 3 == 0)
            row["extracted_at"] = "2026-11-15T00:00:00+00:00"
            rows.append(row)
    return pd.DataFrame(rows)


def _write_real_snapshot(
    root: Path,
    *,
    frame: pd.DataFrame | None = None,
    feature_schema_version: str = "feature_schema_v002",
):
    config = {
        "snapshot_type": "real_source_extract",
        "feature_schema_version": feature_schema_version,
        "output_snapshot_prefix": "real_source",
        "output": {"root": str(root)},
        "project_1": {"enabled": True, "schema": "public", "table": "user_features"},
        "project_2_3": {
            "enabled": True,
            "tables": {"customer_360": {"schema": "marts", "table": "mart_customer_360"}},
        },
    }
    return write_snapshot(
        _real_source_frame() if frame is None else frame,
        config,
        output_root=root,
    )


def _registry_versions(registry_path: Path) -> dict[str, registry.ModelRegistryEntry]:
    return {entry.model_version: entry for entry in registry.load_model_registry(registry_path)}


def test_feature_schema_v002_is_accepted_for_the_52_feature_set(tmp_path: Path) -> None:
    from anomaly_detection.training import train_isolation_forest_from_snapshot

    assert len(MODEL_FEATURE_COLUMNS) == 52
    snapshot = _write_real_snapshot(tmp_path / "snapshots")

    # Default registry: the (mirrored) real one, where v001 already names two other lists.
    artifact = train_isolation_forest_from_snapshot(
        features_path=snapshot.features_path,
        snapshot_metadata_path=snapshot.metadata_path,
        model_version="v003",
        artifact_root_override=tmp_path / "artifacts",
    )

    assert artifact.feature_schema_version == "feature_schema_v002"
    assert artifact.feature_names == MODEL_FEATURE_COLUMNS
    schema = json.loads(Path(artifact.feature_schema_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(artifact.baseline_stats_path).read_text(encoding="utf-8"))
    assert schema["feature_schema_version"] == "feature_schema_v002"
    assert baseline["feature_schema_version"] == "feature_schema_v002"


def test_retrain_writes_bundle_registers_candidate_and_leaves_pointer(tmp_path: Path) -> None:
    from anomaly_detection.retrain import retrain

    snapshot = _write_real_snapshot(tmp_path / "snapshots")
    registry_path = registry.DEFAULT_REGISTRY_PATH
    pointer_path = registry.DEFAULT_ACTIVE_MODEL_PATH
    pointer_before = pointer_path.read_bytes()
    active_before = _registry_versions(registry_path)["v002"]

    result = retrain(
        snapshot=str(snapshot.snapshot_path),
        model_version="v003",
        artifact_root=tmp_path / "artifacts",
    )

    artifact_dir = tmp_path / "artifacts" / "isolation_forest" / "model_version=v003"
    for name in (
        "model.joblib",
        "metadata.json",
        "feature_schema.json",
        "baseline_stats.json",
        "evaluation_report.json",
    ):
        assert (artifact_dir / name).is_file(), name

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["runtime_versions"]) >= {"python", "scikit-learn", "numpy"}
    assert len(metadata["feature_set_fingerprint"]) == 64
    assert metadata["feature_schema_version"] == "feature_schema_v002"

    # Registered as a candidate; the active pointer and production version are untouched.
    assert result.registry_source == registry.POINTER_SOURCE_FILE
    entries = _registry_versions(registry_path)
    assert entries["v003"].status == "candidate"
    assert entries["v003"].approved_for_prod is False
    assert entries["v002"] == active_before
    assert pointer_path.read_bytes() == pointer_before

    report = json.loads((artifact_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["model_version"] == "v003"
    assert report["snapshot_id"] == snapshot.snapshot_id
    assert report["contamination"] == 0.05
    assert report["row_counts"]["snapshot"] == len(ENTITY_TYPES) * ROWS_PER_ENTITY_TYPE
    assert report["row_counts"]["trained"] == len(ENTITY_TYPES) * ROWS_PER_ENTITY_TYPE
    assert 0.0 < report["overall"]["anomaly_rate"] < 0.2
    assert set(report["entity_types"]) == set(ENTITY_TYPES)
    flagged = 0
    for entity_type in ENTITY_TYPES:
        stats = report["entity_types"][entity_type]
        assert stats["row_count"] == ROWS_PER_ENTITY_TYPE
        assert 0.0 <= stats["anomaly_rate"] <= 1.0
        assert stats["score_p5"] <= stats["score_p50"] <= stats["score_p95"]
        flagged += stats["anomaly_count"]
    assert flagged == report["overall"]["anomaly_count"]


def _failed_snapshot(root: Path):
    frame = _real_source_frame()
    frame.loc[0, "total_events"] = np.nan
    return _write_real_snapshot(root, frame=frame)


@pytest.mark.parametrize(
    ("make_snapshot", "expected_reasons"),
    [
        pytest.param(
            _failed_snapshot,
            ["data_quality_status is 'failed'", "trainable is not true", "total_events"],
            id="failed_snapshot",
        ),
        pytest.param(
            lambda root: _write_real_snapshot(root, feature_schema_version="feature_schema_v001"),
            ["feature_schema_version 'feature_schema_v001'"],
            id="wrong_schema_label",
        ),
    ],
)
def test_retrain_refuses_untrainable_snapshot(
    tmp_path: Path,
    make_snapshot,
    expected_reasons: list[str],
) -> None:
    from anomaly_detection.retrain import RetrainError, retrain

    snapshot = make_snapshot(tmp_path / "snapshots")
    registry_before = registry.DEFAULT_REGISTRY_PATH.read_bytes()

    with pytest.raises(RetrainError) as raised:
        retrain(
            snapshot=str(snapshot.snapshot_path),
            model_version="v003",
            artifact_root=tmp_path / "artifacts",
        )

    for reason in expected_reasons:
        assert reason in str(raised.value)
    assert not (tmp_path / "artifacts").exists()
    assert registry.DEFAULT_REGISTRY_PATH.read_bytes() == registry_before


def test_retrain_refuses_an_already_registered_version(tmp_path: Path) -> None:
    from anomaly_detection.retrain import RetrainError, retrain

    snapshot = _write_real_snapshot(tmp_path / "snapshots")

    with pytest.raises(RetrainError, match="v002 is already registered"):
        retrain(
            snapshot=str(snapshot.snapshot_path),
            model_version="v002",
            artifact_root=tmp_path / "artifacts",
        )

    assert not (tmp_path / "artifacts").exists()
