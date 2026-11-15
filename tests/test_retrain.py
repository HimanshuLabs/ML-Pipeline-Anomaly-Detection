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
ROWS_PER_ENTITY_TYPE = 40


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


def _train_real(snapshot, tmp_path: Path, **kwargs):
    from anomaly_detection.training import train_isolation_forest_from_snapshot

    return train_isolation_forest_from_snapshot(
        features_path=snapshot.features_path,
        snapshot_metadata_path=snapshot.metadata_path,
        model_version="v004",
        artifact_root_override=tmp_path / "artifacts",
        **kwargs,
    )


def test_real_source_training_is_segmented_and_labelled_feature_schema_v003(
    tmp_path: Path,
) -> None:
    import joblib

    from anomaly_detection.segmented_model import (
        SEGMENTED_MODEL_KIND,
        SegmentedIsolationForest,
        segment_spec_fingerprint,
    )
    from anomaly_detection.source_extract import SEGMENT_FEATURE_COLUMNS

    assert len(MODEL_FEATURE_COLUMNS) == 52
    snapshot = _write_real_snapshot(tmp_path / "snapshots")

    # Default registry: the (mirrored) real one; v002 there names a single 52-feature model.
    artifact = _train_real(snapshot, tmp_path)

    spec = {t: list(SEGMENT_FEATURE_COLUMNS[t]) for t in sorted(ENTITY_TYPES)}
    assert artifact.feature_schema_version == "feature_schema_v003"
    assert artifact.model_kind == SEGMENTED_MODEL_KIND
    assert artifact.feature_names == MODEL_FEATURE_COLUMNS  # ordered union
    assert artifact.feature_set_fingerprint == segment_spec_fingerprint(spec)

    metadata = json.loads(Path(artifact.metadata_path).read_text(encoding="utf-8"))
    assert metadata["model_kind"] == SEGMENTED_MODEL_KIND
    assert set(metadata["segments"]) == set(ENTITY_TYPES)
    for entity_type, segment in metadata["segments"].items():
        assert segment["feature_names"] == spec[entity_type]
        assert segment["row_count"] == ROWS_PER_ENTITY_TYPE
        assert len(segment["feature_set_fingerprint"]) == 64

    model = joblib.load(artifact.model_path)
    assert isinstance(model, SegmentedIsolationForest)
    assert model.feature_spec() == spec

    schema = json.loads(Path(artifact.feature_schema_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(artifact.baseline_stats_path).read_text(encoding="utf-8"))
    assert schema["feature_schema_version"] == "feature_schema_v003"
    assert schema["model_kind"] == SEGMENTED_MODEL_KIND
    assert baseline["feature_schema_version"] == "feature_schema_v003"
    for entity_type in ENTITY_TYPES:
        segment_baseline = baseline["segments"][entity_type]
        assert segment_baseline["anomaly_rate"] == pytest.approx(0.05, abs=0.03)
        assert segment_baseline["score_p5"] <= segment_baseline["score_p95"]
        # JSON keys are written sorted; the ordered list lives in metadata.json segments.
        assert set(segment_baseline["feature_baselines"]) == set(spec[entity_type])


def test_segment_below_min_rows_per_segment_refuses_training(tmp_path: Path) -> None:
    from anomaly_detection.training import TrainingError

    frame = _real_source_frame()
    campaign_rows = frame.index[frame["entity_type"] == "campaign"][10:]
    snapshot = _write_real_snapshot(tmp_path / "snapshots", frame=frame.drop(campaign_rows))

    with pytest.raises(TrainingError, match=r"campaign has 10 rows.*min_rows_per_segment 30"):
        _train_real(snapshot, tmp_path)

    assert not (tmp_path / "artifacts").exists()


def _registry_with_v003_spec(tmp_path: Path, segments: dict[str, list[str]]) -> Path:
    import joblib

    registry_path = tmp_path / "registry" / "model_registry.json"
    artifact_dir = tmp_path / "existing_model"
    artifact_dir.mkdir()
    joblib.dump({"fake_model": "v900"}, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_kind": "segmented_isolation_forest",
                "feature_names": MODEL_FEATURE_COLUMNS,
                "segments": {
                    t: {"feature_names": names, "row_count": 40} for t, names in segments.items()
                },
            }
        ),
        encoding="utf-8",
    )
    registry.register_model(
        model_name="isolation_forest",
        model_version="v900",
        artifact_path=str(artifact_dir / "model.joblib"),
        dataset_snapshot_id="snapshot_v900",
        feature_schema_version="feature_schema_v003",
        training_timestamp="2026-11-15T10:00:00+00:00",
        registry_path=registry_path,
    )
    return registry_path


def test_label_check_compares_the_segment_spec_not_only_the_union(tmp_path: Path) -> None:
    from anomaly_detection.source_extract import SEGMENT_FEATURE_COLUMNS
    from anomaly_detection.training import TrainingError

    spec = {t: list(SEGMENT_FEATURE_COLUMNS[t]) for t in ENTITY_TYPES}
    # Same types and the same ordered union, but one segment's order differs.
    other = {**spec, "user": list(reversed(spec["user"]))}
    snapshot = _write_real_snapshot(tmp_path / "snapshots")

    with pytest.raises(TrainingError, match="feature_schema_v003.*v900.*segment spec"):
        _train_real(snapshot, tmp_path, registry_path=_registry_with_v003_spec(tmp_path, other))

    same_dir = tmp_path / "same"
    same_dir.mkdir()
    artifact = _train_real(
        snapshot, same_dir, registry_path=_registry_with_v003_spec(same_dir, spec)
    )
    assert artifact.feature_schema_version == "feature_schema_v003"


def test_retrain_writes_bundle_registers_candidate_and_leaves_pointer(tmp_path: Path) -> None:
    from anomaly_detection.retrain import retrain

    snapshot = _write_real_snapshot(tmp_path / "snapshots")
    registry_path = registry.DEFAULT_REGISTRY_PATH
    pointer_path = registry.DEFAULT_ACTIVE_MODEL_PATH
    pointer_before = pointer_path.read_bytes()
    active_before = _registry_versions(registry_path)["v002"]

    result = retrain(
        snapshot=str(snapshot.snapshot_path),
        model_version="v004",
        artifact_root=tmp_path / "artifacts",
    )

    artifact_dir = tmp_path / "artifacts" / "isolation_forest" / "model_version=v004"
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
    assert metadata["feature_schema_version"] == "feature_schema_v003"
    assert metadata["model_kind"] == "segmented_isolation_forest"

    # Registered as a candidate; the active pointer and production version are untouched.
    assert result.registry_source == registry.POINTER_SOURCE_FILE
    entries = _registry_versions(registry_path)
    assert entries["v004"].status == "candidate"
    assert entries["v004"].approved_for_prod is False
    assert entries["v004"].feature_schema_version == "feature_schema_v003"
    assert entries["v002"] == active_before
    assert pointer_path.read_bytes() == pointer_before

    report = json.loads((artifact_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["model_version"] == "v004"
    assert report["model_kind"] == "segmented_isolation_forest"
    for entity_type in ENTITY_TYPES:
        segment = report["segments"][entity_type]
        assert segment["feature_names"] == metadata["segments"][entity_type]["feature_names"]
        assert segment["feature_count"] == len(segment["feature_names"])
        # One model per type: each type's rate is near contamination, not 0% or 76%.
        assert report["entity_types"][entity_type]["anomaly_rate"] == pytest.approx(
            0.05, abs=0.03
        )
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
