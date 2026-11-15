"""Tests for local model registry and production pointer behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from anomaly_detection.registry import (
    RegistryError,
    find_model_entry,
    list_model_versions,
    load_model_registry,
    promote_model_version,
    read_active_model_pointer,
    register_model_from_artifacts,
    write_active_model_pointer,
)
from anomaly_detection.training import train_isolation_forest_from_snapshot
from anomaly_detection.training_snapshot import (
    TrainingSnapshotConfig,
    create_training_snapshot,
)


def _sample_feature_frame(row_count: int = 48) -> pd.DataFrame:
    """Build deterministic feature data that satisfies the model contract."""
    rows = []

    for index in range(row_count):
        is_outlier = index >= row_count - 3
        multiplier = 7.0 if is_outlier else 1.0

        rows.append(
            {
                "entity_id": f"user_{index:03d}",
                "feature_timestamp": pd.Timestamp("2026-06-01T00:00:00Z")
                + pd.Timedelta(minutes=index),
                "schema_version": "feature_schema_v001",
                "avg_cart_value_7d": 110.0 * multiplier + index,
                "event_count_1h": 4.0 * multiplier + (index % 5),
                "avg_api_latency_ms": 100.0 * multiplier + index,
                "fraud_score_avg": min(0.99, 0.03 * multiplier + (index % 4) * 0.01),
                "purchase_probability_delta": min(
                    0.99,
                    0.07 * multiplier + (index % 5) * 0.01,
                ),
                "cart_abandonment_rate": min(
                    0.99,
                    0.14 * multiplier + (index % 4) * 0.02,
                ),
                "campaign_roas": 2.2 * multiplier + (index % 6) * 0.1,
                "conversion_rate": min(0.99, 0.25 + (index % 5) * 0.02),
                "customer_lifetime_value": 700.0 * multiplier + index * 3.0,
                "discount_sensitivity": 0.30 * multiplier + (index % 4) * 0.04,
                "page_load_p95_ms": 230.0 * multiplier + index,
            }
        )

    return pd.DataFrame(rows)


def _write_loadable_artifact(
    artifact_path: Path,
    feature_names: tuple[str, ...] = ("feature_a", "feature_b"),
) -> None:
    """Write a joblib-loadable stand-in model plus a metadata.json beside it."""
    joblib.dump({"fake_model": artifact_path.name}, artifact_path)
    (artifact_path.parent / "metadata.json").write_text(
        json.dumps({"feature_names": list(feature_names)}),
        encoding="utf-8",
    )


def _train_test_artifact(
    *,
    tmp_path: Path,
    model_version: str,
    row_count: int = 48,
) -> Path:
    snapshot_metadata = create_training_snapshot(
        _sample_feature_frame(row_count=row_count),
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path / "snapshots"),
        source_tables=["project4_registry_test_features"],
        source_project_versions={
            "project1": "test",
            "project2_3": "test",
        },
        created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
        overwrite=True,
    )

    artifact_metadata = train_isolation_forest_from_snapshot(
        features_path=snapshot_metadata.features_path,
        snapshot_metadata_path=snapshot_metadata.metadata_path,
        model_version=model_version,
        artifact_root_override=tmp_path / "artifacts",
        overwrite=True,
        training_started_at_utc=datetime(2026, 6, 8, 12, 5, 0, tzinfo=UTC),
        registry_path=tmp_path / "training_label_registry.json",
    )

    return Path(artifact_metadata.metadata_path)


def test_register_model_from_artifacts_writes_candidate_registry_entry(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry" / "model_registry.json"
    metadata_path = _train_test_artifact(tmp_path=tmp_path, model_version="v001")

    entry = register_model_from_artifacts(
        metadata_path=metadata_path,
        registry_path=registry_path,
        status="candidate",
        approved_for_prod=False,
        notes="Candidate model from registry test.",
    )

    assert registry_path.exists()
    assert entry.model_name == "isolation_forest"
    assert entry.model_version == "v001"
    assert entry.status == "candidate"
    assert entry.approved_for_prod is False
    assert entry.baseline_anomaly_rate is not None
    assert entry.baseline_anomaly_rate >= 0.0
    assert entry.baseline_anomaly_rate <= 1.0
    assert entry.snapshot_id is not None
    assert entry.training_dataset_id is not None

    loaded_entries = load_model_registry(registry_path)
    assert len(loaded_entries) == 1
    assert loaded_entries[0].model_version == "v001"

    with registry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["registry_backend"] == "local_json"
    assert payload["models"][0]["status"] == "candidate"


def test_promote_model_version_updates_active_pointer_and_archives_previous_production(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry" / "model_registry.json"
    active_model_path = tmp_path / "configs" / "active_model.yaml"

    metadata_v001 = _train_test_artifact(
        tmp_path=tmp_path,
        model_version="v001",
        row_count=48,
    )
    metadata_v002 = _train_test_artifact(
        tmp_path=tmp_path,
        model_version="v002",
        row_count=52,
    )

    register_model_from_artifacts(
        metadata_path=metadata_v001,
        registry_path=registry_path,
        status="candidate",
    )
    register_model_from_artifacts(
        metadata_path=metadata_v002,
        registry_path=registry_path,
        status="candidate",
    )

    promoted_v001 = promote_model_version(
        model_name="isolation_forest",
        model_version="v001",
        registry_path=registry_path,
        active_model_path=active_model_path,
        notes="Promote v001 for test.",
    )

    assert promoted_v001.status == "production"
    assert promoted_v001.approved_for_prod is True

    pointer_v001 = read_active_model_pointer(active_model_path)
    assert pointer_v001.model_name == "isolation_forest"
    assert pointer_v001.active_model_version == "v001"
    assert pointer_v001.status == "production"
    assert pointer_v001.artifact_path == promoted_v001.artifact_path

    promoted_v002 = promote_model_version(
        model_name="isolation_forest",
        model_version="v002",
        registry_path=registry_path,
        active_model_path=active_model_path,
        notes="Promote v002 for test.",
    )

    assert promoted_v002.status == "production"
    assert promoted_v002.approved_for_prod is True

    v001_after_v002 = find_model_entry(
        model_name="isolation_forest",
        model_version="v001",
        registry_path=registry_path,
    )
    assert v001_after_v002.status == "archived"
    assert v001_after_v002.approved_for_prod is True
    assert v001_after_v002.archived_at_utc is not None

    pointer_v002 = read_active_model_pointer(active_model_path)
    assert pointer_v002.active_model_version == "v002"
    assert pointer_v002.artifact_path == promoted_v002.artifact_path

    production_entries = list_model_versions(
        model_name="isolation_forest",
        status="production",
        registry_path=registry_path,
    )
    archived_entries = list_model_versions(
        model_name="isolation_forest",
        status="archived",
        registry_path=registry_path,
    )

    assert [entry.model_version for entry in production_entries] == ["v002"]
    assert [entry.model_version for entry in archived_entries] == ["v001"]

    with active_model_path.open("r", encoding="utf-8") as file:
        active_payload = yaml.safe_load(file)

    assert active_payload["active_model_version"] == "v002"
    assert active_payload["status"] == "production"


def test_non_production_model_cannot_become_active_pointer(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "model_registry.json"
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    metadata_path = _train_test_artifact(tmp_path=tmp_path, model_version="v001")

    candidate_entry = register_model_from_artifacts(
        metadata_path=metadata_path,
        registry_path=registry_path,
        status="candidate",
    )

    with pytest.raises(RegistryError, match="Only a production model"):
        write_active_model_pointer(
            entry=candidate_entry,
            active_model_path=active_model_path,
        )


def test_register_rejects_invalid_status(tmp_path: Path) -> None:
    metadata_path = _train_test_artifact(tmp_path=tmp_path, model_version="v001")

    with pytest.raises(RegistryError, match="Invalid model status"):
        register_model_from_artifacts(
            metadata_path=metadata_path,
            registry_path=tmp_path / "registry" / "model_registry.json",
            status="experimental",
        )


def test_register_model_writes_lifecycle_record(tmp_path):
    from anomaly_detection.registry import ModelStatus, register_model

    registry_path = tmp_path / "model_registry.json"
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_text("fake model artifact", encoding="utf-8")

    record = register_model(
        model_name="isolation_forest",
        model_version="v900",
        artifact_path=str(artifact_path),
        dataset_snapshot_id="snapshot_20260610_001",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        metrics={
            "baseline_anomaly_rate": 0.04,
            "latency_p95_ms": 125.0,
        },
        status=ModelStatus.CANDIDATE,
        registry_path=registry_path,
    )

    assert record.model_name == "isolation_forest"
    assert record.model_version == "v900"
    assert record.status == "candidate"
    assert record.artifact_path == str(artifact_path)
    assert record.dataset_snapshot_id == "snapshot_20260610_001"
    assert record.feature_schema_version == "feature_schema_v001"

    registry_payload = registry_path.read_text(encoding="utf-8")
    assert "isolation_forest" in registry_payload
    assert "v900" in registry_payload
    assert "baseline_anomaly_rate" in registry_payload


def test_register_model_rejects_duplicate_model_version(tmp_path):
    import pytest

    from anomaly_detection.registry import register_model

    registry_path = tmp_path / "model_registry.json"
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_text("fake model artifact", encoding="utf-8")

    kwargs = {
        "model_name": "isolation_forest",
        "model_version": "v901",
        "artifact_path": str(artifact_path),
        "dataset_snapshot_id": "snapshot_20260610_001",
        "feature_schema_version": "feature_schema_v001",
        "training_timestamp": "2026-06-10T10:00:00+00:00",
        "metrics": {"baseline_anomaly_rate": 0.04},
        "registry_path": registry_path,
    }

    register_model(**kwargs)

    with pytest.raises(ValueError, match="already registered"):
        register_model(**kwargs)


def test_promote_model_updates_active_model_pointer(tmp_path):
    import yaml

    from anomaly_detection.registry import ModelStatus, promote_model, register_model

    registry_path = tmp_path / "model_registry.json"
    active_model_path = tmp_path / "active_model.yaml"
    artifact_path = tmp_path / "model.joblib"
    _write_loadable_artifact(artifact_path)

    register_model(
        model_name="isolation_forest",
        model_version="v902",
        artifact_path=str(artifact_path),
        dataset_snapshot_id="snapshot_20260610_002",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        metrics={
            "baseline_anomaly_rate": 0.05,
            "latency_p95_ms": 118.0,
        },
        status=ModelStatus.CANDIDATE,
        registry_path=registry_path,
    )

    promoted_record = promote_model(
        model_name="isolation_forest",
        model_version="v902",
        target_status=ModelStatus.PRODUCTION,
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )

    assert promoted_record["status"] == "production"
    assert promoted_record["approved_for_prod"] is True

    active_model = yaml.safe_load(active_model_path.read_text(encoding="utf-8"))

    assert active_model["model_name"] == "isolation_forest"
    assert active_model["active_model_version"] == "v902"
    assert active_model["status"] == "production"
    assert active_model["artifact_path"] == str(artifact_path)
    assert active_model["dataset_snapshot_id"] == "snapshot_20260610_002"
    assert active_model["feature_schema_version"] == "feature_schema_v001"


def test_get_active_model_reads_required_pointer_fields(tmp_path):
    import yaml

    from anomaly_detection.registry import get_active_model

    active_model_path = tmp_path / "active_model.yaml"
    expected_config = {
        "model_name": "isolation_forest",
        "active_model_version": "v903",
        "status": "production",
        "artifact_path": "artifacts/models/isolation_forest/model_version=v903/model.joblib",
        "dataset_snapshot_id": "snapshot_20260610_003",
        "feature_schema_version": "feature_schema_v001",
    }

    active_model_path.write_text(
        yaml.safe_dump(expected_config, sort_keys=False),
        encoding="utf-8",
    )

    active_model = get_active_model(active_model_path)

    assert active_model["model_name"] == "isolation_forest"
    assert active_model["active_model_version"] == "v903"
    assert active_model["artifact_path"].endswith("model.joblib")


def test_promoting_new_production_model_archives_previous_production(tmp_path):
    import json

    from anomaly_detection.registry import ModelStatus, promote_model, register_model

    registry_path = tmp_path / "model_registry.json"
    active_model_path = tmp_path / "active_model.yaml"

    artifact_v1 = tmp_path / "model_v1.joblib"
    artifact_v2 = tmp_path / "model_v2.joblib"
    _write_loadable_artifact(artifact_v1)
    _write_loadable_artifact(artifact_v2)

    for version, artifact in [("v904", artifact_v1), ("v905", artifact_v2)]:
        register_model(
            model_name="isolation_forest",
            model_version=version,
            artifact_path=str(artifact),
            dataset_snapshot_id=f"snapshot_{version}",
            feature_schema_version="feature_schema_v001",
            training_timestamp="2026-06-10T10:00:00+00:00",
            metrics={"baseline_anomaly_rate": 0.04},
            status=ModelStatus.CANDIDATE,
            registry_path=registry_path,
        )

    promote_model(
        model_name="isolation_forest",
        model_version="v904",
        target_status=ModelStatus.PRODUCTION,
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )

    promote_model(
        model_name="isolation_forest",
        model_version="v905",
        target_status=ModelStatus.PRODUCTION,
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )

    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    records_by_version = {
        record["model_version"]: record for record in registry_payload["models"]
    }

    assert records_by_version["v904"]["status"] == "archived"
    assert records_by_version["v904"]["approved_for_prod"] is True
    assert records_by_version["v905"]["status"] == "production"
    assert records_by_version["v905"]["approved_for_prod"] is True


REAL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "models"
    / "_registry"
    / "model_registry.json"
)


def test_register_model_round_trips_through_load_and_list(tmp_path: Path) -> None:
    from anomaly_detection.registry import register_model

    registry_path = tmp_path / "model_registry.json"
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_text("fake model artifact", encoding="utf-8")

    register_model(
        model_name="isolation_forest",
        model_version="v910",
        artifact_path=str(artifact_path),
        dataset_snapshot_id="snapshot_v910",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        metrics={"baseline_anomaly_rate": 0.04},
        registry_path=registry_path,
    )

    loaded_entries = load_model_registry(registry_path)
    listed_entries = list_model_versions(
        model_name="isolation_forest",
        registry_path=registry_path,
    )

    assert [entry.model_version for entry in loaded_entries] == ["v910"]
    assert [entry.model_version for entry in listed_entries] == ["v910"]
    assert listed_entries[0].snapshot_id == "snapshot_v910"
    assert listed_entries[0].baseline_anomaly_rate == 0.04
    assert listed_entries[0].status == "candidate"


def test_register_promote_get_active_model_uses_target_model_metadata(
    tmp_path: Path,
) -> None:
    from anomaly_detection.registry import get_active_model, promote_model, register_model

    registry_path = tmp_path / "registry" / "model_registry.json"
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    metadata_path = _train_test_artifact(tmp_path=tmp_path, model_version="v911")
    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    register_model(
        model_name="isolation_forest",
        model_version="v911",
        artifact_path=artifact_metadata["model_path"],
        dataset_snapshot_id="snapshot_v911",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        registry_path=registry_path,
    )
    promote_model(
        model_name="isolation_forest",
        model_version="v911",
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )

    active_model = get_active_model(active_model_path)

    assert active_model["active_model_version"] == "v911"
    assert active_model["status"] == "production"
    assert active_model["dataset_snapshot_id"] == "snapshot_v911"
    assert active_model["feature_count"] == len(artifact_metadata["feature_names"])


def _register_fake_model(registry_path: Path, artifact_dir: Path, model_version: str) -> None:
    from anomaly_detection.registry import register_model

    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "model.joblib"
    _write_loadable_artifact(artifact_path)

    register_model(
        model_name="isolation_forest",
        model_version=model_version,
        artifact_path=str(artifact_path),
        dataset_snapshot_id=f"snapshot_{model_version}",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        registry_path=registry_path,
    )


def test_register_model_rejects_production_status(tmp_path: Path) -> None:
    from anomaly_detection.registry import register_model

    registry_path = tmp_path / "model_registry.json"
    _register_fake_model(registry_path, tmp_path / "v920", "v920")
    registry_before = registry_path.read_bytes()

    artifact_path = tmp_path / "v921" / "model.joblib"
    artifact_path.parent.mkdir()
    artifact_path.write_text("fake model artifact", encoding="utf-8")

    with pytest.raises(RegistryError, match="promote_model"):
        register_model(
            model_name="isolation_forest",
            model_version="v921",
            artifact_path=str(artifact_path),
            dataset_snapshot_id="snapshot_v921",
            feature_schema_version="feature_schema_v001",
            training_timestamp="2026-06-10T10:00:00+00:00",
            status="production",
            approved_for_prod=True,
            registry_path=registry_path,
        )

    assert registry_path.read_bytes() == registry_before


def test_promote_model_refuses_status_change_of_active_version(tmp_path: Path) -> None:
    from anomaly_detection.registry import promote_model

    registry_path = tmp_path / "model_registry.json"
    active_model_path = tmp_path / "active_model.yaml"
    _register_fake_model(registry_path, tmp_path / "v930", "v930")
    promote_model(
        model_name="isolation_forest",
        model_version="v930",
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )
    registry_before = registry_path.read_bytes()
    pointer_before = active_model_path.read_bytes()

    with pytest.raises(RegistryError, match="active"):
        promote_model(
            model_name="isolation_forest",
            model_version="v930",
            target_status="archived",
            registry_path=registry_path,
            active_model_config_path=active_model_path,
        )

    assert registry_path.read_bytes() == registry_before
    assert active_model_path.read_bytes() == pointer_before


def test_promote_model_archives_non_active_version_with_timestamp(tmp_path: Path) -> None:
    from anomaly_detection.registry import promote_model

    registry_path = tmp_path / "model_registry.json"
    active_model_path = tmp_path / "active_model.yaml"
    _register_fake_model(registry_path, tmp_path / "v940", "v940")
    _register_fake_model(registry_path, tmp_path / "v941", "v941")
    promote_model(
        model_name="isolation_forest",
        model_version="v940",
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )

    archived = promote_model(
        model_name="isolation_forest",
        model_version="v941",
        target_status="archived",
        registry_path=registry_path,
        active_model_config_path=active_model_path,
    )

    stored = find_model_entry(
        model_name="isolation_forest",
        model_version="v941",
        registry_path=registry_path,
    )
    assert archived["status"] == "archived"
    assert stored.status == "archived"
    assert stored.archived_at_utc is not None


def _register_without_promoting(
    tmp_path: Path,
    *,
    write_artifact,
) -> Path:
    from anomaly_detection.registry import register_model

    registry_path = tmp_path / "model_registry.json"
    artifact_path = tmp_path / "v950" / "model.joblib"
    artifact_path.parent.mkdir()
    write_artifact(artifact_path)

    register_model(
        model_name="isolation_forest",
        model_version="v950",
        artifact_path=str(artifact_path),
        dataset_snapshot_id="snapshot_v950",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        registry_path=registry_path,
    )
    return registry_path


def _loadable_without_metadata(artifact_path: Path) -> None:
    joblib.dump({"fake_model": "v950"}, artifact_path)


def _unloadable_with_metadata(artifact_path: Path) -> None:
    artifact_path.write_text("this is not a joblib file", encoding="utf-8")
    (artifact_path.parent / "metadata.json").write_text(
        json.dumps({"feature_names": ["feature_a"]}),
        encoding="utf-8",
    )


def _loadable_with_empty_feature_names(artifact_path: Path) -> None:
    _write_loadable_artifact(artifact_path, feature_names=())


@pytest.mark.parametrize(
    ("write_artifact", "expected_message"),
    [
        pytest.param(_loadable_without_metadata, "metadata.json", id="no_metadata"),
        pytest.param(_unloadable_with_metadata, "does not load", id="unloadable_artifact"),
        pytest.param(
            _loadable_with_empty_feature_names,
            "feature_names",
            id="empty_feature_names",
        ),
    ],
)
def test_promotion_requires_loadable_artifact_and_feature_names(
    tmp_path: Path,
    write_artifact,
    expected_message: str,
) -> None:
    registry_path = _register_without_promoting(tmp_path, write_artifact=write_artifact)
    active_model_path = tmp_path / "active_model.yaml"
    registry_before = registry_path.read_bytes()

    with pytest.raises(RegistryError, match=expected_message):
        promote_model_version(
            model_name="isolation_forest",
            model_version="v950",
            registry_path=registry_path,
            active_model_path=active_model_path,
        )

    assert registry_path.read_bytes() == registry_before
    assert not active_model_path.exists()


@pytest.mark.parametrize(
    ("existing_mode", "expected_mode"),
    [
        pytest.param(0o664, 0o664, id="existing_pointer_keeps_mode"),
        pytest.param(None, 0o644, id="new_pointer_gets_0644"),
    ],
)
def test_pointer_write_preserves_or_sets_file_mode(
    tmp_path: Path,
    existing_mode: int | None,
    expected_mode: int,
) -> None:
    import stat

    registry_path = tmp_path / "model_registry.json"
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    _register_fake_model(registry_path, tmp_path / "v960", "v960")

    if existing_mode is not None:
        active_model_path.parent.mkdir(parents=True)
        active_model_path.write_text("model_name: isolation_forest\n", encoding="utf-8")
        active_model_path.chmod(existing_mode)

    promote_model_version(
        model_name="isolation_forest",
        model_version="v960",
        registry_path=registry_path,
        active_model_path=active_model_path,
    )

    assert stat.S_IMODE(active_model_path.stat().st_mode) == expected_mode


def test_real_registry_loads_and_saves_losslessly(tmp_path: Path) -> None:
    from anomaly_detection.registry import save_model_registry

    raw_models = json.loads(REAL_REGISTRY_PATH.read_text(encoding="utf-8"))["models"]

    entries = load_model_registry(REAL_REGISTRY_PATH)
    copy_path = tmp_path / "model_registry.json"
    save_model_registry(entries, copy_path)

    saved_models = json.loads(copy_path.read_text(encoding="utf-8"))["models"]

    assert [entry.to_dict() for entry in entries] == raw_models
    assert saved_models == raw_models


# --- stage 10c: retire v003 through the code path -------------------------------------

V003_NOTE = "Superseded: single model learned entity type; see evaluation_report.json"
V003_FILES = (
    "model.joblib",
    "metadata.json",
    "feature_schema.json",
    "baseline_stats.json",
    "evaluation_report.json",
)


def _v003_file_hashes() -> dict[str, str]:
    import hashlib

    from anomaly_detection.registry import PROJECT_ROOT

    artifact_dir = PROJECT_ROOT / "artifacts" / "models" / "isolation_forest" / "model_version=v003"
    return {
        name: hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
        for name in V003_FILES
    }


def _only_v003_status_and_notes_changed(before, after) -> None:
    assert [entry.model_version for entry in after] == [entry.model_version for entry in before]
    for old, new in zip(before, after, strict=True):
        if old.model_version != "v003":
            assert new == old
            continue
        changed = {key for key, value in new.to_dict().items() if old.to_dict()[key] != value}
        assert changed == {"status", "notes"}
        assert new.status == "retired"
        assert new.notes == V003_NOTE


@pytest.fixture()
def real_v003_registered() -> None:
    """The real v003 entry, as the candidate it was before retirement, in the mirror only.

    The real registry has v003 retired since the owner ran the retirement; the
    per-test mirror is put back to the pre-retirement state so the retirement
    path is exercised against the real v003 entry and files.
    """
    from anomaly_detection import registry
    from anomaly_detection.registry import ModelRegistryEntry, save_model_registry

    entries = load_model_registry(registry.DEFAULT_REGISTRY_PATH)
    if not any(entry.model_version == "v003" for entry in entries):
        pytest.skip("v003 is not in the real registry")

    save_model_registry(
        [
            ModelRegistryEntry(
                **{
                    **entry.to_dict(),
                    "status": "candidate",
                    "notes": "Candidate retrained from real_source_20261115T010701Z.",
                }
            )
            if entry.model_version == "v003"
            else entry
            for entry in entries
        ],
        registry.DEFAULT_REGISTRY_PATH,  # per-test mirror (conftest)
    )


def test_retire_v003_in_file_mode_changes_only_its_status_and_note_and_keeps_files(
    real_v003_registered: None,
) -> None:
    from anomaly_detection import registry
    from anomaly_detection.registry import retire_model_version

    registry_path = registry.DEFAULT_REGISTRY_PATH  # per-test mirror of the real registry
    pointer_before = registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes()
    entries_before = load_model_registry(registry_path)
    files_before = _v003_file_hashes()

    entry = retire_model_version(
        model_name="isolation_forest",
        model_version="v003",
        notes=V003_NOTE,
    )

    assert entry.status == "retired"
    _only_v003_status_and_notes_changed(entries_before, load_model_registry(registry_path))
    assert _v003_file_hashes() == files_before
    assert registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes() == pointer_before


def test_retire_refuses_the_active_version() -> None:
    from anomaly_detection.registry import retire_model_version

    with pytest.raises(RegistryError, match="active model version"):
        retire_model_version(model_name="isolation_forest", model_version="v002", notes="no")


def test_retire_v003_in_database_mode_writes_only_its_row_and_never_the_pointer(
    real_v003_registered: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from anomaly_detection import database, registry
    from anomaly_detection.registry import RegistryState, retire_model_version

    registry_path = registry.DEFAULT_REGISTRY_PATH
    entries_before = load_model_registry(registry_path)
    pointer_before = registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes()
    state = RegistryState(
        entries=entries_before,
        model_name="isolation_forest",
        active_model_version="v002",
        pointer_version=5,
        source=registry.POINTER_SOURCE_POSTGRES,
    )
    upserted: list[dict] = []

    class _Engine:
        @contextmanager
        def begin(self):
            yield object()

        def dispose(self) -> None:
            return None

    def _never(*_args, **_kwargs):
        raise AssertionError("retirement must not write the active pointer or extra rows")

    fake_database = SimpleNamespace(
        get_engine=_Engine,
        lock_active_pointer=lambda _connection, _name: 5,
        read_active_pointer=lambda _connection, _name: {
            "model_version": "v002",
            "artifact_path": "artifacts/models/isolation_forest/model_version=v002/model.joblib",
            "notes": None,
        },
        insert_row_if_absent=lambda _connection, _table, _row: False,
        upsert_model_registry_row=lambda _connection, row: upserted.append(dict(row)),
        model_registry_row=database.model_registry_row,
        snapshot_rows_for_training_metadata=database.snapshot_rows_for_training_metadata,
        set_active_pointer=_never,
        insert_row=_never,
    )
    monkeypatch.setattr(registry, "database_enabled", lambda: True)
    monkeypatch.setattr(registry, "load_registry_state", lambda **_kwargs: state)
    monkeypatch.setattr(registry, "_database", lambda: fake_database)

    retire_model_version(model_name="isolation_forest", model_version="v003", notes=V003_NOTE)

    assert [(row["model_version"], row["status"], row["notes"]) for row in upserted] == [
        ("v003", "retired", V003_NOTE)
    ]
    # The file mirror follows the database; the pointer file is not written.
    _only_v003_status_and_notes_changed(entries_before, load_model_registry(registry_path))
    assert registry.DEFAULT_ACTIVE_MODEL_PATH.read_bytes() == pointer_before


def test_registry_cli_retires_a_version(
    real_v003_registered: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from anomaly_detection import registry

    entries_before = load_model_registry(registry.DEFAULT_REGISTRY_PATH)

    exit_code = registry.main(["retire", "--model-version", "v003", "--notes", V003_NOTE])

    assert exit_code == 0
    assert "status=retired" in capsys.readouterr().out
    _only_v003_status_and_notes_changed(
        entries_before, load_model_registry(registry.DEFAULT_REGISTRY_PATH)
    )
