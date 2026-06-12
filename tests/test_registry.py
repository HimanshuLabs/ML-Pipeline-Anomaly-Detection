"""Tests for local model registry and production pointer behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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
    artifact_path.write_text("fake model artifact", encoding="utf-8")

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
    artifact_v1.write_text("fake v1 model artifact", encoding="utf-8")
    artifact_v2.write_text("fake v2 model artifact", encoding="utf-8")

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
