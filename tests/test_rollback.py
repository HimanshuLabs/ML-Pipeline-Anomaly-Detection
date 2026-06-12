"""Tests for rollback controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anomaly_detection.rollback import (
    RollbackError,
    read_rollback_events_jsonl,
    rollback_active_model,
    select_previous_stable_model,
)


def _write_active_model(path: Path, version: str = "v002") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model_name": "isolation_forest",
                "active_model_version": version,
                "artifact_path": (
                    "artifacts/models/isolation_forest/"
                    f"model_version={version}/model.joblib"
                ),
                "dataset_snapshot_id": f"snapshot_{version}",
                "feature_schema_version": "feature_schema_v001",
                "status": "production",
                "approved_for_prod": True,
                "baseline_anomaly_rate": 0.05,
                "last_updated_at": "2026-06-10T00:00:00+00:00",
                "updated_at": "2026-06-10T00:00:00+00:00",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _registry_records() -> list[dict[str, object]]:
    return [
        {
            "model_name": "isolation_forest",
            "model_version": "v001",
            "artifact_path": "artifacts/models/isolation_forest/model_version=v001/model.joblib",
            "dataset_snapshot_id": "snapshot_v001",
            "feature_schema_version": "feature_schema_v001",
            "status": "archived",
            "approved_for_prod": True,
            "archived_at_utc": "2026-06-10T05:00:00+00:00",
        },
        {
            "model_name": "isolation_forest",
            "model_version": "v002",
            "artifact_path": "artifacts/models/isolation_forest/model_version=v002/model.joblib",
            "dataset_snapshot_id": "snapshot_v002",
            "feature_schema_version": "feature_schema_v001",
            "status": "production",
            "approved_for_prod": True,
            "promoted_at_utc": "2026-06-10T06:00:00+00:00",
        },
        {
            "model_name": "isolation_forest",
            "model_version": "v003",
            "artifact_path": "artifacts/models/isolation_forest/model_version=v003/model.joblib",
            "dataset_snapshot_id": "snapshot_v003",
            "feature_schema_version": "feature_schema_v001",
            "status": "candidate",
            "approved_for_prod": False,
            "registered_at_utc": "2026-06-10T07:00:00+00:00",
        },
    ]


def test_select_previous_stable_model_excludes_current_and_unapproved_models() -> None:
    target = select_previous_stable_model(
        _registry_records(),
        current_model_version="v002",
        model_name="isolation_forest",
    )

    assert target.model_name == "isolation_forest"
    assert target.model_version == "v001"
    assert target.status == "production"
    assert target.approved_for_prod is True
    assert target.selected_from_status == "archived"
    assert target.artifact_path.endswith("model_version=v001/model.joblib")


def test_select_previous_stable_model_rejects_missing_target() -> None:
    with pytest.raises(RollbackError, match="no previous stable approved model"):
        select_previous_stable_model(
            [
                {
                    "model_name": "isolation_forest",
                    "model_version": "v002",
                    "artifact_path": "model.joblib",
                    "status": "production",
                    "approved_for_prod": True,
                }
            ],
            current_model_version="v002",
            model_name="isolation_forest",
        )


def test_rollback_active_model_updates_active_pointer_and_writes_audit_event(
    tmp_path: Path,
) -> None:
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    rollback_events_path = tmp_path / "logs" / "alerts" / "rollback_events.jsonl"

    _write_active_model(active_model_path, version="v002")

    event = rollback_active_model(
        _registry_records(),
        active_model_path=active_model_path,
        rollback_events_path=rollback_events_path,
        rollback_reason="critical drift detected",
        triggered_by="operator",
    )

    updated_active_model = yaml.safe_load(
        active_model_path.read_text(encoding="utf-8")
    )

    assert event.from_model_version == "v002"
    assert event.to_model_version == "v001"
    assert event.rollback_reason == "critical drift detected"
    assert event.triggered_by == "operator"
    assert event.validation_status == "applied"
    assert event.dry_run is False

    assert updated_active_model["model_name"] == "isolation_forest"
    assert updated_active_model["active_model_version"] == "v001"
    assert updated_active_model["status"] == "production"
    assert updated_active_model["approved_for_prod"] is True
    assert updated_active_model["artifact_path"].endswith(
        "model_version=v001/model.joblib"
    )
    assert updated_active_model["dataset_snapshot_id"] == "snapshot_v001"
    assert updated_active_model["feature_schema_version"] == "feature_schema_v001"
    assert updated_active_model["rollback"]["rolled_back_from_model_version"] == "v002"
    assert updated_active_model["rollback"]["rolled_back_to_model_version"] == "v001"

    audit_records = read_rollback_events_jsonl(rollback_events_path)

    assert len(audit_records) == 1
    assert audit_records[0]["from_model_version"] == "v002"
    assert audit_records[0]["to_model_version"] == "v001"
    assert audit_records[0]["validation_status"] == "applied"


def test_rollback_dry_run_does_not_mutate_pointer_or_write_audit_event(
    tmp_path: Path,
) -> None:
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    rollback_events_path = tmp_path / "logs" / "alerts" / "rollback_events.jsonl"

    _write_active_model(active_model_path, version="v002")

    before_payload = active_model_path.read_text(encoding="utf-8")

    event = rollback_active_model(
        _registry_records(),
        active_model_path=active_model_path,
        rollback_events_path=rollback_events_path,
        rollback_reason="manual rollback validation",
        triggered_by="operator",
        dry_run=True,
    )

    after_payload = active_model_path.read_text(encoding="utf-8")

    assert event.from_model_version == "v002"
    assert event.to_model_version == "v001"
    assert event.validation_status == "dry_run_validated"
    assert event.dry_run is True
    assert before_payload == after_payload
    assert not rollback_events_path.exists()


def test_rollback_requires_reason_and_triggered_by(tmp_path: Path) -> None:
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    rollback_events_path = tmp_path / "logs" / "alerts" / "rollback_events.jsonl"

    _write_active_model(active_model_path, version="v002")

    with pytest.raises(RollbackError, match="rollback_reason is required"):
        rollback_active_model(
            _registry_records(),
            active_model_path=active_model_path,
            rollback_events_path=rollback_events_path,
            rollback_reason=" ",
            triggered_by="operator",
        )

    with pytest.raises(RollbackError, match="triggered_by is required"):
        rollback_active_model(
            _registry_records(),
            active_model_path=active_model_path,
            rollback_events_path=rollback_events_path,
            rollback_reason="critical drift detected",
            triggered_by=" ",
        )


def test_read_rollback_events_jsonl_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RollbackError, match="does not exist"):
        read_rollback_events_jsonl(tmp_path / "missing.jsonl")


def test_written_rollback_event_is_valid_jsonl(tmp_path: Path) -> None:
    active_model_path = tmp_path / "configs" / "active_model.yaml"
    rollback_events_path = tmp_path / "logs" / "alerts" / "rollback_events.jsonl"

    _write_active_model(active_model_path, version="v002")

    rollback_active_model(
        _registry_records(),
        active_model_path=active_model_path,
        rollback_events_path=rollback_events_path,
        rollback_reason="latency budget breach",
        triggered_by="operator",
    )

    lines = rollback_events_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    payload = json.loads(lines[0])

    assert payload["rollback_reason"] == "latency budget breach"
    assert payload["model_name"] == "isolation_forest"
    assert payload["rollback_id"].startswith("rollback_")
