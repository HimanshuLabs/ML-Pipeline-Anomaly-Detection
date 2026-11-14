"""Tests for rollback controls."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
import yaml

from anomaly_detection.registry import (
    find_model_entry,
    load_model_registry,
    promote_model,
    promote_model_version,
    register_model,
    register_model_from_artifacts,
)
from anomaly_detection.rollback import (
    RollbackError,
    RollbackServingError,
    read_rollback_events_jsonl,
    rollback_active_model,
    select_previous_stable_model,
)

MODEL_NAME = "isolation_forest"
FEATURES = ["feature_a", "feature_b", "feature_c"]


def _fake_artifact(root: Path, version: str, feature_names: list[str]) -> Path:
    """Write a loadable joblib artifact and its metadata.json."""

    artifact_dir = root / "artifacts" / f"model_version={version}"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "model.joblib"
    joblib.dump({"fake_model": version}, artifact_path)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_version": version,
                "feature_names": feature_names,
                "training_snapshot_metadata": {
                    "snapshot_id": f"snapshot_{version}",
                    "source_projects": [f"project_{version}"],
                    "source_tables": [f"table_{version}"],
                    "snapshot_type": "unit_test",
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _fake_registry(
    tmp_path: Path,
    versions: list[tuple[str, list[str]]],
) -> tuple[Path, Path]:
    """Register and promote versions in order; the last one ends up active."""

    registry_path = tmp_path / "registry" / "model_registry.json"
    active_model_path = tmp_path / "configs" / "active_model.yaml"

    for version, feature_names in versions:
        register_model(
            model_name=MODEL_NAME,
            model_version=version,
            artifact_path=str(_fake_artifact(tmp_path, version, feature_names)),
            dataset_snapshot_id=f"snapshot_{version}",
            feature_schema_version="feature_schema_v001",
            training_timestamp="2026-06-10T10:00:00+00:00",
            registry_path=registry_path,
        )

    for version, _feature_names in versions:
        promote_model_version(
            model_name=MODEL_NAME,
            model_version=version,
            registry_path=registry_path,
            active_model_path=active_model_path,
        )

    return registry_path, active_model_path


def _rollback(
    registry_path: Path,
    active_model_path: Path,
    rollback_events_path: Path,
    *,
    dry_run: bool = False,
    rollback_reason: str = "critical drift detected",
    triggered_by: str = "operator",
):
    return rollback_active_model(
        registry_path=registry_path,
        active_model_path=active_model_path,
        rollback_events_path=rollback_events_path,
        rollback_reason=rollback_reason,
        triggered_by=triggered_by,
        dry_run=dry_run,
    )


def _trained_registry(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    versions: list[str],
) -> tuple[Path, Path]:
    registry_path = tmp_path / "registry" / "model_registry.json"
    active_model_path = tmp_path / "configs" / "active_model.yaml"

    for version in versions:
        register_model_from_artifacts(
            metadata_path=trained_metadata_paths[version],
            registry_path=registry_path,
        )
    for version in versions:
        promote_model_version(
            model_name=MODEL_NAME,
            model_version=version,
            registry_path=registry_path,
            active_model_path=active_model_path,
        )

    return registry_path, active_model_path


# --- target selection -------------------------------------------------------


def test_select_previous_stable_model_excludes_current_and_unapproved_models(
    tmp_path: Path,
) -> None:
    registry_path, _active_model_path = _fake_registry(
        tmp_path,
        [("v001", FEATURES), ("v002", FEATURES)],
    )
    register_model(
        model_name=MODEL_NAME,
        model_version="v003",
        artifact_path=str(_fake_artifact(tmp_path, "v003", FEATURES)),
        dataset_snapshot_id="snapshot_v003",
        feature_schema_version="feature_schema_v001",
        training_timestamp="2026-06-10T10:00:00+00:00",
        registry_path=registry_path,
    )

    target = select_previous_stable_model(
        load_model_registry(registry_path),
        current_model_version="v002",
        model_name=MODEL_NAME,
    )

    assert target.model_name == MODEL_NAME
    assert target.model_version == "v001"
    assert target.status == "production"
    assert target.approved_for_prod is True
    assert target.selected_from_status == "archived"
    assert target.artifact_path.endswith("model_version=v001/model.joblib")


def test_select_previous_stable_model_rejects_missing_target(tmp_path: Path) -> None:
    registry_path, _active_model_path = _fake_registry(tmp_path, [("v002", FEATURES)])

    with pytest.raises(RollbackError, match="No compatible rollback target"):
        select_previous_stable_model(
            load_model_registry(registry_path),
            current_model_version="v002",
            model_name=MODEL_NAME,
        )


def _delete_artifact(tmp_path: Path, registry_path: Path, active_model_path: Path) -> None:
    (tmp_path / "artifacts" / "model_version=v001" / "model.joblib").unlink()


def _corrupt_artifact(tmp_path: Path, registry_path: Path, active_model_path: Path) -> None:
    (tmp_path / "artifacts" / "model_version=v001" / "model.joblib").write_text(
        "this is not a joblib file",
        encoding="utf-8",
    )


def _set_status(status: str):
    def _apply(tmp_path: Path, registry_path: Path, active_model_path: Path) -> None:
        promote_model(
            model_name=MODEL_NAME,
            model_version="v001",
            target_status=status,
            registry_path=registry_path,
            active_model_config_path=active_model_path,
        )

    return _apply


def _no_change(tmp_path: Path, registry_path: Path, active_model_path: Path) -> None:
    return None


@pytest.mark.parametrize(
    ("target_features", "mutate", "expected_reason"),
    [
        pytest.param(FEATURES, _delete_artifact, "does not exist", id="missing_artifact"),
        pytest.param(FEATURES, _corrupt_artifact, "does not load", id="corrupt_artifact"),
        pytest.param(
            ["feature_a", "feature_b", "feature_x"],
            _no_change,
            "feature_names differ",
            id="different_feature_names",
        ),
        pytest.param(
            list(reversed(FEATURES)),
            _no_change,
            "feature_names differ",
            id="same_names_different_order",
        ),
        pytest.param(FEATURES, _set_status("retired"), "status 'retired'", id="retired"),
        pytest.param(
            FEATURES,
            _set_status("rolled_back"),
            "status 'rolled_back'",
            id="rolled_back",
        ),
    ],
)
def test_incompatible_rollback_target_is_not_eligible(
    tmp_path: Path,
    target_features: list[str],
    mutate,
    expected_reason: str,
) -> None:
    registry_path, active_model_path = _fake_registry(
        tmp_path,
        [("v001", target_features), ("v002", FEATURES)],
    )
    mutate(tmp_path, registry_path, active_model_path)

    with pytest.raises(RollbackError, match="No compatible rollback target") as error:
        select_previous_stable_model(
            load_model_registry(registry_path),
            current_model_version="v002",
            model_name=MODEL_NAME,
        )

    assert "v001" in str(error.value)
    assert expected_reason in str(error.value)


def test_incompatible_target_makes_dry_run_fail_too(tmp_path: Path) -> None:
    registry_path, active_model_path = _fake_registry(
        tmp_path,
        [("v001", list(reversed(FEATURES))), ("v002", FEATURES)],
    )

    with pytest.raises(RollbackError, match="No compatible rollback target"):
        _rollback(
            registry_path,
            active_model_path,
            tmp_path / "rollback_events.jsonl",
            dry_run=True,
        )


# --- applying a rollback -----------------------------------------------------


def test_dry_run_validates_compatible_target_without_writing(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    registry_path, active_model_path = _trained_registry(
        tmp_path,
        trained_metadata_paths,
        ["v101", "v102"],
    )
    rollback_events_path = tmp_path / "logs" / "rollback_events.jsonl"
    registry_before = registry_path.read_bytes()
    pointer_before = active_model_path.read_bytes()

    event = _rollback(registry_path, active_model_path, rollback_events_path, dry_run=True)

    assert event.from_model_version == "v102"
    assert event.to_model_version == "v101"
    assert event.validation_status == "dry_run_validated"
    assert event.dry_run is True
    assert registry_path.read_bytes() == registry_before
    assert active_model_path.read_bytes() == pointer_before
    assert not rollback_events_path.exists()


def test_apply_rollback_updates_registry_and_builds_pointer_from_target(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    registry_path, active_model_path = _trained_registry(
        tmp_path,
        trained_metadata_paths,
        ["v101", "v102"],
    )
    rollback_events_path = tmp_path / "logs" / "rollback_events.jsonl"
    target_metadata = json.loads(trained_metadata_paths["v101"].read_text(encoding="utf-8"))

    event = _rollback(registry_path, active_model_path, rollback_events_path)

    rolled_back = find_model_entry(
        model_name=MODEL_NAME, model_version="v102", registry_path=registry_path
    )
    restored = find_model_entry(
        model_name=MODEL_NAME, model_version="v101", registry_path=registry_path
    )
    pointer = yaml.safe_load(active_model_path.read_text(encoding="utf-8"))

    assert event.validation_status == "applied"
    assert rolled_back.status == "rolled_back"
    assert restored.status == "production"
    assert restored.approved_for_prod is True
    assert restored.promoted_at_utc is not None

    assert pointer["active_model_version"] == "v101"
    assert pointer["artifact_path"] == restored.artifact_path
    assert pointer["dataset_snapshot_id"] == restored.snapshot_id
    assert pointer["dataset_snapshot_id"] != rolled_back.snapshot_id
    assert pointer["feature_count"] == len(target_metadata["feature_names"])
    assert pointer["source_tables"] == ["source_table_v101"]
    assert pointer["source_projects"] == []

    audit_records = read_rollback_events_jsonl(rollback_events_path)
    assert len(audit_records) == 1
    assert audit_records[0]["from_model_version"] == "v102"
    assert audit_records[0]["to_model_version"] == "v101"
    assert audit_records[0]["validation_status"] == "applied"


def test_second_rollback_does_not_select_rolled_back_model(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    registry_path, active_model_path = _trained_registry(
        tmp_path,
        trained_metadata_paths,
        ["v101", "v102", "v103"],
    )
    rollback_events_path = tmp_path / "logs" / "rollback_events.jsonl"

    first = _rollback(registry_path, active_model_path, rollback_events_path)
    second = _rollback(registry_path, active_model_path, rollback_events_path)

    assert (first.from_model_version, first.to_model_version) == ("v103", "v102")
    assert (second.from_model_version, second.to_model_version) == ("v102", "v101")
    assert [
        (entry.model_version, entry.status) for entry in load_model_registry(registry_path)
    ] == [("v101", "production"), ("v102", "rolled_back"), ("v103", "rolled_back")]


def test_pointer_write_failure_restores_registry(
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import rollback

    registry_path, active_model_path = _trained_registry(
        tmp_path,
        trained_metadata_paths,
        ["v101", "v102"],
    )
    rollback_events_path = tmp_path / "logs" / "rollback_events.jsonl"
    registry_before = registry_path.read_bytes()
    pointer_before = active_model_path.read_bytes()

    def _failing_pointer_write(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rollback, "write_active_model_pointer", _failing_pointer_write)

    with pytest.raises(RollbackServingError, match="disk full"):
        _rollback(registry_path, active_model_path, rollback_events_path)

    assert registry_path.read_bytes() == registry_before
    assert active_model_path.read_bytes() == pointer_before
    assert not rollback_events_path.exists()


def test_rollback_requires_reason_and_triggered_by(tmp_path: Path) -> None:
    registry_path, active_model_path = _fake_registry(
        tmp_path,
        [("v001", FEATURES), ("v002", FEATURES)],
    )
    rollback_events_path = tmp_path / "logs" / "rollback_events.jsonl"

    with pytest.raises(RollbackError, match="rollback_reason is required"):
        _rollback(registry_path, active_model_path, rollback_events_path, rollback_reason=" ")

    with pytest.raises(RollbackError, match="triggered_by is required"):
        _rollback(registry_path, active_model_path, rollback_events_path, triggered_by=" ")


def test_read_rollback_events_jsonl_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RollbackError, match="does not exist"):
        read_rollback_events_jsonl(tmp_path / "missing.jsonl")


def test_written_rollback_event_is_valid_jsonl(tmp_path: Path) -> None:
    registry_path, active_model_path = _fake_registry(
        tmp_path,
        [("v001", FEATURES), ("v002", FEATURES)],
    )
    rollback_events_path = tmp_path / "logs" / "rollback_events.jsonl"

    _rollback(
        registry_path,
        active_model_path,
        rollback_events_path,
        rollback_reason="latency budget breach",
    )

    lines = rollback_events_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    payload = json.loads(lines[0])

    assert payload["rollback_reason"] == "latency budget breach"
    assert payload["model_name"] == MODEL_NAME
    assert payload["rollback_id"].startswith("rollback_")
