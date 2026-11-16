"""Repo-relative bundle paths for new versions; registration of old bundles from anywhere.

v001/v002 store host paths (/home/...), v003/v004 container paths (/app/...). They are
never edited: registration resolves each stored path, falling back to the file of the
same name next to the metadata.json being read. New training runs store repo-relative
paths, and new rollback events store repo-relative pointer and event-log paths.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anomaly_detection import registry
from tests.conftest import REPO_ROOT, _sample_feature_frame
from tests.test_database import (  # noqa: F401 - pytest fixtures
    database_mode,
    initialized_database,
    postgres_engine,
)

MODELS = REPO_ROOT / "artifacts" / "models" / "isolation_forest"
PATH_FIELDS = ("artifact_dir", "model_path", "metadata_path", "feature_schema_path",
               "baseline_stats_path")


def _real_bundle_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(MODELS.glob("model_version=v00[1-5]/*"))
    }


@pytest.fixture(autouse=True)
def real_bundles_unchanged():
    before = _real_bundle_hashes()
    yield
    assert _real_bundle_hashes() == before


def _metadata(bundle: Path) -> dict:
    return json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))


def _train(artifact_root: Path, work: Path, version: str):
    from anomaly_detection.training import train_isolation_forest_from_snapshot
    from anomaly_detection.training_snapshot import TrainingSnapshotConfig, create_training_snapshot

    snapshot = create_training_snapshot(
        _sample_feature_frame(row_count=48),
        snapshot_config=TrainingSnapshotConfig(output_root=work / "snapshots"),
        source_tables=["source_table_paths"],
        created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )
    return train_isolation_forest_from_snapshot(
        features_path=snapshot.features_path,
        snapshot_metadata_path=snapshot.metadata_path,
        model_version=version,
        artifact_root_override=artifact_root,
        registry_path=work / "training_label_registry.json",
    )


@pytest.fixture()
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in repository root, so tests never write into the real one."""

    root = (tmp_path / "repo").resolve()
    root.mkdir()
    monkeypatch.setattr(registry, "PROJECT_ROOT", root)
    return root


# --- training ----------------------------------------------------------------------


def test_new_training_run_stores_repo_relative_paths(project_root: Path, tmp_path: Path) -> None:
    artifact = _train(project_root / "artifacts" / "models", tmp_path / "work", "v501")

    bundle = "artifacts/models/isolation_forest/model_version=v501"
    expected = {
        "artifact_dir": bundle,
        "model_path": f"{bundle}/model.joblib",
        "metadata_path": f"{bundle}/metadata.json",
        "feature_schema_path": f"{bundle}/feature_schema.json",
        "baseline_stats_path": f"{bundle}/baseline_stats.json",
    }
    stored = _metadata(project_root / bundle)
    assert {field: stored[field] for field in PATH_FIELDS} == expected
    assert {field: getattr(artifact, field) for field in PATH_FIELDS} == expected

    entry = registry.register_model_from_artifacts(
        metadata_path=project_root / bundle / "metadata.json",
        registry_path=tmp_path / "registry.json",
    )
    assert entry.artifact_path == f"{bundle}/model.joblib"


def test_bundle_outside_the_repo_keeps_absolute_paths(project_root: Path, tmp_path: Path) -> None:
    outside = (tmp_path / "outside" / "models").resolve()
    _train(outside, tmp_path / "work", "v502")

    bundle = outside / "isolation_forest" / "model_version=v502"
    stored = _metadata(bundle)
    assert stored["model_path"] == str(bundle / "model.joblib")
    assert all(Path(stored[field]).is_absolute() for field in PATH_FIELDS)


# --- registration of bundles with stored paths from another root -------------------


def _copy_bundle(version: str, destination: Path) -> Path:
    bundle = destination / f"model_version={version}"
    shutil.copytree(MODELS / f"model_version={version}", bundle)
    return bundle


@pytest.mark.parametrize(
    ("version", "stored_prefix"),
    [
        ("v001", "/home/"),  # written on the host
        ("v004", "/app/"),  # written in the tools container
    ],
)
def test_copy_of_an_old_bundle_registers_from_its_own_directory(
    tmp_path: Path, version: str, stored_prefix: str
) -> None:
    from anomaly_detection.registry import load_verified_feature_spec

    copy = _copy_bundle(version, tmp_path / "elsewhere")
    assert _metadata(copy)["model_path"].startswith(stored_prefix)  # untouched stored paths

    entry = registry.register_model_from_artifacts(
        metadata_path=copy / "metadata.json", registry_path=tmp_path / "registry.json"
    )

    assert entry.artifact_path == str(copy / "model.joblib")
    assert entry.artifact_dir == str(copy)
    assert entry.metadata_path == str(copy / "metadata.json")
    assert entry.feature_schema_path == str(copy / "feature_schema.json")
    assert entry.baseline_stats_path == str(copy / "baseline_stats.json")
    assert load_verified_feature_spec(entry).feature_names


def test_relative_path_bundle_registers_inside_the_repo_and_after_a_move(
    project_root: Path, tmp_path: Path
) -> None:
    bundle_dir = "artifacts/models/isolation_forest/model_version=v001"
    bundle = _copy_bundle("v001", project_root / "artifacts/models/isolation_forest")
    metadata = _metadata(bundle)  # the copy only; the real v001 is never edited
    for field in PATH_FIELDS:
        metadata[field] = bundle_dir + ("" if field == "artifact_dir" else
                                        "/" + Path(metadata[field]).name)
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    inside = registry.register_model_from_artifacts(
        metadata_path=bundle / "metadata.json", registry_path=tmp_path / "inside.json"
    )
    assert inside.artifact_path == f"{bundle_dir}/model.joblib"
    assert inside.artifact_dir == bundle_dir

    moved = tmp_path / "moved" / "model_version=v001"
    shutil.move(str(bundle), moved)
    after_move = registry.register_model_from_artifacts(
        metadata_path=moved / "metadata.json", registry_path=tmp_path / "moved.json"
    )
    assert after_move.artifact_path == str(moved / "model.joblib")
    assert after_move.baseline_stats_path == str(moved / "baseline_stats.json")


# --- rollback events ---------------------------------------------------------------


def _two_versions(registry_path: Path, active_model_path: Path, metadata: dict[str, Path]):
    for version in ("v101", "v102"):
        registry.register_model_from_artifacts(
            metadata_path=metadata[version], registry_path=registry_path
        )
    for version in ("v101", "v102"):
        registry.promote_model_version(
            model_name="isolation_forest",
            model_version=version,
            registry_path=registry_path,
            active_model_path=active_model_path,
        )


def test_new_rollback_event_stores_repo_relative_paths_in_jsonl(
    project_root: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.rollback import read_rollback_events_jsonl, rollback_active_model

    registry_path = project_root / "artifacts/models/_registry/model_registry.json"
    active_model_path = project_root / "configs" / "active_model.yaml"
    events_path = project_root / "logs" / "alerts" / "rollback_events.jsonl"
    _two_versions(registry_path, active_model_path, trained_metadata_paths)

    event = rollback_active_model(
        registry_path=registry_path,
        active_model_path=active_model_path,
        rollback_events_path=events_path,
        rollback_reason="pytest",
        triggered_by="pytest",
    )

    assert event.active_model_path == "configs/active_model.yaml"
    assert event.rollback_event_path == "logs/alerts/rollback_events.jsonl"
    [stored] = read_rollback_events_jsonl(events_path)
    assert stored["active_model_path"] == "configs/active_model.yaml"
    assert stored["rollback_event_path"] == "logs/alerts/rollback_events.jsonl"


def test_admin_rollback_response_keeps_an_absolute_event_path(
    isolate_repo_state: Path,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    admin_headers: dict[str, str],
) -> None:
    from fastapi.testclient import TestClient

    from anomaly_detection.rollback import read_rollback_events_jsonl
    from api.main import app, state
    from tests.test_api import _apply_rollback, _install_two_model_registry

    mirror = isolate_repo_state.resolve()
    monkeypatch.setattr(registry, "PROJECT_ROOT", mirror)
    monkeypatch.setattr(state, "inference_service", None)
    _install_two_model_registry(isolate_repo_state, tmp_path, trained_metadata_paths)

    with TestClient(app) as client:
        response = _apply_rollback(client, admin_headers)

    events_path = mirror / "logs" / "alerts" / "rollback_events.jsonl"
    assert response.status_code == 200
    assert response.json()["rollback_event_path"] == str(events_path)  # API value unchanged
    [stored] = read_rollback_events_jsonl(events_path)
    assert stored["rollback_event_path"] == "logs/alerts/rollback_events.jsonl"


@pytest.mark.postgres
def test_new_rollback_event_stores_repo_relative_paths_in_postgres(
    initialized_database,  # noqa: F811 - pytest fixture
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import text

    from anomaly_detection.rollback import rollback_active_model

    for version in ("v101", "v102"):
        registry.register_model_from_artifacts(metadata_path=trained_metadata_paths[version])
    for version in ("v101", "v102"):
        registry.promote_model_version(model_name="isolation_forest", model_version=version)

    # The mirror stands in for the repository only for the rollback being tested.
    monkeypatch.setattr(registry, "PROJECT_ROOT", isolate_repo_state.resolve())
    event = rollback_active_model(rollback_reason="pytest", triggered_by="pytest")

    with initialized_database.connect() as connection:
        row = connection.execute(
            text(
                "SELECT active_model_path, rollback_event_path FROM audit.rollback_events "
                "WHERE rollback_id = :id"
            ),
            {"id": event.rollback_id},
        ).one()
    assert tuple(row) == ("configs/active_model.yaml", "logs/alerts/rollback_events.jsonl")
