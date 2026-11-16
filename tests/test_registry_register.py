"""Stage 15a: `registry register` and `registry mirror`.

register adds a trained bundle to the registry as a candidate (after the serving
check), together with its snapshot and training dataset rows, in FK order; it is
idempotent and refuses a different bundle under a registered version. mirror
rewrites the JSON registry and the pointer YAML from the database, byte for byte
what promote and rollback write for the same state.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from anomaly_detection import database, registry
from tests.conftest import REAL_REGISTRY_PATH, REPO_ROOT
from tests.test_database import (  # noqa: F401 - pytest fixtures
    _count,
    _database_pointer,
    app_mode,
    database_mode,
    initialized_database,
    postgres_engine,
)

# Later than every timestamp in the fixture registry the mirror starts from (conftest):
# archiving a version "now" must not predate its promotion (model_registry_archive_time_chk).
FROZEN_NOW = "2100-01-01T00:00:00+00:00"


def _mirror_paths(isolate_repo_state: Path) -> dict[str, Path]:
    return {
        "registry": isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT),
        "pointer": isolate_repo_state / "configs" / "active_model.yaml",
    }


def _read(paths: dict[str, Path]) -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in paths.items()}


def _freeze_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    from anomaly_detection import rollback

    monkeypatch.setattr(registry, "_utc_now_iso", lambda: FROZEN_NOW)
    monkeypatch.setattr(rollback, "_utc_now", lambda: FROZEN_NOW)


def _dataset_id(metadata_path: Path) -> str:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_id = metadata["training_snapshot_metadata"]["training_dataset_id"]
    assert dataset_id
    return dataset_id


# --- file mode -----------------------------------------------------------------------------


def test_register_in_file_mode_adds_a_candidate_once(
    isolate_repo_state: Path,
    segmented_metadata_paths: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_file = _mirror_paths(isolate_repo_state)["registry"]
    metadata_path = segmented_metadata_paths["v201"]

    assert registry.main(["register", "--metadata-path", str(metadata_path)]) == 0
    entry = registry.find_model_entry(model_name="isolation_forest", model_version="v201")
    assert (entry.status, entry.approved_for_prod) == ("candidate", False)
    assert entry.training_dataset_id == _dataset_id(metadata_path)
    after_first = registry_file.read_bytes()
    assert "registered" in capsys.readouterr().out

    assert registry.main(["register", "--metadata-path", str(metadata_path)]) == 0
    assert registry_file.read_bytes() == after_first
    assert "nothing changed" in capsys.readouterr().out


def test_register_refuses_a_bundle_that_cannot_serve_and_writes_nothing(
    isolate_repo_state: Path,
    segmented_metadata_paths: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "copy" / "model_version=v201"
    shutil.copytree(segmented_metadata_paths["v201"].parent, bundle)
    (bundle / "model.joblib").write_bytes(b"not a model")
    registry_file = _mirror_paths(isolate_repo_state)["registry"]
    before = registry_file.read_bytes()

    assert registry.main(["register", "--metadata-path", str(bundle / "metadata.json")]) == 1

    assert "REFUSED" in capsys.readouterr().err
    assert registry_file.read_bytes() == before


def test_mirror_refuses_without_a_database(capsys: pytest.CaptureFixture[str]) -> None:
    assert not registry.database_enabled()

    assert registry.main(["mirror"]) == 1

    assert "database" in capsys.readouterr().err


# --- database mode -------------------------------------------------------------------------


@pytest.mark.postgres
def test_register_as_project4_app_inserts_snapshot_then_dataset_then_registry_row(
    app_mode,  # noqa: F811 - pytest fixture
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import text

    order: list[str] = []
    real_insert = database.insert_row_if_absent
    real_upsert = database.upsert_model_registry_row

    def _insert(connection, table, row):
        order.append(table)
        return real_insert(connection, table, row)

    def _upsert(connection, row):
        order.append(f"ml.model_registry:{row['model_version']}")
        return real_upsert(connection, row)

    monkeypatch.setattr(database, "insert_row_if_absent", _insert)
    monkeypatch.setattr(database, "upsert_model_registry_row", _upsert)
    with database.get_shared_engine().connect() as connection:
        assert connection.execute(text("SELECT current_user")).scalar_one() == "project4_app"
    pointer_before = _database_pointer(app_mode)
    metadata_path = segmented_metadata_paths["v201"]
    dataset_id = _dataset_id(metadata_path)

    assert registry.main(["register", "--metadata-path", str(metadata_path)]) == 0

    assert order == ["ml.feature_snapshots", "ml.training_datasets", "ml.model_registry:v201"]
    assert _count(
        app_mode,
        "SELECT COUNT(*) FROM ml.training_datasets WHERE training_dataset_id = :id",
        {"id": dataset_id},
    ) == 1
    assert _count(
        app_mode,
        "SELECT COUNT(*) FROM integration.model_metadata WHERE model_version = 'v201' "
        "AND status = 'candidate' AND NOT is_active AND training_dataset_id = :id",
        {"id": dataset_id},
    ) == 1
    assert _database_pointer(app_mode) == pointer_before


@pytest.mark.postgres
def test_register_is_idempotent_and_refuses_different_content(
    app_mode,  # noqa: F811 - pytest fixture
    segmented_metadata_paths: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = segmented_metadata_paths["v201"]
    assert registry.main(["register", "--metadata-path", str(metadata_path)]) == 0
    counts = {
        table: _count(app_mode, f"SELECT COUNT(*) FROM {table}")
        for table in ("ml.feature_snapshots", "ml.training_datasets", "ml.model_registry")
    }
    pointer = _database_pointer(app_mode)
    capsys.readouterr()

    assert registry.main(["register", "--metadata-path", str(metadata_path)]) == 0
    assert "nothing changed" in capsys.readouterr().out

    bundle = tmp_path / "copy" / "model_version=v201"
    shutil.copytree(metadata_path.parent, bundle)
    changed = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    changed["baseline_anomaly_rate"] = 0.5
    (bundle / "metadata.json").write_text(json.dumps(changed), encoding="utf-8")

    assert registry.main(["register", "--metadata-path", str(bundle / "metadata.json")]) == 1
    assert "different content" in capsys.readouterr().err

    for table, count in counts.items():
        assert _count(app_mode, f"SELECT COUNT(*) FROM {table}") == count, table
    assert _database_pointer(app_mode) == pointer


@pytest.mark.postgres
def test_mirror_writes_exactly_what_promote_and_rollback_write(
    initialized_database,  # noqa: F811 - pytest fixture
    isolate_repo_state: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from anomaly_detection.rollback import rollback_active_model

    _freeze_clock(monkeypatch)
    paths = _mirror_paths(isolate_repo_state)
    for version in ("v201", "v202"):
        registry.register_model_from_artifacts(metadata_path=segmented_metadata_paths[version])

    registry.promote_model_version(model_name="isolation_forest", model_version="v201")
    registry.promote_model_version(
        model_name="isolation_forest", model_version="v202", notes="pytest promote"
    )
    after_promote = _read(paths)
    for path in paths.values():
        path.write_text("stale\n", encoding="utf-8")

    assert registry.main(["mirror"]) == 0
    assert _read(paths) == after_promote
    assert "v202" in capsys.readouterr().out

    rollback_active_model(rollback_reason="pytest", triggered_by="pytest")
    after_rollback = _read(paths)
    assert b"v201" in after_rollback["pointer"]
    for path in paths.values():
        path.write_text("stale\n", encoding="utf-8")

    assert registry.main(["mirror"]) == 0
    assert _read(paths) == after_rollback


@pytest.mark.postgres
def test_mirror_reads_only_the_database(
    initialized_database,  # noqa: F811 - pytest fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import event

    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement.lstrip().split()[0].upper())

    engines = []
    real_get_engine = database.get_engine

    def _get_engine(*args, **kwargs):
        engine = real_get_engine(*args, **kwargs)
        event.listen(engine, "before_cursor_execute", _record)
        engines.append(engine)
        return engine

    monkeypatch.setattr(database, "get_engine", _get_engine)

    assert registry.main(["mirror"]) == 0

    assert engines and statements
    assert set(statements) <= {"SELECT", "SHOW"}, statements



def test_register_writes_only_the_given_registry_and_pointer_paths(
    isolate_repo_state: Path,
    segmented_metadata_paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    monkeypatch.delenv("PROJECT4_REGISTRY_FILE_MIRROR", raising=False)
    default_pointer = _mirror_paths(isolate_repo_state)["pointer"]
    default_registry = _mirror_paths(isolate_repo_state)["registry"]
    before = {path: path.read_bytes() for path in (default_pointer, default_registry)}
    registry_path = tmp_path / "own" / "model_registry.json"
    registry_path.parent.mkdir()
    registry_path.write_bytes(default_registry.read_bytes())
    pointer_path = tmp_path / "own" / "active_model.yaml"
    pointer_path.write_bytes(default_pointer.read_bytes())
    # The default pointer path is the mirrored configs/active_model.yaml; make any write
    # to it observable.
    watched: list[Path] = []
    real_write = registry._write_bytes_atomic

    def _watch(path, data) -> None:
        watched.append(Path(path))
        real_write(path, data)

    monkeypatch.setattr(registry, "_write_bytes_atomic", _watch)

    entry, written = registry.register_bundle(
        metadata_path=segmented_metadata_paths["v201"],
        registry_path=registry_path,
        active_model_path=pointer_path,
    )

    assert written and entry.model_version == "v201"
    assert default_pointer not in watched and default_registry not in watched
    assert {path: path.read_bytes() for path in before} == before
    assert yaml.safe_load(pointer_path.read_text(encoding="utf-8"))["active_model_version"]
    assert "v201" in registry_path.read_text(encoding="utf-8")


@pytest.mark.postgres
def test_register_in_database_mode_mirrors_only_to_the_given_paths(
    initialized_database,  # noqa: F811 - pytest fixture
    isolate_repo_state: Path,
    segmented_metadata_paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJECT4_REGISTRY_FILE_MIRROR", "true")
    defaults = _mirror_paths(isolate_repo_state)
    before = _read(defaults)
    registry_path = tmp_path / "own" / "model_registry.json"
    pointer_path = tmp_path / "own" / "active_model.yaml"

    entry, written = registry.register_bundle(
        metadata_path=segmented_metadata_paths["v201"],
        registry_path=registry_path,
        active_model_path=pointer_path,
    )

    assert written and entry.model_version == "v201"
    assert _read(defaults) == before
    assert '"v201"' in registry_path.read_text(encoding="utf-8")
    # Registration never moves the pointer, so no pointer file is written at all.
    assert not pointer_path.exists()
