"""PROJECT4_REGISTRY_FILE_MIRROR: the JSON/YAML mirror of the database registry.

In database mode the registry and pointer live in Postgres and are mirrored to
artifacts/models/_registry/model_registry.json and configs/active_model.yaml after
every committed change. With the switch off (a read-only root filesystem in k8s)
nothing is written to those files, and a failed rollback does not try to restore
them either, so its original error surfaces. In file mode the files are the store
and the switch does not apply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anomaly_detection import registry
from tests.conftest import REAL_REGISTRY_PATH, REPO_ROOT
from tests.test_database import (  # noqa: F401 - pytest fixtures
    _register_trained,
    database_mode,
    initialized_database,
    postgres_engine,
)

MIRROR_ENV = "PROJECT4_REGISTRY_FILE_MIRROR"


def _mirror_files(isolate_repo_state: Path) -> dict[str, bytes]:
    paths = {
        "registry": isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT),
        "pointer": isolate_repo_state / "configs" / "active_model.yaml",
    }
    return {name: path.read_bytes() for name, path in paths.items()}


def _mirror_failures() -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value("registry_mirror_write_failures_total") or 0.0


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (None, True), ("", True), ("true", True), ("TRUE", True), ("1", True), ("yes", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_file_mirror_switch_defaults_on(
    monkeypatch: pytest.MonkeyPatch, value: str | None, enabled: bool
) -> None:
    if value is None:
        monkeypatch.delenv(MIRROR_ENV, raising=False)
    else:
        monkeypatch.setenv(MIRROR_ENV, value)

    assert registry.file_mirror_enabled() is enabled


@pytest.mark.postgres
def test_mirror_off_leaves_the_files_untouched_through_promote_retire_and_rollback(
    initialized_database,  # noqa: F811 - pytest fixture
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from anomaly_detection.rollback import rollback_active_model

    monkeypatch.setenv(MIRROR_ENV, "false")
    files_before = _mirror_files(isolate_repo_state)
    failures_before = _mirror_failures()
    caplog.set_level("INFO", logger="anomaly_detection.registry")

    _register_trained(trained_metadata_paths, ["v101", "v102", "v103"])
    registry.promote_model_version(model_name="isolation_forest", model_version="v101")
    registry.promote_model_version(model_name="isolation_forest", model_version="v102")
    registry.retire_model_version(
        model_name="isolation_forest", model_version="v103", notes="pytest"
    )
    rollback_active_model(rollback_reason="pytest", triggered_by="pytest")

    assert _mirror_files(isolate_repo_state) == files_before
    assert _mirror_failures() == failures_before
    # The database moved on: v102 was rolled back to v101.
    state = registry.load_registry_state(model_name="isolation_forest")
    assert state.active_model_version == "v101"
    skipped = [r for r in caplog.records if "file mirror is off" in r.getMessage()]
    assert skipped and all(r.levelname == "INFO" for r in skipped)


@pytest.mark.postgres
@pytest.mark.parametrize("value", [None, "true"])
def test_mirror_on_by_default_still_writes_the_files(
    initialized_database,  # noqa: F811 - pytest fixture
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv(MIRROR_ENV, raising=False)
    else:
        monkeypatch.setenv(MIRROR_ENV, value)
    files_before = _mirror_files(isolate_repo_state)

    _register_trained(trained_metadata_paths, ["v101"])
    registry.promote_model_version(model_name="isolation_forest", model_version="v101")

    files_after = _mirror_files(isolate_repo_state)
    assert files_after["registry"] != files_before["registry"]
    assert files_after["pointer"] != files_before["pointer"]
    assert b"v101" in files_after["pointer"]


@pytest.mark.postgres
def test_failed_rollback_with_mirror_off_surfaces_the_original_error(
    initialized_database,  # noqa: F811 - pytest fixture
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.rollback import RollbackServingError, rollback_active_model

    _register_trained(trained_metadata_paths, ["v101", "v102"])
    registry.promote_model_version(model_name="isolation_forest", model_version="v101")
    registry.promote_model_version(model_name="isolation_forest", model_version="v102")
    monkeypatch.setenv(MIRROR_ENV, "false")

    file_writes: list[Path] = []
    real_write = registry._write_bytes_atomic

    def _read_only_root(path, data) -> None:
        # /tmp stays writable (an emptyDir in k8s): the rollback's candidate and
        # committed pointers live in project4_* temporary directories there. Every
        # other write (the mirror files, a restore of them) hits the read-only root.
        if any(part.startswith("project4_") for part in Path(path).parts):
            return real_write(path, data)
        file_writes.append(Path(path))
        raise OSError(30, "Read-only file system", str(path))

    monkeypatch.setattr(registry, "_write_bytes_atomic", _read_only_root)

    def _fails_on_the_committed_pointer(pointer_path: Path) -> None:
        if "project4_rollback_committed_" in str(pointer_path):
            raise RuntimeError("simulated load failure of the committed pointer")

    with pytest.raises(RollbackServingError, match="restored"):
        rollback_active_model(
            rollback_reason="pytest",
            triggered_by="pytest",
            verify_service=_fails_on_the_committed_pointer,
        )

    assert file_writes == []  # neither the mirror nor the restore touched the files
    state = registry.load_registry_state(model_name="isolation_forest")
    assert state.active_model_version == "v102"  # the database change was undone


def test_file_mode_ignores_the_switch(
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.registry import register_model_from_artifacts

    assert not registry.database_enabled()
    monkeypatch.setenv(MIRROR_ENV, "false")
    registry_file = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    before = registry_file.read_bytes()

    register_model_from_artifacts(metadata_path=trained_metadata_paths["v101"])

    assert registry_file.read_bytes() != before  # the files are the store
    assert b'"v101"' in registry_file.read_bytes()
