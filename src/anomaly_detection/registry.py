"""Local model registry for anomaly detection model versions.

This module is intentionally local-first for the portfolio build. PostgreSQL
tables already exist for the production metadata shape, but this checkpoint
uses JSON files so model versioning, promotion, and active-model pointers work
without a database dependency.

There is one stored registry schema (ModelRegistryEntry), one function that
writes the registry file (save_model_registry), and one function that writes
the active model pointer (write_active_model_pointer).
"""

from __future__ import annotations

import json
import logging
import os
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib
import yaml

from anomaly_detection.segmented_model import (
    FeatureSpec,
    SegmentedIsolationForest,
    feature_spec_from_metadata,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "_registry" / "model_registry.json"
)
DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


class ModelStatus(StrEnum):
    """Supported lifecycle statuses for registered model versions."""

    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"


VALID_MODEL_STATUSES = {status.value for status in ModelStatus}

# Statuses a version may be created with; anything else goes through promote_model.
REGISTRATION_STATUSES = {ModelStatus.CANDIDATE.value, ModelStatus.STAGING.value}

# Registry metric fields that register_model() accepts through its metrics dict.
REGISTRY_METRIC_FIELDS = (
    "baseline_anomaly_rate",
    "baseline_precision_proxy",
    "baseline_recall_proxy",
    "baseline_f1_proxy",
    "latency_p50_ms",
    "latency_p95_ms",
    "threshold_used",
)


class RegistryError(ValueError):
    """Raised when model registry operations are invalid."""


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Versioned model registry record."""

    model_id: str
    model_name: str
    model_version: str
    algorithm: str
    artifact_path: str
    artifact_dir: str
    metadata_path: str
    feature_schema_path: str
    baseline_stats_path: str
    snapshot_id: str | None
    training_dataset_id: str | None
    feature_schema_version: str
    training_started_at_utc: str | None
    training_finished_at_utc: str | None
    baseline_anomaly_rate: float | None
    baseline_precision_proxy: float | None
    baseline_recall_proxy: float | None
    baseline_f1_proxy: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    threshold_used: float | None
    status: str
    approved_for_prod: bool
    promoted_at_utc: str | None
    archived_at_utc: str | None
    created_at_utc: str
    notes: str | None

    @property
    def dataset_snapshot_id(self) -> str | None:
        """Snapshot id under the name used by prediction and pointer records."""
        return self.snapshot_id

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe registry entry."""
        return asdict(self)


# Former name of the lifecycle record; it is the same stored shape.
ModelRegistryRecord = ModelRegistryEntry

_ENTRY_FIELD_NAMES = {field.name for field in fields(ModelRegistryEntry)}


@dataclass(frozen=True)
class ActiveModelPointer:
    """Active model pointer used by API and future rollback logic."""

    model_name: str
    active_model_version: str | None
    artifact_path: str | None
    feature_schema_version: str
    status: str
    last_updated_at: str | None
    notes: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for YAML writing."""
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)

    if not json_path.exists():
        raise RegistryError(f"JSON file does not exist: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"JSON file is not valid JSON: {json_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RegistryError(f"JSON file must contain an object: {json_path}")

    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_bytes_atomic(path: str | Path, data: bytes) -> None:
    """Replace a file atomically: a failed write leaves the previous file intact."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        "wb",
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(data)
        except BaseException:
            tmp_file.close()
            tmp_path.unlink(missing_ok=True)
            raise

    # NamedTemporaryFile creates 0600; keep the target's mode, or 0644 for a new file.
    target_mode = stat.S_IMODE(target_path.stat().st_mode) if target_path.exists() else 0o644
    tmp_path.chmod(target_mode)
    tmp_path.replace(target_path)


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write YAML atomically: a failed write leaves the previous file intact."""
    _write_bytes_atomic(path, yaml.safe_dump(payload, sort_keys=False).encode("utf-8"))


def _read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise RegistryError(f"YAML file does not exist: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise RegistryError(f"YAML file is not valid YAML: {yaml_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RegistryError(f"YAML file must contain a mapping: {yaml_path}")

    return payload


def _status_value(status: ModelStatus | str) -> str:
    return status.value if isinstance(status, ModelStatus) else str(status)


def _validate_status(status: str) -> None:
    if status not in VALID_MODEL_STATUSES:
        raise RegistryError(
            f"Invalid model status: {status}. "
            f"Valid statuses: {sorted(VALID_MODEL_STATUSES)}"
        )


def _path_exists(path_value: str | None, *, label: str) -> None:
    if not path_value:
        raise RegistryError(f"{label} is missing from artifact metadata.")

    path = _resolve_registry_path(path_value)
    if not path.exists():
        raise RegistryError(f"{label} does not exist: {path}")


def _to_registry_path(path_value: str) -> str:
    """Store project-local paths as repo-relative paths.

    Absolute local paths are poison in source-controlled config because they only
    work on one machine. If an artifact lives inside this repository, store it
    relative to PROJECT_ROOT. External paths remain absolute. Relative input
    paths are taken as relative to PROJECT_ROOT, never to the working directory.
    """
    resolved_path = _resolve_registry_path(path_value).resolve()

    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def _resolve_registry_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_verified_feature_names(entry: ModelRegistryEntry) -> list[str]:
    """Check a version can serve: its artifact loads and metadata lists its features.

    Returns the ordered feature_names from the version's metadata.json (for a
    segmented model, the ordered union of its segments' features).
    Raises RegistryError with a short reason otherwise.
    """
    return list(load_verified_feature_spec(entry).feature_names)


def load_verified_feature_spec(entry: ModelRegistryEntry) -> FeatureSpec:
    """Like load_verified_feature_names, returning the full feature spec.

    Segmented models must have a valid segments spec in metadata.json that
    matches the loaded SegmentedIsolationForest; a single model's metadata must
    not describe a segmented artifact, and the reverse.
    """
    artifact_path = _resolve_registry_path(entry.artifact_path)
    if not artifact_path.exists():
        raise RegistryError(f"artifact does not exist: {entry.artifact_path}")

    try:
        model = joblib.load(artifact_path)
    except Exception as exc:  # noqa: BLE001 - any load failure makes the artifact unusable
        raise RegistryError(
            f"artifact does not load: {entry.artifact_path} ({type(exc).__name__})"
        ) from exc

    metadata_path = _resolve_registry_path(entry.metadata_path)
    if not metadata_path.exists():
        raise RegistryError(f"metadata.json does not exist: {entry.metadata_path}")

    metadata = _read_json(metadata_path)
    feature_names = metadata.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) for name in feature_names)
    ):
        raise RegistryError(
            f"metadata.json has no non-empty feature_names list: {entry.metadata_path}"
        )

    try:
        spec = feature_spec_from_metadata(metadata)
    except ValueError as exc:
        raise RegistryError(
            f"metadata.json has no valid feature spec: {entry.metadata_path} ({exc})"
        ) from exc

    is_segmented_artifact = isinstance(model, SegmentedIsolationForest)
    if spec.is_segmented != is_segmented_artifact:
        raise RegistryError(
            f"metadata.json model_kind {spec.model_kind!r} does not match the artifact "
            f"({type(model).__name__}): {entry.artifact_path}"
        )
    if is_segmented_artifact and model.feature_spec() != {
        entity_type: list(names) for entity_type, names in spec.segments
    }:
        raise RegistryError(
            f"artifact segments do not match metadata.json segments: {entry.artifact_path}"
        )

    return spec


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _stable_model_id(model_name: str, model_version: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{model_name}:{model_version}"))


def _extract_snapshot_id(training_snapshot_metadata: dict[str, Any]) -> str | None:
    value = training_snapshot_metadata.get("snapshot_id")
    return str(value) if value else None


def _extract_training_dataset_id(training_snapshot_metadata: dict[str, Any]) -> str | None:
    value = training_snapshot_metadata.get("training_dataset_id")
    return str(value) if value else None


def _build_entry_from_artifact_metadata(
    *,
    artifact_metadata: dict[str, Any],
    baseline_stats: dict[str, Any],
    status: str,
    approved_for_prod: bool,
    notes: str | None,
) -> ModelRegistryEntry:
    model_name = str(artifact_metadata["model_name"])
    model_version = str(artifact_metadata["model_version"])
    training_snapshot_metadata = artifact_metadata.get("training_snapshot_metadata") or {}
    latency_summary = baseline_stats.get("latency_summary") or {}

    return ModelRegistryEntry(
        model_id=_stable_model_id(model_name, model_version),
        model_name=model_name,
        model_version=model_version,
        algorithm=str(artifact_metadata["algorithm"]),
        artifact_path=_to_registry_path(str(artifact_metadata["model_path"])),
        artifact_dir=_to_registry_path(str(artifact_metadata["artifact_dir"])),
        metadata_path=_to_registry_path(str(artifact_metadata["metadata_path"])),
        feature_schema_path=_to_registry_path(str(artifact_metadata["feature_schema_path"])),
        baseline_stats_path=_to_registry_path(str(artifact_metadata["baseline_stats_path"])),
        snapshot_id=_extract_snapshot_id(training_snapshot_metadata),
        training_dataset_id=_extract_training_dataset_id(training_snapshot_metadata),
        feature_schema_version=str(artifact_metadata["feature_schema_version"]),
        training_started_at_utc=artifact_metadata.get("training_started_at_utc"),
        training_finished_at_utc=artifact_metadata.get("training_finished_at_utc"),
        baseline_anomaly_rate=_as_optional_float(
            artifact_metadata.get("baseline_anomaly_rate")
        ),
        baseline_precision_proxy=None,
        baseline_recall_proxy=None,
        baseline_f1_proxy=None,
        latency_p50_ms=_as_optional_float(latency_summary.get("latency_p50_ms")),
        latency_p95_ms=_as_optional_float(latency_summary.get("latency_p95_ms")),
        threshold_used=None,
        status=status,
        approved_for_prod=approved_for_prod,
        promoted_at_utc=_utc_now_iso() if status == "production" else None,
        archived_at_utc=None,
        created_at_utc=_utc_now_iso(),
        notes=notes,
    )


def load_model_registry(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[ModelRegistryEntry]:
    """Load local model registry entries.

    Missing registry file means no model versions have been registered yet.
    """
    path = Path(registry_path)

    if not path.exists():
        return []

    payload = _read_json(path)
    raw_entries = payload.get("models", [])

    if not isinstance(raw_entries, list):
        raise RegistryError("Registry payload field 'models' must be a list.")

    entries: list[ModelRegistryEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise RegistryError("Every registry entry must be an object.")

        unexpected_fields = sorted(set(raw_entry) - _ENTRY_FIELD_NAMES)
        missing_fields = sorted(_ENTRY_FIELD_NAMES - set(raw_entry))
        if unexpected_fields or missing_fields:
            raise RegistryError(
                "Registry entry does not match the registry schema: "
                f"model_version={raw_entry.get('model_version')!r}, "
                f"unexpected fields={unexpected_fields}, missing fields={missing_fields}"
            )

        entries.append(ModelRegistryEntry(**raw_entry))

    return entries


def save_model_registry(
    entries: list[ModelRegistryEntry],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Persist local model registry entries.

    This is the only function that writes registry content; restore_registry_state
    only puts back bytes captured earlier by snapshot_registry_state.
    """
    payload = {
        "registry_backend": "local_json",
        "updated_at_utc": _utc_now_iso(),
        "models": [entry.to_dict() for entry in entries],
    }
    _write_json(registry_path, payload)


@dataclass(frozen=True)
class RegistryStateSnapshot:
    """Exact bytes of the registry file and active pointer at one moment."""

    registry_path: Path
    registry_bytes: bytes | None
    active_model_path: Path
    pointer_bytes: bytes | None


def snapshot_registry_state(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
) -> RegistryStateSnapshot:
    """Capture the registry file and active pointer so a failed change can be undone."""

    def _read_bytes(path: Path) -> bytes | None:
        return path.read_bytes() if path.exists() else None

    resolved_registry_path = Path(registry_path)
    resolved_active_model_path = Path(active_model_path)
    return RegistryStateSnapshot(
        registry_path=resolved_registry_path,
        registry_bytes=_read_bytes(resolved_registry_path),
        active_model_path=resolved_active_model_path,
        pointer_bytes=_read_bytes(resolved_active_model_path),
    )


def restore_registry_state(snapshot: RegistryStateSnapshot) -> None:
    """Put the registry file and active pointer back to their snapshotted bytes.

    Skipped in database mode with the file mirror off: the files were never written,
    and restoring them on a read-only filesystem would hide the original error.
    """

    if not file_mirror_enabled() and database_enabled():
        return
    for path, data in (
        (snapshot.registry_path, snapshot.registry_bytes),
        (snapshot.active_model_path, snapshot.pointer_bytes),
    ):
        if data is None:
            path.unlink(missing_ok=True)
        else:
            _write_bytes_atomic(path, data)


# --- storage: Postgres when configured, files otherwise ----------------------------
#
# With PROJECT4_POSTGRES_* configured, Postgres is the source of truth for the
# registry and the active pointer; the JSON registry and YAML pointer are kept as
# a mirror after every committed change. Without a database, the files are the
# store, exactly as before.

POINTER_SOURCE_POSTGRES = "postgres"
POINTER_SOURCE_FILE = "file"


class RegistryStorageError(RegistryError):
    """The database could not be read or written; nothing was changed."""


@dataclass(frozen=True)
class RegistryState:
    """Registry entries and active pointer as read from the current store."""

    entries: list[ModelRegistryEntry]
    model_name: str | None
    active_model_version: str | None
    pointer_version: int | None
    source: str


@dataclass(frozen=True)
class RegistryCommit:
    """What a committed database change replaced, so it can be undone."""

    model_name: str
    previous_entries: list[ModelRegistryEntry]
    previous_pointer: dict[str, Any] | None
    pointer_version: int | None


def _database() -> Any:
    from anomaly_detection import database

    return database


def database_enabled() -> bool:
    """True when the PROJECT4_POSTGRES_* settings are configured."""

    return bool(_database().is_configured())


FILE_MIRROR_ENV = "PROJECT4_REGISTRY_FILE_MIRROR"


def file_mirror_enabled() -> bool:
    """PROJECT4_REGISTRY_FILE_MIRROR: on unless set to false/0/no/off.

    Only database mode consults it (the JSON/YAML mirror of the database registry);
    in file mode the files are the store. Off for a read-only root filesystem.
    """

    value = os.environ.get(FILE_MIRROR_ENV, "").strip().lower()
    return value not in {"false", "0", "no", "off"}


def default_model_name(model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> str:
    """The configured model name (configs/model_config.yaml model.name)."""

    path = Path(model_config_path)
    if path.exists():
        name = (_read_yaml(path).get("model") or {}).get("name")
        if name:
            return str(name)
    return "isolation_forest"


def _entry_key(entry: ModelRegistryEntry) -> tuple[str, str]:
    return entry.model_name, entry.model_version


def load_registry_state(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
    model_name: str | None = None,
) -> RegistryState:
    """Read registry entries and the active pointer from the configured store."""

    if not database_enabled():
        pointer_name: str | None = None
        pointer_version_name: str | None = None
        if Path(active_model_path).exists():
            pointer = read_active_model_pointer(active_model_path)
            pointer_name = pointer.model_name
            pointer_version_name = pointer.active_model_version
        return RegistryState(
            entries=load_model_registry(registry_path),
            model_name=model_name or pointer_name,
            active_model_version=pointer_version_name,
            pointer_version=None,
            source=POINTER_SOURCE_FILE,
        )

    database = _database()
    resolved_name = model_name or default_model_name()
    engine = database.get_engine()
    try:
        with engine.connect() as connection:
            entries = [
                ModelRegistryEntry(**values)
                for values in database.read_registry_entries(connection)
            ]
            pointer_row = database.read_active_pointer(connection, resolved_name)
    except Exception as exc:  # noqa: BLE001 - any database failure means "cannot read"
        raise RegistryStorageError(
            f"registry database could not be read: {type(exc).__name__}"
        ) from exc
    finally:
        engine.dispose()

    return RegistryState(
        entries=entries,
        model_name=resolved_name,
        active_model_version=pointer_row["model_version"] if pointer_row else None,
        pointer_version=int(pointer_row["pointer_version"]) if pointer_row else None,
        source=POINTER_SOURCE_POSTGRES,
    )


def ensure_database_initialized(state: RegistryState, model_name: str | None = None) -> None:
    """Refuse a database write until sync-from-files has seeded the registry and pointer.

    Without this, the first write would mirror an empty or partial database
    registry over the committed JSON/YAML files. File-backed state passes.
    """

    if state.source != POINTER_SOURCE_POSTGRES:
        return

    name = model_name or state.model_name
    has_entries = any(entry.model_name == name for entry in state.entries)
    if not has_entries or state.pointer_version is None:
        raise RegistryStorageError(
            f"Project 4 database is not initialized for {name}; "
            "run: python -m anomaly_detection.database sync-from-files"
        )


def snapshot_rows_for_entry(entry: ModelRegistryEntry) -> list[tuple[str, dict[str, Any]]]:
    """Database rows a registry entry's snapshot FK needs, from its metadata.json."""

    if not entry.snapshot_id:
        return []

    snapshot_metadata = _load_artifact_metadata(entry).get("training_snapshot_metadata") or {}
    if snapshot_metadata.get("snapshot_id") != entry.snapshot_id:
        raise RegistryError(
            f"Snapshot {entry.snapshot_id!r} of {entry.model_name} {entry.model_version} "
            "is not described by its metadata.json, so it cannot be stored in the database."
        )
    return _database().snapshot_rows_for_training_metadata(snapshot_metadata)


def _write_file_mirror(
    *,
    entries: list[ModelRegistryEntry],
    registry_path: str | Path,
    pointer_entry: ModelRegistryEntry | None,
    active_model_path: str | Path,
    notes: str | None,
) -> None:
    """Write the JSON/YAML mirror after a committed database change; never raises."""

    if not file_mirror_enabled():
        LOGGER.info(
            "Registry file mirror is off (%s=false); the database change is not mirrored",
            FILE_MIRROR_ENV,
        )
        return
    try:
        save_model_registry(entries, registry_path)
        if pointer_entry is not None:
            write_active_model_pointer(
                entry=pointer_entry,
                active_model_path=active_model_path,
                notes=notes,
            )
    except Exception:  # noqa: BLE001 - the database is authoritative; report and continue
        LOGGER.error(
            "Registry change committed to the database but the file mirror write failed",
            exc_info=True,
        )
        from anomaly_detection.metrics import record_registry_mirror_write_failure

        record_registry_mirror_write_failure()


def commit_registry_change(
    *,
    state: RegistryState,
    updated_entries: list[ModelRegistryEntry],
    registry_path: str | Path,
    active_model_path: str | Path,
    pointer_entry: ModelRegistryEntry | None = None,
    notes: str | None = None,
    updated_by: str = "registry",
    extra_rows: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> RegistryCommit | None:
    """Persist a registry change to the store of `state`.

    Files: the registry file, then the pointer (as before). Postgres: one
    transaction (snapshot rows, changed registry rows, pointer with
    pointer_version + 1, extra rows), then the file mirror. Returns what the
    database change replaced, or None for the file store.
    """

    if state.source == POINTER_SOURCE_FILE:
        save_model_registry(updated_entries, registry_path)
        if pointer_entry is not None:
            write_active_model_pointer(
                entry=pointer_entry,
                active_model_path=active_model_path,
                notes=notes,
            )
        return None

    model_name = (pointer_entry.model_name if pointer_entry else None) or state.model_name
    if model_name is None:
        raise RegistryError("no model name for the registry change")
    ensure_database_initialized(state, model_name)
    database = _database()

    before = {_entry_key(entry): entry for entry in state.entries}
    changed = [
        entry
        for entry in updated_entries
        if before.get(_entry_key(entry)) is None
        or before[_entry_key(entry)].to_dict() != entry.to_dict()
    ]
    snapshot_rows = [row for entry in changed for row in snapshot_rows_for_entry(entry)]

    engine = database.get_engine()
    try:
        with engine.begin() as connection:
            locked_version = database.lock_active_pointer(connection, model_name)
            if locked_version != state.pointer_version:
                raise RegistryStorageError(
                    f"the active pointer of {model_name} changed concurrently "
                    f"({state.pointer_version} -> {locked_version}); nothing was changed, retry"
                )
            previous_pointer = database.read_active_pointer(connection, model_name)

            for table, row in snapshot_rows:
                database.insert_row_if_absent(connection, table, row)
            for entry in changed:
                database.upsert_model_registry_row(connection, database.model_registry_row(entry))

            pointer_version = state.pointer_version
            if pointer_entry is not None:
                pointer_version, _changed = database.set_active_pointer(
                    connection,
                    model_name=pointer_entry.model_name,
                    model_version=pointer_entry.model_version,
                    artifact_path=pointer_entry.artifact_path,
                    updated_by=updated_by,
                    notes=notes,
                )

            for table, row in extra_rows:
                database.insert_row(connection, table, row)
    except RegistryStorageError:
        raise
    except Exception as exc:  # noqa: BLE001 - the transaction rolled back; nothing changed
        raise RegistryStorageError(
            f"registry database write failed; nothing was changed: {type(exc).__name__}"
        ) from exc
    finally:
        engine.dispose()

    _write_file_mirror(
        entries=updated_entries,
        registry_path=registry_path,
        pointer_entry=pointer_entry,
        active_model_path=active_model_path,
        notes=notes,
    )

    return RegistryCommit(
        model_name=model_name,
        previous_entries=[
            before[_entry_key(entry)] for entry in changed if _entry_key(entry) in before
        ],
        previous_pointer=previous_pointer,
        pointer_version=pointer_version,
    )


def restore_database_change(
    commit: RegistryCommit,
    *,
    updated_by: str,
    on_connection: Callable[[Any], None] | None = None,
) -> None:
    """Undo a committed database change in a new transaction.

    The replaced registry rows are written back and the pointer returns to its
    previous model (as a new pointer_version, so pollers notice).
    """

    database = _database()
    engine = database.get_engine()
    try:
        with engine.begin() as connection:
            for entry in commit.previous_entries:
                database.upsert_model_registry_row(connection, database.model_registry_row(entry))
            if commit.previous_pointer is not None:
                database.set_active_pointer(
                    connection,
                    model_name=commit.model_name,
                    model_version=commit.previous_pointer["model_version"],
                    artifact_path=commit.previous_pointer["artifact_path"],
                    updated_by=updated_by,
                    notes=commit.previous_pointer.get("notes"),
                )
            if on_connection is not None:
                on_connection(connection)
    finally:
        engine.dispose()


def materialize_active_pointer(
    target_path: str | Path,
    *,
    model_name: str | None = None,
) -> int:
    """Write the full pointer YAML for the database's active model; its pointer_version.

    Used to build a serving model from the shared pointer (API startup, the
    pointer poller, and post-commit rollback verification).
    """

    state = load_registry_state(model_name=model_name)
    if state.source != POINTER_SOURCE_POSTGRES:
        raise RegistryError("the shared pointer needs the database to be configured")
    if state.active_model_version is None or state.pointer_version is None:
        raise RegistryError(f"no active model pointer for {state.model_name} in the database")

    active_entry = next(
        (
            entry
            for entry in state.entries
            if entry.model_name == state.model_name
            and entry.model_version == state.active_model_version
        ),
        None,
    )
    if active_entry is None:
        raise RegistryError(
            f"active model {state.model_name} {state.active_model_version} is not in the registry"
        )

    write_active_model_pointer(entry=active_entry, active_model_path=target_path)
    return state.pointer_version


def read_shared_pointer_version(model_name: str | None = None) -> int | None:
    """Current pointer_version in the database (raises when it cannot be read)."""

    database = _database()
    engine = database.get_engine()
    try:
        return database.read_pointer_version(engine, model_name or default_model_name())
    finally:
        engine.dispose()


def find_model_entry(
    *,
    model_name: str,
    model_version: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ModelRegistryEntry:
    """Find one model registry entry by name and version."""
    for entry in load_model_registry(registry_path):
        if entry.model_name == model_name and entry.model_version == model_version:
            return entry

    raise RegistryError(f"Model version not registered: {model_name} {model_version}")


def _add_entry(
    entry: ModelRegistryEntry,
    *,
    registry_path: str | Path,
    replace_existing: bool,
) -> ModelRegistryEntry:
    state = load_registry_state(registry_path=registry_path, model_name=entry.model_name)
    ensure_database_initialized(state, entry.model_name)
    entries = list(state.entries)
    existing_index = next(
        (
            index
            for index, existing_entry in enumerate(entries)
            if existing_entry.model_name == entry.model_name
            and existing_entry.model_version == entry.model_version
        ),
        None,
    )

    if existing_index is not None and not replace_existing:
        raise RegistryError(
            "Model version is already registered: "
            f"{entry.model_name} {entry.model_version}"
        )

    if existing_index is None:
        entries.append(entry)
    else:
        entries[existing_index] = entry

    commit_registry_change(
        state=state,
        updated_entries=entries,
        registry_path=registry_path,
        active_model_path=DEFAULT_ACTIVE_MODEL_PATH,
        updated_by="registration",
    )

    return entry


def register_model_from_artifacts(
    *,
    metadata_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    status: str = "candidate",
    approved_for_prod: bool = False,
    notes: str | None = None,
    replace_existing: bool = True,
) -> ModelRegistryEntry:
    """Register a trained local model artifact.

    Args:
        metadata_path: Path to model artifact metadata.json.
        registry_path: Local JSON registry path.
        status: Lifecycle status for the registered version.
        approved_for_prod: Whether this model is approved for production use.
        notes: Optional human-readable registry note.
        replace_existing: Replace an existing same model/version record.

    Returns:
        Registered model entry.
    """
    _validate_status(status)

    if approved_for_prod and status != "production":
        raise RegistryError(
            "approved_for_prod=True is only valid when status='production'."
        )

    artifact_metadata = _read_json(metadata_path)

    required_metadata_fields = [
        "model_name",
        "model_version",
        "algorithm",
        "artifact_dir",
        "model_path",
        "metadata_path",
        "feature_schema_path",
        "baseline_stats_path",
        "feature_schema_version",
    ]

    missing_fields = [
        field
        for field in required_metadata_fields
        if field not in artifact_metadata or artifact_metadata[field] in (None, "")
    ]
    if missing_fields:
        raise RegistryError(f"Artifact metadata missing fields: {missing_fields}")

    _path_exists(artifact_metadata.get("model_path"), label="model_path")
    _path_exists(artifact_metadata.get("feature_schema_path"), label="feature_schema_path")
    _path_exists(artifact_metadata.get("baseline_stats_path"), label="baseline_stats_path")

    baseline_stats = _read_json(_resolve_registry_path(artifact_metadata["baseline_stats_path"]))

    entry = _build_entry_from_artifact_metadata(
        artifact_metadata=artifact_metadata,
        baseline_stats=baseline_stats,
        status=status,
        approved_for_prod=approved_for_prod,
        notes=notes,
    )

    return _add_entry(entry, registry_path=registry_path, replace_existing=replace_existing)


def register_model(
    *,
    model_name: str,
    model_version: str,
    artifact_path: str,
    dataset_snapshot_id: str,
    feature_schema_version: str,
    training_timestamp: str,
    metrics: dict[str, Any] | None = None,
    status: ModelStatus | str = ModelStatus.CANDIDATE,
    approved_for_prod: bool = False,
    registry_path: str | Path | None = None,
) -> ModelRegistryEntry:
    """Register a model version from explicit fields, in the registry schema.

    The artifact's sibling files (metadata.json, feature_schema.json,
    baseline_stats.json) are recorded by the standard artifact layout.
    Registration only creates candidate or staging versions; production
    approval happens through promote_model.
    """
    normalized_status = _status_value(status)

    if normalized_status not in REGISTRATION_STATUSES:
        raise RegistryError(
            f"register_model only accepts status {sorted(REGISTRATION_STATUSES)}; "
            f"got {normalized_status!r}. Use promote_model to change status."
        )

    if approved_for_prod:
        raise RegistryError(
            "register_model cannot set approved_for_prod=True. "
            "Use promote_model to promote a version to production."
        )

    artifact = _resolve_registry_path(artifact_path)
    if not artifact.exists():
        raise RegistryError(f"Model artifact does not exist: {artifact_path}")

    metric_values = dict(metrics or {})
    unknown_metrics = sorted(set(metric_values) - set(REGISTRY_METRIC_FIELDS))
    if unknown_metrics:
        raise RegistryError(
            f"Unsupported registry metrics: {unknown_metrics}. "
            f"Supported metrics: {list(REGISTRY_METRIC_FIELDS)}"
        )

    artifact_dir = artifact.parent
    metadata_path = artifact_dir / "metadata.json"
    artifact_metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    training_snapshot_metadata = artifact_metadata.get("training_snapshot_metadata") or {}

    entry = ModelRegistryEntry(
        model_id=_stable_model_id(model_name, model_version),
        model_name=model_name,
        model_version=model_version,
        algorithm=str(artifact_metadata.get("algorithm") or "unknown"),
        artifact_path=_to_registry_path(str(artifact)),
        artifact_dir=_to_registry_path(str(artifact_dir)),
        metadata_path=_to_registry_path(str(metadata_path)),
        feature_schema_path=_to_registry_path(str(artifact_dir / "feature_schema.json")),
        baseline_stats_path=_to_registry_path(str(artifact_dir / "baseline_stats.json")),
        snapshot_id=dataset_snapshot_id,
        training_dataset_id=_extract_training_dataset_id(training_snapshot_metadata),
        feature_schema_version=feature_schema_version,
        training_started_at_utc=artifact_metadata.get("training_started_at_utc"),
        training_finished_at_utc=training_timestamp,
        **{
            field_name: _as_optional_float(metric_values.get(field_name))
            for field_name in REGISTRY_METRIC_FIELDS
        },
        status=normalized_status,
        approved_for_prod=approved_for_prod,
        promoted_at_utc=_utc_now_iso() if normalized_status == ModelStatus.PRODUCTION else None,
        archived_at_utc=None,
        created_at_utc=_utc_now_iso(),
        notes=None,
    )

    return _add_entry(
        entry,
        registry_path=registry_path or DEFAULT_REGISTRY_PATH,
        replace_existing=False,
    )


def _load_artifact_metadata(entry: ModelRegistryEntry) -> dict[str, Any]:
    metadata_path = _resolve_registry_path(entry.metadata_path)
    if not metadata_path.exists():
        return {}
    return _read_json(metadata_path)


def _load_latency_budget(
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> dict[str, Any] | None:
    path = Path(model_config_path)
    if not path.exists():
        return None
    latency_budget = _read_yaml(path).get("latency_budget")
    return dict(latency_budget) if isinstance(latency_budget, dict) else None


def build_active_model_pointer(
    entry: ModelRegistryEntry,
    artifact_metadata: dict[str, Any],
    *,
    notes: str | None = None,
    latency_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full active model pointer from the target entry and its metadata only.

    Nothing is carried over from a previous pointer, so every lineage field
    describes the model the pointer activates.
    """
    training_snapshot_metadata = artifact_metadata.get("training_snapshot_metadata") or {}
    feature_names = artifact_metadata.get("feature_names")
    source_projects = training_snapshot_metadata.get("source_projects")
    source_tables = training_snapshot_metadata.get("source_tables")
    baseline_anomaly_rate = entry.baseline_anomaly_rate
    if baseline_anomaly_rate is None:
        baseline_anomaly_rate = _as_optional_float(
            artifact_metadata.get("baseline_anomaly_rate")
        )
    now = _utc_now_iso()

    pointer: dict[str, Any] = {
        "model_name": entry.model_name,
        "active_model_version": entry.model_version,
        "artifact_path": entry.artifact_path,
        "feature_schema_version": entry.feature_schema_version,
        "status": entry.status,
        "dataset_snapshot_id": entry.snapshot_id,
        "training_dataset_id": entry.training_dataset_id,
        "feature_count": len(feature_names) if isinstance(feature_names, list) else None,
        "baseline_anomaly_rate": baseline_anomaly_rate,
        "source_projects": list(source_projects) if isinstance(source_projects, list) else [],
        "source_tables": list(source_tables) if isinstance(source_tables, list) else [],
        "snapshot_type": training_snapshot_metadata.get("snapshot_type"),
        "last_updated_at": now,
        "updated_at": now,
        "notes": notes or f"Active production model set to {entry.model_version}.",
    }

    if latency_budget is not None:
        pointer["latency_budget"] = latency_budget

    return pointer


def write_active_model_pointer(
    *,
    entry: ModelRegistryEntry,
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
    notes: str | None = None,
) -> ActiveModelPointer:
    """Write the active model pointer for a production registry entry.

    This is the only function that writes the active model pointer.
    """
    if entry.status != "production":
        raise RegistryError(
            "Only a production model can become the active model pointer. "
            f"Got status={entry.status}"
        )

    if not entry.approved_for_prod:
        raise RegistryError("Production active model must be approved_for_prod=True.")

    pointer_payload = build_active_model_pointer(
        entry,
        _load_artifact_metadata(entry),
        notes=notes,
        latency_budget=_load_latency_budget(),
    )

    _write_yaml(active_model_path, pointer_payload)

    return ActiveModelPointer(
        model_name=pointer_payload["model_name"],
        active_model_version=pointer_payload["active_model_version"],
        artifact_path=pointer_payload["artifact_path"],
        feature_schema_version=pointer_payload["feature_schema_version"],
        status=pointer_payload["status"],
        last_updated_at=pointer_payload["last_updated_at"],
        notes=pointer_payload["notes"],
    )


def read_active_model_pointer(
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
) -> ActiveModelPointer:
    """Read active model YAML pointer."""
    payload = _read_yaml(active_model_path)

    return ActiveModelPointer(
        model_name=str(payload.get("model_name", "isolation_forest")),
        active_model_version=payload.get("active_model_version"),
        artifact_path=payload.get("artifact_path"),
        feature_schema_version=str(
            payload.get("feature_schema_version", "feature_schema_v001")
        ),
        status=str(payload.get("status", "not_deployed")),
        last_updated_at=payload.get("last_updated_at"),
        notes=payload.get("notes"),
    )


def _set_entry_status(
    *,
    model_name: str,
    model_version: str,
    status: str,
    registry_path: str | Path,
    active_model_path: str | Path,
    notes: str | None = None,
) -> ModelRegistryEntry:
    """Change a version's status outside the production promotion path.

    notes, when given, replaces the version's registry note.
    """
    _validate_status(status)
    state = load_registry_state(
        registry_path=registry_path,
        active_model_path=active_model_path,
        model_name=model_name,
    )
    ensure_database_initialized(state, model_name)
    entries = state.entries
    if not any(
        entry.model_name == model_name and entry.model_version == model_version
        for entry in entries
    ):
        raise RegistryError(f"Model version not registered: {model_name} {model_version}")

    if state.model_name == model_name and state.active_model_version == model_version:
        raise RegistryError(
            f"Cannot change status of {model_name} {model_version} to {status!r}: "
            "it is the active model version. Promote another version first."
        )

    def _with_status(entry: ModelRegistryEntry) -> ModelRegistryEntry:
        changes: dict[str, Any] = {"status": status}
        if status == ModelStatus.ARCHIVED and entry.archived_at_utc is None:
            changes["archived_at_utc"] = _utc_now_iso()
        if notes is not None:
            changes["notes"] = notes
        return ModelRegistryEntry(**{**entry.to_dict(), **changes})

    updated_entries = [
        _with_status(entry)
        if entry.model_name == model_name and entry.model_version == model_version
        else entry
        for entry in entries
    ]
    commit_registry_change(
        state=state,
        updated_entries=updated_entries,
        registry_path=registry_path,
        active_model_path=active_model_path,
        updated_by="status-change",
    )

    return next(
        entry
        for entry in updated_entries
        if entry.model_name == model_name and entry.model_version == model_version
    )


def retire_model_version(
    *,
    model_name: str,
    model_version: str,
    notes: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
) -> ModelRegistryEntry:
    """Set a non-active version's status to retired with a note; its files are kept.

    Postgres when configured (with the JSON mirror), else the JSON registry.
    The active pointer is never written.
    """
    if not notes or not notes.strip():
        raise RegistryError("retiring a model version needs a note explaining why")

    return _set_entry_status(
        model_name=model_name,
        model_version=model_version,
        status=ModelStatus.RETIRED.value,
        registry_path=registry_path,
        active_model_path=active_model_path,
        notes=notes,
    )


def promote_model_version(
    *,
    model_name: str,
    model_version: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
    notes: str | None = None,
) -> ModelRegistryEntry:
    """Promote a registered model version to production.

    Any existing production version for the same model is archived. This gives
    rollback logic a clean previous-stable trail later. The target's artifact
    must load and its metadata.json must list feature_names before anything is
    written.
    """
    state = load_registry_state(
        registry_path=registry_path,
        active_model_path=active_model_path,
        model_name=model_name,
    )
    ensure_database_initialized(state, model_name)
    entries = state.entries

    if not entries:
        raise RegistryError("No registered models found.")

    target_index: int | None = None
    now = _utc_now_iso()
    updated_entries: list[ModelRegistryEntry] = []

    for index, entry in enumerate(entries):
        if entry.model_name == model_name and entry.model_version == model_version:
            target_index = index

    if target_index is None:
        raise RegistryError(f"Model version not registered: {model_name} {model_version}")

    load_verified_feature_names(entries[target_index])

    for entry in entries:
        if (
            entry.model_name == model_name
            and entry.status == "production"
            and entry.model_version != model_version
        ):
            updated_entries.append(
                ModelRegistryEntry(
                    **{
                        **entry.to_dict(),
                        "status": "archived",
                        "approved_for_prod": True,
                        "archived_at_utc": now,
                        "notes": entry.notes
                        or "Archived during production promotion and retained as rollback-safe.",
                    }
                )
            )
        elif entry.model_name == model_name and entry.model_version == model_version:
            updated_entries.append(
                ModelRegistryEntry(
                    **{
                        **entry.to_dict(),
                        "status": "production",
                        "approved_for_prod": True,
                        "promoted_at_utc": now,
                        "archived_at_utc": None,
                        "notes": notes or "Promoted to production.",
                    }
                )
            )
        else:
            updated_entries.append(entry)

    promoted_entry = next(
        entry
        for entry in updated_entries
        if entry.model_name == model_name and entry.model_version == model_version
    )

    commit_registry_change(
        state=state,
        updated_entries=updated_entries,
        registry_path=registry_path,
        active_model_path=active_model_path,
        pointer_entry=promoted_entry,
        notes=notes or f"Active production model set to {model_version}.",
        updated_by="promotion",
    )

    return promoted_entry


def promote_model(
    *,
    model_name: str,
    model_version: str,
    target_status: ModelStatus | str = ModelStatus.PRODUCTION,
    registry_path: str | Path | None = None,
    active_model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Move a model version to target_status; production also updates the pointer."""

    normalized_status = _status_value(target_status)
    _validate_status(normalized_status)
    resolved_registry_path = registry_path or DEFAULT_REGISTRY_PATH
    resolved_active_model_path = active_model_config_path or DEFAULT_ACTIVE_MODEL_PATH

    if normalized_status == ModelStatus.PRODUCTION:
        entry = promote_model_version(
            model_name=model_name,
            model_version=model_version,
            registry_path=resolved_registry_path,
            active_model_path=resolved_active_model_path,
        )
    else:
        entry = _set_entry_status(
            model_name=model_name,
            model_version=model_version,
            status=normalized_status,
            registry_path=resolved_registry_path,
            active_model_path=resolved_active_model_path,
        )

    return entry.to_dict()


def list_model_versions(
    *,
    model_name: str | None = None,
    status: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[ModelRegistryEntry]:
    """List registered model versions with optional filters."""
    if status is not None:
        _validate_status(status)

    entries = load_model_registry(registry_path)

    if model_name is not None:
        entries = [entry for entry in entries if entry.model_name == model_name]

    if status is not None:
        entries = [entry for entry in entries if entry.status == status]

    return entries


def get_active_model(
    active_model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the active production model pointer from configs/active_model.yaml."""

    payload = _read_yaml(active_model_config_path or DEFAULT_ACTIVE_MODEL_PATH)

    required_fields = {
        "model_name",
        "active_model_version",
        "status",
        "artifact_path",
        "dataset_snapshot_id",
        "feature_schema_version",
    }
    missing_fields = sorted(required_fields - set(payload))
    if missing_fields:
        raise RegistryError(
            f"Active model config is missing required fields: {missing_fields}"
        )

    return payload


# --- serving check and CLI -----------------------------------------------------------


def verify_servable(entry: ModelRegistryEntry) -> None:
    """Refuse a version the serving code cannot load and score.

    The artifact must load and match its metadata (load_verified_feature_spec);
    then a serving model is built from a temporary pointer for the version, as
    the API would build it, and a probe payload is scored for every segment
    (single models: once). Nothing is written to the registry or the pointer.
    """

    load_verified_feature_spec(entry)

    from tempfile import TemporaryDirectory

    from anomaly_detection.online_inference import OnlineInferenceService

    as_active = ModelRegistryEntry(
        **{**entry.to_dict(), "status": ModelStatus.PRODUCTION.value, "approved_for_prod": True}
    )
    with TemporaryDirectory(prefix="project4_promote_candidate_") as candidate_dir:
        pointer_path = Path(candidate_dir) / "active_model.yaml"
        try:
            write_active_model_pointer(entry=as_active, active_model_path=pointer_path)
            service = OnlineInferenceService(active_model_path=pointer_path)
            probes = service.segments or {"user": service.feature_names}
            for entity_type, feature_names in probes.items():
                service.predict({name: 0.0 for name in feature_names}, entity_type=entity_type)
        except Exception as exc:  # noqa: BLE001 - any failure means serving cannot score it
            raise RegistryError(
                f"{entry.model_name} {entry.model_version} cannot be served: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Registry tools.

    python -m anomaly_detection.registry retire --model-version v003 --notes "..."
    python -m anomaly_detection.registry promote --model-version v004

    Postgres when PROJECT4_POSTGRES_* is configured (with the file mirror), else files.
    """

    import argparse
    import sys

    from anomaly_detection.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="Project 4 model registry tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    retire = commands.add_parser(
        "retire",
        help="Set a non-active version to retired with a note (files are kept).",
    )
    retire.add_argument("--model-version", required=True)
    retire.add_argument("--notes", required=True)
    retire.add_argument("--model-name", default=None, help="Default: model_config.yaml.")
    promote = commands.add_parser(
        "promote",
        help="Make a registered version the active production model after a serving check.",
    )
    promote.add_argument("--model-version", required=True)
    promote.add_argument("--notes", default=None)
    promote.add_argument("--model-name", default=None, help="Default: model_config.yaml.")
    args = parser.parse_args(argv)

    model_name = args.model_name or default_model_name()
    try:
        if args.command == "retire":
            entry = retire_model_version(
                model_name=model_name,
                model_version=args.model_version,
                notes=args.notes,
            )
        else:
            before = load_registry_state(model_name=model_name)
            candidate = next(
                (
                    item
                    for item in before.entries
                    if item.model_name == model_name and item.model_version == args.model_version
                ),
                None,
            )
            if candidate is None:
                raise RegistryError(
                    f"Model version not registered: {model_name} {args.model_version}"
                )
            verify_servable(candidate)
            entry = promote_model_version(
                model_name=model_name,
                model_version=args.model_version,
                notes=args.notes,
            )
        state = load_registry_state(model_name=model_name)
    except RegistryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {entry.model_name} {entry.model_version} status={entry.status}")
    print(f"notes={entry.notes}")
    print(
        f"registry={state.source} active_model_version={state.active_model_version} "
        f"pointer_version={state.pointer_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
