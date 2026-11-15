"""Model rollback controls for Project 4.

Rollback is intentionally implemented as a controlled operation, not a blind
automation. The module reads the active production model pointer, selects a
previous stable approved model that can serve the same feature contract,
updates the registry and the active pointer through the registry's single
writers, and writes durable rollback audit evidence.

Database support is planned through audit.rollback_events. The current
implementation is local-first and writes JSONL rollback evidence so the behavior
can be tested without requiring a live PostgreSQL dependency.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from anomaly_detection.registry import (
    DEFAULT_REGISTRY_PATH,
    ModelRegistryEntry,
    ModelStatus,
    RegistryError,
    RegistryState,
    RegistryStateSnapshot,
    RegistryStorageError,
    commit_registry_change,
    database_enabled,
    ensure_database_initialized,
    load_model_registry,
    load_registry_state,
    load_verified_feature_spec,
    materialize_active_pointer,
    project_relative_path,
    restore_database_change,
    restore_registry_state,
    save_model_registry,
    snapshot_registry_state,
    write_active_model_pointer,
)
from anomaly_detection.segmented_model import FeatureSpec

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_ROLLBACK_EVENTS_PATH = PROJECT_ROOT / "logs" / "alerts" / "rollback_events.jsonl"


class RollbackError(RuntimeError):
    """Raised when rollback cannot be performed safely."""


class RollbackServingError(RuntimeError):
    """Raised when the rollback target cannot be served; the active model is unchanged.

    Deliberately not a RollbackError: the request was valid, the server failed.
    """


@dataclass(frozen=True)
class RollbackTarget:
    """Previous stable model selected as rollback target."""

    model_name: str
    model_version: str
    artifact_path: str
    dataset_snapshot_id: str | None
    feature_schema_version: str | None
    status: str
    approved_for_prod: bool
    selected_from_status: str


@dataclass(frozen=True)
class RollbackEvent:
    """Rollback audit evidence written after a rollback attempt."""

    rollback_id: str
    model_name: str
    from_model_version: str
    to_model_version: str
    rollback_reason: str
    triggered_by: str
    triggered_at_utc: str
    validation_status: str
    dry_run: bool
    active_model_path: str
    rollback_event_path: str | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RollbackError(f"active model pointer does not exist: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise RollbackError(f"active model pointer must be a YAML object: {path}")

    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    return value


def append_rollback_event_jsonl(
    event: RollbackEvent,
    rollback_events_path: Path = DEFAULT_ROLLBACK_EVENTS_PATH,
) -> Path:
    """Append one rollback audit event to a JSONL file."""

    rollback_events_path.parent.mkdir(parents=True, exist_ok=True)

    with rollback_events_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                _json_safe(asdict(event)),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )

    return rollback_events_path


def read_rollback_events_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read rollback JSONL evidence for validation and tests."""

    if not path.exists():
        raise RollbackError(f"rollback event file does not exist: {path}")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        record = json.loads(line)

        if not isinstance(record, dict):
            raise RollbackError(f"rollback event line {line_number} is not a JSON object")

        records.append(record)

    return records


def _rejection_reason(
    entry: ModelRegistryEntry,
    active_spec: FeatureSpec,
) -> str | None:
    """Return why a version cannot be a rollback target, or None if it can.

    Single models need the same ordered feature_names as the active model;
    segmented models the same segment spec; a single and a segmented model are
    never compatible.
    """

    if entry.status != ModelStatus.ARCHIVED:
        return f"status {entry.status!r} is not 'archived'"

    if not entry.approved_for_prod:
        return "not approved_for_prod"

    try:
        spec = load_verified_feature_spec(entry)
    except RegistryError as exc:
        return str(exc)

    difference = spec.difference(active_spec)
    if difference is None:
        return None

    if spec.is_segmented or active_spec.is_segmented:
        return f"{difference}; not compatible with the active model"

    return (
        f"feature_names differ from the active model "
        f"({len(spec.feature_names)} vs {len(active_spec.feature_names)} features)"
    )


def select_previous_stable_model(
    registry_records: list[ModelRegistryEntry],
    *,
    current_model_version: str,
    model_name: str,
) -> RollbackTarget:
    """Select the latest previous stable model that can serve the active contract.

    A rollback target must:
    - have the same model_name and a different version from the active one
    - have status 'archived' and be approved for production
    - have an artifact that exists and loads with joblib
    - have the same feature spec as the active model: equal ordered feature_names for
      single models, the same segment spec for segmented ones (never mixed)

    Among eligible versions the most recently archived one is selected.
    """

    if not current_model_version:
        raise RollbackError("current_model_version is required")

    same_model = [entry for entry in registry_records if entry.model_name == model_name]
    active_entry = next(
        (entry for entry in same_model if entry.model_version == current_model_version),
        None,
    )
    if active_entry is None:
        raise RollbackError(
            f"active model {model_name} {current_model_version} is not in the registry"
        )

    try:
        active_spec = load_verified_feature_spec(active_entry)
    except RegistryError as exc:
        raise RollbackError(
            f"active model {model_name} {current_model_version} cannot be checked: {exc}"
        ) from exc

    eligible: list[ModelRegistryEntry] = []
    rejections: list[str] = []

    for entry in same_model:
        if entry.model_version == current_model_version:
            continue

        reason = _rejection_reason(entry, active_spec)
        if reason is None:
            eligible.append(entry)
        else:
            rejections.append(f"{entry.model_version}: {reason}")

    if not eligible:
        details = "; ".join(rejections) or "no other registered versions"
        raise RollbackError(f"No compatible rollback target for {model_name}: {details}")

    selected = max(eligible, key=lambda entry: entry.archived_at_utc or "")

    return RollbackTarget(
        model_name=selected.model_name,
        model_version=selected.model_version,
        artifact_path=selected.artifact_path,
        dataset_snapshot_id=selected.snapshot_id,
        feature_schema_version=selected.feature_schema_version,
        status=ModelStatus.PRODUCTION.value,
        approved_for_prod=True,
        selected_from_status=selected.status,
    )


def _committed_entries(
    entries: list[ModelRegistryEntry],
    *,
    model_name: str,
    from_model_version: str,
    to_model_version: str,
) -> tuple[list[ModelRegistryEntry], ModelRegistryEntry]:
    """Registry entries after the rollback, and the target entry as production."""

    now = _utc_now()
    updated_entries: list[ModelRegistryEntry] = []
    target_entry: ModelRegistryEntry | None = None

    for entry in entries:
        if entry.model_name == model_name and entry.model_version == from_model_version:
            updated_entries.append(replace(entry, status=ModelStatus.ROLLED_BACK.value))
        elif entry.model_name == model_name and entry.model_version == to_model_version:
            target_entry = replace(
                entry,
                status=ModelStatus.PRODUCTION.value,
                approved_for_prod=True,
                promoted_at_utc=now,
                archived_at_utc=None,
            )
            updated_entries.append(target_entry)
        else:
            updated_entries.append(entry)

    if target_entry is None:
        raise RollbackError(f"rollback target {to_model_version} is not in the registry")

    return updated_entries, target_entry


def _verify_target_serves(
    target_entry: ModelRegistryEntry,
    *,
    notes: str,
    verify_service: Callable[[Path], object],
) -> None:
    """Build the target's pointer in a temporary directory and load a service from it."""

    with TemporaryDirectory(prefix="project4_rollback_candidate_") as candidate_dir:
        candidate_pointer_path = Path(candidate_dir) / "active_model.yaml"
        try:
            write_active_model_pointer(
                entry=target_entry,
                active_model_path=candidate_pointer_path,
                notes=notes,
            )
            verify_service(candidate_pointer_path)
        except Exception as exc:
            raise RollbackServingError(
                f"rollback target {target_entry.model_version} cannot serve; "
                f"nothing was changed: {exc}"
            ) from exc


def rollback_active_model(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    active_model_path: Path = DEFAULT_ACTIVE_MODEL_PATH,
    rollback_events_path: Path = DEFAULT_ROLLBACK_EVENTS_PATH,
    rollback_reason: str,
    triggered_by: str = "manual",
    dry_run: bool = False,
    verify_service: Callable[[Path], object] | None = None,
) -> RollbackEvent:
    """Rollback the active model to the previous compatible stable version.

    verify_service, when given, is called with a pointer path and must load a
    serving model from it (raising on failure). It is called on a candidate
    pointer for the target before anything is written (dry runs included),
    and again on the committed pointer after the registry and pointer are
    written. If the post-commit call fails, the registry and pointer are
    restored to their previous bytes. A dry run writes nothing.
    """

    rollback_reason = rollback_reason.strip()
    triggered_by = triggered_by.strip()

    if not rollback_reason:
        raise RollbackError("rollback_reason is required")

    if not triggered_by:
        raise RollbackError("triggered_by is required")

    registry_path = Path(registry_path)
    active_model_path = Path(active_model_path)

    state: RegistryState | None = None
    if database_enabled():
        state = load_registry_state(
            registry_path=registry_path,
            active_model_path=active_model_path,
        )
        if not dry_run:
            ensure_database_initialized(state)
        current_model_version = state.active_model_version
        model_name = state.model_name
        if not current_model_version:
            raise RollbackError(f"no active model pointer for {model_name} in the database")
        entries = state.entries
    else:
        current_active_model = _read_yaml(active_model_path)
        current_model_version = current_active_model.get("active_model_version")
        model_name = current_active_model.get("model_name")

        if not current_model_version:
            raise RollbackError("active model pointer missing active_model_version")

        if not model_name:
            raise RollbackError("active model pointer missing model_name")

        entries = load_model_registry(registry_path)

    rollback_target = select_previous_stable_model(
        entries,
        current_model_version=str(current_model_version),
        model_name=str(model_name),
    )

    event = RollbackEvent(
        rollback_id=f"rollback_{uuid.uuid4().hex}",
        model_name=str(model_name),
        from_model_version=str(current_model_version),
        to_model_version=rollback_target.model_version,
        rollback_reason=rollback_reason,
        triggered_by=triggered_by,
        triggered_at_utc=_utc_now(),
        validation_status="dry_run_validated" if dry_run else "applied",
        dry_run=dry_run,
        # Repo-relative inside the repository (JSONL and audit.rollback_events alike).
        active_model_path=project_relative_path(active_model_path),
        rollback_event_path=None if dry_run else project_relative_path(rollback_events_path),
    )

    updated_entries, target_entry = _committed_entries(
        entries,
        model_name=str(model_name),
        from_model_version=str(current_model_version),
        to_model_version=rollback_target.model_version,
    )
    pointer_notes = (
        f"Rolled back from {current_model_version} to {rollback_target.model_version}: "
        f"{rollback_reason}"
    )

    if verify_service is not None:
        _verify_target_serves(target_entry, notes=pointer_notes, verify_service=verify_service)

    if dry_run:
        return event

    snapshot = snapshot_registry_state(registry_path, active_model_path)

    if state is not None:
        return _apply_with_database(
            state=state,
            event=event,
            updated_entries=updated_entries,
            target_entry=target_entry,
            pointer_notes=pointer_notes,
            file_snapshot=snapshot,
            registry_path=registry_path,
            active_model_path=active_model_path,
            rollback_events_path=rollback_events_path,
            verify_service=verify_service,
        )

    save_model_registry(updated_entries, registry_path)

    try:
        write_active_model_pointer(
            entry=target_entry,
            active_model_path=active_model_path,
            notes=pointer_notes,
        )
    except Exception as exc:
        restore_registry_state(snapshot)
        raise RollbackServingError(
            f"rollback to {rollback_target.model_version} failed while writing the active "
            f"pointer; registry and pointer restored: {exc}"
        ) from exc

    if verify_service is not None:
        try:
            verify_service(active_model_path)
        except Exception as exc:
            restore_registry_state(snapshot)
            raise RollbackServingError(
                f"rollback to {rollback_target.model_version} was written but the committed "
                f"pointer failed to load; registry and pointer restored: {exc}"
            ) from exc

    append_rollback_event_jsonl(event, rollback_events_path)

    return event


def _apply_with_database(
    *,
    state: RegistryState,
    event: RollbackEvent,
    updated_entries: list[ModelRegistryEntry],
    target_entry: ModelRegistryEntry,
    pointer_notes: str,
    file_snapshot: RegistryStateSnapshot,
    registry_path: Path,
    active_model_path: Path,
    rollback_events_path: Path,
    verify_service: Callable[[Path], object] | None,
) -> RollbackEvent:
    """Commit registry, pointer, and the rollback event in one database transaction.

    The file mirror is written after the commit. If the committed pointer then
    fails to load, a new transaction restores the registry rows and the pointer,
    marks the event 'failed', and the files are restored from their snapshot.
    """

    from anomaly_detection.database import mark_rollback_event_failed, rollback_event_row

    try:
        commit = commit_registry_change(
            state=state,
            updated_entries=updated_entries,
            registry_path=registry_path,
            active_model_path=active_model_path,
            pointer_entry=target_entry,
            notes=pointer_notes,
            updated_by=event.triggered_by,
            extra_rows=[("audit.rollback_events", rollback_event_row(event))],
        )
    except RegistryStorageError as exc:
        raise RollbackServingError(
            f"rollback to {event.to_model_version} could not be committed: {exc}"
        ) from exc

    if verify_service is not None:
        try:
            with TemporaryDirectory(prefix="project4_rollback_committed_") as committed_dir:
                committed_pointer_path = Path(committed_dir) / "active_model.yaml"
                materialize_active_pointer(committed_pointer_path, model_name=event.model_name)
                verify_service(committed_pointer_path)
        except Exception as exc:
            if commit is not None:
                restore_database_change(
                    commit,
                    updated_by=event.triggered_by,
                    on_connection=lambda connection: mark_rollback_event_failed(
                        connection, event.rollback_id
                    ),
                )
            restore_registry_state(file_snapshot)
            raise RollbackServingError(
                f"rollback to {event.to_model_version} was committed but the committed "
                f"pointer failed to load; registry and pointer restored: {exc}"
            ) from exc

    try:
        append_rollback_event_jsonl(event, rollback_events_path)
    except Exception:  # noqa: BLE001 - the database already holds the event
        LOGGER.error("Rollback event committed to the database; JSONL mirror write failed")

    return event
