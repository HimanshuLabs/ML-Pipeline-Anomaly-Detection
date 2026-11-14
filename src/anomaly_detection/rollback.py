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
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from anomaly_detection.registry import (
    DEFAULT_REGISTRY_PATH,
    ModelRegistryEntry,
    ModelStatus,
    RegistryError,
    load_model_registry,
    load_verified_feature_names,
    save_model_registry,
    write_active_model_pointer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_ROLLBACK_EVENTS_PATH = PROJECT_ROOT / "logs" / "alerts" / "rollback_events.jsonl"


class RollbackError(RuntimeError):
    """Raised when rollback cannot be performed safely."""


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
    active_feature_names: list[str],
) -> str | None:
    """Return why a version cannot be a rollback target, or None if it can."""

    if entry.status != ModelStatus.ARCHIVED:
        return f"status {entry.status!r} is not 'archived'"

    if not entry.approved_for_prod:
        return "not approved_for_prod"

    try:
        feature_names = load_verified_feature_names(entry)
    except RegistryError as exc:
        return str(exc)

    if feature_names != active_feature_names:
        return (
            f"feature_names differ from the active model "
            f"({len(feature_names)} vs {len(active_feature_names)} features)"
        )

    return None


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
    - have metadata.json whose feature_names equal the active model's, in order

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
        active_feature_names = load_verified_feature_names(active_entry)
    except RegistryError as exc:
        raise RollbackError(
            f"active model {model_name} {current_model_version} cannot be checked: {exc}"
        ) from exc

    eligible: list[ModelRegistryEntry] = []
    rejections: list[str] = []

    for entry in same_model:
        if entry.model_version == current_model_version:
            continue

        reason = _rejection_reason(entry, active_feature_names)
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


def _apply_registry_and_pointer(
    *,
    entries: list[ModelRegistryEntry],
    registry_path: Path,
    active_model_path: Path,
    model_name: str,
    from_model_version: str,
    to_model_version: str,
    rollback_reason: str,
) -> None:
    """Commit the rollback: registry statuses first, then the pointer.

    If the pointer write fails, the registry is restored to its previous content.
    """

    previous_updated_at = json.loads(registry_path.read_text(encoding="utf-8")).get(
        "updated_at_utc"
    )
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

    save_model_registry(updated_entries, registry_path)

    try:
        write_active_model_pointer(
            entry=target_entry,
            active_model_path=active_model_path,
            notes=(
                f"Rolled back from {from_model_version} to {to_model_version}: "
                f"{rollback_reason}"
            ),
        )
    except Exception as exc:
        save_model_registry(entries, registry_path, updated_at_utc=previous_updated_at)
        raise RollbackError(
            f"rollback to {to_model_version} failed while writing the active pointer; "
            f"registry restored: {exc}"
        ) from exc


def rollback_active_model(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    active_model_path: Path = DEFAULT_ACTIVE_MODEL_PATH,
    rollback_events_path: Path = DEFAULT_ROLLBACK_EVENTS_PATH,
    rollback_reason: str,
    triggered_by: str = "manual",
    dry_run: bool = False,
) -> RollbackEvent:
    """Rollback the active model to the previous compatible stable version.

    A dry run performs exactly the same target checks and writes nothing.
    """

    rollback_reason = rollback_reason.strip()
    triggered_by = triggered_by.strip()

    if not rollback_reason:
        raise RollbackError("rollback_reason is required")

    if not triggered_by:
        raise RollbackError("triggered_by is required")

    registry_path = Path(registry_path)
    active_model_path = Path(active_model_path)

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
        active_model_path=str(active_model_path),
        rollback_event_path=None if dry_run else str(rollback_events_path),
    )

    if dry_run:
        return event

    _apply_registry_and_pointer(
        entries=entries,
        registry_path=registry_path,
        active_model_path=active_model_path,
        model_name=str(model_name),
        from_model_version=str(current_model_version),
        to_model_version=rollback_target.model_version,
        rollback_reason=rollback_reason,
    )
    append_rollback_event_jsonl(event, rollback_events_path)

    return event
