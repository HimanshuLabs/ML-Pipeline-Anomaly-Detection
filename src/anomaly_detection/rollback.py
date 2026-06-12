"""Model rollback controls for Project 4.

Rollback is intentionally implemented as a controlled operation, not a blind
automation. The module reads the active production model pointer, selects a
previous stable approved model from registry-like records, updates the active
pointer, and writes durable rollback audit evidence.

Database support is planned through audit.rollback_events. The current
implementation is local-first and writes JSONL rollback evidence so the behavior
can be tested without requiring a live PostgreSQL dependency.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_ROLLBACK_EVENTS_PATH = PROJECT_ROOT / "logs" / "alerts" / "rollback_events.jsonl"

STABLE_ROLLBACK_STATUSES = {"archived", "rolled_back", "production"}


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


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp_file:
        yaml.safe_dump(
            payload,
            tmp_file,
            sort_keys=False,
            allow_unicode=True,
        )
        tmp_path = Path(tmp_file.name)

    tmp_path.replace(path)


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


def _record_to_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record

    if hasattr(record, "model_dump"):
        payload = record.model_dump()
        if isinstance(payload, dict):
            return payload

    if hasattr(record, "__dict__"):
        return {
            key: value
            for key, value in vars(record).items()
            if not key.startswith("_")
        }

    raise RollbackError(f"unsupported registry record type: {type(record)!r}")


def _parse_sort_timestamp(record: dict[str, Any]) -> str:
    """Return best available timestamp-like value for stable ordering."""

    for key in (
        "promoted_at_utc",
        "archived_at_utc",
        "registered_at_utc",
        "created_at_utc",
        "training_finished_at",
        "training_finished_at_utc",
        "updated_at",
        "last_updated_at",
    ):
        value = record.get(key)
        if value:
            return str(value)

    return ""


def select_previous_stable_model(
    registry_records: list[Any],
    *,
    current_model_version: str,
    model_name: str | None = None,
) -> RollbackTarget:
    """Select the latest previous stable approved model.

    A rollback target must:
    - not be the currently active version
    - match model_name when provided
    - be approved for production
    - have a stable status such as archived, rolled_back, or production
    - include model_version and artifact_path
    """

    if not current_model_version:
        raise RollbackError("current_model_version is required")

    normalized_records = [_record_to_dict(record) for record in registry_records]

    candidates: list[dict[str, Any]] = []

    for record in normalized_records:
        record_model_name = record.get("model_name")
        record_version = record.get("model_version")
        status = str(record.get("status", "")).lower()
        approved_for_prod = bool(record.get("approved_for_prod", False))
        artifact_path = record.get("artifact_path")

        if model_name and record_model_name != model_name:
            continue

        if record_version == current_model_version:
            continue

        if status not in STABLE_ROLLBACK_STATUSES:
            continue

        if not approved_for_prod:
            continue

        if not record_version or not artifact_path:
            continue

        candidates.append(record)

    if not candidates:
        raise RollbackError(
            "no previous stable approved model found for rollback"
        )

    candidates.sort(
        key=lambda item: (
            _parse_sort_timestamp(item),
            str(item.get("model_version", "")),
        ),
        reverse=True,
    )

    selected = candidates[0]

    return RollbackTarget(
        model_name=str(selected.get("model_name") or model_name or ""),
        model_version=str(selected["model_version"]),
        artifact_path=str(selected["artifact_path"]),
        dataset_snapshot_id=(
            str(selected["dataset_snapshot_id"])
            if selected.get("dataset_snapshot_id") is not None
            else None
        ),
        feature_schema_version=(
            str(selected["feature_schema_version"])
            if selected.get("feature_schema_version") is not None
            else None
        ),
        status="production",
        approved_for_prod=True,
        selected_from_status=str(selected.get("status", "")),
    )


def build_rollback_active_model_payload(
    *,
    current_active_model: dict[str, Any],
    rollback_target: RollbackTarget,
) -> dict[str, Any]:
    """Build the replacement active_model.yaml payload."""

    model_name = rollback_target.model_name or current_active_model.get("model_name")

    if not model_name:
        raise RollbackError("rollback target model_name is missing")

    payload = dict(current_active_model)

    payload.update(
        {
            "model_name": model_name,
            "active_model_version": rollback_target.model_version,
            "artifact_path": rollback_target.artifact_path,
            "status": "production",
            "approved_for_prod": True,
            "last_updated_at": _utc_now(),
            "updated_at": _utc_now(),
            "rollback": {
                "rolled_back_from_model_version": current_active_model.get(
                    "active_model_version"
                ),
                "rolled_back_to_model_version": rollback_target.model_version,
                "rollback_applied_at_utc": _utc_now(),
                "selected_from_status": rollback_target.selected_from_status,
            },
        }
    )

    if rollback_target.dataset_snapshot_id is not None:
        payload["dataset_snapshot_id"] = rollback_target.dataset_snapshot_id

    if rollback_target.feature_schema_version is not None:
        payload["feature_schema_version"] = rollback_target.feature_schema_version

    return payload


def rollback_active_model(
    registry_records: list[Any],
    *,
    active_model_path: Path = DEFAULT_ACTIVE_MODEL_PATH,
    rollback_events_path: Path = DEFAULT_ROLLBACK_EVENTS_PATH,
    rollback_reason: str,
    triggered_by: str = "manual",
    dry_run: bool = False,
) -> RollbackEvent:
    """Rollback active model pointer to the previous stable approved version."""

    rollback_reason = rollback_reason.strip()
    triggered_by = triggered_by.strip()

    if not rollback_reason:
        raise RollbackError("rollback_reason is required")

    if not triggered_by:
        raise RollbackError("triggered_by is required")

    current_active_model = _read_yaml(active_model_path)

    current_model_version = current_active_model.get("active_model_version")
    model_name = current_active_model.get("model_name")

    if not current_model_version:
        raise RollbackError("active model pointer missing active_model_version")

    rollback_target = select_previous_stable_model(
        registry_records,
        current_model_version=str(current_model_version),
        model_name=str(model_name) if model_name else None,
    )

    next_active_model = build_rollback_active_model_payload(
        current_active_model=current_active_model,
        rollback_target=rollback_target,
    )

    event = RollbackEvent(
        rollback_id=f"rollback_{uuid.uuid4().hex}",
        model_name=str(model_name or rollback_target.model_name),
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

    _write_yaml_atomic(active_model_path, next_active_model)
    append_rollback_event_jsonl(event, rollback_events_path)

    return event
