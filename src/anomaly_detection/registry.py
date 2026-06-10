"""Local model registry for anomaly detection model versions.

This module is intentionally local-first for the portfolio build. PostgreSQL
tables already exist for the production metadata shape, but this checkpoint
uses JSON files so model versioning, promotion, and active-model pointers work
without a database dependency.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "_registry" / "model_registry.json"
)
DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"

VALID_MODEL_STATUSES = {
    "candidate",
    "staging",
    "production",
    "archived",
    "rolled_back",
    "failed_validation",
}


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

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe registry entry."""
        return asdict(self)


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

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise RegistryError(f"JSON file must contain an object: {json_path}")

    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise RegistryError(f"YAML file does not exist: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    if not isinstance(payload, dict):
        raise RegistryError(f"YAML file must contain a mapping: {yaml_path}")

    return payload


def _validate_status(status: str) -> None:
    if status not in VALID_MODEL_STATUSES:
        raise RegistryError(
            f"Invalid model status: {status}. "
            f"Valid statuses: {sorted(VALID_MODEL_STATUSES)}"
        )


def _path_exists(path_value: str | None, *, label: str) -> None:
    if not path_value:
        raise RegistryError(f"{label} is missing from artifact metadata.")

    path = Path(path_value)
    if not path.exists():
        raise RegistryError(f"{label} does not exist: {path}")


def _to_registry_path(path_value: str) -> str:
    """Store project-local paths as repo-relative paths.

    Absolute local paths are poison in source-controlled config because they only
    work on one machine. If an artifact lives inside this repository, store it
    relative to PROJECT_ROOT. External paths remain absolute.
    """
    resolved_path = Path(path_value).resolve()

    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


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
        entries.append(ModelRegistryEntry(**raw_entry))

    return entries


def save_model_registry(
    entries: list[ModelRegistryEntry],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Persist local model registry entries."""
    payload = {
        "registry_backend": "local_json",
        "updated_at_utc": _utc_now_iso(),
        "models": [entry.to_dict() for entry in entries],
    }
    _write_json(registry_path, payload)


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

    baseline_stats = _read_json(artifact_metadata["baseline_stats_path"])

    entry = _build_entry_from_artifact_metadata(
        artifact_metadata=artifact_metadata,
        baseline_stats=baseline_stats,
        status=status,
        approved_for_prod=approved_for_prod,
        notes=notes,
    )

    entries = load_model_registry(registry_path)
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

    save_model_registry(entries, registry_path)

    return entry


def write_active_model_pointer(
    *,
    entry: ModelRegistryEntry,
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
    notes: str | None = None,
) -> ActiveModelPointer:
    """Write active model YAML pointer from a registry entry."""
    if entry.status != "production":
        raise RegistryError(
            "Only a production model can become the active model pointer. "
            f"Got status={entry.status}"
        )

    if not entry.approved_for_prod:
        raise RegistryError("Production active model must be approved_for_prod=True.")

    pointer = ActiveModelPointer(
        model_name=entry.model_name,
        active_model_version=entry.model_version,
        artifact_path=entry.artifact_path,
        feature_schema_version=entry.feature_schema_version,
        status=entry.status,
        last_updated_at=_utc_now_iso(),
        notes=notes or f"Active production model set to {entry.model_version}.",
    )

    _write_yaml(active_model_path, pointer.to_dict())

    return pointer


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
    rollback logic a clean previous-stable trail later.
    """
    entries = load_model_registry(registry_path)

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

    target_entry = entries[target_index]

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
                        "approved_for_prod": False,
                        "archived_at_utc": now,
                        "notes": entry.notes or "Archived during production promotion.",
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

    save_model_registry(updated_entries, registry_path)

    promoted_entry = find_model_entry(
        model_name=model_name,
        model_version=model_version,
        registry_path=registry_path,
    )

    write_active_model_pointer(
        entry=promoted_entry,
        active_model_path=active_model_path,
        notes=notes or f"Active production model set to {model_version}.",
    )

    return promoted_entry


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


# ---------------------------------------------------------------------------
# Production lifecycle API
# ---------------------------------------------------------------------------
# This compatibility layer adds explicit model lifecycle operations while
# preserving the existing registry implementation above.

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


class ModelStatus(StrEnum):
    """Supported lifecycle statuses for registered model versions."""

    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"
    FAILED_VALIDATION = "failed_validation"


@dataclass(frozen=True)
class ModelRegistryRecord:
    """Metadata required to treat a trained model as a governed production asset."""

    model_name: str
    model_version: str
    status: str
    artifact_path: str
    dataset_snapshot_id: str
    feature_schema_version: str
    training_timestamp: str
    metrics: dict[str, Any]
    approved_for_prod: bool = False
    registry_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if not record["registry_id"]:
            record["registry_id"] = str(uuid4())
        if not record["created_at"]:
            record["created_at"] = datetime.now(UTC).isoformat()
        return record


def _registry_path() -> Path:
    return Path("artifacts/models/_registry/model_registry.json")


def _active_model_config_path() -> Path:
    return Path("configs/active_model.yaml")


def _load_registry_document(registry_path: Path | None = None) -> dict[str, Any]:
    path = registry_path or _registry_path()
    if not path.exists():
        return {"models": []}

    try:
        import json

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to read model registry JSON at {path}: {exc}") from exc

    if isinstance(payload, list):
        return {"models": payload}

    if isinstance(payload, dict):
        if "models" not in payload:
            payload["models"] = []
        if not isinstance(payload["models"], list):
            raise ValueError("Model registry field 'models' must be a list.")
        return payload

    raise ValueError("Model registry must be a JSON object or list.")


def _write_registry_document(payload: dict[str, Any], registry_path: Path | None = None) -> None:
    import json

    path = registry_path or _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _find_model_record(
    payload: dict[str, Any],
    *,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    for record in payload.get("models", []):
        if (
            record.get("model_name") == model_name
            and record.get("model_version") == model_version
        ):
            return record

    raise ValueError(f"Model version not found: {model_name}/{model_version}")


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
    registry_path: Path | None = None,
) -> ModelRegistryRecord:
    """Register a trained model version as a lifecycle-controlled asset."""

    normalized_status = ModelStatus(status).value
    payload = _load_registry_document(registry_path)

    duplicate_exists = any(
        record.get("model_name") == model_name
        and record.get("model_version") == model_version
        for record in payload.get("models", [])
    )
    if duplicate_exists:
        raise ValueError(f"Model version already registered: {model_name}/{model_version}")

    artifact = Path(artifact_path)
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")

    record = ModelRegistryRecord(
        model_name=model_name,
        model_version=model_version,
        status=normalized_status,
        artifact_path=artifact_path,
        dataset_snapshot_id=dataset_snapshot_id,
        feature_schema_version=feature_schema_version,
        training_timestamp=training_timestamp,
        metrics=metrics or {},
        approved_for_prod=approved_for_prod,
    )

    payload.setdefault("models", []).append(record.to_dict())
    _write_registry_document(payload, registry_path)

    return record


def promote_model(
    *,
    model_name: str,
    model_version: str,
    target_status: ModelStatus | str = ModelStatus.PRODUCTION,
    registry_path: Path | None = None,
    active_model_config_path: Path | None = None,
) -> dict[str, Any]:
    """Promote a model version and update the active model pointer for production."""

    normalized_status = ModelStatus(target_status).value
    payload = _load_registry_document(registry_path)
    target_record = _find_model_record(
        payload,
        model_name=model_name,
        model_version=model_version,
    )

    if normalized_status == ModelStatus.PRODUCTION.value:
        for record in payload.get("models", []):
            same_model = record.get("model_name") == model_name
            currently_production = record.get("status") == ModelStatus.PRODUCTION.value
            same_version = record.get("model_version") == model_version

            if same_model and currently_production and not same_version:
                record["status"] = ModelStatus.ARCHIVED.value
                record["approved_for_prod"] = False

        target_record["approved_for_prod"] = True

    target_record["status"] = normalized_status
    target_record["promoted_at"] = datetime.now(UTC).isoformat()

    _write_registry_document(payload, registry_path)

    if normalized_status == ModelStatus.PRODUCTION.value:
        config_path = active_model_config_path or _active_model_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        active_config = {
            "model_name": target_record["model_name"],
            "active_model_version": target_record["model_version"],
            "status": target_record["status"],
            "artifact_path": target_record["artifact_path"],
            "dataset_snapshot_id": target_record["dataset_snapshot_id"],
            "feature_schema_version": target_record["feature_schema_version"],
            "updated_at": target_record["promoted_at"],
        }

        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(active_config, file, sort_keys=False)

    return target_record


def get_active_model(
    active_model_config_path: Path | None = None,
) -> dict[str, Any]:
    """Load the active production model pointer from configs/active_model.yaml."""

    config_path = active_model_config_path or _active_model_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Active model config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

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
        raise ValueError(
            f"Active model config is missing required fields: {missing_fields}"
        )

    return payload
