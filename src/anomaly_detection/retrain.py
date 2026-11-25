"""Retrain the anomaly model from a real-source snapshot and register it as a candidate.

    python -m anomaly_detection.retrain --snapshot <path|snapshot_id|latest> --model-version v004

1. The snapshot must be a real-source extract with data_quality_status "passed",
   trainable, a features.parquet, and the extract label configured in
   source_extract.yaml; otherwise it is refused with every reason.
2. Training reuses training.train_isolation_forest_from_snapshot, which trains
   the segmented model (one IsolationForest per entity type) for real-source
   snapshots (model.joblib, metadata.json, feature_schema.json, baseline_stats.json).
3. evaluation_report.json is written next to the artifact: row counts, the
   configured contamination, the features of each segment, and the anomaly rate
   and decision_function score distribution overall and per entity type.
4. The version is registered as "candidate" (Postgres when PROJECT4_POSTGRES_* is
   configured, with the JSON file mirror; else the JSON registry).

It never promotes and never writes the active model pointer.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.registry import (
    DEFAULT_REGISTRY_PATH,
    ModelRegistryEntry,
    RegistryError,
    _write_bytes_atomic,
    ensure_database_initialized,
    load_registry_state,
    load_verified_feature_spec,
    read_bundle_paths,
    register_model_from_artifacts,
)
from anomaly_detection.segmented_model import (
    FeatureSpec,
    SegmentedIsolationForest,
    feature_spec_for,
    feature_spec_from_metadata,
    score_distribution,
)
from anomaly_detection.source_extract import SEGMENT_FEATURE_COLUMNS, content_sha256
from anomaly_detection.source_extract import load_config as load_source_extract_config
from anomaly_detection.tracking import TrackingError, training_run
from anomaly_detection.training import (
    DEFAULT_MODEL_CONFIG_PATH,
    PROJECT_ROOT,
    ModelArtifactMetadata,
    TrainingError,
    build_training_feature_matrix,
    feature_set_fingerprint,
    load_trained_model,
    load_training_config,
    train_isolation_forest_from_snapshot,
)
from anomaly_detection.training_snapshot import read_training_snapshot

LOGGER = logging.getLogger(__name__)

DEFAULT_TRAINING_ROOT = PROJECT_ROOT / "data" / "features" / "training"
DEFAULT_SOURCE_EXTRACT_CONFIG_PATH = PROJECT_ROOT / "configs" / "source_extract.yaml"
REAL_SOURCE_SNAPSHOT_TYPE = "real_source_extract"
EVALUATION_REPORT_NAME = "evaluation_report.json"
# Entity types source_extract produces; each is reported even when absent.
EXPECTED_ENTITY_TYPES = tuple(SEGMENT_FEATURE_COLUMNS)
SCORE_DEFINITION = (
    "IsolationForest.decision_function on the training rows (segmented models: "
    "each row's own entity-type model); lower is more anomalous; a row is an "
    "anomaly when predict() == -1."
)


class RetrainError(ValueError):
    """Raised when a retrain is refused; nothing was trained or registered."""


@dataclass(frozen=True)
class RetrainResult:
    """What a retrain wrote and registered."""

    artifact: ModelArtifactMetadata
    registry_entry: ModelRegistryEntry
    registry_source: str
    active_model_version_before: str | None
    active_model_version_after: str | None
    evaluation_report_path: Path
    evaluation_report: dict[str, Any]


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def resolve_snapshot_dir(snapshot: str | Path, training_root: str | Path) -> Path:
    """Snapshot directory for a path, a snapshot id, or 'latest'.

    'latest' is the most recently created real-source snapshot, even a failed
    one, so a failed extraction is refused instead of silently training on an
    older snapshot.
    """

    value = str(snapshot)
    root = Path(training_root)

    if value == "latest":
        candidates = []
        for metadata_path in root.glob("snapshot_date=*/snapshot_id=*/metadata.json"):
            metadata = _read_metadata(metadata_path) or {}
            if metadata.get("snapshot_type") == REAL_SOURCE_SNAPSHOT_TYPE:
                candidates.append((str(metadata.get("created_at") or ""), str(metadata_path)))
        if not candidates:
            raise RetrainError(f"No real-source snapshot found under: {root}")
        return Path(max(candidates)[1]).parent

    path = Path(value)
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent

    matches = sorted(root.glob(f"snapshot_date=*/snapshot_id={value}/metadata.json"))
    if not matches:
        raise RetrainError(f"Could not resolve snapshot {value!r} under: {root}")
    return matches[-1].parent


def snapshot_refusal_reasons(
    snapshot_dir: Path,
    metadata: dict[str, Any] | None,
    *,
    expected_feature_schema_version: str,
) -> list[str]:
    """Every reason the snapshot cannot be trained on; empty when it can."""

    if metadata is None:
        return [f"metadata.json is missing or unreadable in {snapshot_dir}"]

    reasons: list[str] = []
    snapshot_type = metadata.get("snapshot_type")
    if snapshot_type != REAL_SOURCE_SNAPSHOT_TYPE:
        reasons.append(f"snapshot_type is {snapshot_type!r}, not {REAL_SOURCE_SNAPSHOT_TYPE!r}")

    status = metadata.get("data_quality_status")
    if status != "passed":
        reasons.append(f"data_quality_status is {status!r}")

    if metadata.get("trainable") is not True:
        reasons.append("trainable is not true")

    validation_errors = metadata.get("validation_errors") or []
    if validation_errors:
        reasons.append("validation errors: " + " | ".join(str(e) for e in validation_errors))

    if not (snapshot_dir / "features.parquet").is_file():
        reasons.append("features.parquet is missing")

    label = metadata.get("feature_schema_version")
    if label != expected_feature_schema_version:
        reasons.append(
            f"feature_schema_version {label!r} is not the configured extract label "
            f"{expected_feature_schema_version!r} (source_extract.yaml "
            "source_extract.feature_schema_version)"
        )

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        reasons.append("feature_columns is missing or empty")
    elif metadata.get("feature_set_fingerprint") not in (
        None,
        feature_set_fingerprint([str(column) for column in feature_columns]),
    ):
        reasons.append("feature_set_fingerprint does not match feature_columns")

    return reasons


def _score_snapshot(
    metadata_path: str | Path,
    snapshot_frame: pd.DataFrame,
    snapshot_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(feature matrix, entity types, scores, is_anomaly) of one bundle on a snapshot."""

    if "entity_type" not in snapshot_frame.columns:
        raise RetrainError("Snapshot features have no entity_type column.")

    feature_matrix = build_training_feature_matrix(
        snapshot_frame,
        snapshot_metadata=snapshot_metadata,
    )
    model = load_trained_model(read_bundle_paths(metadata_path)["model_path"])
    entity_types = snapshot_frame["entity_type"].astype(str).to_numpy()
    # A segmented model routes each row by entity_type to its own segment.
    scoring_input = (
        feature_matrix.assign(entity_type=entity_types)
        if isinstance(model, SegmentedIsolationForest)
        else feature_matrix
    )
    scores = np.asarray(model.decision_function(scoring_input), dtype=float)
    is_anomaly = np.asarray(model.predict(scoring_input)) == -1
    return feature_matrix, entity_types, scores, is_anomaly


def _expected_feature_spec(
    snapshot_frame: pd.DataFrame, snapshot_metadata: dict[str, Any]
) -> FeatureSpec:
    """The segment spec training will produce from this snapshot (as training resolves it)."""

    present = sorted(set(snapshot_frame["entity_type"].astype(str)))
    segments = {
        t: list(SEGMENT_FEATURE_COLUMNS[t]) for t in present if t in SEGMENT_FEATURE_COLUMNS
    }
    names = [
        str(name)
        for name in snapshot_metadata.get("feature_columns") or []
        if any(name in features for features in segments.values())
    ]
    return feature_spec_for(names, segments)


def resolve_comparison_entry(
    compare_with: str,
    *,
    entries: list[ModelRegistryEntry],
    model_name: str,
    expected_spec: FeatureSpec,
) -> ModelRegistryEntry:
    """The registered version to compare with; refused unless it has the same segment spec.

    Same segment spec is what makes the two versions rollback targets of each other.
    """

    entry = next(
        (e for e in entries if e.model_name == model_name and e.model_version == compare_with),
        None,
    )
    if entry is None:
        raise RetrainError(f"--compare-with {compare_with} is not registered for {model_name}")
    try:
        previous_spec = load_verified_feature_spec(entry)
    except RegistryError as exc:
        raise RetrainError(f"--compare-with {compare_with} cannot be loaded: {exc}") from exc
    difference = previous_spec.difference(expected_spec)
    if difference is not None:
        raise RetrainError(
            f"--compare-with {compare_with}: its segment spec is not the new model's "
            f"({difference}); a rollback between them would be refused"
        )
    return entry


def _snapshot_sha256(snapshot_metadata: dict[str, Any], training_root: str | Path) -> str | None:
    """content_sha256 of a model's training snapshot: from its metadata, else its directory."""

    recorded = snapshot_metadata.get("content_sha256")
    if recorded:
        return str(recorded)
    snapshot_id = snapshot_metadata.get("snapshot_id")
    root = Path(training_root)
    if not snapshot_id or not root.is_dir():
        return None
    matches = sorted(root.rglob(f"snapshot_id={snapshot_id}/features.parquet"))
    return content_sha256(read_training_snapshot(matches[-1])) if matches else None


def _flag_comparison(
    keys: np.ndarray, previous: np.ndarray, new: np.ndarray
) -> dict[str, Any]:
    rows = int(len(keys))
    flagged_previous = set(keys[previous].tolist())
    flagged_new = set(keys[new].tolist())
    both = flagged_previous & flagged_new
    union = flagged_previous | flagged_new
    return {
        "row_count": rows,
        "previous_flag_rate": float(previous.mean()) if rows else None,
        "new_flag_rate": float(new.mean()) if rows else None,
        "flagged_previous": len(flagged_previous),
        "flagged_new": len(flagged_new),
        "flagged_both": len(both),
        "jaccard": len(both) / len(union) if union else None,
    }


def build_comparison(
    *,
    previous_entry: ModelRegistryEntry,
    artifact: ModelArtifactMetadata,
    snapshot_frame: pd.DataFrame,
    snapshot_metadata: dict[str, Any],
    training_root: str | Path,
) -> dict[str, Any]:
    """Score the new snapshot with the previous and the new version and compare the flags.

    Evidence only (no gate): flag rates, and the Jaccard overlap of flagged entities
    (entity_type|entity_id), overall and per entity type.
    """

    new_spec = feature_spec_from_metadata(artifact.to_dict())
    difference = load_verified_feature_spec(previous_entry).difference(new_spec)
    if difference is not None:
        raise RetrainError(
            f"{artifact.model_version} and {previous_entry.model_version}: segment spec "
            f"differs ({difference}); the bundle is written but not registered"
        )

    previous_metadata = json.loads(
        read_bundle_paths(previous_entry.metadata_path)["metadata_path"].read_text(
            encoding="utf-8"
        )
    )
    previous_snapshot = previous_metadata.get("training_snapshot_metadata") or {}
    _, entity_types, _, previous_flags = _score_snapshot(
        previous_entry.metadata_path, snapshot_frame, snapshot_metadata
    )
    _, _, _, new_flags = _score_snapshot(artifact.metadata_path, snapshot_frame, snapshot_metadata)
    keys = np.asarray(
        [
            f"{t}|{e}"
            for t, e in zip(entity_types, snapshot_frame["entity_id"].astype(str), strict=True)
        ]
    )

    return {
        "previous_model_version": previous_entry.model_version,
        "new_model_version": artifact.model_version,
        "segment_spec_equal": True,
        "scored_snapshot_id": snapshot_metadata.get("snapshot_id"),
        "new_snapshot_content_sha256": snapshot_metadata.get("content_sha256")
        or content_sha256(snapshot_frame),
        "previous_snapshot_id": previous_snapshot.get("snapshot_id"),
        "previous_snapshot_content_sha256": _snapshot_sha256(previous_snapshot, training_root),
        "overall": _flag_comparison(keys, previous_flags, new_flags),
        "entity_types": {
            entity_type: _flag_comparison(
                keys[entity_types == entity_type],
                previous_flags[entity_types == entity_type],
                new_flags[entity_types == entity_type],
            )
            for entity_type in sorted(set(entity_types))
        },
    }


def build_evaluation_report(
    *,
    artifact: ModelArtifactMetadata,
    snapshot_frame: pd.DataFrame,
    snapshot_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Anomaly rate and score distribution of the trained model, overall and per entity type."""

    feature_matrix, entity_types, scores, is_anomaly = _score_snapshot(
        artifact.metadata_path, snapshot_frame, snapshot_metadata
    )

    present = sorted(set(entity_types))
    unexpected = [t for t in present if t not in EXPECTED_ENTITY_TYPES]
    per_entity_type = {
        entity_type: score_distribution(
            scores[entity_types == entity_type],
            is_anomaly[entity_types == entity_type],
        )
        for entity_type in (*EXPECTED_ENTITY_TYPES, *unexpected)
    }

    return {
        "report_type": "retrain_evaluation",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "model_name": artifact.model_name,
        "model_version": artifact.model_version,
        "feature_schema_version": artifact.feature_schema_version,
        "feature_set_fingerprint": artifact.feature_set_fingerprint,
        "feature_count": len(artifact.feature_names),
        "snapshot_id": snapshot_metadata.get("snapshot_id"),
        "row_counts": {
            "snapshot": snapshot_metadata.get("row_count"),
            "trained": int(len(feature_matrix)),
            "by_entity_type": {
                entity_type: stats["row_count"] for entity_type, stats in per_entity_type.items()
            },
        },
        "contamination": artifact.model_parameters["contamination"],
        "model_kind": artifact.model_kind,
        "segments": {
            entity_type: {
                "feature_names": segment["feature_names"],
                "feature_count": len(segment["feature_names"]),
            }
            for entity_type, segment in (artifact.segments or {}).items()
        },
        "score_definition": SCORE_DEFINITION,
        "overall": score_distribution(scores, is_anomaly),
        "entity_types": per_entity_type,
        "missing_entity_types": [t for t in EXPECTED_ENTITY_TYPES if t not in present],
    }


def retrain(
    *,
    snapshot: str | Path,
    model_version: str,
    training_root: str | Path = DEFAULT_TRAINING_ROOT,
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    artifact_root: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    source_extract_config_path: str | Path = DEFAULT_SOURCE_EXTRACT_CONFIG_PATH,
    compare_with: str | None = None,
) -> RetrainResult:
    """Train a candidate from a real-source snapshot, evaluate it, and register it.

    compare_with: a registered version with the same segment spec (refused before
    training otherwise); evaluation_report.json then gets a "comparison" section.
    """

    if not re.fullmatch(r"v\d+", model_version):
        raise RetrainError(f"model version must look like v003; got {model_version!r}")

    config = load_training_config(config_path, artifact_root_override=artifact_root)

    snapshot_dir = resolve_snapshot_dir(snapshot, training_root)
    metadata_path = snapshot_dir / "metadata.json"
    snapshot_metadata = _read_metadata(metadata_path)
    reasons = snapshot_refusal_reasons(
        snapshot_dir,
        snapshot_metadata,
        # The snapshot label describes the extracted columns, not the model's spec.
        expected_feature_schema_version=str(
            load_source_extract_config(source_extract_config_path)["feature_schema_version"]
        ),
    )
    if reasons:
        raise RetrainError(f"Snapshot {snapshot_dir} is not trainable: " + "; ".join(reasons))
    assert snapshot_metadata is not None

    # Refuse before training: the registry store must be usable and the version new.
    state = load_registry_state(registry_path=registry_path, model_name=config.model_name)
    ensure_database_initialized(state, config.model_name)
    if any(
        entry.model_name == config.model_name and entry.model_version == model_version
        for entry in state.entries
    ):
        raise RetrainError(
            f"{config.model_name} {model_version} is already registered ({state.source}); "
            "choose a new --model-version."
        )

    features_path = snapshot_dir / "features.parquet"
    snapshot_frame = read_training_snapshot(features_path)
    previous_entry = None
    if compare_with is not None:
        # Before training: the version to compare with must be rollback-compatible.
        previous_entry = resolve_comparison_entry(
            compare_with,
            entries=state.entries,
            model_name=config.model_name,
            expected_spec=_expected_feature_spec(snapshot_frame, snapshot_metadata),
        )

    # Strict tracking (PROJECT4_MLFLOW_REQUIRED) refuses here, before any file is written.
    with training_run(
        entrypoint="retrain", model_version=model_version, model_name=config.model_name
    ) as run:
        artifact = train_isolation_forest_from_snapshot(
            features_path=features_path,
            snapshot_metadata_path=metadata_path,
            model_version=model_version,
            config_path=config_path,
            artifact_root_override=artifact_root,
            registry_path=registry_path,
            training_run=run,
        )

        report = build_evaluation_report(
            artifact=artifact,
            snapshot_frame=snapshot_frame,
            snapshot_metadata=snapshot_metadata,
        )
        if previous_entry is not None:
            report["comparison"] = build_comparison(
                previous_entry=previous_entry,
                artifact=artifact,
                snapshot_frame=snapshot_frame,
                snapshot_metadata=snapshot_metadata,
                training_root=training_root,
            )
        bundle_paths = read_bundle_paths(artifact.metadata_path)
        report_path = bundle_paths["artifact_dir"] / EVALUATION_REPORT_NAME
        report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        _write_bytes_atomic(report_path, report_text.encode("utf-8"))
        run.log_evaluation(report=report, report_path=report_path)

        entry = register_model_from_artifacts(
            metadata_path=bundle_paths["metadata_path"],
            registry_path=registry_path,
            status="candidate",
            approved_for_prod=False,
            notes=(
                f"Candidate retrained from {snapshot_metadata.get('snapshot_id')}; "
                f"see {EVALUATION_REPORT_NAME}. Not promoted."
            ),
            replace_existing=False,
        )

    after = load_registry_state(registry_path=registry_path, model_name=config.model_name)

    return RetrainResult(
        artifact=artifact,
        registry_entry=entry,
        registry_source=state.source,
        active_model_version_before=state.active_model_version,
        active_model_version_after=after.active_model_version,
        evaluation_report_path=report_path,
        evaluation_report=report,
    )


def _format(value: Any, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def print_summary(result: RetrainResult) -> None:
    report = result.evaluation_report
    overall = report["overall"]

    print("OK: retrained candidate (not promoted)")
    print(
        f"model={result.artifact.model_name} {result.artifact.model_version} "
        f"status={result.registry_entry.status} registry={result.registry_source}"
    )
    print(
        f"snapshot_id={report['snapshot_id']} "
        f"feature_schema_version={report['feature_schema_version']} "
        f"feature_count={report['feature_count']}"
    )
    print(
        f"rows: snapshot={report['row_counts']['snapshot']} "
        f"trained={report['row_counts']['trained']}"
    )
    print(
        f"contamination={report['contamination']} "
        f"overall_anomaly_rate={_format(overall['anomaly_rate'], '.4f')}"
    )
    print(f"model_kind={report['model_kind']}")
    print(
        f"{'entity_type':<14}{'feats':>6}{'rows':>8}{'anom_rate':>11}{'p5':>10}{'p50':>10}{'p95':>10}"
    )
    for entity_type, stats in report["entity_types"].items():
        print(
            f"{entity_type:<14}"
            f"{report['segments'].get(entity_type, {}).get('feature_count', '-'):>6}"
            f"{stats['row_count']:>8}"
            f"{_format(stats['anomaly_rate'], '.4f'):>11}"
            f"{_format(stats['score_p5'], '.4f'):>10}"
            f"{_format(stats['score_p50'], '.4f'):>10}"
            f"{_format(stats['score_p95'], '.4f'):>10}"
        )
    print(
        f"active_model_version={result.active_model_version_after} "
        f"(before: {result.active_model_version_before})"
    )
    comparison = report.get("comparison")
    if comparison:
        print(
            f"compared with {comparison['previous_model_version']} on this snapshot "
            "(flag rates previous -> new, Jaccard of flagged entities):"
        )
        groups = {"overall": comparison["overall"], **comparison["entity_types"]}
        for group, stats in groups.items():
            print(
                f"  {group:<14}{_format(stats['previous_flag_rate'], '.4f'):>8} -> "
                f"{_format(stats['new_flag_rate'], '.4f'):<8}"
                f"jaccard={_format(stats['jaccard'], '.3f')}"
            )
    print(f"artifact_dir={result.artifact.artifact_dir}")
    print(f"evaluation_report={result.evaluation_report_path}")
    print(f"mlflow_run_id={result.artifact.mlflow_run_id}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a candidate model from a passed real-source snapshot, write "
            "evaluation_report.json, and register it as candidate. Never promotes."
        )
    )
    parser.add_argument(
        "--snapshot", required=True, help="Snapshot path, snapshot id, or 'latest'."
    )
    parser.add_argument("--model-version", required=True, help="New model version, e.g. v003.")
    parser.add_argument("--training-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--artifact-root", default=None, help="Optional artifact root override.")
    parser.add_argument(
        "--compare-with",
        default=None,
        help="A registered version with the same segment spec (e.g. the active one): "
        "evaluation_report.json then compares its flags with the new model's.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from anomaly_detection.logging_config import configure_logging

    configure_logging()
    args = parse_args(argv)

    try:
        result = retrain(
            snapshot=args.snapshot,
            model_version=args.model_version,
            training_root=args.training_root,
            config_path=args.config,
            artifact_root=args.artifact_root,
            compare_with=args.compare_with,
        )
    except (RetrainError, TrainingError, RegistryError, TrackingError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
