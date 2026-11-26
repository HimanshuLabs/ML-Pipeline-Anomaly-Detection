"""Fault injection: the share of synthetic faults the active model flags.

    python scripts/fault_injection.py [--active-model-path configs/active_model.yaml]
        [--snapshot <features.parquet>] [--seed 20261126]
        [--output-dir data/validation/fault_injection]

Reads the active pointer FILE, its model bundle and its training snapshot, read-only.
Per segment it copies a seeded 1% of the rows (at least 5), applies one documented fault
per feature family the segment has, and scores the clean and the faulted rows in memory
with the active model. It writes only its report (JSON and a short text table): no
database, no prediction sink, no batch run.

The model has no labels and the faults are synthetic, so the report gives, per segment
and fault type, the share of synthetic faults flagged, next to the clean flag rate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from anomaly_detection.batch_inference import (  # noqa: E402
    DEFAULT_ACTIVE_MODEL_PATH,
    ActiveModelArtifacts,
    BatchInferenceError,
    load_active_model_artifacts,
    prepare_feature_matrix,
    score_entity_matrix,
    segment_feature_names,
)

DEFAULT_SEED = 20261126
SAMPLE_FRACTION = 0.01
MIN_FAULTED_ROWS = 5
# A segmented model flags a row when its decision score is below 0.
FLAG_THRESHOLD = 0.0
MEASURE = "share of synthetic faults flagged"
TRAINING_ROOT = PROJECT_ROOT / "data" / "features" / "training"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "fault_injection"

# The money feature of each segment that has one (funnel_day has none).
MONEY_FEATURE = {
    "user": "avg_event_price",
    "customer": "lifetime_value",
    "campaign": "total_spend",
    "product": "product_revenue",
    "web_customer": "avg_event_value",
}


@dataclass(frozen=True)
class Fault:
    """One documented fault: the features it needs in a segment, and how it changes them."""

    family: str
    change: str
    needs: Callable[[str], tuple[str, ...]]
    apply: Callable[[pd.DataFrame, str], None]


def _money_feature(entity_type: str) -> tuple[str, ...]:
    return (MONEY_FEATURE[entity_type],) if entity_type in MONEY_FEATURE else ()


def _money_x10(matrix: pd.DataFrame, entity_type: str) -> None:
    name = MONEY_FEATURE[entity_type]
    matrix[name] = matrix[name] * 10


def _latency_feature(entity_type: str) -> tuple[str, ...]:
    return ("avg_api_latency_ms",)


def _latency_x20(matrix: pd.DataFrame, entity_type: str) -> None:
    matrix["avg_api_latency_ms"] = matrix["avg_api_latency_ms"] * 20


def _event_count_features(entity_type: str) -> tuple[str, ...]:
    return ("total_events", "purchase_count")


def _purchases_without_events(matrix: pd.DataFrame, entity_type: str) -> None:
    matrix["total_events"] = 0.0
    matrix["purchase_count"] = matrix["purchase_count"].clip(lower=1.0)


FAULTS: dict[str, Fault] = {
    "money_x10": Fault("money", "one money feature x10", _money_feature, _money_x10),
    "latency_x20": Fault("latency", "avg_api_latency_ms x20", _latency_feature, _latency_x20),
    "purchases_without_events": Fault(
        "event consistency",
        "total_events 0, purchase_count at least 1",
        _event_count_features,
        _purchases_without_events,
    ),
}


def snapshot_path(snapshot_id: str | None, training_root: Path = TRAINING_ROOT) -> Path:
    """features.parquet of a snapshot id under the training root (exactly one match)."""

    if not snapshot_id:
        raise BatchInferenceError("the active pointer names no dataset_snapshot_id")
    matches = sorted(
        training_root.glob(f"snapshot_date=*/snapshot_id={snapshot_id}/features.parquet")
    )
    if len(matches) != 1:
        raise BatchInferenceError(
            f"expected one features.parquet for snapshot {snapshot_id}; found {len(matches)}"
        )
    return matches[0]


def _snapshot_label(path: Path) -> str:
    """The snapshot id from its folder name, else the file name (never a machine path)."""

    prefix = "snapshot_id="
    folder = path.parent.name
    return folder[len(prefix) :] if folder.startswith(prefix) else path.name


def _flags(artifacts: ActiveModelArtifacts, matrix: pd.DataFrame, entity_type: str):
    return score_entity_matrix(artifacts, matrix, entity_type) < FLAG_THRESHOLD


def build_report(
    features: pd.DataFrame,
    artifacts: ActiveModelArtifacts,
    *,
    seed: int,
    snapshot_label: str,
) -> dict[str, Any]:
    """Per segment and fault type: the share of synthetic faults flagged."""

    segments = segment_feature_names(artifacts)
    if segments is None:
        raise BatchInferenceError("fault injection needs a segmented model")
    if "entity_type" not in features.columns:
        raise BatchInferenceError("the snapshot has no entity_type column")
    types = features["entity_type"].astype(str).to_numpy()
    unknown = sorted(set(types) - set(segments))
    if unknown:
        raise BatchInferenceError(f"the model has no segment for entity type(s): {unknown}")

    rng = np.random.default_rng(seed)
    report_segments: dict[str, Any] = {}
    for entity_type in sorted(set(types)):
        names = segments[entity_type]
        matrix = prepare_feature_matrix(features.loc[types == entity_type], names)
        matrix = matrix.reset_index(drop=True)
        clean = _flags(artifacts, matrix, entity_type)
        size = min(len(matrix), max(MIN_FAULTED_ROWS, math.ceil(SAMPLE_FRACTION * len(matrix))))
        rows = np.sort(rng.choice(len(matrix), size=size, replace=False))

        faults: dict[str, Any] = {}
        for fault_type, fault in FAULTS.items():
            needed = fault.needs(entity_type)
            if not needed or not set(needed) <= set(names):
                continue
            faulted = matrix.iloc[rows].copy()
            fault.apply(faulted, entity_type)
            flagged = int(_flags(artifacts, faulted, entity_type).sum())
            faults[fault_type] = {
                "family": fault.family,
                "change": fault.change,
                "features": list(needed),
                "flagged": flagged,
                "share_of_synthetic_faults_flagged": flagged / size,
            }

        report_segments[entity_type] = {
            "rows": len(matrix),
            "clean_flag_rate": float(clean.mean()),
            "faulted_rows": int(size),
            "clean_flagged_in_sample": int(clean[rows].sum()),
            "faults": faults,
        }

    return {
        "measure": MEASURE,
        "model_name": artifacts.model_name,
        "model_version": artifacts.model_version,
        "snapshot": snapshot_label,
        "seed": seed,
        "sample_fraction": SAMPLE_FRACTION,
        "min_faulted_rows": MIN_FAULTED_ROWS,
        "segments": report_segments,
    }


def render_table(report: dict[str, Any]) -> str:
    lines = [
        f"Fault injection: {report['model_name']} {report['model_version']}, "
        f"snapshot {report['snapshot']}, seed {report['seed']}",
        f"{MEASURE}: faulted rows flagged / faulted rows; clean flag rate: whole segment",
        "",
        f"{'segment':<14}{'fault':<26}{'faulted':>8}{'flagged':>9}"
        f"{'share flagged':>15}{'clean flag rate':>17}",
    ]
    for entity_type, segment in report["segments"].items():
        for fault_type, fault in segment["faults"].items():
            lines.append(
                f"{entity_type:<14}{fault_type:<26}{segment['faulted_rows']:>8}"
                f"{fault['flagged']:>9}{fault['share_of_synthetic_faults_flagged']:>15.4f}"
                f"{segment['clean_flag_rate']:>17.4f}"
            )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """The JSON report and the table; refuses to overwrite either."""

    stem = f"fault_injection_{report['model_version']}_seed{report['seed']}"
    paths = [output_dir / f"{stem}.json", output_dir / f"{stem}.txt"]
    existing = [path.name for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"report already exists, not overwritten: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = [json.dumps(report, indent=2, sort_keys=True) + "\n", render_table(report)]
    for path, content in zip(paths, contents, strict=True):
        with path.open("x", encoding="utf-8") as file:
            file.write(content)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--active-model-path", default=str(DEFAULT_ACTIVE_MODEL_PATH))
    parser.add_argument(
        "--snapshot",
        default=None,
        help="features.parquet to read (default: the pointer's training snapshot).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        artifacts = load_active_model_artifacts(Path(args.active_model_path))
        path = (
            Path(args.snapshot)
            if args.snapshot
            else snapshot_path(artifacts.dataset_snapshot_id)
        )
        report = build_report(
            pd.read_parquet(path),
            artifacts,
            seed=args.seed,
            snapshot_label=_snapshot_label(path),
        )
        write_report(report, Path(args.output_dir))
    except (BatchInferenceError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(render_table(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
