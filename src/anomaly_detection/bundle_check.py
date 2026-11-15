"""Image build check: the ACTIVE pointer's model bundle and the registry are present.

    python -m anomaly_detection.bundle_check --active-model configs/active_model.yaml

The runtime and tools targets of docker/Dockerfile run this at build time. It
reads the pointer that is being baked into the image (never a hardcoded
version), lists every missing bundle file, and then loads the bundle with the
serving loader so a bundle that cannot serve also fails the build.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVE_MODEL_PATH = PROJECT_ROOT / "configs" / "active_model.yaml"
DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "_registry" / "model_registry.json"
)
BUNDLE_FILES = ("model.joblib", "metadata.json", "feature_schema.json", "baseline_stats.json")


class BundleCheckError(RuntimeError):
    """The active model bundle (or the registry) is missing or cannot serve."""


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def check_active_bundle(
    active_model_path: str | Path = DEFAULT_ACTIVE_MODEL_PATH,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Return the active bundle directory; raise BundleCheckError naming what is wrong."""

    pointer_path = Path(active_model_path)
    if not pointer_path.is_file():
        raise BundleCheckError(f"active model pointer not found: {pointer_path}")

    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, dict):
        raise BundleCheckError(f"active model pointer is not a YAML mapping: {pointer_path}")

    version = pointer.get("active_model_version")
    artifact_path = pointer.get("artifact_path")
    if not version or not artifact_path:
        raise BundleCheckError(
            "active model pointer needs active_model_version and artifact_path"
        )

    model_path = Path(str(artifact_path))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    bundle_dir = model_path.parent

    missing = [bundle_dir / name for name in BUNDLE_FILES if not (bundle_dir / name).is_file()]
    if not Path(registry_path).is_file():
        missing.append(Path(registry_path))
    if missing:
        raise BundleCheckError(
            f"active model {version} is missing: " + ", ".join(_display(p) for p in missing)
        )

    from anomaly_detection.batch_inference import BatchInferenceError, load_active_model_artifacts

    try:
        artifacts = load_active_model_artifacts(pointer_path)
    except Exception as exc:  # noqa: BLE001 - any load failure fails the build
        detail = str(exc) if isinstance(exc, BatchInferenceError) else type(exc).__name__
        raise BundleCheckError(f"active model {version} does not load: {detail}") from exc

    if artifacts.model_path.parent.resolve() != bundle_dir.resolve():
        raise BundleCheckError(f"active model {version} resolved to an unexpected bundle")

    return bundle_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the active model bundle is present.")
    parser.add_argument("--active-model", default=str(DEFAULT_ACTIVE_MODEL_PATH))
    args = parser.parse_args(argv)

    try:
        bundle_dir = check_active_bundle(args.active_model)
    except BundleCheckError as exc:
        print(f"Missing or unusable active model bundle: {exc}", file=sys.stderr)
        return 1

    print(f"OK: active model bundle is present and loads: {_display(bundle_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
