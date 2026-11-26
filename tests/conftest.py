"""Shared test isolation for this pipeline.

Every test runs against a per-test mirror of the repository's mutable state
(configs/, the model registry, logs/, data/) instead of the real files. The
mirror's registry and active pointer come from tests/fixtures/repo_state, not
from the committed lifecycle state, so committing a promote, retire or rollback
cannot change what the tests see; only tests that check the committed files on
purpose read REAL_*. Real model artifacts stay read-only inputs. A guard fails
any test that changes or creates files under the real state directories or the
fixtures, and any test in which the API silently failed to persist prediction
evidence.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ACTIVE_MODEL_PATH = REPO_ROOT / "configs" / "active_model.yaml"
REAL_REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "_registry" / "model_registry.json"
# A realistic registry and pointer (v001 retired, v002 archived, v003 retired, v004
# production, v005 rolled_back) over the real, immutable bundles. They may diverge from
# the committed files; that is the point.
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"
FIXTURE_REGISTRY_PATH = FIXTURES_ROOT / "repo_state" / "model_registry.json"
FIXTURE_ACTIVE_MODEL_PATH = FIXTURES_ROOT / "repo_state" / "active_model.yaml"

# Directories whose contents tests must never write to.
REDIRECTED_ROOTS = (
    REPO_ROOT / "logs",
    REPO_ROOT / "configs",
    REPO_ROOT / "data",
    REPO_ROOT / "artifacts" / "models" / "_registry",
)
GUARDED_ROOTS = (
    REPO_ROOT / "logs",
    REPO_ROOT / "configs",
    REPO_ROOT / "data",
    REPO_ROOT / "artifacts",
    FIXTURES_ROOT,
)


def _project_modules() -> list[ModuleType]:
    import anomaly_detection

    modules = [
        importlib.import_module(f"anomaly_detection.{info.name}")
        for info in pkgutil.iter_modules(anomaly_detection.__path__)
    ]
    modules.append(importlib.import_module("api.main"))
    return modules


def _redirect(value: Any, mirror_root: Path) -> Any:
    if not isinstance(value, Path) or not value.is_absolute():
        return value

    for root in REDIRECTED_ROOTS:
        if value == root or root in value.parents:
            return mirror_root / value.relative_to(REPO_ROOT)

    return value


def _patch_function_defaults(
    monkeypatch: pytest.MonkeyPatch,
    function: Any,
    mirror_root: Path,
) -> None:
    defaults = function.__defaults__
    if defaults:
        redirected = tuple(_redirect(value, mirror_root) for value in defaults)
        if redirected != defaults:
            monkeypatch.setattr(function, "__defaults__", redirected)

    kwdefaults = function.__kwdefaults__
    if kwdefaults:
        redirected_kw = {key: _redirect(value, mirror_root) for key, value in kwdefaults.items()}
        if redirected_kw != kwdefaults:
            monkeypatch.setattr(function, "__kwdefaults__", redirected_kw)


def _snapshot_real_state() -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for root in GUARDED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[str(path.relative_to(REPO_ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return state


def _build_mirror(mirror_root: Path) -> None:
    shutil.copytree(REPO_ROOT / "configs", mirror_root / "configs")
    # The fixture pointer replaces the committed one; the other configs stay real.
    shutil.copy2(FIXTURE_ACTIVE_MODEL_PATH, mirror_root / "configs" / "active_model.yaml")
    registry_mirror = mirror_root / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    registry_mirror.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_REGISTRY_PATH, registry_mirror)
    for relative in ("logs/predictions", "logs/alerts", "data/features/training"):
        (mirror_root / relative).mkdir(parents=True)


@pytest.fixture(autouse=True)
def isolate_repo_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    caplog: pytest.LogCaptureFixture,
    request: pytest.FixtureRequest,
) -> Iterator[Path]:
    """Redirect all mutable repo paths to a per-test mirror and guard the real ones."""

    mirror_root = tmp_path_factory.mktemp("repo_state")
    _build_mirror(mirror_root)

    for module in _project_modules():
        for name, value in list(vars(module).items()):
            redirected = _redirect(value, mirror_root)
            if redirected is not value:
                monkeypatch.setattr(module, name, redirected)
            elif inspect.isfunction(value) and value.__module__ == module.__name__:
                _patch_function_defaults(monkeypatch, value, mirror_root)
            elif inspect.isclass(value) and value.__module__ == module.__name__:
                for member in vars(value).values():
                    if inspect.isfunction(member):
                        _patch_function_defaults(monkeypatch, member, mirror_root)

    # training resolves relative config paths (e.g. artifact_root) against PROJECT_ROOT.
    from anomaly_detection import training

    monkeypatch.setattr(
        training._resolve_path,
        "__kwdefaults__",
        {"base_dir": mirror_root},
    )

    # CWD-relative paths (api/main.py rollback pointer, registry helpers,
    # feature_validation default config) resolve inside the mirror.
    monkeypatch.chdir(mirror_root)

    # Admin endpoints are disabled unless a test opts in through admin_headers.
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    # Each test starts outside any persistence failure burst, so the
    # "Failed to persist" guard below always sees the first failure.
    import api.main

    monkeypatch.setattr(api.main, "PERSIST_FAILING", {"jsonl": False, "postgres": False})

    # File-backed registry and pointer unless a postgres test opts in explicitly;
    # this also keeps the API's pointer poller from ever starting in tests.
    for name in (
        "PROJECT4_POSTGRES_HOST",
        "PROJECT4_POSTGRES_PORT",
        "PROJECT4_POSTGRES_DB",
        "PROJECT4_POSTGRES_USER",
        "PROJECT4_POSTGRES_PASSWORD",
        "PROJECT4_POINTER_POLL_SECONDS",
        # MLflow tracking stays off unless a test opts in through mlflow_tracking.
        "MLFLOW_TRACKING_URI",
        "PROJECT4_MLFLOW_REQUIRED",
    ):
        monkeypatch.delenv(name, raising=False)

    before = _snapshot_real_state()

    yield mirror_root

    after = _snapshot_real_state()
    changed = sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    if changed:
        pytest.fail(f"test modified real repository state: {changed}")

    persist_failures = [
        record.getMessage()
        for record in caplog.records
        if record.name == "api.main" and "Failed to persist" in record.getMessage()
    ]
    if persist_failures and request.node.get_closest_marker("expect_persist_failure") is None:
        pytest.fail(f"API failed to persist prediction evidence: {persist_failures}")


def _sample_feature_frame(row_count: int) -> Any:
    import pandas as pd

    rows = []
    for index in range(row_count):
        multiplier = 7.0 if index >= row_count - 3 else 1.0
        rows.append(
            {
                "entity_id": f"user_{index:03d}",
                "feature_timestamp": pd.Timestamp("2026-06-01T00:00:00Z")
                + pd.Timedelta(minutes=index),
                "schema_version": "feature_schema_v001",
                "avg_cart_value_7d": 110.0 * multiplier + index,
                "event_count_1h": 4.0 * multiplier + (index % 5),
                "avg_api_latency_ms": 100.0 * multiplier + index,
                "fraud_score_avg": min(0.99, 0.03 * multiplier + (index % 4) * 0.01),
                "purchase_probability_delta": min(0.99, 0.07 * multiplier + (index % 5) * 0.01),
                "cart_abandonment_rate": min(0.99, 0.14 * multiplier + (index % 4) * 0.02),
                "campaign_roas": 2.2 * multiplier + (index % 6) * 0.1,
                "conversion_rate": min(0.99, 0.25 + (index % 5) * 0.02),
                "customer_lifetime_value": 700.0 * multiplier + index * 3.0,
                "discount_sensitivity": 0.30 * multiplier + (index % 4) * 0.04,
                "page_load_p95_ms": 230.0 * multiplier + index,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_metadata_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Three real models trained on identical features, with distinct source tables.

    Artifacts live outside the repository and are shared read-only by tests.
    """
    from datetime import UTC, datetime

    from anomaly_detection.training import train_isolation_forest_from_snapshot
    from anomaly_detection.training_snapshot import (
        TrainingSnapshotConfig,
        create_training_snapshot,
    )

    root = tmp_path_factory.mktemp("trained_models")
    metadata_paths: dict[str, Path] = {}

    for index, version in enumerate(("v101", "v102", "v103")):
        snapshot = create_training_snapshot(
            _sample_feature_frame(row_count=48 + index * 4),
            snapshot_config=TrainingSnapshotConfig(output_root=root / "snapshots" / version),
            source_tables=[f"source_table_{version}"],
            created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
        )
        artifact = train_isolation_forest_from_snapshot(
            features_path=snapshot.features_path,
            snapshot_metadata_path=snapshot.metadata_path,
            model_version=version,
            artifact_root_override=root / "artifacts",
            training_started_at_utc=datetime(2026, 6, 8, 12, 5, 0, tzinfo=UTC),
            registry_path=root / "training_label_registry.json",
        )
        metadata_paths[version] = Path(artifact.metadata_path)

    return metadata_paths


@pytest.fixture()
def mlflow_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Opt-in MLflow tracking to a sqlite store with artifacts under tmp_path (no server)."""

    store = tmp_path / "mlflow"
    store.mkdir()
    uri = f"sqlite:///{store / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


TEST_ADMIN_TOKEN = "pytest-admin-token-0123456789abcdef"


@pytest.fixture()
def admin_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Enable /admin/* with a test token and return the header that authorizes it."""

    monkeypatch.setenv("ADMIN_TOKEN", TEST_ADMIN_TOKEN)
    return {"X-Admin-Token": TEST_ADMIN_TOKEN}


@pytest.fixture()
def active_model_pointer() -> dict[str, Any]:
    """The fixture active model pointer: the one every per-test mirror starts from."""

    payload = yaml.safe_load(FIXTURE_ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture()
def active_model_version(active_model_pointer: dict[str, Any]) -> str:
    """The active model version from the fixture pointer."""

    return str(active_model_pointer["active_model_version"])


SEGMENTED_TEST_TYPES = ("user", "campaign", "product")


def real_source_test_frame(
    entity_types: tuple[str, ...] = SEGMENTED_TEST_TYPES,
    rows_per_type: int = 40,
    seed: int = 7,
) -> Any:
    """Real-source-shaped rows (all 52 columns) that pass validate_real_source_features."""

    import numpy as np
    import pandas as pd

    from anomaly_detection.source_extract import MODEL_FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    rows = []
    for entity_type in entity_types:
        for index in range(rows_per_type):
            row = {
                "source_project": "project_1" if entity_type == "user" else "project_2_3",
                "source_table": f"test.{entity_type}",
                "entity_type": entity_type,
                "entity_id": f"{entity_type}_{index:03d}",
                "source_event_timestamp": "2026-11-01T00:00:00+00:00",
            }
            row.update({column: float(rng.random()) for column in MODEL_FEATURE_COLUMNS})
            row["repeat_purchase_signal"] = float(index % 2)
            row["never_ordered"] = float(index % 3 == 0)
            row["extracted_at"] = "2026-11-15T00:00:00+00:00"
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def segmented_metadata_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Two segmented models (same segment spec) trained by the real training path.

    Artifacts live outside the repository and are shared read-only by tests.
    """
    from anomaly_detection.source_extract import write_snapshot
    from anomaly_detection.training import train_isolation_forest_from_snapshot

    root = tmp_path_factory.mktemp("segmented_models")
    metadata_paths: dict[str, Path] = {}
    for index, version in enumerate(("v201", "v202")):
        snapshot_root = root / "snapshots" / version
        snapshot = write_snapshot(
            real_source_test_frame(seed=7 + index),
            {
                "snapshot_type": "real_source_extract",
                "feature_schema_version": "feature_schema_v002",
                "output_snapshot_prefix": f"real_source_{version}",
                "output": {"root": str(snapshot_root)},
                "project_1": {"enabled": True, "schema": "public", "table": "user_features"},
                "project_2_3": {
                    "enabled": True,
                    "tables": {"campaign": {"schema": "marts", "table": "mart_campaign"}},
                },
            },
            output_root=snapshot_root,
        )
        artifact = train_isolation_forest_from_snapshot(
            features_path=snapshot.features_path,
            snapshot_metadata_path=snapshot.metadata_path,
            model_version=version,
            artifact_root_override=root / "artifacts",
            registry_path=root / "training_label_registry.json",
        )
        metadata_paths[version] = Path(artifact.metadata_path)
    return metadata_paths
