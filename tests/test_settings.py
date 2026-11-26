"""Settings templates list only what something reads (stage 13d removed the rest).

Every key here has a reader: the code (LOG_LEVEL, MLFLOW_*, PROJECT4_* via
database.py/api/main.py/tracking.py), docker compose interpolation (PROJECT4_ADMIN_TOKEN,
the Postgres/Grafana/MLflow passwords, PROJECT4_GRAFANA_ROOT_URL), or MLflow/GitPython
themselves (MLFLOW_DISABLE_TELEMETRY, GIT_PYTHON_REFRESH).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

KEPT_ENV_KEYS = sorted([
    "GIT_PYTHON_REFRESH",
    "LOG_LEVEL",
    "MLFLOW_DISABLE_TELEMETRY",
    "MLFLOW_TRACKING_URI",
    "PROJECT4_ADMIN_TOKEN",
    "PROJECT4_APP_PASSWORD",
    "PROJECT4_GRAFANA_ADMIN_PASSWORD",
    "PROJECT4_GRAFANA_ROOT_URL",
    "PROJECT4_MLFLOW_DB_PASSWORD",
    "PROJECT4_MLFLOW_REQUIRED",
    "PROJECT4_P5_READER_PASSWORD",
    "PROJECT4_POSTGRES_DB",
    "PROJECT4_POSTGRES_HOST",
    "PROJECT4_POSTGRES_PASSWORD",
    "PROJECT4_POSTGRES_PORT",
    "PROJECT4_POSTGRES_USER",
])
REMOVED_TRAINING_KEYS = ("snapshot_root", "model_registry_backend", "approval_required")


def test_env_example_holds_exactly_the_settings_something_reads() -> None:
    keys = [
        line.split("=", 1)[0]
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert sorted(keys) == KEPT_ENV_KEYS
    assert len(keys) == len(set(keys))


def test_model_config_training_section_holds_only_keys_the_loader_reads() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "model_config.yaml").read_text())

    for key in REMOVED_TRAINING_KEYS:
        assert key not in config["training"], key
    assert set(config["training"]) == {"artifact_root", "min_rows_per_segment"}


def test_unread_settings_template_with_project2_placeholders_does_not_exist() -> None:
    # configs/database.yaml.example held the batch lakehouse's warehouse placeholders and
    # had no reader; this pipeline's database settings come only from PROJECT4_POSTGRES_*
    # (database.py).
    assert not (REPO_ROOT / "configs" / "database.yaml.example").exists()
    assert "database.yaml" not in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
