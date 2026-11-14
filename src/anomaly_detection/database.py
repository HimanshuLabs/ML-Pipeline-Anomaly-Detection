"""Project 4 PostgreSQL database: connection settings, schema, and row mapping.

Connection settings come only from PROJECT4_POSTGRES_* environment variables.
The password is never logged, printed, or included in reprs.

CLI:
    python -m anomaly_detection.database apply-schema
    python -m anomaly_detection.database check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = PROJECT_ROOT / "sql"

# Order matters: later files reference tables created by earlier ones.
SCHEMA_FILES = (
    "create_ml_schemas.sql",
    "create_feature_tables.sql",
    "create_model_registry_tables.sql",
    "create_prediction_tables.sql",
    "create_monitoring_tables.sql",
)
VALIDATION_QUERIES_FILE = "validation_queries.sql"
PROJECT4_SCHEMAS = ("ml", "monitoring", "audit")

ENV_HOST = "PROJECT4_POSTGRES_HOST"
ENV_PORT = "PROJECT4_POSTGRES_PORT"
ENV_DATABASE = "PROJECT4_POSTGRES_DB"
ENV_USER = "PROJECT4_POSTGRES_USER"
ENV_PASSWORD = "PROJECT4_POSTGRES_PASSWORD"
REQUIRED_ENV_VARS = (ENV_HOST, ENV_PORT, ENV_DATABASE, ENV_USER, ENV_PASSWORD)

DRIVER = "postgresql+psycopg2"

INSERTABLE_TABLES = frozenset(
    {
        "ml.feature_snapshots",
        "ml.training_datasets",
        "ml.feature_baselines",
        "ml.feature_current_stats",
        "ml.model_registry",
        "ml.model_metrics",
        "ml.batch_predictions",
        "ml.online_predictions",
        "monitoring.drift_events",
        "monitoring.alert_events",
        "audit.rollback_events",
    }
)
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class DatabaseConfigError(RuntimeError):
    """Raised when database settings are missing or invalid."""


@dataclass(frozen=True)
class DatabaseSettings:
    """Connection settings; the password is excluded from repr and str."""

    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)

    def url(self) -> URL:
        """SQLAlchemy URL; its string form masks the password."""

        return URL.create(
            DRIVER,
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


def load_settings(environ: Mapping[str, str] | None = None) -> DatabaseSettings | None:
    """Read PROJECT4_POSTGRES_* settings; None when any of them is unset or empty."""

    env = os.environ if environ is None else environ
    values = {name: env.get(name, "").strip() for name in REQUIRED_ENV_VARS}
    if not all(values.values()):
        return None

    try:
        port = int(values[ENV_PORT])
    except ValueError as exc:
        raise DatabaseConfigError(f"{ENV_PORT} must be an integer") from exc

    if not 1 <= port <= 65535:
        raise DatabaseConfigError(f"{ENV_PORT} must be between 1 and 65535")

    return DatabaseSettings(
        host=values[ENV_HOST],
        port=port,
        database=values[ENV_DATABASE],
        user=values[ENV_USER],
        password=env.get(ENV_PASSWORD, ""),
    )


def is_configured(environ: Mapping[str, str] | None = None) -> bool:
    """True when every PROJECT4_POSTGRES_* setting is present and valid."""

    try:
        return load_settings(environ) is not None
    except DatabaseConfigError:
        return False


def get_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Create a SQLAlchemy engine (psycopg2, as in source_extract). Does not connect."""

    resolved_settings = settings or load_settings()
    if resolved_settings is None:
        raise DatabaseConfigError(
            "Project 4 database is not configured; set " + ", ".join(REQUIRED_ENV_VARS)
        )

    return create_engine(resolved_settings.url(), pool_pre_ping=True, hide_parameters=True)


def schema_file_paths() -> list[Path]:
    """The DDL files in the order they must run."""

    return [SQL_DIR / name for name in SCHEMA_FILES]


def apply_schema(engine: Engine) -> list[str]:
    """Run every DDL file in order inside one transaction; safe to run repeatedly."""

    applied: list[str] = []
    with engine.begin() as connection:
        for path in schema_file_paths():
            connection.exec_driver_sql(path.read_text(encoding="utf-8"))
            applied.append(path.name)
            LOGGER.info("Applied %s", path.name)
    return applied


def check(engine: Engine) -> dict[str, Any]:
    """Connect and report the server version and the Project 4 tables."""

    with engine.connect() as connection:
        server_version = connection.execute(text("SHOW server_version")).scalar_one()
        tables = [
            f"{schema}.{table}"
            for schema, table in connection.execute(
                text(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema = ANY(:schemas) ORDER BY table_schema, table_name"
                ),
                {"schemas": list(PROJECT4_SCHEMAS)},
            )
        ]
    return {"connection": "ok", "server_version": server_version, "tables": tables}


def validation_statements() -> list[str]:
    """SQL statements of validation_queries.sql, without psql meta-commands."""

    sql_text = "\n".join(
        line
        for line in (SQL_DIR / VALIDATION_QUERIES_FILE).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("\\")
    )
    return [statement.strip() for statement in sql_text.split(";") if statement.strip()]


# --- row mapping ---------------------------------------------------------------
# Each function returns a row whose keys are column names of the target table.
# JSONB values are JSON-encoded strings so they bind as plain parameters.


def _as_mapping(record: Any) -> dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    return dict(record)


def model_registry_row(entry: Any) -> dict[str, Any]:
    """ml.model_registry row from a registry.ModelRegistryEntry."""

    values = _as_mapping(entry)
    return {
        "model_id": values["model_id"],
        "model_name": values["model_name"],
        "model_version": values["model_version"],
        "algorithm": values["algorithm"],
        "artifact_path": values["artifact_path"],
        "artifact_dir": values["artifact_dir"],
        "metadata_path": values["metadata_path"],
        "feature_schema_path": values["feature_schema_path"],
        "baseline_stats_path": values["baseline_stats_path"],
        "snapshot_id": values["snapshot_id"],
        "training_dataset_id": values["training_dataset_id"],
        "feature_schema_version": values["feature_schema_version"],
        "training_started_at": values["training_started_at_utc"],
        "training_finished_at": values["training_finished_at_utc"],
        "baseline_anomaly_rate": values["baseline_anomaly_rate"],
        "baseline_precision_proxy": values["baseline_precision_proxy"],
        "baseline_recall_proxy": values["baseline_recall_proxy"],
        "baseline_f1_proxy": values["baseline_f1_proxy"],
        "latency_p50_ms": values["latency_p50_ms"],
        "latency_p95_ms": values["latency_p95_ms"],
        "threshold_used": values["threshold_used"],
        "status": values["status"],
        "approved_for_prod": values["approved_for_prod"],
        "promoted_at": values["promoted_at_utc"],
        "archived_at": values["archived_at_utc"],
        "created_at": values["created_at_utc"],
        "notes": values["notes"],
    }


def online_prediction_row(record: Any) -> dict[str, Any]:
    """ml.online_predictions row from an online_inference.OnlinePredictionRecord."""

    values = _as_mapping(record)
    return {
        "prediction_id": values["prediction_id"],
        "model_name": values["model_name"],
        "model_version": values["model_version"],
        "dataset_snapshot_id": values["dataset_snapshot_id"],
        "training_dataset_id": values["training_dataset_id"],
        "feature_schema_version": values["feature_schema_version"],
        "entity_type": values["entity_type"],
        "entity_id": values["entity_id"],
        "request_timestamp": values["prediction_timestamp"],
        "anomaly_score": values["anomaly_score"],
        "is_anomaly": values["is_anomaly"],
        "threshold_used": values["threshold_used"],
        "drift_status": values["drift_status"],
        "feature_payload_hash": values["feature_payload_hash"],
        "latency_ms": values["latency_ms"],
        "prediction_status": values["prediction_status"],
        "error_message": values["error_message"],
    }


def batch_prediction_row(record: Any) -> dict[str, Any]:
    """ml.batch_predictions row from a batch_inference.BatchPredictionRecord."""

    values = _as_mapping(record)
    return {**values, "feature_payload": json.dumps(values["feature_payload"])}


def rollback_event_row(event: Any) -> dict[str, Any]:
    """audit.rollback_events row from a rollback.RollbackEvent."""

    return _as_mapping(event)


def drift_event_row(event: Any) -> dict[str, Any]:
    """monitoring.drift_events row from a drift.FeatureDriftEvent."""

    return _as_mapping(event)


def alert_event_row(event: Any) -> dict[str, Any]:
    """monitoring.alert_events row from an alerts.AlertEvent."""

    values = _as_mapping(event)
    return {**values, "details": json.dumps(values["details"])}


def insert_row(connection: Connection, table: str, row: Mapping[str, Any]) -> None:
    """Insert one mapped row into a known Project 4 table."""

    if table not in INSERTABLE_TABLES:
        raise ValueError(f"not a Project 4 table: {table}")

    columns = list(row)
    invalid = [column for column in columns if not _IDENTIFIER.match(column)]
    if invalid:
        raise ValueError(f"invalid column names: {invalid}")

    statement = text(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + column for column in columns)})"
    )
    connection.execute(statement, dict(row))


# --- CLI -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project 4 PostgreSQL schema tools.")
    parser.add_argument("command", choices=["apply-schema", "check"])
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except DatabaseConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if settings is None:
        print(
            "error: database not configured; set " + ", ".join(REQUIRED_ENV_VARS),
            file=sys.stderr,
        )
        return 2

    engine = get_engine(settings)
    try:
        if args.command == "apply-schema":
            for name in apply_schema(engine):
                print(f"applied {name}")
            return 0

        report = check(engine)
        print(f"connection {report['connection']}")
        print(f"server_version {report['server_version']}")
        for table in report["tables"]:
            print(f"table {table}")
        return 0
    except SQLAlchemyError as exc:
        detail = getattr(exc, "orig", None) or exc
        print(f"error: {type(exc).__name__}: {detail}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
