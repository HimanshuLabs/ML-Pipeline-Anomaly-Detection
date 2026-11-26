"""Project 4 PostgreSQL database: connection settings, schema, and row mapping.

Connection settings come only from PROJECT4_POSTGRES_* environment variables.
The password is never logged, printed, or included in reprs.

CLI:
    python -m anomaly_detection.database apply-schema
    python -m anomaly_detection.database check
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
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
    "create_integration_views.sql",
)
INTEGRATION_SCHEMA = "integration"
INTEGRATION_VIEWS = (
    "integration.model_metadata",
    "integration.prediction_outputs",
    "integration.drift_validation_rollback",
)
PROJECT5_READER_ROLE = "project5_reader"
ENV_P5_READER_PASSWORD = "PROJECT4_P5_READER_PASSWORD"
# Non-superuser role for the API (lifespan, pointer poller, /predict, /admin/rollback),
# the registry CLI (register/promote/retire) and the drift job: exactly the tables they
# read and write, no DDL. Schema changes and batch scoring stay with the owner role.
PROJECT4_APP_ROLE = "project4_app"
ENV_APP_PASSWORD = "PROJECT4_APP_PASSWORD"
APP_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    # registry reads; upserts on register/promote/retire/rollback and their undo
    "ml.model_registry": ("SELECT", "INSERT", "UPDATE"),
    # pointer reads, poller version reads, SELECT ... FOR UPDATE lock, upsert
    "ml.active_model_pointer": ("SELECT", "INSERT", "UPDATE"),
    # snapshot rows a registered entry needs (INSERT ... ON CONFLICT DO NOTHING)
    "ml.feature_snapshots": ("INSERT",),
    "ml.training_datasets": ("INSERT",),
    # /predict and /predict/batch evidence
    "ml.online_predictions": ("INSERT",),
    # drift job input (batch scoring itself runs as the owner)
    "ml.batch_predictions": ("SELECT",),
    # drift job writes; the API reads the latest run per segment
    "monitoring.drift_events": ("SELECT", "INSERT"),
    # drift job critical alerts
    "monitoring.alert_events": ("INSERT",),
    # rollback events, and marking one failed when a committed rollback is undone
    "audit.rollback_events": ("SELECT", "INSERT", "UPDATE"),
}
SCRAM_ITERATIONS = 4096
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
        "ml.active_model_pointer",
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


@functools.lru_cache(maxsize=4)
def _shared_engine(settings: DatabaseSettings) -> Engine:
    return create_engine(settings.url(), pool_pre_ping=True, hide_parameters=True)


def get_shared_engine() -> Engine:
    """A pooled engine reused across calls (per-request writes); raises when unconfigured."""

    settings = load_settings()
    if settings is None:
        raise DatabaseConfigError(
            "Project 4 database is not configured; set " + ", ".join(REQUIRED_ENV_VARS)
        )
    return _shared_engine(settings)


def _insert_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Insert rows into one table in a single transaction; returns the row count."""

    if not rows:
        return 0
    with get_shared_engine().begin() as connection:
        for row in rows:
            insert_row(connection, table, row)
    return len(rows)


def insert_online_predictions(records: Sequence[Any]) -> int:
    """ml.online_predictions rows for one API request, in one transaction."""

    return _insert_rows("ml.online_predictions", [online_prediction_row(r) for r in records])


BATCH_PREDICTION_INSERT_ROWS = 5_000


def insert_batch_predictions(records: Sequence[Any]) -> int:
    """ml.batch_predictions rows for one batch run, in one transaction.

    Rows are mapped and sent in chunks (multi-row INSERTs), so a large run never
    holds every mapped row at once; a failure in any chunk rolls back the run.
    After the commit the table is analyzed, so the planner knows the new run.
    """

    if not records:
        return 0
    engine = get_shared_engine()
    with engine.begin() as connection:
        for start in range(0, len(records), BATCH_PREDICTION_INSERT_ROWS):
            insert_rows(
                connection,
                "ml.batch_predictions",
                [
                    batch_prediction_row(record)
                    for record in records[start : start + BATCH_PREDICTION_INSERT_ROWS]
                ],
            )
    _analyze_batch_predictions(engine)
    return len(records)


def _analyze_batch_predictions(engine: Engine) -> None:
    """Refresh the planner statistics of ml.batch_predictions after a run has committed.

    Its own transaction, after the run's, so the run's rows and their single xmin are
    unchanged. The rows are already stored, so a failure is logged, not raised.
    """

    try:
        with engine.begin() as connection:
            connection.execute(text("ANALYZE ml.batch_predictions"))
    except SQLAlchemyError as exc:
        LOGGER.warning(
            "ANALYZE ml.batch_predictions failed after the run committed: %s",
            type(exc).__name__,
        )


# Per-row hash of batch_inference.batch_run_row_hash, in SQL: the first 60 bits of
# sha256(entity_type TAB entity_id TAB feature_payload_hash).
BATCH_RUN_ROW_HASH_SQL = (
    "('x' || substr(encode(sha256(convert_to(entity_type || E'\\t' || entity_id || E'\\t' "
    "|| coalesce(feature_payload_hash, ''), 'UTF8')), 'hex'), 1, 15))::bit(60)::bigint"
)


def batch_run_fingerprints(model_name: str, model_version: str, row_count: int) -> dict[str, int]:
    """batch_run_id -> hash sum, for this model's runs with exactly row_count rows."""

    query = text(
        "SELECT batch_run_id::text AS batch_run_id, "
        f"sum({BATCH_RUN_ROW_HASH_SQL})::text AS hash_sum "
        "FROM ml.batch_predictions "
        "WHERE model_name = :model_name AND model_version = :model_version "
        "GROUP BY batch_run_id HAVING count(*) = :row_count"
    )
    params = {"model_name": model_name, "model_version": model_version, "row_count": row_count}
    with get_shared_engine().connect() as connection:
        rows = connection.execute(query, params).mappings().all()
    return {row["batch_run_id"]: int(row["hash_sum"]) for row in rows}


def batch_run_exists(batch_run_id: str) -> bool:
    """True when ml.batch_predictions holds any row of this run."""

    query = text(
        "SELECT EXISTS (SELECT 1 FROM ml.batch_predictions "
        "WHERE batch_run_id = CAST(:run AS uuid))"
    )
    with get_shared_engine().connect() as connection:
        return bool(connection.execute(query, {"run": batch_run_id}).scalar_one())


def insert_drift_events(events: Sequence[Any]) -> int:
    """monitoring.drift_events rows, in one transaction."""

    return _insert_rows("monitoring.drift_events", [drift_event_row(e) for e in events])


def insert_alert_events(events: Sequence[Any]) -> int:
    """monitoring.alert_events rows, in one transaction."""

    return _insert_rows("monitoring.alert_events", [alert_event_row(e) for e in events])


_BATCH_FEATURE_PAYLOADS_SQL = (
    "SELECT entity_type, feature_payload FROM ml.batch_predictions "
    "WHERE model_name = :model_name AND model_version = :model_version "
    "AND prediction_status = 'success' AND score_timestamp >= :since"
)


def read_batch_feature_payloads(
    *,
    model_name: str,
    model_version: str,
    since: datetime,
) -> list[dict[str, Any]]:
    """entity_type and feature_payload of successful batch predictions since a time."""

    with get_shared_engine().connect() as connection:
        rows = connection.execute(
            text(_BATCH_FEATURE_PAYLOADS_SQL),
            {"model_name": model_name, "model_version": model_version, "since": since},
        ).mappings()
        return [dict(row) for row in rows]


def iter_batch_feature_payloads(
    *,
    model_name: str,
    model_version: str,
    since: datetime,
    chunk_rows: int,
) -> Iterator[list[dict[str, Any]]]:
    """The rows of read_batch_feature_payloads, chunk_rows at a time (server-side cursor).

    cursor_tuple_fraction 1.0 plans the cursor like the one-piece query, so the rows
    come in the same order.
    """

    with get_shared_engine().connect() as connection:
        connection.execute(text("SET LOCAL cursor_tuple_fraction = 1.0"))
        result = connection.execution_options(
            stream_results=True, max_row_buffer=chunk_rows
        ).execute(
            text(_BATCH_FEATURE_PAYLOADS_SQL),
            {"model_name": model_name, "model_version": model_version, "since": since},
        ).mappings()
        for partition in result.partitions(chunk_rows):
            yield [dict(row) for row in partition]


def read_last_batch_prediction(*, model_name: str) -> tuple[datetime, str] | None:
    """(score_timestamp, model_version) of the model name's latest successful batch
    prediction, any version; None if there never was one."""

    with get_shared_engine().connect() as connection:
        row = connection.execute(
            text(
                "SELECT score_timestamp, model_version FROM ml.batch_predictions "
                "WHERE model_name = :model_name AND prediction_status = 'success' "
                "ORDER BY score_timestamp DESC LIMIT 1"
            ),
            {"model_name": model_name},
        ).first()
    return None if row is None else (row[0], str(row[1]))


def read_latest_drift_rows(*, model_name: str, model_version: str) -> list[dict[str, Any]]:
    """Drift events of the latest run (greatest detected_at) per entity_type."""

    with get_shared_engine().connect() as connection:
        rows = connection.execute(
            text(
                "SELECT d.model_name, d.model_version, d.entity_type, d.feature_name, "
                "d.drift_status, d.mean_delta, d.variance_delta, d.detected_at "
                "FROM monitoring.drift_events AS d "
                "WHERE d.model_name = :model_name AND d.model_version = :model_version "
                "AND d.detected_at = ("
                "  SELECT max(l.detected_at) FROM monitoring.drift_events AS l"
                "  WHERE l.model_name = d.model_name AND l.model_version = d.model_version"
                "  AND l.entity_type IS NOT DISTINCT FROM d.entity_type)"
            ),
            {"model_name": model_name, "model_version": model_version},
        ).mappings()
        return [dict(row) for row in rows]


def schema_file_paths() -> list[Path]:
    """The DDL files in the order they must run."""

    return [SQL_DIR / name for name in SCHEMA_FILES]


def scram_sha256_verifier(password: str, *, salt: bytes | None = None) -> str:
    """PostgreSQL SCRAM-SHA-256 verifier, so the plain password never reaches the server."""

    salt = salt if salt is not None else os.urandom(16)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, SCRAM_ITERATIONS)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()

    def _b64(value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    return (
        f"SCRAM-SHA-256${SCRAM_ITERATIONS}:{_b64(salt)}"
        f"${_b64(stored_key)}:{_b64(server_key)}"
    )


def configure_project5_reader(connection: Connection, password: str | None) -> str:
    """Create/update the read-only role Project 5 uses: SELECT on integration views only.

    With a password the role can log in (stored as a SCRAM verifier); without one
    it is NOLOGIN. Returns a short status line that never contains the password.
    """

    role = PROJECT5_READER_ROLE
    connection.exec_driver_sql(
        "DO $$ BEGIN "
        f"IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN "
        f"CREATE ROLE {role} NOLOGIN; END IF; END $$"
    )
    if password:
        connection.exec_driver_sql(
            f"ALTER ROLE {role} WITH LOGIN PASSWORD %(verifier)s",
            {"verifier": scram_sha256_verifier(password)},
        )
        status = f"role {role}: LOGIN (password from {ENV_P5_READER_PASSWORD})"
    else:
        connection.exec_driver_sql(f"ALTER ROLE {role} WITH NOLOGIN")
        status = f"role {role}: NOLOGIN ({ENV_P5_READER_PASSWORD} is not set)"

    # Explicit CONNECT: the role keeps connecting after sql/restrict_project4_ml_connect.sql
    # revokes CONNECT from PUBLIC (also on a fresh database or a re-created role).
    connection.exec_driver_sql(
        "DO $$ BEGIN EXECUTE 'GRANT CONNECT ON DATABASE ' "
        f"|| quote_ident(current_database()) || ' TO {role}'; END $$"
    )
    for schema in PROJECT4_SCHEMAS:
        connection.exec_driver_sql(f"REVOKE ALL ON ALL TABLES IN SCHEMA {schema} FROM {role}")
        connection.exec_driver_sql(f"REVOKE ALL ON SCHEMA {schema} FROM {role}")
    connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA {INTEGRATION_SCHEMA} TO {role}")
    for view in INTEGRATION_VIEWS:
        connection.exec_driver_sql(f"GRANT SELECT ON {view} TO {role}")
    return status


def configure_project4_app(connection: Connection, password: str | None) -> str:
    """Create/update the application role: exactly APP_TABLE_PRIVILEGES, no DDL.

    Not a superuser, cannot create databases or roles. With a password the role can
    log in (stored as a SCRAM verifier); without one it is NOLOGIN. Every privilege
    is revoked and granted again, so the result is exact on every run. Returns a
    short status line that never contains the password.
    """

    role = PROJECT4_APP_ROLE
    connection.exec_driver_sql(
        "DO $$ BEGIN "
        f"IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN "
        f"CREATE ROLE {role} NOLOGIN; END IF; END $$"
    )
    connection.exec_driver_sql(f"ALTER ROLE {role} WITH NOSUPERUSER NOCREATEDB NOCREATEROLE")
    if password:
        connection.exec_driver_sql(
            f"ALTER ROLE {role} WITH LOGIN PASSWORD %(verifier)s",
            {"verifier": scram_sha256_verifier(password)},
        )
        status = f"role {role}: LOGIN (password from {ENV_APP_PASSWORD})"
    else:
        connection.exec_driver_sql(f"ALTER ROLE {role} WITH NOLOGIN")
        status = f"role {role}: NOLOGIN ({ENV_APP_PASSWORD} is not set)"

    # Explicit CONNECT: still connects after PUBLIC's CONNECT is revoked.
    connection.exec_driver_sql(
        "DO $$ BEGIN EXECUTE 'GRANT CONNECT ON DATABASE ' "
        f"|| quote_ident(current_database()) || ' TO {role}'; END $$"
    )
    for schema in (*PROJECT4_SCHEMAS, INTEGRATION_SCHEMA):
        connection.exec_driver_sql(f"REVOKE ALL ON ALL TABLES IN SCHEMA {schema} FROM {role}")
        connection.exec_driver_sql(f"REVOKE ALL ON SCHEMA {schema} FROM {role}")
    for schema in PROJECT4_SCHEMAS:
        connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
    for table, privileges in APP_TABLE_PRIVILEGES.items():
        connection.exec_driver_sql(f"GRANT {', '.join(privileges)} ON {table} TO {role}")
    return status


def apply_schema(
    engine: Engine,
    *,
    reader_password: str | None = None,
    app_password: str | None = None,
) -> list[str]:
    """Run every DDL file in order and configure project4_app and project5_reader,
    in one transaction.

    Safe to run repeatedly. reader_password defaults to PROJECT4_P5_READER_PASSWORD,
    app_password to PROJECT4_APP_PASSWORD.
    """

    if reader_password is None:
        reader_password = os.environ.get(ENV_P5_READER_PASSWORD) or None
    if app_password is None:
        app_password = os.environ.get(ENV_APP_PASSWORD) or None

    applied: list[str] = []
    with engine.begin() as connection:
        for path in schema_file_paths():
            connection.exec_driver_sql(path.read_text(encoding="utf-8"))
            applied.append(path.name)
            LOGGER.info("Applied %s", path.name)
        applied.append(configure_project4_app(connection, app_password))
        applied.append(configure_project5_reader(connection, reader_password))
    return applied


def check(engine: Engine) -> dict[str, Any]:
    """Connect and report the server version, the Project 4 tables, and integration views."""

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
        views = [
            f"{schema}.{view}"
            for schema, view in connection.execute(
                text(
                    "SELECT table_schema, table_name FROM information_schema.views "
                    "WHERE table_schema = :schema ORDER BY table_name"
                ),
                {"schema": INTEGRATION_SCHEMA},
            )
        ]
    return {
        "connection": "ok",
        "server_version": server_version,
        "tables": tables,
        "views": views,
    }


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


def _checked_columns(row: Mapping[str, Any]) -> list[str]:
    columns = list(row)
    invalid = [column for column in columns if not _IDENTIFIER.match(column)]
    if invalid:
        raise ValueError(f"invalid column names: {invalid}")
    return columns


def insert_row(connection: Connection, table: str, row: Mapping[str, Any]) -> None:
    """Insert one mapped row into a known Project 4 table."""

    if table not in INSERTABLE_TABLES:
        raise ValueError(f"not a Project 4 table: {table}")

    columns = _checked_columns(row)
    statement = text(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + column for column in columns)})"
    )
    connection.execute(statement, dict(row))


def insert_rows(connection: Connection, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Insert mapped rows with the same columns into a known Project 4 table.

    One executemany of a Core INSERT, which SQLAlchemy sends as multi-row VALUES.
    """

    from sqlalchemy import column as sql_column
    from sqlalchemy import insert
    from sqlalchemy import table as sql_table

    if table not in INSERTABLE_TABLES:
        raise ValueError(f"not a Project 4 table: {table}")
    if not rows:
        return
    columns = _checked_columns(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"rows for {table} do not all have the same columns")

    schema, name = table.split(".", 1)
    target = sql_table(name, *(sql_column(c) for c in columns), schema=schema)
    connection.execute(insert(target), [dict(row) for row in rows])


# --- registry, snapshots, and the shared active pointer -----------------------------

REAL_SOURCE_SNAPSHOT_WRITER = "project4_source_extract"

_REGISTRY_TIMESTAMP_COLUMNS = {
    "training_started_at": "training_started_at_utc",
    "training_finished_at": "training_finished_at_utc",
    "promoted_at": "promoted_at_utc",
    "archived_at": "archived_at_utc",
    "created_at": "created_at_utc",
}


def real_source_snapshot_row(snapshot_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """ml.feature_snapshots row from a real-source snapshot's metadata.

    This is the shape source_extract.write_snapshot writes and training stores
    under metadata.json training_snapshot_metadata (v002 and later).
    """

    source_projects = sorted(snapshot_metadata.get("source_projects") or [])
    feature_columns = snapshot_metadata.get("feature_columns") or []
    return {
        "snapshot_id": snapshot_metadata["snapshot_id"],
        "snapshot_name": snapshot_metadata["snapshot_id"],
        "source_project": "+".join(source_projects) if source_projects else "unknown",
        "source_layer": snapshot_metadata.get("snapshot_type") or "real_source_extract",
        "source_tables": json.dumps(list(snapshot_metadata.get("source_tables") or [])),
        "source_date_start": None,
        "source_date_end": None,
        "feature_schema_version": snapshot_metadata["feature_schema_version"],
        "feature_count": len(feature_columns),
        "row_count": int(snapshot_metadata["row_count"]),
        "snapshot_path": None,
        "data_quality_status": snapshot_metadata.get("data_quality_status") or "pending",
        "validation_report_path": None,
        "created_at": snapshot_metadata.get("created_at"),
        "created_by": REAL_SOURCE_SNAPSHOT_WRITER,
        "notes": snapshot_metadata.get("notes"),
    }


def real_source_training_dataset_row(snapshot_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """ml.training_datasets row for a real-source snapshot that carries a training_dataset_id.

    Built only from the bundle's metadata.json training_snapshot_metadata (never the
    snapshot directory, which is not committed). feature_schema_version is the
    snapshot's extract label, as in its ml.feature_snapshots row; the model's own
    label lives in ml.model_registry.
    """

    sha256 = str(snapshot_metadata["content_sha256"])
    return {
        "training_dataset_id": snapshot_metadata["training_dataset_id"],
        "dataset_name": snapshot_metadata["snapshot_id"],
        "snapshot_id": snapshot_metadata["snapshot_id"],
        "dataset_version": sha256[:12],
        "feature_schema_version": snapshot_metadata["feature_schema_version"],
        "train_row_count": int(snapshot_metadata["row_count"]),
        "validation_row_count": 0,
        "test_row_count": 0,
        "feature_columns": json.dumps(list(snapshot_metadata.get("feature_columns") or [])),
        "label_column": None,
        "dataset_path": None,
        "data_quality_status": snapshot_metadata.get("data_quality_status") or "pending",
        "created_at": snapshot_metadata.get("created_at"),
        "created_by": REAL_SOURCE_SNAPSHOT_WRITER,
        "notes": f"content_sha256={sha256}",
    }


def snapshot_rows_for_training_metadata(
    snapshot_metadata: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Snapshot (and training dataset) rows from metadata.json training_snapshot_metadata.

    In FK order: ml.feature_snapshots, then ml.training_datasets. Real-source
    metadata without a training_dataset_id (v002-v004) gives the snapshot row only.
    """

    if snapshot_metadata.get("snapshot_type") == "real_source_extract":
        rows = [("ml.feature_snapshots", real_source_snapshot_row(snapshot_metadata))]
        if snapshot_metadata.get("training_dataset_id"):
            rows.append(
                ("ml.training_datasets", real_source_training_dataset_row(snapshot_metadata))
            )
        return rows

    from anomaly_detection.training_snapshot import (
        TrainingSnapshotMetadata,
        metadata_to_database_records,
    )

    records = metadata_to_database_records(TrainingSnapshotMetadata(**snapshot_metadata))
    return [
        ("ml.feature_snapshots", records["ml.feature_snapshots"]),
        ("ml.training_datasets", records["ml.training_datasets"]),
    ]


def insert_row_if_absent(connection: Connection, table: str, row: Mapping[str, Any]) -> bool:
    """Insert unless a row with the same key already exists. True when inserted."""

    if table not in INSERTABLE_TABLES:
        raise ValueError(f"not a Project 4 table: {table}")

    columns = _checked_columns(row)
    result = connection.execute(
        text(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({', '.join(':' + column for column in columns)}) "
            "ON CONFLICT DO NOTHING RETURNING 1"
        ),
        dict(row),
    )
    return result.first() is not None


def upsert_model_registry_row(connection: Connection, row: Mapping[str, Any]) -> str:
    """Insert or update one registry row by (model_name, model_version).

    Returns 'inserted', 'updated', or 'unchanged'.
    """

    columns = _checked_columns(row)
    updatable = [
        column for column in columns if column not in ("model_id", "model_name", "model_version")
    ]
    result = connection.execute(
        text(
            f"INSERT INTO ml.model_registry ({', '.join(columns)}) "
            f"VALUES ({', '.join(':' + column for column in columns)}) "
            "ON CONFLICT (model_name, model_version) DO UPDATE SET "
            + ", ".join(f"{column} = EXCLUDED.{column}" for column in updatable)
            + " WHERE ("
            + ", ".join(f"ml.model_registry.{column}" for column in updatable)
            + ") IS DISTINCT FROM ("
            + ", ".join(f"EXCLUDED.{column}" for column in updatable)
            + ") RETURNING (xmax = 0) AS inserted"
        ),
        dict(row),
    ).first()
    if result is None:
        return "unchanged"
    return "inserted" if result.inserted else "updated"


def _iso_utc(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def registry_entry_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """registry.ModelRegistryEntry field values from an ml.model_registry row."""

    values: dict[str, Any] = {}
    for column, value in row.items():
        if column in ("artifact_uri", "created_by"):
            continue
        if isinstance(value, Decimal):
            value = float(value)
        elif isinstance(value, uuid.UUID):
            value = str(value)
        values[_REGISTRY_TIMESTAMP_COLUMNS.get(column, column)] = _iso_utc(value)
    return values


def read_registry_entries(connection: Connection) -> list[dict[str, Any]]:
    """All registry rows as ModelRegistryEntry field values, oldest first."""

    rows = connection.execute(
        text(
            "SELECT * FROM ml.model_registry ORDER BY created_at, model_name, model_version"
        )
    ).mappings()
    return [registry_entry_values(row) for row in rows]


def read_active_pointer(connection: Connection, model_name: str) -> dict[str, Any] | None:
    """The shared active pointer row for model_name, or None."""

    row = connection.execute(
        text(
            "SELECT model_name, model_version, artifact_path, pointer_version, "
            "updated_at, updated_by, notes FROM ml.active_model_pointer "
            "WHERE model_name = :model_name"
        ),
        {"model_name": model_name},
    ).mappings().first()
    return dict(row) if row is not None else None


def lock_active_pointer(connection: Connection, model_name: str) -> int | None:
    """Lock the pointer row for this transaction; its pointer_version or None."""

    return connection.execute(
        text(
            "SELECT pointer_version FROM ml.active_model_pointer "
            "WHERE model_name = :model_name FOR UPDATE"
        ),
        {"model_name": model_name},
    ).scalar_one_or_none()


def set_active_pointer(
    connection: Connection,
    *,
    model_name: str,
    model_version: str,
    artifact_path: str,
    updated_by: str | None,
    notes: str | None,
    only_if_changed: bool = False,
) -> tuple[int, bool]:
    """Point model_name at model_version. Returns (pointer_version, changed).

    A new row starts at pointer_version 1; every change adds 1. With
    only_if_changed, an identical model_version/artifact_path is left as is.
    """

    statement = (
        "INSERT INTO ml.active_model_pointer "
        "(model_name, model_version, artifact_path, pointer_version, updated_by, notes) "
        "VALUES (:model_name, :model_version, :artifact_path, 1, :updated_by, :notes) "
        "ON CONFLICT (model_name) DO UPDATE SET "
        "model_version = EXCLUDED.model_version, artifact_path = EXCLUDED.artifact_path, "
        "pointer_version = ml.active_model_pointer.pointer_version + 1, "
        "updated_at = NOW(), updated_by = EXCLUDED.updated_by, notes = EXCLUDED.notes"
    )
    if only_if_changed:
        statement += (
            " WHERE (ml.active_model_pointer.model_version, ml.active_model_pointer.artifact_path)"
            " IS DISTINCT FROM (EXCLUDED.model_version, EXCLUDED.artifact_path)"
        )
    result = connection.execute(
        text(statement + " RETURNING pointer_version"),
        {
            "model_name": model_name,
            "model_version": model_version,
            "artifact_path": artifact_path,
            "updated_by": updated_by,
            "notes": notes,
        },
    ).scalar_one_or_none()
    if result is None:
        current = read_active_pointer(connection, model_name)
        return int(current["pointer_version"]), False
    return int(result), True


def read_pointer_version(engine: Engine, model_name: str) -> int | None:
    """Current pointer_version for model_name (a cheap poll)."""

    with engine.connect() as connection:
        version = connection.execute(
            text(
                "SELECT pointer_version FROM ml.active_model_pointer "
                "WHERE model_name = :model_name"
            ),
            {"model_name": model_name},
        ).scalar_one_or_none()
    return int(version) if version is not None else None


def mark_rollback_event_failed(connection: Connection, rollback_id: str) -> None:
    """Record that an applied rollback was undone after commit."""

    connection.execute(
        text(
            "UPDATE audit.rollback_events SET validation_status = 'failed' "
            "WHERE rollback_id = :rollback_id"
        ),
        {"rollback_id": rollback_id},
    )


def sync_from_files(
    engine: Engine,
    *,
    registry_path: Path | None = None,
    active_model_path: Path | None = None,
) -> dict[str, Any]:
    """Backfill the database from the committed registry JSON and pointer YAML.

    Idempotent, one transaction, never modifies the files.
    """

    from anomaly_detection import registry

    entries = registry.load_model_registry(registry_path or registry.DEFAULT_REGISTRY_PATH)
    pointer = registry.read_active_model_pointer(
        active_model_path or registry.DEFAULT_ACTIVE_MODEL_PATH
    )
    summary: dict[str, Any] = {
        "snapshot_rows_inserted": 0,
        "registry_inserted": 0,
        "registry_updated": 0,
        "registry_unchanged": 0,
    }

    with engine.begin() as connection:
        for entry in entries:
            for table, row in registry.snapshot_rows_for_entry(entry):
                summary["snapshot_rows_inserted"] += insert_row_if_absent(connection, table, row)
            outcome = upsert_model_registry_row(connection, model_registry_row(entry))
            summary[f"registry_{outcome}"] += 1

        if not pointer.active_model_version or not pointer.artifact_path:
            raise DatabaseConfigError("active model pointer file has no active_model_version")

        pointer_version, changed = set_active_pointer(
            connection,
            model_name=pointer.model_name,
            model_version=pointer.active_model_version,
            artifact_path=pointer.artifact_path,
            updated_by="sync-from-files",
            notes=pointer.notes,
            only_if_changed=True,
        )

    summary.update(
        {
            "pointer_model": pointer.model_name,
            "pointer_model_version": pointer.active_model_version,
            "pointer_version": pointer_version,
            "pointer_changed": changed,
        }
    )
    return summary


RESTRICT_CONNECT_FILE = "restrict_project4_ml_connect.sql"


def restrict_connect(engine: Engine) -> list[tuple[str, bool, bool]]:
    """Run sql/restrict_project4_ml_connect.sql (plain SQL) on the connected database.

    Grants CONNECT to project5_reader, revokes CONNECT and TEMPORARY from PUBLIC, and
    returns its final table: (login role, superuser, can connect). Re-runnable.
    """

    with engine.begin() as connection:
        result = connection.exec_driver_sql(
            (SQL_DIR / RESTRICT_CONNECT_FILE).read_text(encoding="utf-8")
        )
        return [(str(role), bool(superuser), bool(can)) for role, superuser, can in result]


# --- CLI -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from anomaly_detection.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Project 4 PostgreSQL schema tools.")
    parser.add_argument(
        "command", choices=["apply-schema", "check", "sync-from-files", "restrict-connect"]
    )
    parser.add_argument(
        "--if-empty",
        action="store_true",
        help="sync-from-files only: seed only when the model has no active pointer yet; "
        "a populated database is never overwritten (the seed job always passes it).",
    )
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
                print(name if name.startswith("role ") else f"applied {name}")
            return 0

        if args.command == "sync-from-files":
            if args.if_empty:
                from anomaly_detection.registry import default_model_name

                model_name = default_model_name()
                with engine.connect() as connection:
                    pointer = read_active_pointer(connection, model_name)
                if pointer is not None:
                    print(
                        f"skipped: the database already has an active pointer for {model_name} "
                        f"({pointer['model_version']}, pointer_version "
                        f"{pointer['pointer_version']}); nothing was overwritten"
                    )
                    return 0
            for key, value in sync_from_files(engine).items():
                print(f"{key} {value}")
            return 0

        if args.command == "restrict-connect":
            print(f"{'role':<24}{'superuser':<11}can_connect")
            for role, superuser, can_connect in restrict_connect(engine):
                print(f"{role:<24}{str(superuser).lower():<11}{str(can_connect).lower()}")
            return 0

        report = check(engine)
        print(f"connection {report['connection']}")
        print(f"server_version {report['server_version']}")
        for table in report["tables"]:
            print(f"table {table}")
        for view in report["views"]:
            print(f"view {view}")
        return 0
    except SQLAlchemyError as exc:
        detail = getattr(exc, "orig", None) or exc
        print(f"error: {type(exc).__name__}: {detail}", file=sys.stderr)
        return 1
    except (DatabaseConfigError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
