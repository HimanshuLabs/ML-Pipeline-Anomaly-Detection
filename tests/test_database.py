"""Tests for the Project 4 PostgreSQL schema, settings, and row mapping.

Unit tests need no database. Tests marked ``postgres`` run only when
PROJECT4_TEST_DATABASE_URL points at a disposable PostgreSQL database; with
PROJECT4_REQUIRE_TEST_DATABASE=1 a missing URL fails instead of skipping.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anomaly_detection import database
from anomaly_detection.alerts import AlertEvent, create_alert_event
from anomaly_detection.batch_inference import BatchPredictionRecord
from anomaly_detection.drift import FeatureDriftEvent
from anomaly_detection.online_inference import OnlinePredictionRecord
from anomaly_detection.registry import ModelRegistryEntry, ModelStatus, load_model_registry
from anomaly_detection.rollback import RollbackEvent
from anomaly_detection.training_snapshot import (
    TrainingSnapshotConfig,
    create_training_snapshot,
    metadata_to_database_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "_registry" / "model_registry.json"
COLUMN_TYPES = "UUID|TEXT|BIGINT|INTEGER|NUMERIC|BOOLEAN|JSONB|DATE|TIMESTAMPTZ"
SECRET_PASSWORD = "do-not-print-this-password-7f3a"


# --- DDL parsing helpers ---------------------------------------------------------


def _table_blocks() -> list[tuple[str, str, str]]:
    """(file name, table, body) for every CREATE TABLE in schema order."""

    blocks = []
    for path in database.schema_file_paths():
        sql_text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"CREATE TABLE IF NOT EXISTS (\w+\.\w+) \((.*?)\n\);",
            sql_text,
            re.S,
        ):
            blocks.append((path.name, match.group(1), match.group(2)))
    return blocks


def _parse_columns(body: str) -> dict[str, str]:
    """Column name -> its full definition text (including following CHECK lines)."""

    columns: dict[str, str] = {}
    current: str | None = None
    for line in body.splitlines():
        column_match = re.match(rf"\s*([a-z_][a-z0-9_]*)\s+({COLUMN_TYPES})\b", line)
        if column_match:
            current = column_match.group(1)
            columns[current] = line
        elif line.strip().startswith("CONSTRAINT"):
            current = None
        elif current is not None:
            columns[current] += "\n" + line
    return columns


def _ddl() -> dict[str, dict[str, object]]:
    tables: dict[str, dict[str, object]] = {}
    for _file_name, table, body in _table_blocks():
        columns = _parse_columns(body)
        table_name = table.split(".")[1]
        constraints = set(re.findall(r"CONSTRAINT (\w+)", body))
        constraints |= {
            f"{table_name}_{column}_check"
            for column, definition in columns.items()
            if "CHECK" in definition
        }
        tables[table] = {"columns": list(columns), "constraints": constraints, "body": body}
    return tables


def _columns(table: str) -> set[str]:
    return set(_ddl()[table]["columns"])


# --- schema files ------------------------------------------------------------------


def test_schema_files_run_in_dependency_order() -> None:
    assert database.SCHEMA_FILES == (
        "create_ml_schemas.sql",
        "create_feature_tables.sql",
        "create_model_registry_tables.sql",
        "create_prediction_tables.sql",
        "create_monitoring_tables.sql",
    )
    assert all(path.exists() for path in database.schema_file_paths())

    defined: set[str] = set()
    for file_name, table, body in _table_blocks():
        for referenced in re.findall(r"REFERENCES (\w+\.\w+)", body):
            assert referenced in defined, f"{file_name}: {table} references {referenced} early"
        defined.add(table)


def test_every_index_references_existing_columns() -> None:
    tables = _ddl()
    for path in database.schema_file_paths():
        for table, column_list in re.findall(
            r"CREATE INDEX IF NOT EXISTS \w+\s+ON (\w+\.\w+)\s*\(([^)]*)\)",
            path.read_text(encoding="utf-8"),
        ):
            for column in column_list.split(","):
                column_name = column.strip().split()[0]
                assert column_name in tables[table]["columns"], f"{table}.{column_name}"


def test_every_schema_statement_is_rerunnable() -> None:
    for path in database.schema_file_paths():
        sql_text = path.read_text(encoding="utf-8")
        for statement in re.findall(
            r"^\s*CREATE (?:EXTENSION|SCHEMA|TABLE|INDEX)\b[^\n]*",
            sql_text,
            re.M,
        ):
            assert "IF NOT EXISTS" in statement, f"{path.name}: {statement.strip()}"


def test_validation_queries_reference_only_real_tables_columns_and_constraints() -> None:
    tables = _ddl()
    sql_text = (database.SQL_DIR / database.VALIDATION_QUERIES_FILE).read_text(encoding="utf-8")

    def _section(start: str, end: str) -> str:
        return sql_text.split(start, 1)[1].split(end, 1)[0]

    for schema, table in re.findall(
        r"\('(\w+)', '(\w+)'\)",
        _section("WITH expected_tables", "missing_tables AS"),
    ):
        assert f"{schema}.{table}" in tables

    for schema, table, constraint in re.findall(
        r"\('(\w+)', '(\w+)', '(\w+)'\)",
        _section("WITH expected_constraints", "missing_constraints AS"),
    ):
        assert constraint in tables[f"{schema}.{table}"]["constraints"], constraint

    for schema, table, column in re.findall(
        r"\('(\w+)', '(\w+)', '(\w+)'\)",
        _section("WITH expected_columns", "missing_columns AS"),
    ):
        assert column in tables[f"{schema}.{table}"]["columns"], f"{schema}.{table}.{column}"

    for table in re.findall(r"FROM ((?:ml|monitoring|audit)\.\w+)", sql_text):
        assert table in tables


def test_model_registry_status_check_matches_model_status() -> None:
    status_definition = _parse_columns(_ddl()["ml.model_registry"]["body"])["status"]

    assert set(re.findall(r"'(\w+)'", status_definition)) == {
        status.value for status in ModelStatus
    }


def test_snapshot_id_columns_accept_text_ids() -> None:
    for table, column in [
        ("ml.feature_snapshots", "snapshot_id"),
        ("ml.training_datasets", "snapshot_id"),
        ("ml.feature_baselines", "snapshot_id"),
        ("ml.feature_current_stats", "snapshot_id"),
        ("ml.model_registry", "snapshot_id"),
        ("ml.model_metrics", "dataset_snapshot_id"),
        ("ml.batch_predictions", "dataset_snapshot_id"),
        ("ml.online_predictions", "dataset_snapshot_id"),
    ]:
        definition = _parse_columns(_ddl()[table]["body"])[column]
        assert re.match(rf"\s*{column}\s+TEXT\b", definition), f"{table}.{column}"


# --- row mapping matches the DDL 1:1 -------------------------------------------------


def _sample_online_prediction() -> OnlinePredictionRecord:
    return OnlinePredictionRecord(
        prediction_id="8a7f9c3e-0b4e-4f0c-9b7a-1d2e3f4a5b6c",
        model_name="isolation_forest",
        model_version="v001",
        dataset_snapshot_id="real_source_20260610T060738Z",
        training_dataset_id=None,
        feature_schema_version="feature_schema_v001",
        entity_type="user",
        entity_id="user_1",
        prediction_timestamp="2026-06-10T10:00:00+00:00",
        anomaly_score=0.1,
        is_anomaly=False,
        threshold_used=0.0,
        drift_status="not_evaluated",
        feature_payload_hash="a" * 64,
        latency_ms=1.5,
    )


def _sample_batch_prediction() -> BatchPredictionRecord:
    return BatchPredictionRecord(
        prediction_id="9b8f9c3e-0b4e-4f0c-9b7a-1d2e3f4a5b6c",
        batch_run_id="1c8f9c3e-0b4e-4f0c-9b7a-1d2e3f4a5b6c",
        model_name="isolation_forest",
        model_version="v001",
        dataset_snapshot_id="real_source_20260610T060738Z",
        training_dataset_id=None,
        feature_schema_version="feature_schema_v001",
        entity_type="customer",
        entity_id="customer_1",
        source_project="project_2_3",
        source_table="marts.mart_customer_360",
        source_record_id=None,
        score_timestamp="2026-06-10T10:00:00+00:00",
        anomaly_score=-0.2,
        is_anomaly=True,
        threshold_used=0.0,
        feature_payload_hash="b" * 64,
        feature_payload={"feature_a": 1.0},
        inference_latency_ms=0.4,
    )


def _sample_rollback_event() -> RollbackEvent:
    return RollbackEvent(
        rollback_id="rollback_test_0001",
        model_name="isolation_forest",
        from_model_version="v002",
        to_model_version="v001",
        rollback_reason="test",
        triggered_by="pytest",
        triggered_at_utc="2026-06-10T10:00:00+00:00",
        validation_status="applied",
        dry_run=False,
        active_model_path="/tmp/active_model.yaml",
        rollback_event_path="/tmp/rollback_events.jsonl",
    )


def _sample_drift_event() -> FeatureDriftEvent:
    return FeatureDriftEvent(
        drift_event_id="2d8f9c3e-0b4e-4f0c-9b7a-1d2e3f4a5b6c",
        model_name="isolation_forest",
        model_version="v001",
        feature_schema_version="feature_schema_v001",
        feature_name="feature_a",
        feature_dtype="float64",
        baseline_mean=1.0,
        current_mean=1.5,
        mean_delta=0.5,
        mean_delta_percent=0.5,
        baseline_variance=0.1,
        current_variance=0.2,
        variance_delta=0.1,
        variance_delta_percent=1.0,
        mean_warning_threshold=0.15,
        mean_critical_threshold=0.30,
        variance_warning_threshold=0.25,
        variance_critical_threshold=0.50,
        drift_status="critical",
        detection_method="mean_variance_threshold",
        observation_window_start="2026-06-10T09:00:00+00:00",
        observation_window_end="2026-06-10T10:00:00+00:00",
        detected_at="2026-06-10T10:00:00+00:00",
        notes="test",
    )


def _sample_alert_event(drift_event_id: str) -> AlertEvent:
    return create_alert_event(
        alert_type="drift_critical",
        severity="critical",
        message="test alert",
        model_name="isolation_forest",
        model_version="v001",
        drift_event_id=drift_event_id,
        metric_name="feature_drift_delta_percent",
        metric_value=1.0,
        threshold_value=0.3,
        comparison_operator=">=",
        details={"feature_name": "feature_a"},
    )


@pytest.mark.parametrize(
    ("table", "record", "mapper"),
    [
        pytest.param(
            "ml.online_predictions",
            _sample_online_prediction(),
            database.online_prediction_row,
            id="online_prediction",
        ),
        pytest.param(
            "ml.batch_predictions",
            _sample_batch_prediction(),
            database.batch_prediction_row,
            id="batch_prediction",
        ),
        pytest.param(
            "audit.rollback_events",
            _sample_rollback_event(),
            database.rollback_event_row,
            id="rollback_event",
        ),
        pytest.param(
            "monitoring.drift_events",
            _sample_drift_event(),
            database.drift_event_row,
            id="drift_event",
        ),
        pytest.param(
            "monitoring.alert_events",
            _sample_alert_event("2d8f9c3e-0b4e-4f0c-9b7a-1d2e3f4a5b6c"),
            database.alert_event_row,
            id="alert_event",
        ),
    ],
)
def test_record_mappers_cover_every_field_with_a_real_column(table, record, mapper) -> None:
    row = mapper(record)

    assert set(row) <= _columns(table)
    assert len(row) == len(fields(record))


def test_model_registry_mapper_covers_every_entry_field() -> None:
    for entry in load_model_registry(REAL_REGISTRY_PATH):
        row = database.model_registry_row(entry)

        assert set(row) <= _columns("ml.model_registry")
        assert len(row) == len(fields(ModelRegistryEntry))
        assert row["snapshot_id"] == entry.snapshot_id


def test_metadata_to_database_records_keys_equal_table_columns(tmp_path: Path) -> None:
    from conftest import _sample_feature_frame

    metadata = create_training_snapshot(
        _sample_feature_frame(row_count=12),
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
        source_project_versions={"project1": "test", "project2_3": "test"},
        created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )

    records = metadata_to_database_records(metadata)

    assert set(records["ml.feature_snapshots"]) == _columns("ml.feature_snapshots")
    assert set(records["ml.training_datasets"]) == _columns("ml.training_datasets")
    assert records["ml.feature_snapshots"]["source_project"] == "project1+project2_3"
    assert json.loads(records["ml.training_datasets"]["feature_columns"]) == (
        metadata.feature_columns
    )


# --- settings, engine, CLI ---------------------------------------------------------


def _full_environ(**overrides: str) -> dict[str, str]:
    environ = {
        database.ENV_HOST: "127.0.0.1",
        database.ENV_PORT: "5435",
        database.ENV_DATABASE: "project4_ml",
        database.ENV_USER: "manshu",
        database.ENV_PASSWORD: SECRET_PASSWORD,
    }
    environ.update(overrides)
    return environ


def test_settings_are_parsed_from_environment() -> None:
    settings = database.load_settings(_full_environ())

    assert settings is not None
    assert (settings.host, settings.port, settings.database, settings.user) == (
        "127.0.0.1",
        5435,
        "project4_ml",
        "manshu",
    )
    assert settings.password == SECRET_PASSWORD
    assert database.is_configured(_full_environ()) is True


@pytest.mark.parametrize("missing", database.REQUIRED_ENV_VARS)
def test_is_configured_false_when_any_setting_is_missing_or_empty(missing: str) -> None:
    without = _full_environ()
    del without[missing]
    empty = _full_environ(**{missing: ""})

    assert database.load_settings(without) is None
    assert database.is_configured(without) is False
    assert database.is_configured(empty) is False


def test_invalid_port_is_not_configured_and_error_names_only_the_variable() -> None:
    environ = _full_environ(**{database.ENV_PORT: "not-a-port"})

    assert database.is_configured(environ) is False
    with pytest.raises(database.DatabaseConfigError) as error:
        database.load_settings(environ)
    assert "not-a-port" not in str(error.value)
    assert SECRET_PASSWORD not in str(error.value)


def test_password_never_appears_in_repr_str_url_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    settings = database.load_settings(_full_environ())
    assert settings is not None

    engine = database.get_engine(settings)
    try:
        for rendered in (repr(settings), str(settings), repr(settings.url()), str(engine.url)):
            assert SECRET_PASSWORD not in rendered
    finally:
        engine.dispose()

    for record in caplog.records:
        assert SECRET_PASSWORD not in record.getMessage()


def test_apply_schema_runs_files_in_order_in_one_transaction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    executed: list[str] = []
    transactions: list[int] = []

    class _FakeConnection:
        def exec_driver_sql(self, sql_text: str) -> None:
            executed.append(sql_text)

    class _FakeEngine:
        @contextmanager
        def begin(self):
            transactions.append(1)
            yield _FakeConnection()

    applied = database.apply_schema(_FakeEngine())

    assert applied == list(database.SCHEMA_FILES)
    assert len(transactions) == 1
    assert executed == [
        path.read_text(encoding="utf-8") for path in database.schema_file_paths()
    ]


def test_cli_refuses_to_run_without_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in database.REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    assert database.main(["check"]) == 2
    assert "not configured" in capsys.readouterr().err


def test_validation_statements_skip_psql_meta_commands() -> None:
    statements = database.validation_statements()

    assert len(statements) == 7
    assert not any(line.lstrip().startswith("\\") for s in statements for line in s.splitlines())


# --- integration (real PostgreSQL) -------------------------------------------------


@pytest.fixture(scope="module")
def postgres_engine():
    url = os.environ.get("PROJECT4_TEST_DATABASE_URL")
    if not url:
        if os.environ.get("PROJECT4_REQUIRE_TEST_DATABASE") == "1":
            pytest.fail("PROJECT4_REQUIRE_TEST_DATABASE=1 but PROJECT4_TEST_DATABASE_URL is unset")
        pytest.skip("PROJECT4_TEST_DATABASE_URL is not set")

    from sqlalchemy import create_engine

    engine = create_engine(url, hide_parameters=True)
    yield engine
    engine.dispose()


@pytest.mark.postgres
def test_apply_schema_is_idempotent_on_a_real_database(postgres_engine) -> None:
    database.apply_schema(postgres_engine)
    database.apply_schema(postgres_engine)

    report = database.check(postgres_engine)

    assert report["connection"] == "ok"
    assert set(report["tables"]) == set(database.INSERTABLE_TABLES)


@pytest.mark.postgres
def test_one_row_per_table_then_every_validation_query_passes(
    postgres_engine,
    tmp_path: Path,
) -> None:
    from conftest import _sample_feature_frame
    from sqlalchemy import text

    database.apply_schema(postgres_engine)

    snapshot = create_training_snapshot(
        _sample_feature_frame(row_count=12),
        snapshot_config=TrainingSnapshotConfig(output_root=tmp_path),
        source_project_versions={"project1": "test"},
        created_at_utc=datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC),
    )
    snapshot_rows = metadata_to_database_records(snapshot)

    def _registry_entry(version: str) -> ModelRegistryEntry:
        return ModelRegistryEntry(
            model_id=f"00000000-0000-4000-8000-00000000000{version[-1]}",
            model_name="isolation_forest",
            model_version=version,
            algorithm="IsolationForest",
            artifact_path=f"artifacts/models/isolation_forest/model_version={version}/model.joblib",
            artifact_dir=f"artifacts/models/isolation_forest/model_version={version}",
            metadata_path="metadata.json",
            feature_schema_path="feature_schema.json",
            baseline_stats_path="baseline_stats.json",
            snapshot_id=snapshot.snapshot_id,
            training_dataset_id=snapshot.training_dataset_id,
            feature_schema_version="feature_schema_v001",
            training_started_at_utc="2026-06-10T09:00:00+00:00",
            training_finished_at_utc="2026-06-10T09:05:00+00:00",
            baseline_anomaly_rate=0.05,
            baseline_precision_proxy=None,
            baseline_recall_proxy=None,
            baseline_f1_proxy=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            threshold_used=0.0,
            status="production" if version == "v001" else "rolled_back",
            approved_for_prod=True,
            promoted_at_utc="2026-06-10T10:00:00+00:00",
            archived_at_utc=None,
            created_at_utc="2026-06-10T09:06:00+00:00",
            notes=None,
        )

    drift_event = _sample_drift_event()
    connection = postgres_engine.connect()
    transaction = connection.begin()
    try:
        database.insert_row(
            connection, "ml.feature_snapshots", snapshot_rows["ml.feature_snapshots"]
        )
        database.insert_row(
            connection, "ml.training_datasets", snapshot_rows["ml.training_datasets"]
        )
        database.insert_row(
            connection,
            "ml.feature_baselines",
            {
                "snapshot_id": snapshot.snapshot_id,
                "training_dataset_id": snapshot.training_dataset_id,
                "model_version": "v001",
                "feature_schema_version": "feature_schema_v001",
                "feature_name": "feature_a",
                "feature_dtype": "float64",
                "baseline_mean": 1.0,
                "baseline_variance": 0.1,
                "baseline_row_count": 12,
            },
        )
        database.insert_row(
            connection,
            "ml.feature_current_stats",
            {
                "snapshot_id": snapshot.snapshot_id,
                "model_version": "v001",
                "feature_schema_version": "feature_schema_v001",
                "feature_name": "feature_a",
                "feature_dtype": "float64",
                "observation_window_start": "2026-06-10T09:00:00+00:00",
                "observation_window_end": "2026-06-10T10:00:00+00:00",
                "current_mean": 1.5,
                "current_variance": 0.2,
                "current_row_count": 12,
            },
        )
        for version in ("v001", "v002"):
            database.insert_row(
                connection,
                "ml.model_registry",
                database.model_registry_row(_registry_entry(version)),
            )
        database.set_active_pointer(
            connection,
            model_name="isolation_forest",
            model_version="v001",
            artifact_path="artifacts/models/isolation_forest/model_version=v001/model.joblib",
            updated_by="pytest",
            notes=None,
        )
        database.insert_row(
            connection,
            "ml.model_metrics",
            {
                "model_id": "00000000-0000-4000-8000-000000000001",
                "model_name": "isolation_forest",
                "model_version": "v001",
                "metric_scope": "baseline",
                "metric_name": "baseline_anomaly_rate",
                "metric_value": 0.05,
            },
        )
        batch = _sample_batch_prediction()
        online = _sample_online_prediction()
        database.insert_row(
            connection,
            "ml.batch_predictions",
            database.batch_prediction_row(
                {**batch.__dict__, "dataset_snapshot_id": snapshot.snapshot_id}
            ),
        )
        database.insert_row(
            connection,
            "ml.online_predictions",
            database.online_prediction_row(
                {**online.__dict__, "dataset_snapshot_id": snapshot.snapshot_id}
            ),
        )
        database.insert_row(
            connection, "monitoring.drift_events", database.drift_event_row(drift_event)
        )
        database.insert_row(
            connection,
            "monitoring.alert_events",
            database.alert_event_row(_sample_alert_event(drift_event.drift_event_id)),
        )
        database.insert_row(
            connection,
            "audit.rollback_events",
            database.rollback_event_row(_sample_rollback_event()),
        )

        for table in sorted(database.INSERTABLE_TABLES):
            count = connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            assert count >= 1, table

        for statement in database.validation_statements():
            result = connection.execute(text(statement))
            for row in result.mappings():
                for key in ("status", "overall_status"):
                    if key in row:
                        assert row[key] == "PASS", dict(row)
    finally:
        transaction.rollback()
        connection.close()


# --- stage 8a: shared registry and pointer -------------------------------------------


def test_real_source_snapshot_row_matches_feature_snapshots_columns() -> None:
    metadata_path = (
        REPO_ROOT / "artifacts" / "models" / "isolation_forest" / "model_version=v002"
    ) / "metadata.json"
    snapshot_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))[
        "training_snapshot_metadata"
    ]

    row = database.real_source_snapshot_row(snapshot_metadata)

    assert set(row) == _columns("ml.feature_snapshots")
    assert row["snapshot_id"] == "real_source_20260610T060738Z"
    assert row["source_project"] == "project_1+project_2_3"
    assert row["feature_count"] == 51
    for not_null in ("snapshot_name", "source_layer", "feature_schema_version", "row_count"):
        assert row[not_null] is not None


def test_snapshot_rows_exist_for_every_real_registry_entry() -> None:
    from anomaly_detection.registry import snapshot_rows_for_entry

    for entry in load_model_registry(REAL_REGISTRY_PATH):
        rows = snapshot_rows_for_entry(entry)

        assert rows, entry.model_version
        assert rows[0][0] == "ml.feature_snapshots"
        assert rows[0][1]["snapshot_id"] == entry.snapshot_id
        for table, row in rows:
            assert set(row) == _columns(table)


def test_registry_rows_round_trip_back_to_entries() -> None:
    import uuid
    from decimal import Decimal

    for entry in load_model_registry(REAL_REGISTRY_PATH):
        row = database.model_registry_row(entry)
        stored = {
            **row,
            "model_id": uuid.UUID(row["model_id"]),
            "artifact_uri": None,
            "created_by": "manshu",
        }
        for column in ("training_started_at", "training_finished_at", "promoted_at",
                       "archived_at", "created_at"):
            if stored[column] is not None:
                stored[column] = datetime.fromisoformat(stored[column])
        for column in ("baseline_anomaly_rate", "latency_p50_ms", "latency_p95_ms"):
            if stored[column] is not None:
                stored[column] = Decimal(repr(stored[column]))

        assert database.registry_entry_values(stored) == entry.to_dict()


@pytest.fixture()
def database_mode(postgres_engine, monkeypatch: pytest.MonkeyPatch):
    """Empty, schema-applied test database used as the registry store."""

    from sqlalchemy import text
    from sqlalchemy.engine import make_url

    url = make_url(os.environ["PROJECT4_TEST_DATABASE_URL"])
    monkeypatch.setenv(database.ENV_HOST, url.host or "127.0.0.1")
    monkeypatch.setenv(database.ENV_PORT, str(url.port or 5432))
    monkeypatch.setenv(database.ENV_DATABASE, url.database or "")
    monkeypatch.setenv(database.ENV_USER, url.username or "")
    monkeypatch.setenv(database.ENV_PASSWORD, url.password or "")

    database.apply_schema(postgres_engine)
    with postgres_engine.begin() as connection:
        connection.execute(
            text(f"TRUNCATE {', '.join(sorted(database.INSERTABLE_TABLES))} CASCADE")
        )
    return postgres_engine


@pytest.fixture()
def initialized_database(database_mode):
    """database_mode seeded by sync-from-files (v001, v002, pointer v002 at version 1)."""

    database.sync_from_files(database_mode)
    return database_mode


def _database_entries(engine) -> dict[str, ModelRegistryEntry]:
    with engine.connect() as connection:
        return {
            values["model_version"]: ModelRegistryEntry(**values)
            for values in database.read_registry_entries(connection)
        }


def _database_pointer(engine, model_name: str = "isolation_forest"):
    with engine.connect() as connection:
        return database.read_active_pointer(connection, model_name)


def _register_trained(trained_metadata_paths, versions) -> None:
    from anomaly_detection.registry import register_model_from_artifacts

    for version in versions:
        register_model_from_artifacts(metadata_path=trained_metadata_paths[version])


def _assert_mirror_matches_database(engine, isolate_repo_state: Path) -> None:
    import yaml

    mirror_registry = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    mirror_pointer = yaml.safe_load(
        (isolate_repo_state / "configs" / "active_model.yaml").read_text(encoding="utf-8")
    )
    assert {
        entry.model_version: entry.to_dict() for entry in load_model_registry(mirror_registry)
    } == {version: entry.to_dict() for version, entry in _database_entries(engine).items()}
    assert mirror_pointer["active_model_version"] == _database_pointer(engine)["model_version"]


@pytest.mark.postgres
def test_sync_from_files_is_idempotent_and_never_touches_the_files(
    database_mode,
    isolate_repo_state: Path,
) -> None:
    mirror_registry = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    mirror_pointer = isolate_repo_state / "configs" / "active_model.yaml"
    files_before = (mirror_registry.read_bytes(), mirror_pointer.read_bytes())

    first = database.sync_from_files(database_mode)
    second = database.sync_from_files(database_mode)

    assert first["registry_inserted"] == 2
    assert first["pointer_version"] == 1 and first["pointer_changed"] is True
    assert second["registry_unchanged"] == 2
    assert second["snapshot_rows_inserted"] == 0
    assert second["pointer_version"] == 1 and second["pointer_changed"] is False
    assert (mirror_registry.read_bytes(), mirror_pointer.read_bytes()) == files_before
    assert {
        version: entry.to_dict() for version, entry in _database_entries(database_mode).items()
    } == {entry.model_version: entry.to_dict() for entry in load_model_registry(mirror_registry)}


@pytest.mark.postgres
def test_promotion_and_rollback_commit_registry_pointer_and_event_together(
    initialized_database,
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from sqlalchemy import text

    from anomaly_detection.registry import promote_model_version
    from anomaly_detection.rollback import rollback_active_model

    _register_trained(trained_metadata_paths, ["v101", "v102"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    promote_model_version(model_name="isolation_forest", model_version="v102")

    pointer = _database_pointer(initialized_database)
    assert (pointer["model_version"], pointer["pointer_version"]) == ("v102", 3)
    _assert_mirror_matches_database(initialized_database, isolate_repo_state)

    event = rollback_active_model(rollback_reason="pytest", triggered_by="pytest")

    entries = _database_entries(initialized_database)
    pointer = _database_pointer(initialized_database)
    with initialized_database.connect() as connection:
        stored_event = connection.execute(
            text("SELECT validation_status FROM audit.rollback_events WHERE rollback_id = :id"),
            {"id": event.rollback_id},
        ).scalar_one()
    assert (entries["v101"].status, entries["v102"].status) == ("production", "rolled_back")
    assert (pointer["model_version"], pointer["pointer_version"]) == ("v101", 4)
    assert pointer["updated_by"] == "pytest"
    assert stored_event == "applied"
    _assert_mirror_matches_database(initialized_database, isolate_repo_state)


@pytest.mark.postgres
def test_forced_database_failure_changes_neither_database_nor_files(
    initialized_database,
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.registry import RegistryStorageError, promote_model_version

    _register_trained(trained_metadata_paths, ["v101", "v102"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    mirror_registry = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    mirror_pointer = isolate_repo_state / "configs" / "active_model.yaml"
    files_before = (mirror_registry.read_bytes(), mirror_pointer.read_bytes())
    entries_before = {
        version: entry.to_dict()
        for version, entry in _database_entries(initialized_database).items()
    }
    pointer_before = _database_pointer(initialized_database)

    def _failing_pointer_write(*_args, **_kwargs):
        raise RuntimeError("simulated failure inside the transaction")

    monkeypatch.setattr(database, "set_active_pointer", _failing_pointer_write)

    with pytest.raises(RegistryStorageError, match="nothing was changed"):
        promote_model_version(model_name="isolation_forest", model_version="v102")

    assert {
        version: entry.to_dict()
        for version, entry in _database_entries(initialized_database).items()
    } == entries_before
    assert _database_pointer(initialized_database) == pointer_before
    assert (mirror_registry.read_bytes(), mirror_pointer.read_bytes()) == files_before


@pytest.mark.postgres
def test_two_serving_processes_converge_on_a_new_pointer_version(
    initialized_database,
    trained_metadata_paths: dict[str, Path],
) -> None:
    import time as time_module

    from anomaly_detection.registry import promote_model_version
    from api.main import (
        AppState,
        PointerPoller,
        load_service_from_shared_pointer,
        swap_inference_service,
    )

    _register_trained(trained_metadata_paths, ["v101", "v102"])
    promote_model_version(model_name="isolation_forest", model_version="v101")

    processes = []
    for _ in range(2):
        holder = AppState()
        service, pointer_version = load_service_from_shared_pointer()
        swap_inference_service(
            holder, service, pointer_version=pointer_version, pointer_source="postgres"
        )
        poller = PointerPoller(holder, interval_seconds=0.1)
        poller.start()
        processes.append((holder, poller))

    try:
        assert {h.inference_service.active_model_info().model_version for h, _ in processes} == {
            "v101"
        }

        promote_model_version(model_name="isolation_forest", model_version="v102")

        deadline = time_module.monotonic() + 20
        while time_module.monotonic() < deadline and any(
            holder.inference_service.active_model_info().model_version != "v102"
            for holder, _ in processes
        ):
            time_module.sleep(0.1)
    finally:
        for _holder, poller in processes:
            poller.stop()

    for holder, _ in processes:
        assert holder.inference_service.active_model_info().model_version == "v102"
        assert holder.pointer_version == _database_pointer(initialized_database)["pointer_version"]


@pytest.mark.postgres
def test_poller_keeps_serving_when_the_database_becomes_unreachable(
    initialized_database,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.registry import promote_model_version
    from api.main import (
        AppState,
        PointerPoller,
        load_service_from_shared_pointer,
        swap_inference_service,
    )

    _register_trained(trained_metadata_paths, ["v101"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    holder = AppState()
    service, pointer_version = load_service_from_shared_pointer()
    swap_inference_service(
        holder, service, pointer_version=pointer_version, pointer_source="postgres"
    )

    monkeypatch.setenv(database.ENV_PORT, "1")  # nothing listens there: an outage

    poller = PointerPoller(holder, interval_seconds=30)
    assert poller.poll_once() is False
    assert holder.inference_service is service
    assert holder.inference_service.active_model_info().model_version == "v101"


# --- stage 8a follow-up: refuse writes to an uninitialized database ------------------

NOT_INITIALIZED = "Project 4 database is not initialized for isolation_forest"


def _write_attempts(trained_metadata_paths: dict[str, Path]):
    from anomaly_detection.registry import (
        promote_model,
        promote_model_version,
        register_model_from_artifacts,
    )
    from anomaly_detection.rollback import rollback_active_model

    return {
        "register": lambda: register_model_from_artifacts(
            metadata_path=trained_metadata_paths["v101"]
        ),
        "promote": lambda: promote_model_version(
            model_name="isolation_forest", model_version="v002"
        ),
        "status_change": lambda: promote_model(
            model_name="isolation_forest", model_version="v001", target_status="archived"
        ),
        "rollback": lambda: rollback_active_model(
            rollback_reason="pytest", triggered_by="pytest"
        ),
    }


def _mirror_files(isolate_repo_state: Path) -> tuple[bytes, bytes]:
    return (
        (isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)).read_bytes(),
        (isolate_repo_state / "configs" / "active_model.yaml").read_bytes(),
    )


@pytest.mark.parametrize(
    ("entries_kind", "pointer_version"),
    [
        pytest.param("none", None, id="empty_registry_no_pointer"),
        pytest.param("real", None, id="entries_but_no_pointer"),
        pytest.param("none", 1, id="pointer_but_empty_registry"),
    ],
)
@pytest.mark.parametrize("attempt", ["register", "promote", "status_change", "rollback"])
def test_writes_refuse_an_uninitialized_database_without_touching_anything(
    attempt: str,
    entries_kind: str,
    pointer_version: int | None,
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import registry, rollback
    from anomaly_detection.registry import RegistryState, RegistryStorageError

    entries = [] if entries_kind == "none" else load_model_registry(REAL_REGISTRY_PATH)
    fake_state = RegistryState(
        entries=entries,
        model_name="isolation_forest",
        active_model_version="v002" if pointer_version else None,
        pointer_version=pointer_version,
        source=registry.POINTER_SOURCE_POSTGRES,
    )

    def _fake_load_registry_state(**_kwargs):
        return fake_state

    def _no_database_access():
        raise AssertionError("an uninitialized database must not be written to")

    monkeypatch.setattr(registry, "database_enabled", lambda: True)
    monkeypatch.setattr(rollback, "database_enabled", lambda: True)
    monkeypatch.setattr(registry, "load_registry_state", _fake_load_registry_state)
    monkeypatch.setattr(rollback, "load_registry_state", _fake_load_registry_state)
    monkeypatch.setattr(registry, "_database", _no_database_access)
    files_before = _mirror_files(isolate_repo_state)

    with pytest.raises(RegistryStorageError, match=NOT_INITIALIZED) as error:
        _write_attempts(trained_metadata_paths)[attempt]()

    assert "sync-from-files" in str(error.value)
    assert _mirror_files(isolate_repo_state) == files_before


@pytest.mark.postgres
@pytest.mark.parametrize("attempt", ["register", "promote", "status_change", "rollback"])
def test_writes_refuse_an_empty_real_database_and_change_nothing(
    attempt: str,
    database_mode,
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from sqlalchemy import text

    from anomaly_detection.registry import RegistryStorageError

    files_before = _mirror_files(isolate_repo_state)

    with pytest.raises(RegistryStorageError, match=NOT_INITIALIZED):
        _write_attempts(trained_metadata_paths)[attempt]()

    with database_mode.connect() as connection:
        for table in sorted(database.INSERTABLE_TABLES):
            assert connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one() == 0
    assert _mirror_files(isolate_repo_state) == files_before
