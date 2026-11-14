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
