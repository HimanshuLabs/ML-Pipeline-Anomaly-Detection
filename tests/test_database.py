"""Tests for the Project 4 PostgreSQL schema, settings, and row mapping.

Unit tests need no database. Tests marked ``postgres`` run only when
PROJECT4_TEST_DATABASE_URL points at a disposable PostgreSQL database; with
PROJECT4_REQUIRE_TEST_DATABASE=1 a missing URL fails instead of skipping.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import fields
from datetime import UTC, datetime, timedelta
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
from tests.conftest import FIXTURE_REGISTRY_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
# The committed registry: read on purpose by the tests that check every committed entry maps
# to database rows, and used to locate the registry inside the per-test mirror.
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
        "create_integration_views.sql",
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
        from_model_version="v009",
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
        def exec_driver_sql(self, sql_text: str, parameters=None) -> None:
            executed.append(sql_text)

    class _FakeEngine:
        @contextmanager
        def begin(self):
            transactions.append(1)
            yield _FakeConnection()

    applied = database.apply_schema(_FakeEngine(), reader_password="", app_password="")

    assert applied[: len(database.SCHEMA_FILES)] == list(database.SCHEMA_FILES)
    assert applied[len(database.SCHEMA_FILES):] == [
        "role project4_app: NOLOGIN (PROJECT4_APP_PASSWORD is not set)",
        "role project5_reader: NOLOGIN (PROJECT4_P5_READER_PASSWORD is not set)",
    ]
    assert len(transactions) == 1
    assert executed[: len(database.SCHEMA_FILES)] == [
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
        # Other tests leave rows behind (e.g. the seeded v001-v004); start from empty
        # tables inside this transaction, which the rollback below restores.
        connection.execute(
            text(f"TRUNCATE {', '.join(sorted(database.INSERTABLE_TABLES))} CASCADE")
        )
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
        for version in ("v001", "v009"):
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
    """database_mode seeded by sync-from-files from the mirror (pointer at version 1)."""

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


def _serve_own_single_model(trained_metadata_paths) -> tuple[str, list[str]]:
    """Register and promote this test's own single model (through the database).

    Returns its version and ordered feature names, so a test does not depend on
    which model the repository pointer makes live.
    """
    from anomaly_detection.registry import promote_model_version

    _register_trained(trained_metadata_paths, ["v101"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    metadata = json.loads(trained_metadata_paths["v101"].read_text(encoding="utf-8"))
    return "v101", list(metadata["feature_names"])


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

    registered_count = len(load_model_registry(mirror_registry))

    first = database.sync_from_files(database_mode)
    second = database.sync_from_files(database_mode)

    assert first["registry_inserted"] == registered_count
    assert first["pointer_version"] == 1 and first["pointer_changed"] is True
    assert second["registry_unchanged"] == registered_count
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
def test_promote_refuses_a_retired_version_in_the_database_and_changes_nothing(
    initialized_database,
    isolate_repo_state: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from anomaly_detection.registry import (
        RegistryError,
        promote_model_version,
        retire_model_version,
    )

    _register_trained(trained_metadata_paths, ["v101", "v102"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    retire_model_version(model_name="isolation_forest", model_version="v102", notes="pytest")
    mirror_registry = isolate_repo_state / REAL_REGISTRY_PATH.relative_to(REPO_ROOT)
    mirror_pointer = isolate_repo_state / "configs" / "active_model.yaml"
    files_before = (mirror_registry.read_bytes(), mirror_pointer.read_bytes())
    entries_before = {
        version: entry.to_dict()
        for version, entry in _database_entries(initialized_database).items()
    }
    pointer_before = _database_pointer(initialized_database)

    with pytest.raises(RegistryError, match=r"v102 is retired.*--allow-retired"):
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
            model_name="isolation_forest", model_version="v101"
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

    entries = [] if entries_kind == "none" else load_model_registry(FIXTURE_REGISTRY_PATH)
    fake_state = RegistryState(
        entries=entries,
        model_name="isolation_forest",
        active_model_version="v101" if pointer_version else None,
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


# --- stage 8b: integration views, project5_reader, persistence ------------------------

P5_METADATA_COLUMNS = (
    "source_project",
    "source_system",
    "source_entity",
    "source_record_id",
    "event_timestamp",
    "schema_version",
    "domain_tag",
)
VIEW_DOMAIN_TAGS = {
    "integration.model_metadata": "ml_model_registry",
    "integration.prediction_outputs": "anomaly_predictions",
    "integration.drift_validation_rollback": "ml_observability",
}
# integration.drift_validation_rollback columns before stage 11, in order (P5 contract).
DRIFT_VALIDATION_ROLLBACK_COLUMNS = (
    "source_project",
    "source_system",
    "source_entity",
    "source_record_id",
    "event_timestamp",
    "schema_version",
    "domain_tag",
    "recorded_at",
    "event_type",
    "model_name",
    "model_version",
    "status",
    "severity",
    "subject",
    "message",
    "metric_name",
    "metric_value",
    "threshold_value",
    "from_model_version",
    "to_model_version",
    "triggered_by",
)


def test_integration_views_expose_project5_metadata_columns() -> None:
    sql_text = (database.SQL_DIR / "create_integration_views.sql").read_text(encoding="utf-8")

    assert database.SCHEMA_FILES[-1] == "create_integration_views.sql"
    for view, domain_tag in VIEW_DOMAIN_TAGS.items():
        body = sql_text.split(f"CREATE OR REPLACE VIEW {view} AS", 1)[1].split(";", 1)[0]
        first_select = body.split("UNION ALL", 1)[0]
        for column in P5_METADATA_COLUMNS:
            assert f"AS {column}" in first_select, f"{view}.{column}"
        assert f"'{domain_tag}'::text AS domain_tag" in first_select
        assert f"'{view.split('.')[1]}'::text AS source_entity" in first_select
        assert "artifact_path" not in body


def test_scram_verifier_is_postgres_format_and_hides_the_password() -> None:
    verifier = database.scram_sha256_verifier(SECRET_PASSWORD, salt=b"0123456789abcdef")

    assert re.fullmatch(r"SCRAM-SHA-256\$4096:[A-Za-z0-9+/=]+\$[A-Za-z0-9+/=]+:[A-Za-z0-9+/=]+",
                        verifier)
    assert SECRET_PASSWORD not in verifier
    assert verifier == database.scram_sha256_verifier(SECRET_PASSWORD, salt=b"0123456789abcdef")
    assert verifier != database.scram_sha256_verifier("other", salt=b"0123456789abcdef")


def test_reader_role_setup_never_sends_or_returns_the_password() -> None:
    statements: list[tuple[str, object]] = []

    class _FakeConnection:
        def exec_driver_sql(self, sql_text: str, parameters=None) -> None:
            statements.append((sql_text, parameters))

    status = database.configure_project5_reader(_FakeConnection(), SECRET_PASSWORD)

    assert SECRET_PASSWORD not in status and "LOGIN" in status
    for sql_text, parameters in statements:
        assert SECRET_PASSWORD not in sql_text
        assert SECRET_PASSWORD not in repr(parameters)
    texts = [sql_text for sql_text, _ in statements]
    assert any(
        "GRANT CONNECT ON DATABASE" in text and "current_database()" in text
        and "project5_reader" in text for text in texts
    )  # explicit CONNECT: still connects after PUBLIC's CONNECT is revoked
    assert "GRANT USAGE ON SCHEMA integration TO project5_reader" in texts
    for view in database.INTEGRATION_VIEWS:
        assert f"GRANT SELECT ON {view} TO project5_reader" in texts
    assert not any(text.startswith("GRANT") and " ml." in text for text in texts)


def _count(engine, sql: str, params: dict | None = None) -> int:
    from sqlalchemy import text

    with engine.connect() as connection:
        return connection.execute(text(sql), params or {}).scalar_one()


@pytest.mark.postgres
def test_predict_and_predict_batch_write_online_rows_in_one_transaction(
    initialized_database,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    version, feature_names = _serve_own_single_model(trained_metadata_paths)
    payload = {name: 1.0 for name in feature_names}
    with TestClient(app) as client:
        single = client.post("/predict", json={"feature_payload": payload})
        batch = client.post("/predict/batch", json={"feature_payloads": [payload] * 3})

    assert single.status_code == 200 and batch.status_code == 200
    assert single.json()["model_version"] == version
    assert _count(
        initialized_database,
        "SELECT COUNT(*) FROM ml.online_predictions WHERE prediction_id = :id",
        {"id": single.json()["prediction_id"]},
    ) == 1
    batch_ids = [prediction["prediction_id"] for prediction in batch.json()["predictions"]]
    assert _count(
        initialized_database,
        "SELECT COUNT(*) FROM ml.online_predictions WHERE prediction_id::text = ANY(:ids)",
        {"ids": batch_ids},
    ) == 3
    assert _count(
        initialized_database,
        "SELECT COUNT(DISTINCT xmin::text) FROM ml.online_predictions "
        "WHERE prediction_id::text = ANY(:ids)",
        {"ids": batch_ids},
    ) == 1


@pytest.mark.postgres
def test_batch_inference_run_writes_all_rows_in_one_transaction(
    initialized_database,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    import pandas as pd

    from anomaly_detection.batch_inference import run_batch_inference

    version, feature_names = _serve_own_single_model(trained_metadata_paths)
    features = pd.DataFrame(
        [{"entity_id": f"customer_{index}", **dict.fromkeys(feature_names, float(index))}
         for index in range(4)]
    )

    predictions = run_batch_inference(features, output_path=tmp_path / "batch.jsonl")

    assert {prediction.model_version for prediction in predictions} == {version}

    assert _count(
        initialized_database,
        "SELECT COUNT(*) FROM ml.batch_predictions WHERE batch_run_id = :run",
        {"run": predictions[0].batch_run_id},
    ) == 4
    assert _count(
        initialized_database,
        "SELECT COUNT(DISTINCT xmin::text) FROM ml.batch_predictions WHERE batch_run_id = :run",
        {"run": predictions[0].batch_run_id},
    ) == 1


def _drift_and_alert(model_version: str):
    from dataclasses import replace

    drift_event = replace(_sample_drift_event(), model_version=model_version)
    alert_event = replace(
        _sample_alert_event(drift_event.drift_event_id), model_version=model_version
    )
    return drift_event, alert_event


@pytest.mark.postgres
def test_drift_and_alert_database_writers_insert_valid_rows(initialized_database) -> None:
    from anomaly_detection.alerts import insert_alert_events_database
    from anomaly_detection.drift import insert_drift_events_database

    # Events of whichever model the synced pointer makes live.
    drift_event, alert_event = _drift_and_alert(
        _database_pointer(initialized_database)["model_version"]
    )

    assert insert_drift_events_database([drift_event]) == 1
    assert insert_alert_events_database([alert_event]) == 1
    assert _count(initialized_database, "SELECT COUNT(*) FROM monitoring.drift_events") == 1
    assert _count(
        initialized_database,
        "SELECT COUNT(*) FROM monitoring.alert_events WHERE drift_event_id = :id",
        {"id": drift_event.drift_event_id},
    ) == 1


@pytest.mark.postgres
def test_integration_views_return_expected_rows_and_columns(
    initialized_database,
    tmp_path: Path,
    trained_metadata_paths: dict[str, Path],
) -> None:
    from dataclasses import replace

    import pandas as pd
    from sqlalchemy import text

    from anomaly_detection.alerts import insert_alert_events_database
    from anomaly_detection.batch_inference import run_batch_inference
    from anomaly_detection.drift import insert_drift_events_database

    version, feature_names = _serve_own_single_model(trained_metadata_paths)
    database.insert_online_predictions(
        [
            {
                **_sample_online_prediction().__dict__,
                "model_version": version,
                "dataset_snapshot_id": _database_entries(initialized_database)[
                    version
                ].snapshot_id,
            }
        ]
    )
    run_batch_inference(
        pd.DataFrame([{"entity_id": "customer_1", **dict.fromkeys(feature_names, 1.0)}]),
        output_path=tmp_path / "batch.jsonl",
    )
    drift_event, alert_event = _drift_and_alert(version)
    drift_event = replace(drift_event, entity_type="campaign")
    insert_drift_events_database([drift_event])
    insert_alert_events_database([alert_event])
    # Rollback events reference registered versions; use this test's own two models.
    _register_trained(trained_metadata_paths, ["v102"])
    with initialized_database.begin() as connection:
        database.insert_row(
            connection,
            "audit.rollback_events",
            database.rollback_event_row(
                replace(
                    _sample_rollback_event(),
                    from_model_version=version,
                    to_model_version="v102",
                )
            ),
        )

    with initialized_database.connect() as connection:
        models = connection.execute(
            text("SELECT * FROM integration.model_metadata ORDER BY model_version")
        ).mappings().all()
        predictions = connection.execute(
            text("SELECT * FROM integration.prediction_outputs")
        ).mappings().all()
        events = connection.execute(
            text("SELECT * FROM integration.drift_validation_rollback")
        ).mappings().all()

    for view, rows in (
        ("integration.model_metadata", models),
        ("integration.prediction_outputs", predictions),
        ("integration.drift_validation_rollback", events),
    ):
        assert rows, view
        for row in rows:
            for column in P5_METADATA_COLUMNS:
                assert row[column] is not None, f"{view}.{column}"
            assert row["source_project"] == "project4"
            assert row["source_system"] == "postgresql"
            assert row["source_entity"] == view.split(".")[1]
            assert row["domain_tag"] == VIEW_DOMAIN_TAGS[view]

    # Every registered version, and only the pointer's version active.
    active_version = _database_pointer(initialized_database)["model_version"]
    assert active_version == version
    assert [(row["model_version"], row["is_active"]) for row in models] == [
        (registered, registered == active_version)
        for registered in sorted(_database_entries(initialized_database))
    ]
    assert sorted(row["prediction_source"] for row in predictions) == ["batch", "online"]
    assert sorted(row["event_type"] for row in events) == ["alert", "drift", "rollback"]

    # entity_type is appended as the last column; every existing column keeps its place.
    for row in events:
        assert list(row.keys()) == [*DRIFT_VALIDATION_ROLLBACK_COLUMNS, "entity_type"]
    assert {row["event_type"]: row["entity_type"] for row in events} == {
        "drift": "campaign",
        "alert": None,
        "rollback": None,
    }


@pytest.mark.postgres
def test_project5_reader_reads_views_only(
    initialized_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import ProgrammingError

    reader_password = "project5-reader-test-password"
    applied = database.apply_schema(initialized_database, reader_password=reader_password)
    assert applied[-1].startswith("role project5_reader: LOGIN")
    assert reader_password not in " ".join(applied)

    reader_url = make_url(os.environ["PROJECT4_TEST_DATABASE_URL"]).set(
        username=database.PROJECT5_READER_ROLE, password=reader_password
    )
    reader_engine = create_engine(reader_url)
    try:
        with reader_engine.connect() as connection:
            for view in database.INTEGRATION_VIEWS:
                connection.execute(text(f"SELECT * FROM {view}")).fetchall()
        with reader_engine.connect() as connection, pytest.raises(
            ProgrammingError, match="permission denied"
        ):
            connection.execute(text("SELECT * FROM ml.model_registry")).fetchall()
    finally:
        reader_engine.dispose()
        database.apply_schema(initialized_database, reader_password="")


@pytest.mark.postgres
def test_apply_schema_twice_with_reader_password_stays_idempotent(database_mode) -> None:
    for _ in range(2):
        applied = database.apply_schema(database_mode, reader_password="idempotent-test-pw")
        assert applied[-1].startswith("role project5_reader: LOGIN")

    report = database.check(database_mode)
    assert set(report["tables"]) == set(database.INSERTABLE_TABLES)
    assert report["views"] == sorted(database.INTEGRATION_VIEWS)
    database.apply_schema(database_mode, reader_password="")


# --- stage 13g: CONNECT on the Project 4 database --------------------------------------

RESTRICT_CONNECT_SQL = REPO_ROOT / "sql" / "restrict_project4_ml_connect.sql"
CONNECT_READER_PASSWORD = "connect-test-reader-password"
MLFLOW_ROLE = "project4_mlflow"  # same name as the real MLflow role; test instance only
MLFLOW_ROLE_PASSWORD = "connect-test-mlflow-password"


def _run_restrict_connect_script(engine) -> list[tuple]:
    """Run the script as the superuser; returns its final SELECT (role, super, connect)."""

    with engine.begin() as connection:
        result = connection.exec_driver_sql(RESTRICT_CONNECT_SQL.read_text(encoding="utf-8"))
        return [tuple(row) for row in result.fetchall()]


def _connect_as(role: str, password: str, sql: str = "SELECT 1"):
    """Connect to the test database as role and run sql; returns the rows."""

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    url = make_url(os.environ["PROJECT4_TEST_DATABASE_URL"]).set(
        username=role, password=password
    )
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return connection.execute(text(sql)).fetchall()
    finally:
        engine.dispose()


def _database_acl(engine) -> str:
    return _count(engine, "SELECT datacl::text FROM pg_database WHERE datname = current_database()")


@pytest.fixture()
def connect_roles(database_mode):
    """project5_reader (LOGIN) and a LOGIN project4_mlflow role; restores PUBLIC after."""

    database.apply_schema(database_mode, reader_password=CONNECT_READER_PASSWORD)
    with database_mode.begin() as connection:
        connection.exec_driver_sql(
            "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = "
            f"'{MLFLOW_ROLE}') THEN CREATE ROLE {MLFLOW_ROLE} LOGIN; END IF; END $$"
        )
        connection.exec_driver_sql(
            f"ALTER ROLE {MLFLOW_ROLE} WITH LOGIN PASSWORD %(verifier)s",
            {"verifier": database.scram_sha256_verifier(MLFLOW_ROLE_PASSWORD)},
        )
    yield database_mode
    with database_mode.begin() as connection:
        connection.exec_driver_sql(
            "DO $$ BEGIN EXECUTE 'GRANT CONNECT, TEMPORARY ON DATABASE ' "
            "|| quote_ident(current_database()) || ' TO PUBLIC'; END $$"
        )
        connection.exec_driver_sql(f"DROP ROLE IF EXISTS {MLFLOW_ROLE}")
    database.apply_schema(database_mode, reader_password="")


@pytest.mark.postgres
def test_restrict_connect_keeps_project5_reader_and_the_owner_and_shuts_out_others(
    connect_roles,
) -> None:
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import OperationalError

    owner = make_url(os.environ["PROJECT4_TEST_DATABASE_URL"]).username
    # Before: every LOGIN role connects through PUBLIC.
    assert _connect_as(MLFLOW_ROLE, MLFLOW_ROLE_PASSWORD) == [(1,)]

    rows = _run_restrict_connect_script(connect_roles)
    print("\nrestrict_project4_ml_connect.sql final SELECT (role, superuser, can_connect):")
    for row in rows:
        print("  ", row)

    status = {role: (superuser, can_connect) for role, superuser, can_connect in rows}
    assert status[owner] == (True, True)
    assert status[database.PROJECT5_READER_ROLE] == (False, True)
    assert status[MLFLOW_ROLE] == (False, False)

    view_rows = _connect_as(
        database.PROJECT5_READER_ROLE, CONNECT_READER_PASSWORD,
        "SELECT count(*) FROM integration.model_metadata",
    )
    assert view_rows[0][0] >= 0
    with pytest.raises(OperationalError, match="permission denied for database"):
        _connect_as(MLFLOW_ROLE, MLFLOW_ROLE_PASSWORD)
    assert _count(connect_roles, "SELECT 1") == 1  # the superuser still connects


@pytest.mark.postgres
def test_restrict_connect_script_is_idempotent(connect_roles) -> None:
    first_rows = _run_restrict_connect_script(connect_roles)
    first_acl = _database_acl(connect_roles)

    second_rows = _run_restrict_connect_script(connect_roles)

    assert second_rows == first_rows
    assert _database_acl(connect_roles) == first_acl
    entries = first_acl.strip("{}").split(",")
    assert not any(entry.startswith("=") for entry in entries)  # no PUBLIC (empty grantee)
    assert any(entry.startswith("project5_reader=c/") for entry in entries)


@pytest.mark.postgres
def test_restrict_connect_refuses_and_changes_nothing_without_project5_reader(
    connect_roles,
) -> None:
    from sqlalchemy.exc import DBAPIError

    acl_before = _database_acl(connect_roles)
    with connect_roles.connect() as connection:
        transaction = connection.begin()
        connection.exec_driver_sql(f"DROP OWNED BY {database.PROJECT5_READER_ROLE}")
        connection.exec_driver_sql(f"DROP ROLE {database.PROJECT5_READER_ROLE}")
        with pytest.raises(DBAPIError, match="project5_reader does not exist"):
            connection.exec_driver_sql(RESTRICT_CONNECT_SQL.read_text(encoding="utf-8"))
        transaction.rollback()

    assert _database_acl(connect_roles) == acl_before


@pytest.mark.postgres
def test_apply_schema_after_the_revoke_gives_project5_reader_connect_again(
    connect_roles,
) -> None:
    from sqlalchemy.exc import OperationalError

    _run_restrict_connect_script(connect_roles)
    # A fresh or re-created reader role has no CONNECT once PUBLIC lost it.
    with connect_roles.begin() as connection:
        connection.exec_driver_sql(
            "DO $$ BEGIN EXECUTE 'REVOKE CONNECT ON DATABASE ' "
            "|| quote_ident(current_database()) || ' FROM project5_reader'; END $$"
        )
    with pytest.raises(OperationalError, match="permission denied for database"):
        _connect_as(database.PROJECT5_READER_ROLE, CONNECT_READER_PASSWORD)

    database.apply_schema(connect_roles, reader_password=CONNECT_READER_PASSWORD)

    assert _connect_as(database.PROJECT5_READER_ROLE, CONNECT_READER_PASSWORD) == [(1,)]


# --- stage 13e1: the non-superuser application role project4_app -----------------------

APP_ROLE = "project4_app"
APP_PASSWORD = "project4-app-role-test-password"
# What the API (lifespan, pointer poller, /predict, /admin/rollback), the registry
# CLI (register/promote/retire) and the drift job read and write; nothing else.
EXPECTED_APP_TABLE_PRIVILEGES = {
    "ml.model_registry": {"SELECT", "INSERT", "UPDATE"},
    "ml.active_model_pointer": {"SELECT", "INSERT", "UPDATE"},
    "ml.feature_snapshots": {"INSERT"},
    "ml.training_datasets": {"INSERT"},
    "ml.online_predictions": {"INSERT"},
    "ml.batch_predictions": {"SELECT"},
    "monitoring.drift_events": {"SELECT", "INSERT"},
    "monitoring.alert_events": {"INSERT"},
    "audit.rollback_events": {"SELECT", "INSERT", "UPDATE"},
}


def test_app_role_setup_never_sends_the_password_and_grants_exactly_its_tables() -> None:
    statements: list[tuple[str, object]] = []

    class _FakeConnection:
        def exec_driver_sql(self, sql_text: str, parameters=None) -> None:
            statements.append((sql_text, parameters))

    status = database.configure_project4_app(_FakeConnection(), SECRET_PASSWORD)

    assert SECRET_PASSWORD not in status and "LOGIN" in status
    for sql_text, parameters in statements:
        assert SECRET_PASSWORD not in sql_text and SECRET_PASSWORD not in repr(parameters)
    texts = [sql_text for sql_text, _ in statements]
    assert any("GRANT CONNECT ON DATABASE" in t and "current_database()" in t
               and APP_ROLE in t for t in texts)
    for schema in ("ml", "monitoring", "audit"):
        assert f"GRANT USAGE ON SCHEMA {schema} TO {APP_ROLE}" in texts
    table_grants = {
        t.split(" ON ")[1].split(" TO ")[0]: set(t[len("GRANT "):].split(" ON ")[0].split(", "))
        for t in texts
        if t.startswith("GRANT ") and " ON SCHEMA " not in t and "DATABASE" not in t
    }
    assert table_grants == EXPECTED_APP_TABLE_PRIVILEGES
    assert not any(t.startswith("GRANT") and "integration" in t for t in texts)


@pytest.fixture()
def app_mode(initialized_database, monkeypatch: pytest.MonkeyPatch):
    """The seeded test database, with the code connecting as project4_app (not the owner)."""

    database.apply_schema(initialized_database, app_password=APP_PASSWORD)
    monkeypatch.setenv(database.ENV_USER, APP_ROLE)
    monkeypatch.setenv(database.ENV_PASSWORD, APP_PASSWORD)
    yield initialized_database
    database.apply_schema(initialized_database, app_password="")


def _as_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy.engine import make_url

    url = make_url(os.environ["PROJECT4_TEST_DATABASE_URL"])
    monkeypatch.setenv(database.ENV_USER, url.username or "")
    monkeypatch.setenv(database.ENV_PASSWORD, url.password or "")


@pytest.mark.postgres
def test_project4_app_serves_predicts_promotes_and_rolls_back(
    app_mode,
    trained_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    from anomaly_detection.registry import promote_model_version, read_shared_pointer_version
    from anomaly_detection.rollback import rollback_active_model
    from api.main import app, state

    with database.get_shared_engine().connect() as connection:
        assert connection.execute(text("SELECT current_user")).scalar_one() == APP_ROLE

    _register_trained(trained_metadata_paths, ["v101", "v102"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    promote_model_version(model_name="isolation_forest", model_version="v102")
    event = rollback_active_model(rollback_reason="pytest", triggered_by="pytest")

    pointer = _database_pointer(app_mode)
    assert pointer["model_version"] == "v101"
    assert read_shared_pointer_version("isolation_forest") == pointer["pointer_version"]
    assert _count(
        app_mode,
        "SELECT COUNT(*) FROM audit.rollback_events WHERE rollback_id = :id "
        "AND validation_status = 'applied'",
        {"id": event.rollback_id},
    ) == 1

    metadata = json.loads(trained_metadata_paths["v101"].read_text(encoding="utf-8"))
    payload = {name: 1.0 for name in metadata["feature_names"]}
    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        health = client.get("/health").json()
        response = client.post("/predict", json={"feature_payload": payload})

    assert (health["pointer_source"], health["active_model_version"]) == ("postgres", "v101")
    assert response.status_code == 200
    assert _count(
        app_mode,
        "SELECT COUNT(*) FROM ml.online_predictions WHERE prediction_id = :id",
        {"id": response.json()["prediction_id"]},
    ) == 1


@pytest.mark.postgres
def test_project4_app_runs_the_drift_job(
    app_mode,
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from anomaly_detection.batch_inference import run_batch_inference
    from anomaly_detection.drift_job import main as drift_job_main
    from anomaly_detection.drift_job import run_drift_job
    from anomaly_detection.registry import promote_model_version, register_model_from_artifacts
    from tests.test_drift_job import SHIFTED_TYPE, _current_frame

    register_model_from_artifacts(metadata_path=segmented_metadata_paths["v202"])
    promote_model_version(model_name="isolation_forest", model_version="v202")

    # Nothing scored yet: nothing to evaluate is not an error.
    assert drift_job_main(["--source", "batch-predictions", "--hours", "24"]) == 0
    assert "not_evaluated" in capsys.readouterr().out

    _as_owner(monkeypatch)  # batch scoring is a tools job run as the owner
    artifact_dir = segmented_metadata_paths["v202"].parent
    run_batch_inference(_current_frame(artifact_dir), output_path=tmp_path / "batch.jsonl")
    monkeypatch.setenv(database.ENV_USER, APP_ROLE)
    monkeypatch.setenv(database.ENV_PASSWORD, APP_PASSWORD)

    result = run_drift_job(source="batch-predictions", hours=24, min_rows=30)

    assert result.segments[SHIFTED_TYPE].status == "critical"
    assert _count(
        app_mode, "SELECT COUNT(*) FROM monitoring.drift_events WHERE model_version = 'v202'"
    ) == result.drift_event_count > 0
    assert _count(
        app_mode, "SELECT COUNT(*) FROM monitoring.alert_events WHERE model_version = 'v202'"
    ) == result.alert_count > 0


def _serve_v202_as_app(segmented_metadata_paths: dict[str, Path]) -> Path:
    from anomaly_detection.registry import promote_model_version, register_model_from_artifacts

    register_model_from_artifacts(metadata_path=segmented_metadata_paths["v202"])
    promote_model_version(model_name="isolation_forest", model_version="v202")
    return segmented_metadata_paths["v202"].parent


def _score_as_owner(monkeypatch: pytest.MonkeyPatch, artifact_dir: Path, tmp_path: Path) -> None:
    from anomaly_detection.batch_inference import run_batch_inference
    from tests.test_drift_job import _current_frame

    _as_owner(monkeypatch)  # batch scoring is a tools job run as the owner
    run_batch_inference(_current_frame(artifact_dir), output_path=tmp_path / "batch.jsonl")
    monkeypatch.setenv(database.ENV_USER, APP_ROLE)
    monkeypatch.setenv(database.ENV_PASSWORD, APP_PASSWORD)


def _drift_job_records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "anomaly_detection.drift_job" and record.levelno == level
    ]


def _score_v201_then_serve_v202(
    monkeypatch: pytest.MonkeyPatch, segmented_metadata_paths: dict[str, Path], tmp_path: Path
) -> None:
    """v201 is batch-scored while active, then v202 is promoted and has no rows yet."""

    from anomaly_detection.registry import promote_model_version, register_model_from_artifacts

    for version in ("v201", "v202"):
        register_model_from_artifacts(metadata_path=segmented_metadata_paths[version])
    promote_model_version(model_name="isolation_forest", model_version="v201")
    _score_as_owner(monkeypatch, segmented_metadata_paths["v201"].parent, tmp_path)
    promote_model_version(model_name="isolation_forest", model_version="v202")


@pytest.mark.postgres
def test_drift_job_another_version_scored_in_the_window_logs_info_naming_it(
    app_mode,
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from anomaly_detection.drift_job import run_drift_job

    _score_v201_then_serve_v202(monkeypatch, segmented_metadata_paths, tmp_path)

    with caplog.at_level(logging.INFO, logger="anomaly_detection.drift_job"):
        result = run_drift_job(source="batch-predictions", hours=24)

    assert (result.model_version, result.status) == ("v202", "not_evaluated")
    assert _drift_job_records(caplog, logging.WARNING) == []
    infos = [m for m in _drift_job_records(caplog, logging.INFO) if "No current rows" in m]
    assert len(infos) == 1
    assert "no rows for v202 yet; last batch run used v201 at " in infos[0]


@pytest.mark.postgres
def test_drift_job_another_version_scored_before_the_window_warns_naming_it(
    app_mode,
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sqlalchemy import text

    from anomaly_detection.drift_job import run_drift_job

    _score_v201_then_serve_v202(monkeypatch, segmented_metadata_paths, tmp_path)
    with app_mode.connect() as connection:
        last = connection.execute(
            text("SELECT max(score_timestamp) FROM ml.batch_predictions")
        ).scalar_one()

    with caplog.at_level(logging.INFO, logger="anomaly_detection.drift_job"):
        result = run_drift_job(
            source="batch-predictions", hours=24, now=datetime.now(UTC) + timedelta(days=2)
        )

    assert (result.model_version, result.status) == ("v202", "not_evaluated")
    warnings = _drift_job_records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert (
        f"batch scoring stalled? last batch prediction at {last.astimezone(UTC).isoformat()} "
        "(model v201), none in the last 24h"
    ) in warnings[0]


@pytest.mark.postgres
def test_drift_job_never_scored_model_logs_info_not_a_warning(
    app_mode,
    segmented_metadata_paths: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    from anomaly_detection.drift_job import run_drift_job

    _serve_v202_as_app(segmented_metadata_paths)

    with caplog.at_level(logging.INFO, logger="anomaly_detection.drift_job"):
        result = run_drift_job(source="batch-predictions", hours=24)

    assert result.status == "not_evaluated"
    assert _drift_job_records(caplog, logging.WARNING) == []
    assert any("No current rows" in m for m in _drift_job_records(caplog, logging.INFO))


@pytest.mark.postgres
def test_drift_job_scored_only_before_the_window_warns_with_the_last_timestamp(
    app_mode,
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture,
) -> None:
    from sqlalchemy import text

    from anomaly_detection.drift_job import print_summary, run_drift_job

    _score_as_owner(monkeypatch, _serve_v202_as_app(segmented_metadata_paths), tmp_path)
    with app_mode.connect() as connection:
        last = connection.execute(
            text("SELECT max(score_timestamp) FROM ml.batch_predictions")
        ).scalar_one()

    # Two days later: the last 24 hours are empty, the earlier scoring is not.
    with caplog.at_level(logging.INFO, logger="anomaly_detection.drift_job"):
        result = run_drift_job(
            source="batch-predictions", hours=24, now=datetime.now(UTC) + timedelta(days=2)
        )

    assert result.status == "not_evaluated" and result.segments == {}
    warnings = _drift_job_records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "stalled?" in warnings[0] and "none in the last 24h" in warnings[0]
    assert f"last batch prediction at {last.astimezone(UTC).isoformat()}" in warnings[0]
    print_summary(result)
    assert capsys.readouterr().out.startswith("OK: nothing to evaluate for model=isolation_forest")
    assert _count(app_mode, "SELECT COUNT(*) FROM monitoring.drift_events") == 0


@pytest.mark.postgres
def test_drift_job_scored_in_the_window_is_evaluated_without_a_warning(
    app_mode,
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from anomaly_detection.drift_job import run_drift_job
    from tests.test_drift_job import SHIFTED_TYPE

    _score_as_owner(monkeypatch, _serve_v202_as_app(segmented_metadata_paths), tmp_path)

    with caplog.at_level(logging.INFO, logger="anomaly_detection.drift_job"):
        result = run_drift_job(source="batch-predictions", hours=24, min_rows=30)

    assert result.status == "evaluated"
    assert result.segments[SHIFTED_TYPE].status == "critical"
    assert _drift_job_records(caplog, logging.WARNING) == []


@pytest.mark.postgres
def test_project4_app_cannot_change_the_schema_or_read_beyond_its_tables(app_mode) -> None:
    from sqlalchemy.exc import ProgrammingError

    for statement, message in (
        ("CREATE TABLE ml.app_ddl_probe (id integer)", "permission denied for schema ml"),
        ("CREATE TABLE public.app_ddl_probe (id integer)", "permission denied for schema public"),
        ("ALTER TABLE ml.model_registry ADD COLUMN probe integer", "must be owner"),
        ("DROP TABLE ml.online_predictions", "must be owner"),
        ("SELECT * FROM integration.model_metadata", "permission denied for schema integration"),
        ("SELECT * FROM ml.model_metrics", "permission denied for table model_metrics"),
        ("DELETE FROM ml.model_registry", "permission denied for table model_registry"),
    ):
        with pytest.raises(ProgrammingError, match=message):
            _connect_as(APP_ROLE, APP_PASSWORD, statement)


@pytest.mark.postgres
def test_apply_schema_twice_keeps_project4_app_grants_exact(database_mode) -> None:
    try:
        for _ in range(2):
            applied = database.apply_schema(database_mode, app_password=APP_PASSWORD)
            assert f"role {APP_ROLE}: LOGIN (password from PROJECT4_APP_PASSWORD)" in applied

        with database_mode.connect() as connection:
            from sqlalchemy import text

            rows = connection.execute(text(
                "SELECT table_schema || '.' || table_name, privilege_type "
                "FROM information_schema.role_table_grants WHERE grantee = :role"
            ), {"role": APP_ROLE}).fetchall()
            role = connection.execute(text(
                "SELECT rolsuper, rolcreatedb, rolcreaterole, rolcanlogin "
                "FROM pg_roles WHERE rolname = :role"
            ), {"role": APP_ROLE}).one()
            schemas = {
                schema: connection.execute(text(
                    "SELECT has_schema_privilege(:role, :schema, 'USAGE'), "
                    "has_schema_privilege(:role, :schema, 'CREATE')"
                ), {"role": APP_ROLE, "schema": schema}).one()
                for schema in ("ml", "monitoring", "audit", "integration", "public")
            }
            can_connect = connection.execute(text(
                "SELECT has_database_privilege(:role, current_database(), 'CONNECT')"
            ), {"role": APP_ROLE}).scalar_one()

        granted: dict[str, set[str]] = {}
        for table, privilege in rows:
            granted.setdefault(table, set()).add(privilege)
        assert granted == EXPECTED_APP_TABLE_PRIVILEGES
        assert tuple(role) == (False, False, False, True)
        assert {s: tuple(v) for s, v in schemas.items()} == {
            "ml": (True, False), "monitoring": (True, False), "audit": (True, False),
            "integration": (False, False), "public": (True, False),
        }
        assert can_connect is True
    finally:
        database.apply_schema(database_mode, app_password="")


@pytest.mark.postgres
def test_restrict_connect_leaves_project4_app_connecting(connect_roles) -> None:
    database.apply_schema(
        connect_roles, reader_password=CONNECT_READER_PASSWORD, app_password=APP_PASSWORD
    )

    rows = _run_restrict_connect_script(connect_roles)

    assert (APP_ROLE, False, True) in rows
    assert _connect_as(APP_ROLE, APP_PASSWORD) == [(1,)]


# --- stage 13e3: seed-job CLI steps ------------------------------------------------------


@pytest.mark.postgres
def test_restrict_connect_cli_runs_the_script_and_prints_the_role_table(
    connect_roles, capsys: pytest.CaptureFixture
) -> None:
    assert database.main(["restrict-connect"]) == 0

    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split()[1:] for line in out.splitlines()[1:] if line.strip()}
    assert out.splitlines()[0].split() == ["role", "superuser", "can_connect"]
    assert rows[database.PROJECT5_READER_ROLE] == ["false", "true"]
    assert rows[MLFLOW_ROLE] == ["false", "false"]
    with pytest.raises(Exception, match="permission denied for database"):
        _connect_as(MLFLOW_ROLE, MLFLOW_ROLE_PASSWORD)


@pytest.mark.postgres
def test_sync_from_files_if_empty_never_overwrites_a_populated_database(
    database_mode,
    trained_metadata_paths: dict[str, Path],
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.registry import promote_model_version

    # As in the cluster: the file mirror is off, so the files keep the image's state.
    monkeypatch.setenv("PROJECT4_REGISTRY_FILE_MIRROR", "false")

    # First seed: an empty database is filled from the files.
    assert database.main(["sync-from-files", "--if-empty"]) == 0
    seeded = _database_pointer(database_mode)
    assert seeded["pointer_version"] == 1
    capsys.readouterr()

    # The database moves on (a promote; a rollback is the same kind of change).
    _register_trained(trained_metadata_paths, ["v101"])
    promote_model_version(model_name="isolation_forest", model_version="v101")
    moved = _database_pointer(database_mode)
    entries_moved = {v: e.to_dict() for v, e in _database_entries(database_mode).items()}
    assert moved["model_version"] == "v101"

    # Seeding again is a no-op that says so.
    assert database.main(["sync-from-files", "--if-empty"]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out and "already" in out
    assert _database_pointer(database_mode) == moved
    assert {v: e.to_dict() for v, e in _database_entries(database_mode).items()} == entries_moved

    # Without --if-empty the files win: the pointer goes back to the file's version.
    # That is why the seed job always passes --if-empty.
    database.sync_from_files(database_mode)
    assert _database_pointer(database_mode)["model_version"] == seeded["model_version"]


# --- stage 15a: real-source training dataset rows ------------------------------------------

V004_METADATA = (
    REPO_ROOT / "artifacts" / "models" / "isolation_forest" / "model_version=v004" / "metadata.json"
)


def test_v004_style_metadata_gives_only_the_snapshot_row() -> None:
    snapshot_metadata = json.loads(V004_METADATA.read_text(encoding="utf-8"))[
        "training_snapshot_metadata"
    ]
    assert "training_dataset_id" not in snapshot_metadata

    rows = database.snapshot_rows_for_training_metadata(snapshot_metadata)

    assert [table for table, _row in rows] == ["ml.feature_snapshots"]
    assert rows[0][1] == database.real_source_snapshot_row(snapshot_metadata)


def test_real_source_metadata_with_a_dataset_id_gives_snapshot_then_dataset_row() -> None:
    snapshot_metadata = {
        **json.loads(V004_METADATA.read_text(encoding="utf-8"))["training_snapshot_metadata"],
        "snapshot_id": "real_source_20261201T000000Z",
        "content_sha256": "c" * 64,
        "training_dataset_id": "6f1f7c52-5c1e-5b8e-9c43-0a4c7e7c6b11",
    }

    rows = database.snapshot_rows_for_training_metadata(snapshot_metadata)

    assert [table for table, _row in rows] == ["ml.feature_snapshots", "ml.training_datasets"]
    dataset = rows[1][1]
    assert set(dataset) == _columns("ml.training_datasets")
    assert dataset["training_dataset_id"] == snapshot_metadata["training_dataset_id"]
    assert dataset["snapshot_id"] == dataset["dataset_name"] == "real_source_20261201T000000Z"
    assert dataset["dataset_version"] == "c" * 12
    assert dataset["feature_schema_version"] == snapshot_metadata["feature_schema_version"]
    assert dataset["train_row_count"] == snapshot_metadata["row_count"]
    assert json.loads(dataset["feature_columns"]) == snapshot_metadata["feature_columns"]
    assert dataset["dataset_path"] is None  # bundle metadata only, never the snapshot dir
    assert dataset["created_by"] == database.REAL_SOURCE_SNAPSHOT_WRITER
    assert "c" * 64 in dataset["notes"]
