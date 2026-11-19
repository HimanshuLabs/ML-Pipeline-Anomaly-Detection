"""The 4 lakh round: guards and scale paths that must give today's results unchanged."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from anomaly_detection import batch_inference
from anomaly_detection.batch_inference import ActiveModelArtifacts, score_feature_batch
from anomaly_detection.segmented_model import fit_segmented_isolation_forest
from anomaly_detection.source_extract import (
    MODEL_FEATURE_COLUMNS,
    SEGMENT_FEATURE_COLUMNS,
    ExtractionReport,
    content_sha256,
    sort_feature_frame,
    transform_customer_360,
    transform_project1_user_features,
    transform_web_events,
    write_snapshot,
)
from tests.test_source_extract import (
    _customer_360,
    _customer_dimension,
    _p1_user_features,
    _snapshot_config,
    _web_events,
)

RUN_COLUMNS = ["extracted_at"]

def _without_run_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=RUN_COLUMNS).reset_index(drop=True)

# --- item 1: P1 columns P4 does not map never reach features, specs or the hash ----------

P1_COLUMNS_P4_DOES_NOT_MAP = ("user_name", "membership_tier", "total_spend",
                              "recency_hours", "features_computed_at")

def test_p1_columns_p4_does_not_map_change_nothing() -> None:
    with_extras = _p1_user_features(total_spend=99999.0, recency_hours=12345.0,
                                    features_computed_at="2030-01-01T00:00:00+00:00")
    without_extras = with_extras.drop(columns=list(P1_COLUMNS_P4_DOES_NOT_MAP))

    first = transform_project1_user_features(with_extras)
    second = transform_project1_user_features(without_extras)

    pd.testing.assert_frame_equal(_without_run_columns(first), _without_run_columns(second))
    assert content_sha256(first) == content_sha256(second)
    # total_spend is a campaign feature: a user row gets the 0.0 padding, not P1's value.
    assert first.loc[0, "total_spend"] == 0.0
    assert "total_spend" not in SEGMENT_FEATURE_COLUMNS["user"]
    assert "session_count" in SEGMENT_FEATURE_COLUMNS["user"]
    assert first.loc[0, "source_event_timestamp"] == "2026-06-10T10:00:00+05:30"
    assert list(first.columns) == ["source_project", "source_table", "entity_type", "entity_id",
                                   "source_event_timestamp", *MODEL_FEATURE_COLUMNS,
                                   "extracted_at"]

# --- item 2: a user and a customer with the same id stay two entities --------------------

SHARED_ID = "C010"

def _shared_id_frame() -> pd.DataFrame:
    return pd.concat(
        [
            transform_project1_user_features(_p1_user_features(user_id=SHARED_ID)),
            transform_customer_360(_customer_360(customer_id=SHARED_ID)),
            transform_web_events(_web_events(), _customer_dimension()),
        ],
        ignore_index=True,
    )

def _segmented_artifacts(entity_types: tuple[str, ...]) -> ActiveModelArtifacts:
    from conftest import real_source_test_frame

    training = real_source_test_frame(entity_types=entity_types, rows_per_type=40)
    model = fit_segmented_isolation_forest(
        training,
        segment_features={t: SEGMENT_FEATURE_COLUMNS[t] for t in entity_types},
        build_model=lambda: IsolationForest(n_estimators=20, random_state=42),
        min_rows_per_segment=30,
    )
    return ActiveModelArtifacts(
        model_name="isolation_forest", model_version="v900", model_path=None,
        metadata_path=None, baseline_stats_path=None, feature_schema_path=None,
        feature_schema_version="feature_schema_v003", dataset_snapshot_id=None,
        training_dataset_id=None, model=model,
        metadata={"feature_names": list(MODEL_FEATURE_COLUMNS)},
        baseline_stats={}, feature_schema={"entity_key": "entity_id"},
    )

def test_user_and_customer_with_the_same_id_stay_distinct(tmp_path) -> None:
    frame = _shared_id_frame()
    assert list(frame["entity_id"]) == [SHARED_ID] * 3

    ordered = sort_feature_frame(frame)
    assert list(zip(ordered["entity_type"], ordered["entity_id"], strict=True)) == [
        ("customer", SHARED_ID), ("user", SHARED_ID), ("web_customer", SHARED_ID)]

    full = content_sha256(frame)
    for entity_type in ("user", "customer", "web_customer"):
        assert content_sha256(frame[frame["entity_type"] != entity_type]) != full
    relabelled = frame.assign(entity_type=frame["entity_type"].replace({"user": "customer"}))
    assert content_sha256(relabelled) != full

    snapshot = write_snapshot(frame, _snapshot_config(tmp_path), output_root=tmp_path)
    metadata = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
    assert metadata["entity_type_counts"] == {"customer": 1, "user": 1, "web_customer": 1}

    records = score_feature_batch(frame, _segmented_artifacts(("customer", "user", "web_customer")))
    assert [(r.entity_type, r.entity_id) for r in records] == [
        ("user", SHARED_ID), ("customer", SHARED_ID), ("web_customer", SHARED_ID)]
    assert len({r.batch_run_id for r in records}) == 1
    assert {tuple(sorted(r.feature_payload)) for r in records} == {
        tuple(sorted(SEGMENT_FEATURE_COLUMNS[t])) for t in ("user", "customer", "web_customer")}

    # retrain's comparison keys
    keys = {f"{t}|{e}" for t, e in zip(frame["entity_type"], frame["entity_id"], strict=True)}
    assert len(keys) == 3

def test_a_browse_only_user_gets_zero_prices_counted_as_fills_and_passes_validation() -> None:
    from anomaly_detection.feature_validation import validate_real_source_features

    report = ExtractionReport()
    output = transform_project1_user_features(
        _p1_user_features(avg_event_price=None, max_event_price=None), report=report)

    assert output.loc[0, ["avg_event_price", "max_event_price"]].tolist() == [0.0, 0.0]
    assert report.tables["public.user_features"].null_fills == {
        "avg_event_price": 1, "max_event_price": 1}
    assert validate_real_source_features(output, MODEL_FEATURE_COLUMNS).is_valid

# --- item 4: content_sha256 streams the CSV and gives the same hash ------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V005_METADATA = (PROJECT_ROOT / "artifacts" / "models" / "isolation_forest"
                 / "model_version=v005" / "metadata.json")
V005_SNAPSHOT = (PROJECT_ROOT / "data" / "features" / "training" / "snapshot_date=2026-11-16"
                 / "snapshot_id=real_source_20261116T035043Z" / "features.parquet")

def _whole_csv_sha256(frame: pd.DataFrame) -> str:
    """content_sha256 as main c2823ee computes it: one CSV string of the whole frame."""

    import hashlib

    from anomaly_detection.source_extract import SNAPSHOT_RUN_COLUMNS, _snapshot_column_order

    columns = [c for c in _snapshot_column_order(frame) if c not in SNAPSHOT_RUN_COLUMNS]
    stable_csv = sort_feature_frame(frame).loc[:, columns].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(stable_csv.encode("utf-8")).hexdigest()

def _hash_fixtures() -> dict[str, pd.DataFrame]:
    from conftest import real_source_test_frame

    return {
        "unified": real_source_test_frame(entity_types=("user", "campaign", "product"),
                                          rows_per_type=5),
        "shared_ids": _shared_id_frame(),
        "one_row": _shared_id_frame().iloc[:1],
        "with_nan_and_quotes": _shared_id_frame().assign(
            source_event_timestamp=['a,"b"', "", "line\nbreak"],
            avg_fraud_score=[np.nan, 1e-300, 1.0 / 3.0]),
    }

@pytest.mark.parametrize("name", sorted(_hash_fixtures()))
@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 100_000])
def test_streamed_hash_equals_the_whole_csv_hash(name: str, chunk_rows: int) -> None:
    frame = _hash_fixtures()[name]

    assert content_sha256(frame, chunk_rows=chunk_rows) == _whole_csv_sha256(frame)
    assert content_sha256(frame) == _whole_csv_sha256(frame)

@pytest.mark.parametrize("kind", ["datetime", "timedelta", "category"])
def test_streamed_hash_refuses_columns_formatted_per_column(kind: str) -> None:
    from anomaly_detection.source_extract import SourceExtractionError

    frame = _shared_id_frame()
    values = {
        "datetime": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "timedelta": pd.to_timedelta([1, 2, 3], unit="s"),
        "category": pd.Categorical(["a", "b", "a"]),
    }[kind]
    with pytest.raises(SourceExtractionError, match="extra_column"):
        content_sha256(frame.assign(extra_column=values))

def test_streamed_hash_of_the_local_v005_snapshot_is_the_committed_hash() -> None:
    if not V005_SNAPSHOT.is_file():
        pytest.skip(
            f"local snapshot {V005_SNAPSHOT.relative_to(PROJECT_ROOT)} is absent (snapshots are "
            "gitignored, so CI has none); run on the machine that trained v005 to compare with "
            "the committed hash"
        )
    committed = json.loads(V005_METADATA.read_text(encoding="utf-8"))
    expected = committed["training_snapshot_metadata"]["content_sha256"]
    assert expected == "ad2c8596da09eae4b77875b3d96c698de948ac1508a42a482bb9e4f44f02f6f1"

    frame = pd.read_parquet(V005_SNAPSHOT)

    assert content_sha256(frame) == expected
    assert content_sha256(frame, chunk_rows=97) == expected
    assert _whole_csv_sha256(frame) == expected

# --- item 5c: fact_web_events read in buckets of business customers ------------------------

WEB_MEASURES = ["engagement_score", "purchase_probability", "cart_abandonment_probability",
                "event_value", "api_latency_ms", "page_load_time_ms", "fraud_score"]

def _scale_dimension() -> pd.DataFrame:
    """Sentinels 0 and -1, C011 with two SCD2 versions (11, 12), C030 with three."""

    rows = [(0, "UNKNOWN"), (-1, "INVALID"), (10, "C010"), (11, "C011"), (12, "C011"),
            (30, "C030"), (31, "C030"), (32, "C030")]
    rows += [(sk, f"C{sk:03d}") for sk in range(13, 25)]
    return pd.DataFrame({"customer_sk": [sk for sk, _ in rows],
                         "customer_id": [cid for _, cid in rows],
                         "is_current": True})

def _scale_events(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sks = [10, 11, 12, 11, 0, -1, 30, 31, 32, 30] + [int(s) for s in rng.choice(
        [0, -1, 10, 11, 12, *range(13, 25), 30, 31, 32], size=150)]
    count = len(sks)
    events = pd.DataFrame({
        "event_id": [f"E{index:04d}" for index in range(count)],
        "session_id": [f"S{int(s)}" for s in rng.integers(0, 40, size=count)],
        "customer_sk": sks,
        "event_type": rng.choice(["page_view", "product_view", "add_to_cart", "purchase",
                                  "search"], size=count),
        "event_timestamp": pd.Timestamp("2026-06-01")
        + pd.to_timedelta(rng.integers(0, 10**6, size=count), unit="s"),
    })
    for column in WEB_MEASURES:
        values = np.round(rng.random(count) * 1000, 4) / 7.0  # sums sensitive to the order
        values[rng.random(count) < 0.15] = np.nan
        events[column] = values
    return events

def _fake_source(events: pd.DataFrame, dimension: pd.DataFrame):
    reads: list[list[int] | None] = []

    def read_events(customer_sks):
        reads.append(customer_sks)
        if customer_sks is None:
            return events.copy()
        return events[events["customer_sk"].isin(customer_sks)].reset_index(drop=True)

    def count_unmapped() -> int:
        known = set(dimension["customer_sk"])
        return int((events["customer_sk"].isna() | ~events["customer_sk"].isin(known)).sum())

    return read_events, count_unmapped, reads

def _stats(report: ExtractionReport) -> dict:
    return {name: asdict(stats) for name, stats in report.tables.items()}

@pytest.mark.parametrize("bucket_count", [1, 2, 3, 160])
def test_bucketed_web_events_equal_the_whole_table(bucket_count: int) -> None:
    from anomaly_detection.source_extract import transform_web_events_in_buckets

    events, dimension = _scale_events(), _scale_dimension()
    whole_report, bucketed_report = ExtractionReport(), ExtractionReport()
    whole = transform_web_events(events, dimension, report=whole_report)
    read_events, count_unmapped, reads = _fake_source(events, dimension)

    bucketed = transform_web_events_in_buckets(
        read_events, count_unmapped, dimension, report=bucketed_report, bucket_count=bucket_count)

    pd.testing.assert_frame_equal(_without_run_columns(bucketed), _without_run_columns(whole),
                                  check_exact=True)
    assert _stats(bucketed_report) == _stats(whole_report)
    assert content_sha256(bucketed) == content_sha256(whole)
    from anomaly_detection.source_extract import web_event_buckets

    expected_reads = [None] if bucket_count == 1 else web_event_buckets(dimension, bucket_count)
    assert reads == expected_reads
    assert {"C011", "C030"} <= set(whole["entity_id"])
    assert whole_report.tables["warehouse.fact_web_events"].sentinel_rows_dropped > 0

def test_web_event_order_within_a_customer_does_not_change_any_value() -> None:
    events, dimension = _scale_events(), _scale_dimension()
    shuffled = events.sample(frac=1.0, random_state=11).reset_index(drop=True)

    first = transform_web_events(events, dimension)
    second = transform_web_events(shuffled, dimension)

    pd.testing.assert_frame_equal(_without_run_columns(first), _without_run_columns(second),
                                  check_exact=True)

@pytest.mark.parametrize("bucket_count", [2, 3, 40])
def test_buckets_hold_every_version_of_a_customer_and_every_sentinel_in_one_bucket(
    bucket_count: int,
) -> None:
    from anomaly_detection.source_extract import web_event_buckets

    dimension = _scale_dimension()
    buckets = web_event_buckets(dimension, bucket_count)

    flat = [sk for bucket in buckets for sk in bucket]
    assert sorted(flat) == sorted(dimension["customer_sk"])  # every sk exactly once
    assert all(buckets)
    bucket_of = {sk: index for index, bucket in enumerate(buckets) for sk in bucket}
    for _, versions in dimension.groupby("customer_id")["customer_sk"]:
        assert len({bucket_of[sk] for sk in versions}) == 1
    assert bucket_of[0] == bucket_of[-1]
    assert len(buckets) == min(bucket_count, dimension["customer_id"].nunique() - 2)

@pytest.mark.parametrize("bucket_count", [1, 3])
@pytest.mark.parametrize("bad_sk", [None, 999])
def test_a_null_or_unmapped_customer_sk_is_still_refused(bucket_count, bad_sk) -> None:
    from anomaly_detection.source_extract import (
        SourceExtractionError,
        transform_web_events_in_buckets,
    )

    events, dimension = _scale_events(), _scale_dimension()
    events["customer_sk"] = events["customer_sk"].astype("float64")
    events.loc[3, "customer_sk"] = np.nan if bad_sk is None else bad_sk
    read_events, count_unmapped, _ = _fake_source(events, dimension)

    with pytest.raises(SourceExtractionError,
                       match=r"has 1 event\(s\) whose customer_sk is not in warehouse\.dim_cu"):
        transform_web_events_in_buckets(read_events, count_unmapped, dimension,
                                        report=ExtractionReport(), bucket_count=bucket_count)
    with pytest.raises(SourceExtractionError, match="has 1 event"):
        transform_web_events(events, dimension)

def test_bucket_count_follows_the_number_of_business_customers(monkeypatch) -> None:
    from anomaly_detection import source_extract
    from anomaly_detection.source_extract import SourceExtractionError, web_event_bucket_count

    dimension = _scale_dimension()  # 15 business customers besides the sentinels
    assert web_event_bucket_count(dimension, None) == 1
    monkeypatch.setattr(source_extract, "WEB_EVENT_BUCKET_CUSTOMERS", 5)
    assert web_event_bucket_count(dimension, None) == 3
    assert web_event_bucket_count(dimension, "") == 3
    assert web_event_bucket_count(dimension, "7") == 7
    for bad in ("0", "-2", "eight"):
        with pytest.raises(SourceExtractionError, match="P4_WEB_EVENT_BUCKETS"):
            web_event_bucket_count(dimension, bad)

@pytest.mark.postgres
def test_bucketed_reads_on_a_real_database_equal_the_whole_table(monkeypatch) -> None:
    """The real SQL (= ANY of the bucket's sks, NOT EXISTS count) on a read-only session."""

    import os

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    from anomaly_detection import source_extract
    from anomaly_detection.source_extract import SourceExtractionError, extract_web_event_features

    url_text = os.environ.get("PROJECT4_TEST_DATABASE_URL")
    if not url_text:
        if os.environ.get("PROJECT4_REQUIRE_TEST_DATABASE") == "1":
            pytest.fail("PROJECT4_REQUIRE_TEST_DATABASE=1 but PROJECT4_TEST_DATABASE_URL is unset")
        pytest.skip("PROJECT4_TEST_DATABASE_URL is not set")
    url = make_url(url_text)
    owner = create_engine(url, hide_parameters=True)
    schema = "p4_scale_round_test"
    events, dimension = _scale_events(), _scale_dimension()
    monkeypatch.setenv("PROJECT4_TEST_SOURCE_PASSWORD", url.password or "")
    reader = source_extract.build_postgres_engine({
        "host": url.host, "port": url.port, "database": url.database, "username": url.username,
        "password_env_var": "PROJECT4_TEST_SOURCE_PASSWORD"})
    web_config = {"schema": schema, "table": "fact_web_events",
                  "customer_dimension_table": "dim_customer"}

    def _extract(bucket_count: int) -> tuple[pd.DataFrame, dict]:
        monkeypatch.setenv("P4_WEB_EVENT_BUCKETS", str(bucket_count))
        report = ExtractionReport()
        return extract_web_event_features(reader, web_config, dimension, report=report), _stats(
            report)

    try:
        with owner.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            connection.execute(text(f"CREATE SCHEMA {schema}"))
            # An extra column the extract must not need, like P2's 35-column table.
            events.assign(page_url="/p").to_sql("fact_web_events", connection, schema=schema,
                                                index=False)
            dimension.to_sql("dim_customer", connection, schema=schema, index=False)

        whole_report = ExtractionReport()
        whole = transform_web_events(events, dimension, report=whole_report)
        for bucket_count in (1, 3, 40):
            frame, stats = _extract(bucket_count)
            pd.testing.assert_frame_equal(_without_run_columns(frame), _without_run_columns(whole),
                                          check_exact=True)
            assert stats == _stats(whole_report)
            assert content_sha256(frame) == content_sha256(whole)

        for bad_sk in ("NULL", "999"):
            with owner.begin() as connection:
                connection.execute(text(
                    f"INSERT INTO {schema}.fact_web_events (event_id, customer_sk) "
                    f"VALUES ('BAD', {bad_sk})"))
            for bucket_count in (1, 3):
                with pytest.raises(SourceExtractionError, match="has 1 event"):
                    _extract(bucket_count)
            with owner.begin() as connection:
                connection.execute(text(
                    f"DELETE FROM {schema}.fact_web_events WHERE event_id = 'BAD'"))
    finally:
        with owner.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        owner.dispose()
        reader.dispose()

# --- items 5f and 6: batch inference at scale -----------------------------------------------

RUN_VARYING_FIELDS = {"prediction_id", "score_timestamp"}

def _records_as_main_builds_them(features, artifacts, *, scores, anomaly_flags, threshold_used,
                                 row_feature_names, row_entity_types, per_record_latency_ms,
                                 source_project, source_table, batch_id):
    """batch_inference._build_records as main c2823ee has it (one iterrows pass)."""

    entity_key = str(artifacts.feature_schema.get("entity_key") or "entity_id")
    records = []
    for row_number, (_row_index, row) in enumerate(features.iterrows()):
        feature_payload = {name: batch_inference._json_safe(row[name])
                           for name in row_feature_names[row_number]}
        records.append(dict(
            batch_run_id=batch_id,
            entity_type=row_entity_types[row_number],
            entity_id=batch_inference._get_entity_id(row, row_number, entity_key),
            source_project=source_project, source_table=source_table,
            source_record_id=batch_inference._get_source_record_id(row),
            anomaly_score=float(scores[row_number]), is_anomaly=bool(anomaly_flags[row_number]),
            threshold_used=float(threshold_used),
            feature_payload_hash=batch_inference.canonical_feature_payload_hash(
                feature_payload, row_feature_names[row_number]),
            feature_payload=feature_payload, inference_latency_ms=float(per_record_latency_ms),
        ))
    return records

def _built_values(records, expected: list[dict]) -> list[dict]:
    assert len(records) == len(expected)
    return [{name: getattr(record, name) for name in wanted}
            for record, wanted in zip(records, expected, strict=True)]

def _record_values(record) -> dict:
    return {f.name: getattr(record, f.name) for f in fields(record)
            if f.name not in RUN_VARYING_FIELDS}

def _feature_frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(3)
    numeric = pd.DataFrame({  # only numbers: DataFrame.values is float64 here
        "entity_id": np.arange(6),
        "feature_a": rng.random(6),
        "feature_b": np.arange(6, dtype="int64"),
    })
    mixed = pd.DataFrame({  # strings, NaN ids, an int feature, a fallback user_id
        "entity_id": ["e0", None, "e2", "e3", None, "e5"],
        "user_id": ["u0", "u1", None, "u3", None, "u5"],
        "event_id": [None, "ev1", "ev2", None, None, "ev5"],
        "session_id": ["s0", None, None, "s3", None, None],
        "feature_a": rng.random(6),
        "feature_b": [1, 2, 3, 4, 5, 6],
        "note": ["x", "y", "z", "x", "y", "z"],
    })
    return {"numeric": numeric, "mixed": mixed}

@pytest.mark.parametrize("name", ["numeric", "mixed"])
def test_records_are_built_as_main_builds_them(tmp_path, name) -> None:
    from tests.test_batch_inference import _build_artifact_bundle

    artifacts = batch_inference.load_active_model_artifacts(_build_artifact_bundle(tmp_path))
    features = _feature_frames()[name]
    arguments = dict(
        scores=np.linspace(-0.5, 0.5, len(features)),
        anomaly_flags=np.arange(len(features)) % 2 == 0, threshold_used=0.0,
        row_feature_names=[["feature_a", "feature_b"]] * len(features),
        row_entity_types=["customer"] * len(features), per_record_latency_ms=0.5,
        source_project="p", source_table="t", batch_id="run-1")

    built = batch_inference._build_records(features, artifacts, **arguments)
    expected = _records_as_main_builds_them(features, artifacts, **arguments)

    assert _built_values(built, expected) == expected
    assert [json.dumps(r.feature_payload, sort_keys=True) for r in built] == [
        json.dumps(e["feature_payload"], sort_keys=True) for e in expected]

def test_segmented_records_are_built_as_main_builds_them() -> None:
    artifacts = _segmented_artifacts(("customer", "user", "web_customer"))
    frame = _shared_id_frame()
    frame = pd.concat([frame] * 3, ignore_index=True)
    frame["entity_id"] = [f"C{index}" for index in range(len(frame))]

    records = score_feature_batch(frame, artifacts, batch_run_id="run-2")
    names = {t: SEGMENT_FEATURE_COLUMNS[t] for t in ("customer", "user", "web_customer")}
    expected = _records_as_main_builds_them(
        frame, artifacts, scores=[r.anomaly_score for r in records],
        anomaly_flags=[r.is_anomaly for r in records], threshold_used=0.0,
        row_feature_names=[list(names[t]) for t in frame["entity_type"]],
        row_entity_types=list(frame["entity_type"]),
        per_record_latency_ms=records[0].inference_latency_ms,
        source_project=None, source_table=None, batch_id="run-2")

    assert _built_values(records, expected) == expected

def test_jsonl_is_written_in_chunks_with_every_record_in_order(tmp_path, monkeypatch) -> None:
    from tests.test_batch_inference import _build_artifact_bundle

    artifacts = batch_inference.load_active_model_artifacts(_build_artifact_bundle(tmp_path))
    features = _feature_frames()["mixed"]
    records = batch_inference._build_records(
        features, artifacts, scores=np.zeros(len(features)),
        anomaly_flags=np.zeros(len(features), dtype=bool), threshold_used=0.0,
        row_feature_names=[["feature_a", "feature_b"]] * len(features),
        row_entity_types=["customer"] * len(features), per_record_latency_ms=0.1,
        source_project=None, source_table=None, batch_id="run-3")
    calls = []
    original = batch_inference.write_batch_predictions_jsonl

    def _counting(chunk, path):
        calls.append(len(chunk))
        return original(chunk, path)

    monkeypatch.setattr(batch_inference, "BATCH_WRITE_CHUNK_ROWS", 4)
    monkeypatch.setattr(batch_inference, "write_batch_predictions_jsonl", _counting)
    output = batch_inference.write_predictions_jsonl(records, tmp_path / "p.jsonl")

    assert calls == [4, 2]
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [w["prediction_id"] for w in written] == [r.prediction_id for r in records]
    assert {w["batch_run_id"] for w in written} == {"run-3"}

def test_batch_cli_prints_the_run_id_and_rows_and_anomalies_per_entity_type(
    tmp_path, monkeypatch, capsys,
) -> None:
    from tests.test_batch_inference import _build_artifact_bundle

    active_path = _build_artifact_bundle(tmp_path)
    input_path = tmp_path / "features.csv"
    pd.DataFrame({
        "entity_id": ["a", "b"], "feature_a": [1.0, 10.0], "feature_b": [2.0, 20.0],
    }).to_csv(input_path, index=False)
    from anomaly_detection import database

    monkeypatch.setattr(database, "is_configured", lambda: False)
    monkeypatch.setattr("sys.argv", [
        "batch_inference", "--input-path", str(input_path),
        "--output-path", str(tmp_path / "out.jsonl"), "--active-model-path", str(active_path)])

    batch_inference.main()

    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    written = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert printed["batch_run_id"] == written[0]["batch_run_id"]
    assert printed["records_scored"] == 2
    assert printed["entity_types"] == {
        "customer": {"rows": 2, "anomalies": 1},
    }
    assert printed["anomalies_detected"] == sum(w["is_anomaly"] for w in written)

# --- item 7 and amendment D: the README documents the run day and the drift limitation ----

def test_readme_documents_the_scale_run_day_in_order() -> None:
    from tests.test_readme import _section

    batch = _section("Batch inference")
    steps = [
        "tools python -m anomaly_detection.database apply-schema",
        "tools python -m anomaly_detection.retrain --snapshot latest --model-version v006 "
        "--compare-with v004",
        "tools python -m anomaly_detection.registry promote --model-version v006",
        "tools python -m anomaly_detection.batch_inference",
    ]
    positions = [batch.index(step) for step in steps]
    assert positions == sorted(positions)
    for text in ("### Scale run", "P4_WEB_EVENT_BUCKETS", "batch_run_id", "null_fills",
                 "content_sha256", "entity_type_counts"):
        assert text in batch
    assert "drift job reads every batch payload of the last 24 hours" in _section(
        "Known limitations")
