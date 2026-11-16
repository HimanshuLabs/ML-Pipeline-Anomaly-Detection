from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from anomaly_detection.source_extract import (
    DAYS_SINCE_LAST_ORDER_NEVER_ORDERED,
    MODEL_FEATURE_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
    ExtractionReport,
    SourceExtractionError,
    transform_campaign_performance,
    transform_customer_360,
    transform_marketing_funnel,
    transform_product_sales,
    transform_project1_user_features,
    transform_web_events,
    write_snapshot,
)

# --- frames shaped like the current P1 and P2 tables ----------------------------------


def _p1_user_features(**overrides) -> pd.DataFrame:
    row = {
        "user_id": 101,
        "user_name": "Asha",
        "membership_tier": "gold",
        "total_events": 10,
        "page_view_count": 4,
        "product_view_count": 3,
        "add_to_cart_count": 2,
        "purchase_count": 1,
        "search_count": 1,
        "avg_event_price": 299.5,
        "max_event_price": 999.0,
        "avg_engagement_score": 7.2,
        "avg_purchase_probability": 0.44,
        "last_event_timestamp": "2026-06-10T10:00:00+05:30",
        "unique_products_interacted": 5,
        "total_spend": 1999.0,
        "session_count": 3,
        "features_computed_at": "2026-06-10T12:00:00+00:00",
        "recency_hours": 2.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _customer_360(**overrides) -> pd.DataFrame:
    row = {
        "customer_id": "C001",
        "member_type": "known",
        "total_order_count": 3,
        "gross_revenue": 1500.0,
        "total_discount_amount": 100.0,
        "lifetime_value": 1600.0,
        "average_order_value": 500.0,
        "total_units_purchased": 6,
        "product_categories_purchased": 2,
        "distinct_products_purchased": 4,
        "days_since_last_order": 12,
        "last_order_timestamp": "2026-06-09T00:00:00",
        "repeat_purchase_signal": True,
        "web_event_count": 20,
        "session_count": 8,
        "purchase_intent_event_count": 4,
        "customer_conversion_rate_pct": 12.5,
        "avg_engagement_score": 0.67,
        "avg_purchase_probability": 0.31,
        "avg_cart_abandonment_probability": 0.22,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _campaign_performance(**overrides) -> pd.DataFrame:
    row = {
        "campaign_id": "CMP001",
        "member_type": "known",
        "campaign_end_date": "2026-06-10",
        "campaign_budget": 10000.0,
        "total_spend": 7000.0,
        "budget_remaining": 3000.0,
        "impressions": 100000,
        "clicks": 1200,
        "ad_platform_conversions": 70,
        "web_event_count": 900,
        "session_count": 400,
        "purchase_event_count": 65,
        "order_count": 60,
        "attributed_revenue": 25000.0,
        "average_order_value": 416.67,
        "roas": 3.57,
        "click_through_rate_pct": 1.2,
        "cost_per_click": 5.83,
        "cost_per_order": 116.67,
        "session_to_order_conversion_rate_pct": 15.0,
        "avg_engagement_score": 0.6,
        "avg_purchase_probability": 0.25,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _marketing_funnel(**overrides) -> pd.DataFrame:
    row = {
        "campaign_id": "CMP001",
        "member_type": "known",
        "event_date": "2026-06-10",
        "channel_name": "email",
        "total_event_count": 100,
        "session_count": 80,
        "product_view_event_count": 30,
        "cart_event_count": 10,
        "purchase_event_count": 5,
        "funnel_conversion_rate_pct": 6.25,
        "avg_engagement_score": 0.5,
        "avg_purchase_probability": 0.2,
        "avg_cart_abandonment_probability": 0.3,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _product_sales(**overrides) -> pd.DataFrame:
    row = {
        "product_id": "P001",
        "member_type": "known",
        "last_sold_timestamp": "2026-06-09T00:00:00",
        "original_price": 1000.0,
        "current_price": 850.0,
        "inventory_remaining": 50,
        "order_count": 12,
        "units_sold": 18,
        "product_revenue": 15300.0,
        "average_selling_price": 850.0,
        "product_web_event_count": 60,
        "product_session_count": 40,
        "product_view_event_count": 35,
        "cart_event_count": 14,
        "purchase_event_count": 9,
        "product_conversion_rate_pct": 22.5,
        "avg_product_engagement_score": 0.71,
        "category_revenue_rank": 2,
        "category_revenue_share_pct": 18.5,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _web_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "session_id": ["S1", "S1", "S2"],
            "customer_sk": [10, 10, 10],
            "customer_id": ["C010", "C010", "C010"],
            "event_type": ["page_view", "product_view", "purchase"],
            "event_timestamp": [
                "2026-06-10T10:00:00",
                "2026-06-10T10:01:00",
                "2026-06-10T10:02:00",
            ],
            "time_on_page_sec": [20, 30, 10],
            "scroll_depth_percent": [80.0, 90.0, 70.0],
            "engagement_score": [0.5, 0.8, 0.9],
            "purchase_probability": [0.1, 0.4, 0.9],
            "cart_abandonment_probability": [0.7, 0.4, 0.1],
            "event_value": [0.0, 100.0, 500.0],
            "api_latency_ms": [50, 70, 90],
            "page_load_time_ms": [800, 900, 1000],
            "fraud_score": [0.01, 0.02, 0.03],
        }
    )


def _customer_dimension() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_sk": [0, -1, 10, 11, 12],
            "customer_id": ["UNKNOWN", "INVALID", "C010", "C011", "C011"],
            "is_current": [True, True, True, False, True],
        }
    )


# --- existing behaviour (same assertions, full current-shaped inputs) --------------------


def test_transform_project1_user_features_outputs_unified_contract() -> None:
    output = transform_project1_user_features(_p1_user_features())

    assert len(output) == 1
    assert output.loc[0, "source_project"] == "project_1"
    assert output.loc[0, "source_table"] == "public.user_features"
    assert output.loc[0, "entity_type"] == "user"
    assert output.loc[0, "entity_id"] == "101"
    assert output.loc[0, "total_events"] == 10
    assert output.loc[0, "avg_purchase_probability"] == 0.44

    for column in MODEL_FEATURE_COLUMNS:
        assert column in output.columns


def test_transform_customer_360_maps_business_features() -> None:
    output = transform_customer_360(_customer_360())

    assert output.loc[0, "source_project"] == "project_2_3"
    assert output.loc[0, "entity_type"] == "customer"
    assert output.loc[0, "entity_id"] == "C001"
    assert output.loc[0, "order_count"] == 3
    assert output.loc[0, "lifetime_value"] == 1600.0
    assert output.loc[0, "repeat_purchase_signal"] == 1


def test_transform_campaign_performance_maps_marketing_features() -> None:
    output = transform_campaign_performance(_campaign_performance())

    assert output.loc[0, "entity_type"] == "campaign"
    assert output.loc[0, "entity_id"] == "CMP001"
    assert output.loc[0, "roas"] == 3.57
    assert output.loc[0, "clicks"] == 1200


def test_transform_marketing_funnel_builds_composite_entity_id() -> None:
    output = transform_marketing_funnel(_marketing_funnel())

    assert output.loc[0, "entity_type"] == "funnel_day"
    assert output.loc[0, "entity_id"] == "CMP001|2026-06-10|email"
    assert output.loc[0, "total_events"] == 100
    assert output.loc[0, "funnel_conversion_rate_pct"] == 6.25


def test_transform_product_sales_maps_product_features() -> None:
    output = transform_product_sales(_product_sales())

    assert output.loc[0, "entity_type"] == "product"
    assert output.loc[0, "entity_id"] == "P001"
    assert output.loc[0, "product_revenue"] == 15300.0
    assert output.loc[0, "product_conversion_rate_pct"] == 22.5


def test_transform_web_events_aggregates_by_customer() -> None:
    output = transform_web_events(_web_events(), _customer_dimension())

    assert len(output) == 1
    assert output.loc[0, "entity_type"] == "web_customer"
    assert output.loc[0, "entity_id"] == "C010"
    assert output.loc[0, "total_events"] == 3
    assert output.loc[0, "page_view_count"] == 1
    assert output.loc[0, "product_view_count"] == 1
    assert output.loc[0, "purchase_count"] == 1
    assert output.loc[0, "session_count"] == 2
    assert output.loc[0, "avg_api_latency_ms"] == pytest.approx(70.0)


def test_write_snapshot_creates_parquet_and_metadata(tmp_path) -> None:
    dataframe = transform_project1_user_features(
        pd.concat(
            [_p1_user_features(), _p1_user_features(user_id=102, total_events=20)],
            ignore_index=True,
        )
    )

    result = write_snapshot(dataframe, _snapshot_config(tmp_path), output_root=tmp_path)

    assert result.features_path.exists()
    assert result.metadata_path.exists()
    assert result.row_count == 2

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["snapshot_type"] == "real_source_extract"
    assert metadata["feature_schema_version"] == "feature_schema_v001"
    assert metadata["row_count"] == 2
    assert "project_1" in metadata["source_projects"]
    assert "project_2_3" in metadata["source_projects"]
    assert "public.user_features" in metadata["source_tables"]
    assert metadata["data_quality_status"] == "passed"


def _snapshot_config(tmp_path) -> dict:
    return {
        "snapshot_type": "real_source_extract",
        "feature_schema_version": "feature_schema_v001",
        "output_snapshot_prefix": "test_real_source",
        "output": {"root": str(tmp_path)},
        "project_1": {"enabled": True, "schema": "public", "table": "user_features"},
        "project_2_3": {
            "enabled": True,
            "tables": {
                "customer_360": {"schema": "marts", "table": "mart_customer_360"},
                "web_events": {"schema": "warehouse", "table": "fact_web_events"},
            },
        },
    }


# --- H4: required columns fail loudly ---------------------------------------------------

_TRANSFORMS = {
    "public.user_features": (_p1_user_features, transform_project1_user_features),
    "marts.mart_customer_360": (_customer_360, transform_customer_360),
    "marts.mart_campaign_performance": (_campaign_performance, transform_campaign_performance),
    "marts.mart_marketing_funnel": (_marketing_funnel, transform_marketing_funnel),
    "marts.mart_product_sales": (_product_sales, transform_product_sales),
}

_REQUIRED_CASES = [
    pytest.param(table, column, id=f"{table}.{column}")
    for table in _TRANSFORMS
    for column in REQUIRED_SOURCE_COLUMNS[table]
]


@pytest.mark.parametrize(("table", "column"), _REQUIRED_CASES)
def test_missing_required_column_raises_naming_table_and_column(table: str, column: str) -> None:
    build, transform = _TRANSFORMS[table]

    with pytest.raises(SourceExtractionError) as error:
        transform(build().drop(columns=[column]))

    assert table in str(error.value)
    assert column in str(error.value)


@pytest.mark.parametrize(
    ("frame", "column"),
    [
        pytest.param("events", column, id=f"fact_web_events.{column}")
        for column in REQUIRED_SOURCE_COLUMNS["warehouse.fact_web_events"]
    ]
    + [
        pytest.param("dimension", column, id=f"dim_customer.{column}")
        for column in REQUIRED_SOURCE_COLUMNS["warehouse.dim_customer"]
    ],
)
def test_missing_required_web_event_or_dimension_column_raises(frame: str, column: str) -> None:
    events, dimension = _web_events(), _customer_dimension()
    if frame == "events":
        events = events.drop(columns=[column])
    else:
        dimension = dimension.drop(columns=[column])

    with pytest.raises(SourceExtractionError, match=column):
        transform_web_events(events, dimension)


def test_non_numeric_source_value_fails_instead_of_becoming_zero() -> None:
    with pytest.raises(SourceExtractionError, match="lifetime_value"):
        transform_customer_360(_customer_360(lifetime_value="not-a-number"))


# --- sentinels, meaningful NULLs, SCD2 grain, P1 sessions -------------------------------


@pytest.mark.parametrize("table", [t for t in _TRANSFORMS if t != "public.user_features"])
def test_sentinel_member_rows_are_dropped_and_counted(table: str) -> None:
    build, transform = _TRANSFORMS[table]
    raw = pd.concat(
        [
            build(),
            build(member_type="unknown_member"),
            build(member_type="invalid_reference"),
        ],
        ignore_index=True,
    )
    report = ExtractionReport()

    output = transform(raw, report=report)

    assert len(output) == 1
    stats = report.tables[table]
    assert (stats.rows_read, stats.rows_after_filter, stats.sentinel_rows_dropped) == (3, 1, 2)


def test_unexpected_member_type_fails() -> None:
    with pytest.raises(SourceExtractionError, match="member_type"):
        transform_customer_360(_customer_360(member_type="mystery"))


def test_never_ordered_customer_gets_flag_and_documented_fill() -> None:
    report = ExtractionReport()
    output = transform_customer_360(
        pd.concat(
            [
                _customer_360(),
                _customer_360(
                    customer_id="C002",
                    total_order_count=0,
                    days_since_last_order=None,
                    last_order_timestamp=None,
                    repeat_purchase_signal=False,
                ),
            ],
            ignore_index=True,
        ),
        report=report,
    )

    ordered, never = output.iloc[0], output.iloc[1]
    assert (ordered["never_ordered"], ordered["days_since_last_order"]) == (0, 12)
    assert never["never_ordered"] == 1
    assert never["days_since_last_order"] == DAYS_SINCE_LAST_ORDER_NEVER_ORDERED
    assert never["days_since_last_order"] != 0
    assert report.tables["marts.mart_customer_360"].null_fills["days_since_last_order"] == 1
    assert "never_ordered" in MODEL_FEATURE_COLUMNS


def test_web_events_of_one_business_customer_with_two_sk_versions_form_one_entity() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3", "E4"],
            "session_id": ["S1", "S2", "S3", "S4"],
            "customer_sk": [11, 12, 0, -1],
            "customer_id": ["C011", "C011", "C999", "C998"],
            "event_type": ["page_view", "purchase", "page_view", "page_view"],
            "event_timestamp": ["2026-06-01", "2026-06-05", "2026-06-06", "2026-06-07"],
            "engagement_score": [0.2, 0.4, 0.9, 0.9],
            "purchase_probability": [0.1, 0.3, 0.5, 0.5],
            "cart_abandonment_probability": [0.5, 0.5, 0.5, 0.5],
            "event_value": [0.0, 200.0, 0.0, 0.0],
            "api_latency_ms": [100, None, 50, 50],
            "page_load_time_ms": [500, 700, 400, 400],
            "fraud_score": [0.01, 0.03, 0.02, 0.02],
        }
    )
    report = ExtractionReport()

    output = transform_web_events(events, _customer_dimension(), report=report)

    assert list(output["entity_id"]) == ["C011"]
    assert output.loc[0, "total_events"] == 2
    assert output.loc[0, "purchase_count"] == 1
    assert output.loc[0, "session_count"] == 2
    # Averages skip NULLs instead of treating them as 0.
    assert output.loc[0, "avg_api_latency_ms"] == pytest.approx(100.0)
    assert report.tables["warehouse.fact_web_events"].sentinel_rows_dropped == 2


def test_project1_session_count_is_carried_through() -> None:
    output = transform_project1_user_features(_p1_user_features(session_count=7))

    assert output.loc[0, "session_count"] == 7


def test_null_fills_are_counted_per_column() -> None:
    report = ExtractionReport()

    transform_project1_user_features(
        _p1_user_features(avg_event_price=None, max_event_price=None), report=report
    )

    assert report.tables["public.user_features"].null_fills == {
        "avg_event_price": 1,
        "max_event_price": 1,
    }


# --- N8: real-source validation decides data_quality_status -----------------------------


def _clean_unified_frame() -> pd.DataFrame:
    return pd.concat(
        [
            transform_project1_user_features(_p1_user_features()),
            transform_customer_360(_customer_360()),
            transform_campaign_performance(_campaign_performance(budget_remaining=-50.0)),
            transform_product_sales(_product_sales()),
        ],
        ignore_index=True,
    )


def test_validation_passes_on_clean_data_including_negative_budget_remaining() -> None:
    from anomaly_detection.feature_validation import validate_real_source_features

    result = validate_real_source_features(_clean_unified_frame(), MODEL_FEATURE_COLUMNS)

    assert result.is_valid, result.errors


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        pytest.param("order_count", -1.0, "negative", id="negative_count"),
        pytest.param("customer_conversion_rate_pct", 120.0, "0-100", id="pct_above_100"),
        pytest.param("avg_fraud_score", 1.5, "0-1", id="probability_above_1"),
        pytest.param("never_ordered", 2.0, "0/1", id="flag_not_binary"),
        pytest.param("lifetime_value", math.inf, "infinite", id="inf"),
        pytest.param("roas", math.nan, "missing", id="nan"),
    ],
)
def test_validation_fails_with_reasons(column: str, value: float, expected: str) -> None:
    from anomaly_detection.feature_validation import validate_real_source_features

    frame = _clean_unified_frame()
    frame.loc[0, column] = value

    result = validate_real_source_features(frame, MODEL_FEATURE_COLUMNS)

    assert not result.is_valid
    assert any(column in error and expected in error for error in result.errors), result.errors


def test_failed_snapshot_records_reasons_and_is_not_trainable(tmp_path) -> None:
    frame = _clean_unified_frame()
    frame.loc[0, "order_count"] = -3.0

    result = write_snapshot(frame, _snapshot_config(tmp_path), output_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_quality_status"] == "failed"
    assert metadata["trainable"] is False
    assert any("order_count" in error for error in metadata["validation_errors"])
    assert not result.features_path.exists()


def test_snapshot_metadata_records_extraction_statistics_and_fingerprint(tmp_path) -> None:
    from anomaly_detection.training import feature_set_fingerprint

    report = ExtractionReport()
    frame = pd.concat(
        [
            transform_project1_user_features(
                _p1_user_features(avg_event_price=None), report=report
            ),
            transform_customer_360(
                pd.concat(
                    [_customer_360(), _customer_360(member_type="unknown_member")],
                    ignore_index=True,
                ),
                report=report,
            ),
        ],
        ignore_index=True,
    )

    result = write_snapshot(
        frame, _snapshot_config(tmp_path), output_root=tmp_path, extraction_report=report
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_quality_status"] == "passed" and metadata["trainable"] is True
    assert metadata["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert metadata["feature_set_fingerprint"] == feature_set_fingerprint(MODEL_FEATURE_COLUMNS)
    assert metadata["sentinel_rows_dropped"] == 1
    tables = metadata["source_table_stats"]
    assert tables["marts.mart_customer_360"]["rows_read"] == 2
    assert tables["marts.mart_customer_360"]["rows_after_filter"] == 1
    assert tables["public.user_features"]["null_fills"] == {"avg_event_price": 1}


# --- stage 15a: deterministic content hash and training_dataset_id -----------------------


def _unified_frame_for_hashing() -> pd.DataFrame:
    from conftest import real_source_test_frame

    return real_source_test_frame(entity_types=("user", "campaign", "product"), rows_per_type=5)


def test_training_dataset_id_is_a_uuid5_of_snapshot_hash_and_fingerprint() -> None:
    import uuid

    from anomaly_detection.source_extract import training_dataset_id_for

    first = training_dataset_id_for("real_source_x", "a" * 64, "f" * 64)

    assert first == training_dataset_id_for("real_source_x", "a" * 64, "f" * 64)
    key = f"project4.training_dataset|real_source_x|{'a' * 64}|{'f' * 64}"
    assert first == str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    assert first != training_dataset_id_for("real_source_y", "a" * 64, "f" * 64)
    assert first != training_dataset_id_for("real_source_x", "b" * 64, "f" * 64)
    assert first != training_dataset_id_for("real_source_x", "a" * 64, "e" * 64)


def test_same_data_in_another_row_order_gives_the_same_hash_and_parquet(tmp_path) -> None:
    from anomaly_detection.source_extract import content_sha256

    frame = _unified_frame_for_hashing()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert not shuffled.equals(frame)

    first = write_snapshot(frame, _snapshot_config(tmp_path / "a"), output_root=tmp_path / "a")
    second = write_snapshot(
        shuffled, _snapshot_config(tmp_path / "b"), output_root=tmp_path / "b"
    )

    first_meta = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    second_meta = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert first_meta["content_sha256"] == second_meta["content_sha256"] == content_sha256(frame)
    assert len(first_meta["content_sha256"]) == 64
    assert first.features_path.read_bytes() == second.features_path.read_bytes()
    # Written sorted by entity_type, then entity_id.
    written = pd.read_parquet(first.features_path)
    keys = list(zip(written["entity_type"], written["entity_id"], strict=True))
    assert keys == sorted(keys)


def test_content_hash_ignores_the_run_timestamp_but_not_the_data() -> None:
    from anomaly_detection.source_extract import content_sha256

    frame = _unified_frame_for_hashing()
    later_run = frame.assign(extracted_at="2027-01-01T00:00:00+00:00")
    changed = frame.copy()
    changed.loc[0, "total_events"] = changed.loc[0, "total_events"] + 1.0

    assert content_sha256(later_run) == content_sha256(frame)
    assert content_sha256(changed) != content_sha256(frame)


def test_snapshot_metadata_records_content_hash_and_training_dataset_id(tmp_path) -> None:
    from anomaly_detection.source_extract import content_sha256, training_dataset_id_for
    from anomaly_detection.training import feature_set_fingerprint

    frame = _unified_frame_for_hashing()
    result = write_snapshot(frame, _snapshot_config(tmp_path), output_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["content_sha256"] == content_sha256(frame)
    assert metadata["training_dataset_id"] == training_dataset_id_for(
        metadata["snapshot_id"],
        metadata["content_sha256"],
        feature_set_fingerprint(MODEL_FEATURE_COLUMNS),
    )
    assert result.training_dataset_id == metadata["training_dataset_id"]


def test_failed_snapshot_has_no_training_dataset_id(tmp_path) -> None:
    frame = _clean_unified_frame()
    frame.loc[0, "order_count"] = -3.0

    result = write_snapshot(frame, _snapshot_config(tmp_path), output_root=tmp_path)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_quality_status"] == "failed"
    assert len(metadata["content_sha256"]) == 64
    assert metadata["training_dataset_id"] is None


def test_source_tables_include_the_customer_dimension_read_for_web_events(tmp_path) -> None:
    result = write_snapshot(
        _unified_frame_for_hashing(), _snapshot_config(tmp_path), output_root=tmp_path
    )

    tables = json.loads(result.metadata_path.read_text(encoding="utf-8"))["source_tables"]
    assert tables.index("warehouse.dim_customer") == tables.index("warehouse.fact_web_events") + 1


# --- stage 15a: read-only source sessions --------------------------------------------------

SOURCE_SESSION_OPTIONS = "-c default_transaction_read_only=on -c statement_timeout=120000"


def _source_config(**overrides) -> dict:
    config = {
        "host": "127.0.0.1",
        "port": 5999,
        "database": "source_db",
        "username": "reader",
        "password_env_var": "PROJECT4_TEST_SOURCE_PASSWORD",
    }
    config.update(overrides)
    return config


def test_source_engine_is_read_only_with_a_statement_timeout(monkeypatch) -> None:
    from anomaly_detection import source_extract

    captured: dict = {}
    real_create_engine = source_extract.create_engine

    def _capture(url, **kwargs):
        captured.update(kwargs)
        return real_create_engine(url, **kwargs)

    monkeypatch.setattr(source_extract, "create_engine", _capture)
    monkeypatch.setenv("PROJECT4_TEST_SOURCE_PASSWORD", "not-a-real-password")

    engine = source_extract.build_postgres_engine(_source_config())

    assert captured["connect_args"] == {"options": SOURCE_SESSION_OPTIONS}
    assert engine.get_execution_options()["postgresql_readonly"] is True
    assert "not-a-real-password" not in repr(engine.url)


def test_project4_engine_is_not_made_read_only(monkeypatch) -> None:
    from anomaly_detection import database

    captured: dict = {}
    real_create_engine = database.create_engine

    def _capture(url, **kwargs):
        captured.update(kwargs)
        return real_create_engine(url, **kwargs)

    monkeypatch.setattr(database, "create_engine", _capture)
    settings = database.DatabaseSettings(
        host="127.0.0.1", port=5999, database="project4_ml", user="u", password="p"
    )

    engine = database.get_engine(settings)

    assert "connect_args" not in captured
    assert "postgresql_readonly" not in engine.get_execution_options()


class _ProbeConnection:
    def __init__(self, read_only: str) -> None:
        self.read_only = read_only
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, statement):
        self.statements.append(str(statement))

        class _Result:
            def __init__(self, value) -> None:
                self.value = value

            def scalar_one(self):
                return self.value

        return _Result(self.read_only)


class _ProbeEngine:
    def __init__(self, read_only: str) -> None:
        self.connection = _ProbeConnection(read_only)

    def connect(self):
        return self.connection


@pytest.mark.parametrize("value", ["off", "", "OFF"])
def test_read_only_probe_refuses_before_reading_when_the_session_is_writable(
    monkeypatch, value: str
) -> None:
    from anomaly_detection import source_extract

    def _must_not_read(*args, **kwargs):
        raise AssertionError("the table was read on a writable session")

    monkeypatch.setattr(source_extract.pd, "read_sql_query", _must_not_read)
    engine = _ProbeEngine(value)

    with pytest.raises(SourceExtractionError, match="not read-only"):
        source_extract.read_table(engine, "public", "user_features")

    assert engine.connection.statements == ["SHOW transaction_read_only"]


def test_read_only_probe_lets_a_read_only_session_read(monkeypatch) -> None:
    from anomaly_detection import source_extract

    calls: list = []
    monkeypatch.setattr(
        source_extract.pd,
        "read_sql_query",
        lambda query, connection: calls.append(str(query)) or pd.DataFrame({"a": [1]}),
    )
    engine = _ProbeEngine("on")

    frame = source_extract.read_table(engine, "public", "user_features")

    assert len(frame) == 1
    assert engine.connection.statements == ["SHOW transaction_read_only"]
    assert calls == ['SELECT * FROM "public"."user_features"']


@pytest.mark.postgres
def test_source_engine_session_is_read_only_on_a_real_database(monkeypatch) -> None:
    import os

    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import DBAPIError

    from anomaly_detection import source_extract

    url_text = os.environ.get("PROJECT4_TEST_DATABASE_URL")
    if not url_text:
        if os.environ.get("PROJECT4_REQUIRE_TEST_DATABASE") == "1":
            pytest.fail("PROJECT4_REQUIRE_TEST_DATABASE=1 but PROJECT4_TEST_DATABASE_URL is unset")
        pytest.skip("PROJECT4_TEST_DATABASE_URL is not set")
    url = make_url(url_text)
    monkeypatch.setenv("PROJECT4_TEST_SOURCE_PASSWORD", url.password or "")
    engine = source_extract.build_postgres_engine(
        _source_config(
            host=url.host, port=url.port, database=url.database, username=url.username
        )
    )
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
            assert connection.execute(text("SHOW statement_timeout")).scalar_one() == "2min"
            with pytest.raises(DBAPIError, match="read-only transaction"):
                connection.execute(text("CREATE TEMPORARY TABLE p4_must_fail (a int)"))

        frame = source_extract.read_table(engine, "pg_catalog", "pg_namespace")
        assert not frame.empty
    finally:
        engine.dispose()


def test_a_sort_or_hash_failure_leaves_no_snapshot_directory(tmp_path, monkeypatch) -> None:
    from anomaly_detection import source_extract

    def _fails(frame):
        raise TypeError("cannot order these rows")

    monkeypatch.setattr(source_extract, "sort_feature_frame", _fails)

    with pytest.raises(TypeError, match="cannot order"):
        write_snapshot(
            _unified_frame_for_hashing(), _snapshot_config(tmp_path), output_root=tmp_path
        )

    assert list(tmp_path.iterdir()) == []
