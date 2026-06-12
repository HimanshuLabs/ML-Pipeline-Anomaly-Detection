from __future__ import annotations

import json

import pandas as pd
import pytest

from anomaly_detection.source_extract import (
    MODEL_FEATURE_COLUMNS,
    transform_campaign_performance,
    transform_customer_360,
    transform_marketing_funnel,
    transform_product_sales,
    transform_project1_user_features,
    transform_web_events,
    write_snapshot,
)


def test_transform_project1_user_features_outputs_unified_contract() -> None:
    raw = pd.DataFrame(
        {
            "user_id": [101],
            "total_events": [10],
            "page_view_count": [4],
            "product_view_count": [3],
            "add_to_cart_count": [2],
            "purchase_count": [1],
            "search_count": [1],
            "avg_event_price": [299.5],
            "max_event_price": [999.0],
            "avg_engagement_score": [0.72],
            "avg_purchase_probability": [0.44],
            "unique_products_interacted": [5],
            "last_event_timestamp": ["2026-06-10T10:00:00+05:30"],
        }
    )

    output = transform_project1_user_features(raw)

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
    raw = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "total_order_count": [3],
            "gross_revenue": [1500.0],
            "lifetime_value": [1600.0],
            "average_order_value": [500.0],
            "web_event_count": [20],
            "session_count": [8],
            "purchase_intent_event_count": [4],
            "customer_conversion_rate_pct": [12.5],
            "avg_engagement_score": [0.67],
            "avg_purchase_probability": [0.31],
            "avg_cart_abandonment_probability": [0.22],
            "repeat_purchase_signal": [True],
            "last_order_timestamp": ["2026-06-09T00:00:00"],
        }
    )

    output = transform_customer_360(raw)

    assert output.loc[0, "source_project"] == "project_2_3"
    assert output.loc[0, "entity_type"] == "customer"
    assert output.loc[0, "entity_id"] == "C001"
    assert output.loc[0, "order_count"] == 3
    assert output.loc[0, "lifetime_value"] == 1600.0
    assert output.loc[0, "repeat_purchase_signal"] == 1


def test_transform_campaign_performance_maps_marketing_features() -> None:
    raw = pd.DataFrame(
        {
            "campaign_id": ["CMP001"],
            "campaign_budget": [10000.0],
            "total_spend": [7000.0],
            "budget_remaining": [3000.0],
            "impressions": [100000],
            "clicks": [1200],
            "order_count": [60],
            "attributed_revenue": [25000.0],
            "roas": [3.57],
            "click_through_rate_pct": [1.2],
            "cost_per_click": [5.83],
            "cost_per_order": [116.67],
            "avg_engagement_score": [0.6],
            "avg_purchase_probability": [0.25],
            "campaign_end_date": ["2026-06-10"],
        }
    )

    output = transform_campaign_performance(raw)

    assert output.loc[0, "entity_type"] == "campaign"
    assert output.loc[0, "entity_id"] == "CMP001"
    assert output.loc[0, "roas"] == 3.57
    assert output.loc[0, "clicks"] == 1200


def test_transform_marketing_funnel_builds_composite_entity_id() -> None:
    raw = pd.DataFrame(
        {
            "campaign_id": ["CMP001"],
            "event_date": ["2026-06-10"],
            "channel_name": ["email"],
            "total_event_count": [100],
            "session_count": [80],
            "product_view_event_count": [30],
            "cart_event_count": [10],
            "purchase_event_count": [5],
            "funnel_conversion_rate_pct": [6.25],
            "avg_engagement_score": [0.5],
            "avg_purchase_probability": [0.2],
            "avg_cart_abandonment_probability": [0.3],
        }
    )

    output = transform_marketing_funnel(raw)

    assert output.loc[0, "entity_type"] == "funnel_day"
    assert output.loc[0, "entity_id"] == "CMP001|2026-06-10|email"
    assert output.loc[0, "total_events"] == 100
    assert output.loc[0, "funnel_conversion_rate_pct"] == 6.25


def test_transform_product_sales_maps_product_features() -> None:
    raw = pd.DataFrame(
        {
            "product_id": ["P001"],
            "original_price": [1000.0],
            "current_price": [850.0],
            "inventory_remaining": [50],
            "order_count": [12],
            "units_sold": [18],
            "product_revenue": [15300.0],
            "average_selling_price": [850.0],
            "product_web_event_count": [60],
            "product_session_count": [40],
            "product_view_event_count": [35],
            "cart_event_count": [14],
            "purchase_event_count": [9],
            "product_conversion_rate_pct": [22.5],
            "avg_product_engagement_score": [0.71],
            "category_revenue_rank": [2],
            "category_revenue_share_pct": [18.5],
            "last_sold_timestamp": ["2026-06-09T00:00:00"],
        }
    )

    output = transform_product_sales(raw)

    assert output.loc[0, "entity_type"] == "product"
    assert output.loc[0, "entity_id"] == "P001"
    assert output.loc[0, "product_revenue"] == 15300.0
    assert output.loc[0, "product_conversion_rate_pct"] == 22.5


def test_transform_web_events_aggregates_by_customer() -> None:
    raw = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "session_id": ["S1", "S1", "S2"],
            "customer_sk": [10, 10, 10],
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

    output = transform_web_events(raw)

    assert len(output) == 1
    assert output.loc[0, "entity_type"] == "web_customer"
    assert output.loc[0, "entity_id"] == "10"
    assert output.loc[0, "total_events"] == 3
    assert output.loc[0, "page_view_count"] == 1
    assert output.loc[0, "product_view_count"] == 1
    assert output.loc[0, "purchase_count"] == 1
    assert output.loc[0, "session_count"] == 2
    assert output.loc[0, "avg_api_latency_ms"] == pytest.approx(70.0)


def test_write_snapshot_creates_parquet_and_metadata(tmp_path) -> None:
    dataframe = transform_project1_user_features(
        pd.DataFrame(
            {
                "user_id": [101, 102],
                "total_events": [10, 20],
                "page_view_count": [4, 8],
                "product_view_count": [3, 6],
                "add_to_cart_count": [2, 4],
                "purchase_count": [1, 2],
                "search_count": [1, 2],
                "avg_event_price": [299.5, 399.5],
                "max_event_price": [999.0, 1299.0],
                "avg_engagement_score": [0.72, 0.82],
                "avg_purchase_probability": [0.44, 0.54],
                "unique_products_interacted": [5, 9],
            }
        )
    )

    config = {
        "snapshot_type": "real_source_extract",
        "feature_schema_version": "feature_schema_v001",
        "output_snapshot_prefix": "test_real_source",
        "output": {"root": str(tmp_path)},
        "project_1": {
            "enabled": True,
            "schema": "public",
            "table": "user_features",
        },
        "project_2_3": {
            "enabled": True,
            "tables": {
                "customer_360": {"schema": "marts", "table": "mart_customer_360"},
                "web_events": {"schema": "warehouse", "table": "fact_web_events"},
            },
        },
    }

    result = write_snapshot(dataframe, config, output_root=tmp_path)

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
