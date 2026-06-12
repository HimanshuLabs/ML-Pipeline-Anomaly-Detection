"""Extract real Project 1 and Project 2/3 features into a Project 4 training snapshot.

This module intentionally reads from local PostgreSQL sources instead of Kafka/Spark.
For retraining, Project 4 needs stable historical feature snapshots, not a live replay dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger(__name__)

MODEL_FEATURE_COLUMNS = [
    "total_events",
    "page_view_count",
    "product_view_count",
    "add_to_cart_count",
    "purchase_count",
    "search_count",
    "avg_event_price",
    "max_event_price",
    "avg_engagement_score",
    "avg_purchase_probability",
    "unique_products_interacted",
    "web_event_count",
    "session_count",
    "purchase_intent_event_count",
    "customer_conversion_rate_pct",
    "avg_cart_abandonment_probability",
    "lifetime_value",
    "average_order_value",
    "gross_revenue",
    "total_discount_amount",
    "total_units_purchased",
    "product_categories_purchased",
    "distinct_products_purchased",
    "days_since_last_order",
    "repeat_purchase_signal",
    "campaign_budget",
    "total_spend",
    "budget_remaining",
    "impressions",
    "clicks",
    "ad_platform_conversions",
    "purchase_event_count",
    "order_count",
    "attributed_revenue",
    "roas",
    "click_through_rate_pct",
    "cost_per_click",
    "cost_per_order",
    "session_to_order_conversion_rate_pct",
    "funnel_conversion_rate_pct",
    "product_revenue",
    "inventory_remaining",
    "units_sold",
    "product_conversion_rate_pct",
    "avg_product_engagement_score",
    "category_revenue_rank",
    "category_revenue_share_pct",
    "avg_api_latency_ms",
    "avg_page_load_time_ms",
    "avg_fraud_score",
    "avg_event_value",
]


@dataclass(frozen=True)
class SnapshotWriteResult:
    """Details for a written source extraction snapshot."""

    snapshot_id: str
    snapshot_path: Path
    features_path: Path
    metadata_path: Path
    row_count: int
    feature_schema_version: str
    snapshot_type: str


class SourceExtractionError(RuntimeError):
    """Raised when real-source extraction cannot produce a valid snapshot."""


def configure_logging() -> None:
    """Configure module logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the source extraction YAML config."""

    path = Path(config_path)
    if not path.exists():
        raise SourceExtractionError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "source_extract" not in config:
        raise SourceExtractionError("Config must contain top-level 'source_extract' key")

    source_config = config["source_extract"]

    required_keys = [
        "snapshot_type",
        "feature_schema_version",
        "project_1",
        "project_2_3",
        "output",
    ]
    missing = [key for key in required_keys if key not in source_config]
    if missing:
        raise SourceExtractionError(f"Missing source_extract config keys: {missing}")

    return source_config


def _password_from_env(env_var_name: str) -> str:
    password = os.getenv(env_var_name)
    if not password:
        raise SourceExtractionError(
            f"Required database password environment variable is not set: {env_var_name}"
        )
    return password


def build_postgres_engine(source: dict[str, Any]) -> Engine:
    """Create a SQLAlchemy engine from one configured PostgreSQL source."""

    password = quote_plus(_password_from_env(source["password_env_var"]))
    username = quote_plus(str(source["username"]))
    host = source["host"]
    port = source["port"]
    database = source["database"]

    url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True)


def read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    """Read a PostgreSQL table safely with quoted schema/table names."""

    query = text(f'SELECT * FROM "{schema}"."{table}"')
    with engine.connect() as connection:
        dataframe = pd.read_sql_query(query, connection)

    if dataframe.empty:
        raise SourceExtractionError(f"Source table returned zero rows: {schema}.{table}")

    return dataframe


def _numeric_series(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(default).astype("float64")


def _bool_series(dataframe: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype="bool")
    return dataframe[column].fillna(default).astype("bool")


def _text_series(dataframe: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype="string")
    return dataframe[column].fillna(default).astype("string")


def _empty_feature_frame(index: pd.Index) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    for column in MODEL_FEATURE_COLUMNS:
        features[column] = 0.0
    return features


def _finalize_feature_frame(
    dataframe: pd.DataFrame,
    *,
    source_project: str,
    source_table: str,
    entity_type: str,
    entity_id: pd.Series,
    event_timestamp: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach common lineage/entity columns and enforce numeric feature columns."""

    output = dataframe.copy()

    for column in MODEL_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)

    output.insert(0, "source_project", source_project)
    output.insert(1, "source_table", source_table)
    output.insert(2, "entity_type", entity_type)
    output.insert(3, "entity_id", entity_id.astype("string").fillna("unknown"))

    if event_timestamp is not None:
        output.insert(4, "source_event_timestamp", event_timestamp.astype("string").fillna(""))
    else:
        output.insert(4, "source_event_timestamp", "")

    output["extracted_at"] = datetime.now(UTC).isoformat()

    return output[
        [
            "source_project",
            "source_table",
            "entity_type",
            "entity_id",
            "source_event_timestamp",
            *MODEL_FEATURE_COLUMNS,
            "extracted_at",
        ]
    ]


def transform_project1_user_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform Project 1 public.user_features into unified training features."""

    features = _empty_feature_frame(dataframe.index)

    passthrough_columns = [
        "total_events",
        "page_view_count",
        "product_view_count",
        "add_to_cart_count",
        "purchase_count",
        "search_count",
        "avg_event_price",
        "max_event_price",
        "avg_engagement_score",
        "avg_purchase_probability",
        "unique_products_interacted",
    ]

    for column in passthrough_columns:
        features[column] = _numeric_series(dataframe, column)

    if "last_event_timestamp" in dataframe.columns:
        event_timestamp = dataframe["last_event_timestamp"]
    else:
        event_timestamp = None

    return _finalize_feature_frame(
        features,
        source_project="project_1",
        source_table="public.user_features",
        entity_type="user",
        entity_id=_text_series(dataframe, "user_id"),
        event_timestamp=event_timestamp,
    )


def transform_customer_360(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform Project 2/3 customer mart into unified training features."""

    features = _empty_feature_frame(dataframe.index)

    column_mapping = {
        "total_order_count": "order_count",
        "gross_revenue": "gross_revenue",
        "total_discount_amount": "total_discount_amount",
        "lifetime_value": "lifetime_value",
        "average_order_value": "average_order_value",
        "total_units_purchased": "total_units_purchased",
        "product_categories_purchased": "product_categories_purchased",
        "distinct_products_purchased": "distinct_products_purchased",
        "days_since_last_order": "days_since_last_order",
        "web_event_count": "web_event_count",
        "session_count": "session_count",
        "purchase_intent_event_count": "purchase_intent_event_count",
        "customer_conversion_rate_pct": "customer_conversion_rate_pct",
        "avg_engagement_score": "avg_engagement_score",
        "avg_purchase_probability": "avg_purchase_probability",
        "avg_cart_abandonment_probability": "avg_cart_abandonment_probability",
    }

    for source_column, target_column in column_mapping.items():
        features[target_column] = _numeric_series(dataframe, source_column)

    features["repeat_purchase_signal"] = _bool_series(
        dataframe,
        "repeat_purchase_signal",
    ).astype("int64")

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table="marts.mart_customer_360",
        entity_type="customer",
        entity_id=_text_series(dataframe, "customer_id"),
        event_timestamp=dataframe.get("last_order_timestamp"),
    )


def transform_campaign_performance(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform Project 2/3 campaign performance mart into unified training features."""

    features = _empty_feature_frame(dataframe.index)

    column_mapping = {
        "campaign_budget": "campaign_budget",
        "total_spend": "total_spend",
        "budget_remaining": "budget_remaining",
        "impressions": "impressions",
        "clicks": "clicks",
        "ad_platform_conversions": "ad_platform_conversions",
        "web_event_count": "web_event_count",
        "session_count": "session_count",
        "purchase_event_count": "purchase_event_count",
        "order_count": "order_count",
        "attributed_revenue": "attributed_revenue",
        "average_order_value": "average_order_value",
        "roas": "roas",
        "click_through_rate_pct": "click_through_rate_pct",
        "cost_per_click": "cost_per_click",
        "cost_per_order": "cost_per_order",
        "session_to_order_conversion_rate_pct": "session_to_order_conversion_rate_pct",
        "avg_engagement_score": "avg_engagement_score",
        "avg_purchase_probability": "avg_purchase_probability",
    }

    for source_column, target_column in column_mapping.items():
        features[target_column] = _numeric_series(dataframe, source_column)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table="marts.mart_campaign_performance",
        entity_type="campaign",
        entity_id=_text_series(dataframe, "campaign_id"),
        event_timestamp=dataframe.get("campaign_end_date"),
    )


def transform_marketing_funnel(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform Project 2/3 marketing funnel mart into unified training features."""

    features = _empty_feature_frame(dataframe.index)

    column_mapping = {
        "total_event_count": "total_events",
        "session_count": "session_count",
        "product_view_event_count": "product_view_count",
        "cart_event_count": "add_to_cart_count",
        "purchase_event_count": "purchase_count",
        "funnel_conversion_rate_pct": "funnel_conversion_rate_pct",
        "avg_engagement_score": "avg_engagement_score",
        "avg_purchase_probability": "avg_purchase_probability",
        "avg_cart_abandonment_probability": "avg_cart_abandonment_probability",
    }

    for source_column, target_column in column_mapping.items():
        features[target_column] = _numeric_series(dataframe, source_column)

    entity_id = (
        _text_series(dataframe, "campaign_id")
        + "|"
        + _text_series(dataframe, "event_date")
        + "|"
        + _text_series(dataframe, "channel_name")
    )

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table="marts.mart_marketing_funnel",
        entity_type="funnel_day",
        entity_id=entity_id,
        event_timestamp=dataframe.get("event_date"),
    )


def transform_product_sales(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform Project 2/3 product mart into unified training features."""

    features = _empty_feature_frame(dataframe.index)

    column_mapping = {
        "original_price": "avg_event_price",
        "current_price": "max_event_price",
        "inventory_remaining": "inventory_remaining",
        "order_count": "order_count",
        "units_sold": "units_sold",
        "product_revenue": "product_revenue",
        "average_selling_price": "average_order_value",
        "product_web_event_count": "web_event_count",
        "product_session_count": "session_count",
        "product_view_event_count": "product_view_count",
        "cart_event_count": "add_to_cart_count",
        "purchase_event_count": "purchase_count",
        "product_conversion_rate_pct": "product_conversion_rate_pct",
        "avg_product_engagement_score": "avg_product_engagement_score",
        "category_revenue_rank": "category_revenue_rank",
        "category_revenue_share_pct": "category_revenue_share_pct",
    }

    for source_column, target_column in column_mapping.items():
        features[target_column] = _numeric_series(dataframe, source_column)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table="marts.mart_product_sales",
        entity_type="product",
        entity_id=_text_series(dataframe, "product_id"),
        event_timestamp=dataframe.get("last_sold_timestamp"),
    )


def transform_web_events(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Project 2/3 warehouse web events by customer into training features."""

    working = dataframe.copy()

    for column in [
        "time_on_page_sec",
        "scroll_depth_percent",
        "engagement_score",
        "purchase_probability",
        "cart_abandonment_probability",
        "event_value",
        "api_latency_ms",
        "page_load_time_ms",
        "fraud_score",
    ]:
        working[column] = _numeric_series(working, column)

    working["event_type"] = _text_series(working, "event_type")
    working["customer_sk"] = _text_series(working, "customer_sk")

    grouped = working.groupby("customer_sk", dropna=False).agg(
        total_events=("event_id", "count"),
        page_view_count=("event_type", lambda values: (values == "page_view").sum()),
        product_view_count=("event_type", lambda values: (values == "product_view").sum()),
        add_to_cart_count=("event_type", lambda values: (values == "add_to_cart").sum()),
        purchase_count=("event_type", lambda values: (values == "purchase").sum()),
        avg_engagement_score=("engagement_score", "mean"),
        avg_purchase_probability=("purchase_probability", "mean"),
        avg_cart_abandonment_probability=("cart_abandonment_probability", "mean"),
        avg_event_value=("event_value", "mean"),
        avg_api_latency_ms=("api_latency_ms", "mean"),
        avg_page_load_time_ms=("page_load_time_ms", "mean"),
        avg_fraud_score=("fraud_score", "mean"),
        session_count=("session_id", pd.Series.nunique),
        source_event_timestamp=("event_timestamp", "max"),
    )

    grouped = grouped.reset_index()

    features = _empty_feature_frame(grouped.index)

    for column in [
        "total_events",
        "page_view_count",
        "product_view_count",
        "add_to_cart_count",
        "purchase_count",
        "avg_engagement_score",
        "avg_purchase_probability",
        "avg_cart_abandonment_probability",
        "avg_event_value",
        "avg_api_latency_ms",
        "avg_page_load_time_ms",
        "avg_fraud_score",
        "session_count",
    ]:
        features[column] = _numeric_series(grouped, column)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table="warehouse.fact_web_events",
        entity_type="web_customer",
        entity_id=_text_series(grouped, "customer_sk"),
        event_timestamp=grouped["source_event_timestamp"],
    )


def extract_project1_features(config: dict[str, Any]) -> pd.DataFrame:
    """Extract and transform Project 1 configured source."""

    if not config.get("enabled", True):
        return pd.DataFrame()

    engine = build_postgres_engine(config)
    raw = read_table(engine, config["schema"], config["table"])
    LOGGER.info("Extracted Project 1 rows: %s", len(raw))
    return transform_project1_user_features(raw)


def extract_project23_features(config: dict[str, Any]) -> pd.DataFrame:
    """Extract and transform Project 2/3 configured sources."""

    if not config.get("enabled", True):
        return pd.DataFrame()

    engine = build_postgres_engine(config)
    table_configs = config["tables"]

    frames = []

    customer_cfg = table_configs["customer_360"]
    frames.append(
        transform_customer_360(
            read_table(engine, customer_cfg["schema"], customer_cfg["table"])
        )
    )

    campaign_cfg = table_configs["campaign_performance"]
    frames.append(
        transform_campaign_performance(
            read_table(engine, campaign_cfg["schema"], campaign_cfg["table"])
        )
    )

    funnel_cfg = table_configs["marketing_funnel"]
    frames.append(
        transform_marketing_funnel(
            read_table(engine, funnel_cfg["schema"], funnel_cfg["table"])
        )
    )

    product_cfg = table_configs["product_sales"]
    frames.append(
        transform_product_sales(
            read_table(engine, product_cfg["schema"], product_cfg["table"])
        )
    )

    web_cfg = table_configs["web_events"]
    frames.append(
        transform_web_events(
            read_table(engine, web_cfg["schema"], web_cfg["table"])
        )
    )

    output = pd.concat(frames, ignore_index=True)
    LOGGER.info("Extracted Project 2/3 transformed rows: %s", len(output))
    return output


def build_unified_feature_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    """Build the unified Project 4 real-source feature dataframe."""

    frames = [
        extract_project1_features(config["project_1"]),
        extract_project23_features(config["project_2_3"]),
    ]

    non_empty_frames = [frame for frame in frames if not frame.empty]

    if not non_empty_frames:
        raise SourceExtractionError("No source projects produced rows")

    unified = pd.concat(non_empty_frames, ignore_index=True)

    minimum_total_rows = int(config.get("minimum_total_rows", 1))
    if len(unified) < minimum_total_rows:
        raise SourceExtractionError(
            f"Unified feature dataframe has {len(unified)} rows, below minimum {minimum_total_rows}"
        )

    if unified["entity_id"].isna().any():
        raise SourceExtractionError("Unified feature dataframe contains null entity_id values")

    return unified


def _source_table_names(config: dict[str, Any]) -> list[str]:
    tables = []

    project1 = config["project_1"]
    if project1.get("enabled", True):
        tables.append(f'{project1["schema"]}.{project1["table"]}')

    project23 = config["project_2_3"]
    if project23.get("enabled", True):
        for table_config in project23["tables"].values():
            tables.append(f'{table_config["schema"]}.{table_config["table"]}')

    return tables


def write_snapshot(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    output_root: str | Path | None = None,
) -> SnapshotWriteResult:
    """Write the unified dataframe and metadata as a training snapshot."""

    now = datetime.now(UTC)
    snapshot_date = now.date().isoformat()
    snapshot_prefix = config.get("output_snapshot_prefix", "real_source")
    snapshot_id = f"{snapshot_prefix}_{now.strftime('%Y%m%dT%H%M%SZ')}"

    root = Path(output_root or config["output"]["root"])
    snapshot_path = root / f"snapshot_date={snapshot_date}" / f"snapshot_id={snapshot_id}"
    snapshot_path.mkdir(parents=True, exist_ok=False)

    features_path = snapshot_path / "features.parquet"
    metadata_path = snapshot_path / "metadata.json"

    dataframe.to_parquet(features_path, index=False)

    source_projects = []
    if config["project_1"].get("enabled", True):
        source_projects.append("project_1")
    if config["project_2_3"].get("enabled", True):
        source_projects.append("project_2_3")

    metadata = {
        "snapshot_id": snapshot_id,
        "snapshot_type": config["snapshot_type"],
        "feature_schema_version": config["feature_schema_version"],
        "created_at": now.isoformat(),
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "source_projects": source_projects,
        "source_tables": _source_table_names(config),
        "source_paths": [
            "~/Desktop/Project-1- Real time suggestions/PostgreSQL/public.user_features",
            "~/Desktop/Project-2-Batch-Lakehouse-Marketing-Analytics/PostgreSQL/marts+warehouse",
        ],
        "entity_type_counts": dataframe["entity_type"].value_counts().to_dict(),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "data_quality_status": "passed",
        "notes": (
            "Real-source extract from Project 1 PostgreSQL user_features and "
            "Project 2/3 PostgreSQL warehouse/marts."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    LOGGER.info("Wrote real-source snapshot: %s", snapshot_path)

    return SnapshotWriteResult(
        snapshot_id=snapshot_id,
        snapshot_path=snapshot_path,
        features_path=features_path,
        metadata_path=metadata_path,
        row_count=len(dataframe),
        feature_schema_version=config["feature_schema_version"],
        snapshot_type=config["snapshot_type"],
    )


def run_source_extract(
    config_path: str | Path,
    output_root: str | Path | None = None,
) -> SnapshotWriteResult:
    """Run end-to-end source extraction and snapshot writing."""

    config = load_config(config_path)
    dataframe = build_unified_feature_dataframe(config)
    return write_snapshot(dataframe, config, output_root=output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract real-source features for Project 4 training.",
    )
    parser.add_argument("--config", required=True, help="Path to source_extract.yaml")
    parser.add_argument("--output-root", default=None, help="Output root for training snapshots")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    result = run_source_extract(args.config, output_root=args.output_root)

    print("OK: real-source feature snapshot written")
    print(f"snapshot_id={result.snapshot_id}")
    print(f"snapshot_type={result.snapshot_type}")
    print(f"feature_schema_version={result.feature_schema_version}")
    print(f"row_count={result.row_count}")
    print(f"features_path={result.features_path}")
    print(f"metadata_path={result.metadata_path}")


if __name__ == "__main__":
    main()
