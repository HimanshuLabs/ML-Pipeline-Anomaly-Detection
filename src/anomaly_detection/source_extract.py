"""Extract real Project 1 and Project 2/3 features into a Project 4 training snapshot.

This module intentionally reads from local PostgreSQL sources instead of Kafka/Spark.
For retraining, Project 4 needs stable historical feature snapshots, not a live replay dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    # 1 when mart_customer_360.days_since_last_order is NULL (never ordered).
    "never_ordered",
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
    data_quality_status: str = "passed"


class SourceExtractionError(RuntimeError):
    """Raised when real-source extraction cannot produce a valid snapshot."""


def configure_logging() -> None:
    """Configure module logging for CLI usage."""

    from anomaly_detection.logging_config import configure_logging as configure_project_logging

    configure_project_logging()


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


# --- source contracts --------------------------------------------------------------
#
# Every source column a transform reads is REQUIRED: a missing one raises
# SourceExtractionError naming the table and the column(s). Nothing is silently
# created as 0.0 for a required input; features a table does not provide are
# padded with 0.0 by design (one unified feature frame across entity types).

P1_TABLE = "public.user_features"
CUSTOMER_360_TABLE = "marts.mart_customer_360"
CAMPAIGN_TABLE = "marts.mart_campaign_performance"
FUNNEL_TABLE = "marts.mart_marketing_funnel"
PRODUCT_TABLE = "marts.mart_product_sales"
WEB_EVENTS_TABLE = "warehouse.fact_web_events"
CUSTOMER_DIMENSION_TABLE = "warehouse.dim_customer"
DEFAULT_CUSTOMER_DIMENSION = "dim_customer"

# P2 marts mark rows for facts without a usable dimension member (member_type).
KNOWN_MEMBER_TYPE = "known"
SENTINEL_MEMBER_TYPES = frozenset({"unknown_member", "invalid_reference"})
# dim_customer sentinel members (customer_sk 0 and -1).
SENTINEL_CUSTOMER_IDS = frozenset({"UNKNOWN", "INVALID"})

# mart_customer_360.days_since_last_order is NULL when a customer never ordered.
# It becomes this constant (10 years, beyond the data span) together with
# never_ordered = 1: a constant keeps training and serving identical (a data
# derived value such as max + 1 would change per snapshot), and it keeps the
# feature's direction (never ordered ~ longest ago).
DAYS_SINCE_LAST_ORDER_NEVER_ORDERED = 3650.0

P1_PASSTHROUGH_COLUMNS = (
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
    "session_count",
)

CUSTOMER_360_MAPPING = {
    "total_order_count": "order_count",
    "gross_revenue": "gross_revenue",
    "total_discount_amount": "total_discount_amount",
    "lifetime_value": "lifetime_value",
    "average_order_value": "average_order_value",
    "total_units_purchased": "total_units_purchased",
    "product_categories_purchased": "product_categories_purchased",
    "distinct_products_purchased": "distinct_products_purchased",
    "web_event_count": "web_event_count",
    "session_count": "session_count",
    "purchase_intent_event_count": "purchase_intent_event_count",
    "customer_conversion_rate_pct": "customer_conversion_rate_pct",
    "avg_engagement_score": "avg_engagement_score",
    "avg_purchase_probability": "avg_purchase_probability",
    "avg_cart_abandonment_probability": "avg_cart_abandonment_probability",
}

CAMPAIGN_MAPPING = {
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

FUNNEL_MAPPING = {
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

PRODUCT_MAPPING = {
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

# Averaged per business customer; NULL measures are skipped, not counted as 0.
WEB_EVENT_MEASURES = {
    "engagement_score": "avg_engagement_score",
    "purchase_probability": "avg_purchase_probability",
    "cart_abandonment_probability": "avg_cart_abandonment_probability",
    "event_value": "avg_event_value",
    "api_latency_ms": "avg_api_latency_ms",
    "page_load_time_ms": "avg_page_load_time_ms",
    "fraud_score": "avg_fraud_score",
}

# Derived from mart_customer_360 by transform_customer_360 (not 1:1 mapped).
CUSTOMER_360_DERIVED_FEATURES = ("days_since_last_order", "never_ordered", "repeat_purchase_signal")

# Counted per business customer from warehouse.fact_web_events by transform_web_events.
WEB_EVENT_COUNT_FEATURES = (
    "total_events",
    "page_view_count",
    "product_view_count",
    "add_to_cart_count",
    "purchase_count",
    "session_count",
)

ENTITY_TYPE_USER = "user"
ENTITY_TYPE_CUSTOMER = "customer"
ENTITY_TYPE_CAMPAIGN = "campaign"
ENTITY_TYPE_FUNNEL_DAY = "funnel_day"
ENTITY_TYPE_PRODUCT = "product"
ENTITY_TYPE_WEB_CUSTOMER = "web_customer"


def _in_model_order(features: Sequence[str]) -> tuple[str, ...]:
    wanted = set(features)
    return tuple(column for column in MODEL_FEATURE_COLUMNS if column in wanted)


# The features each entity type's source actually provides (the rest of the 52
# columns are 0.0 padding for that type). Built from the transforms' own
# mappings above; the segmented model trains and scores each type on these.
SEGMENT_FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    ENTITY_TYPE_USER: _in_model_order(P1_PASSTHROUGH_COLUMNS),
    ENTITY_TYPE_CUSTOMER: _in_model_order(
        [*CUSTOMER_360_MAPPING.values(), *CUSTOMER_360_DERIVED_FEATURES]
    ),
    ENTITY_TYPE_CAMPAIGN: _in_model_order(list(CAMPAIGN_MAPPING.values())),
    ENTITY_TYPE_FUNNEL_DAY: _in_model_order(list(FUNNEL_MAPPING.values())),
    ENTITY_TYPE_PRODUCT: _in_model_order(list(PRODUCT_MAPPING.values())),
    ENTITY_TYPE_WEB_CUSTOMER: _in_model_order(
        [*WEB_EVENT_COUNT_FEATURES, *WEB_EVENT_MEASURES.values()]
    ),
}

REQUIRED_SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    P1_TABLE: ("user_id", "last_event_timestamp", *P1_PASSTHROUGH_COLUMNS),
    CUSTOMER_360_TABLE: (
        "customer_id",
        "member_type",
        "last_order_timestamp",
        "days_since_last_order",
        "repeat_purchase_signal",
        *CUSTOMER_360_MAPPING,
    ),
    CAMPAIGN_TABLE: ("campaign_id", "member_type", "campaign_end_date", *CAMPAIGN_MAPPING),
    FUNNEL_TABLE: ("campaign_id", "member_type", "event_date", "channel_name", *FUNNEL_MAPPING),
    PRODUCT_TABLE: ("product_id", "member_type", "last_sold_timestamp", *PRODUCT_MAPPING),
    WEB_EVENTS_TABLE: (
        "event_id",
        "session_id",
        "customer_sk",
        "event_type",
        "event_timestamp",
        *WEB_EVENT_MEASURES,
    ),
    CUSTOMER_DIMENSION_TABLE: ("customer_sk", "customer_id"),
}


@dataclass
class TableExtractionStats:
    """What happened to one source table during extraction."""

    rows_read: int = 0
    rows_after_filter: int = 0
    sentinel_rows_dropped: int = 0
    null_fills: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractionReport:
    """Per-table extraction statistics, recorded in the snapshot metadata."""

    tables: dict[str, TableExtractionStats] = field(default_factory=dict)

    def table(self, name: str) -> TableExtractionStats:
        return self.tables.setdefault(name, TableExtractionStats())

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_table_stats": {name: asdict(stats) for name, stats in self.tables.items()},
            "sentinel_rows_dropped": sum(s.sentinel_rows_dropped for s in self.tables.values()),
        }


def _require_columns(dataframe: pd.DataFrame, table: str, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise SourceExtractionError(
            f"{table} is missing required column(s): {', '.join(missing)}"
        )


def _numeric_column(dataframe: pd.DataFrame, column: str, table: str) -> pd.Series:
    """Numeric values of a required column; NULLs stay NaN, text fails loudly."""

    values = pd.to_numeric(dataframe[column], errors="coerce")
    non_numeric = int((values.isna() & dataframe[column].notna()).sum())
    if non_numeric:
        raise SourceExtractionError(
            f"{table}.{column} has {non_numeric} non-numeric value(s)"
        )
    return values.astype("float64")


def _filled(values: pd.Series, column: str, stats: TableExtractionStats, fill: float) -> pd.Series:
    nulls = int(values.isna().sum())
    if nulls:
        stats.null_fills[column] = stats.null_fills.get(column, 0) + nulls
    return values.fillna(fill)


def _text_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return dataframe[column].astype("string")


def _drop_sentinel_members(
    dataframe: pd.DataFrame,
    table: str,
    stats: TableExtractionStats,
) -> pd.DataFrame:
    member_types = dataframe["member_type"].astype("string")
    unexpected = sorted(
        set(member_types.dropna().unique()) - SENTINEL_MEMBER_TYPES - {KNOWN_MEMBER_TYPE}
    )
    if unexpected or member_types.isna().any():
        raise SourceExtractionError(
            f"{table}.member_type has unexpected value(s): {unexpected or ['NULL']}"
        )

    is_sentinel = member_types.isin(SENTINEL_MEMBER_TYPES)
    stats.sentinel_rows_dropped += int(is_sentinel.sum())
    if is_sentinel.any():
        LOGGER.info("Dropped %s sentinel member row(s) from %s", int(is_sentinel.sum()), table)
    return dataframe.loc[~is_sentinel].reset_index(drop=True)


def _empty_feature_frame(index: pd.Index) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    for column in MODEL_FEATURE_COLUMNS:
        features[column] = 0.0
    return features


def _finalize_feature_frame(
    features: pd.DataFrame,
    *,
    source_project: str,
    source_table: str,
    entity_type: str,
    entity_id: pd.Series,
    event_timestamp: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach lineage/entity columns to a frame that already has every feature."""

    output = features.copy()
    output.insert(0, "source_project", source_project)
    output.insert(1, "source_table", source_table)
    output.insert(2, "entity_type", entity_type)
    output.insert(3, "entity_id", entity_id.astype("string").to_numpy())

    if event_timestamp is not None:
        output.insert(
            4, "source_event_timestamp", event_timestamp.astype("string").fillna("").to_numpy()
        )
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
    ].reset_index(drop=True)


def _mapped_features(
    dataframe: pd.DataFrame,
    mapping: dict[str, str],
    table: str,
    stats: TableExtractionStats,
) -> pd.DataFrame:
    features = _empty_feature_frame(dataframe.index)
    for source_column, feature in mapping.items():
        features[feature] = _filled(
            _numeric_column(dataframe, source_column, table), source_column, stats, 0.0
        )
    return features


def _start(table: str, dataframe: pd.DataFrame, report: ExtractionReport | None):
    _require_columns(dataframe, table, REQUIRED_SOURCE_COLUMNS[table])
    stats = (report or ExtractionReport()).table(table)
    stats.rows_read += len(dataframe)
    return stats


def transform_project1_user_features(
    dataframe: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Transform Project 1 public.user_features into unified training features."""

    stats = _start(P1_TABLE, dataframe, report)
    features = _mapped_features(
        dataframe,
        {column: column for column in P1_PASSTHROUGH_COLUMNS},
        P1_TABLE,
        stats,
    )
    stats.rows_after_filter += len(dataframe)

    return _finalize_feature_frame(
        features,
        source_project="project_1",
        source_table=P1_TABLE,
        entity_type=ENTITY_TYPE_USER,
        entity_id=_text_series(dataframe, "user_id"),
        event_timestamp=dataframe["last_event_timestamp"],
    )


def transform_customer_360(
    dataframe: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Transform Project 2/3 customer mart into unified training features."""

    stats = _start(CUSTOMER_360_TABLE, dataframe, report)
    dataframe = _drop_sentinel_members(dataframe, CUSTOMER_360_TABLE, stats)
    features = _mapped_features(dataframe, CUSTOMER_360_MAPPING, CUSTOMER_360_TABLE, stats)

    days = _numeric_column(dataframe, "days_since_last_order", CUSTOMER_360_TABLE)
    features["never_ordered"] = days.isna().astype("float64")
    features["days_since_last_order"] = _filled(
        days, "days_since_last_order", stats, DAYS_SINCE_LAST_ORDER_NEVER_ORDERED
    )

    repeat = dataframe["repeat_purchase_signal"]
    nulls = int(repeat.isna().sum())
    if nulls:
        stats.null_fills["repeat_purchase_signal"] = nulls
    features["repeat_purchase_signal"] = repeat.fillna(False).astype("bool").astype("float64")
    stats.rows_after_filter += len(dataframe)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table=CUSTOMER_360_TABLE,
        entity_type=ENTITY_TYPE_CUSTOMER,
        entity_id=_text_series(dataframe, "customer_id"),
        event_timestamp=dataframe["last_order_timestamp"],
    )


def transform_campaign_performance(
    dataframe: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Transform Project 2/3 campaign performance mart into unified training features."""

    stats = _start(CAMPAIGN_TABLE, dataframe, report)
    dataframe = _drop_sentinel_members(dataframe, CAMPAIGN_TABLE, stats)
    features = _mapped_features(dataframe, CAMPAIGN_MAPPING, CAMPAIGN_TABLE, stats)
    stats.rows_after_filter += len(dataframe)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table=CAMPAIGN_TABLE,
        entity_type=ENTITY_TYPE_CAMPAIGN,
        entity_id=_text_series(dataframe, "campaign_id"),
        event_timestamp=dataframe["campaign_end_date"],
    )


def transform_marketing_funnel(
    dataframe: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Transform Project 2/3 marketing funnel mart into unified training features."""

    stats = _start(FUNNEL_TABLE, dataframe, report)
    dataframe = _drop_sentinel_members(dataframe, FUNNEL_TABLE, stats)
    features = _mapped_features(dataframe, FUNNEL_MAPPING, FUNNEL_TABLE, stats)
    stats.rows_after_filter += len(dataframe)

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
        source_table=FUNNEL_TABLE,
        entity_type=ENTITY_TYPE_FUNNEL_DAY,
        entity_id=entity_id,
        event_timestamp=dataframe["event_date"],
    )


def transform_product_sales(
    dataframe: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Transform Project 2/3 product mart into unified training features."""

    stats = _start(PRODUCT_TABLE, dataframe, report)
    dataframe = _drop_sentinel_members(dataframe, PRODUCT_TABLE, stats)
    features = _mapped_features(dataframe, PRODUCT_MAPPING, PRODUCT_TABLE, stats)
    stats.rows_after_filter += len(dataframe)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table=PRODUCT_TABLE,
        entity_type=ENTITY_TYPE_PRODUCT,
        entity_id=_text_series(dataframe, "product_id"),
        event_timestamp=dataframe["last_sold_timestamp"],
    )


def transform_web_events(
    events: pd.DataFrame,
    customer_dimension: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Aggregate warehouse web events per customer BUSINESS key into training features.

    fact_web_events.customer_sk is an SCD2 version key (several per customer), so
    events are mapped to dim_customer.customer_id first. Events of the dimension's
    sentinel members (UNKNOWN, INVALID) are dropped.
    """

    stats = _start(WEB_EVENTS_TABLE, events, report)
    _require_columns(
        customer_dimension,
        CUSTOMER_DIMENSION_TABLE,
        REQUIRED_SOURCE_COLUMNS[CUSTOMER_DIMENSION_TABLE],
    )

    business_keys = (
        customer_dimension[["customer_sk", "customer_id"]]
        .drop_duplicates(subset=["customer_sk"])
        .rename(columns={"customer_id": "customer_business_id"})
    )
    working = events.merge(business_keys, on="customer_sk", how="left", validate="many_to_one")
    unmapped = int(working["customer_business_id"].isna().sum())
    if unmapped:
        raise SourceExtractionError(
            f"{WEB_EVENTS_TABLE} has {unmapped} event(s) whose customer_sk is not in "
            f"{CUSTOMER_DIMENSION_TABLE}"
        )

    is_sentinel = working["customer_business_id"].isin(SENTINEL_CUSTOMER_IDS)
    stats.sentinel_rows_dropped += int(is_sentinel.sum())
    if is_sentinel.any():
        LOGGER.info(
            "Dropped %s event(s) of sentinel customers from %s",
            int(is_sentinel.sum()),
            WEB_EVENTS_TABLE,
        )
    working = working.loc[~is_sentinel].reset_index(drop=True)
    stats.rows_after_filter += len(working)

    for column in WEB_EVENT_MEASURES:
        working[column] = _numeric_column(working, column, WEB_EVENTS_TABLE)
    working["event_type"] = working["event_type"].astype("string")

    grouped = working.groupby("customer_business_id", dropna=False, sort=True).agg(
        total_events=("event_id", "count"),
        page_view_count=("event_type", lambda values: (values == "page_view").sum()),
        product_view_count=("event_type", lambda values: (values == "product_view").sum()),
        add_to_cart_count=("event_type", lambda values: (values == "add_to_cart").sum()),
        purchase_count=("event_type", lambda values: (values == "purchase").sum()),
        session_count=("session_id", pd.Series.nunique),
        source_event_timestamp=("event_timestamp", "max"),
        **{feature: (column, "mean") for column, feature in WEB_EVENT_MEASURES.items()},
    ).reset_index()

    features = _empty_feature_frame(grouped.index)
    for column in WEB_EVENT_COUNT_FEATURES:
        features[column] = grouped[column].astype("float64")
    for column, feature in WEB_EVENT_MEASURES.items():
        # A customer whose events all have NULL here gets 0, counted per source column.
        features[feature] = _filled(grouped[feature].astype("float64"), column, stats, 0.0)

    return _finalize_feature_frame(
        features,
        source_project="project_2_3",
        source_table=WEB_EVENTS_TABLE,
        entity_type=ENTITY_TYPE_WEB_CUSTOMER,
        entity_id=grouped["customer_business_id"],
        event_timestamp=grouped["source_event_timestamp"],
    )


def extract_project1_features(
    config: dict[str, Any],
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Extract and transform Project 1 configured source."""

    if not config.get("enabled", True):
        return pd.DataFrame()

    engine = build_postgres_engine(config)
    raw = read_table(engine, config["schema"], config["table"])
    LOGGER.info("Extracted Project 1 rows: %s", len(raw))
    return transform_project1_user_features(raw, report=report)


def extract_project23_features(
    config: dict[str, Any],
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Extract and transform Project 2/3 configured sources (read-only SELECTs)."""

    if not config.get("enabled", True):
        return pd.DataFrame()

    engine = build_postgres_engine(config)
    table_configs = config["tables"]

    def _read(name: str) -> pd.DataFrame:
        table_config = table_configs[name]
        return read_table(engine, table_config["schema"], table_config["table"])

    web_config = table_configs["web_events"]
    customer_dimension = read_table(
        engine,
        web_config["schema"],
        web_config.get("customer_dimension_table", DEFAULT_CUSTOMER_DIMENSION),
    )

    frames = [
        transform_customer_360(_read("customer_360"), report=report),
        transform_campaign_performance(_read("campaign_performance"), report=report),
        transform_marketing_funnel(_read("marketing_funnel"), report=report),
        transform_product_sales(_read("product_sales"), report=report),
        transform_web_events(_read("web_events"), customer_dimension, report=report),
    ]

    output = pd.concat(frames, ignore_index=True)
    LOGGER.info("Extracted Project 2/3 transformed rows: %s", len(output))
    return output


def build_unified_feature_dataframe(
    config: dict[str, Any],
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Build the unified Project 4 real-source feature dataframe."""

    frames = [
        extract_project1_features(config["project_1"], report=report),
        extract_project23_features(config["project_2_3"], report=report),
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
    *,
    extraction_report: ExtractionReport | None = None,
) -> SnapshotWriteResult:
    """Validate the unified dataframe and write the snapshot.

    data_quality_status is "passed" only when the real-source validation passes.
    A failed snapshot gets metadata.json with the reasons and trainable=false, and
    no features.parquet, so it cannot be trained on.
    """

    from anomaly_detection.feature_validation import validate_real_source_features
    from anomaly_detection.training import feature_set_fingerprint

    now = datetime.now(UTC)
    snapshot_date = now.date().isoformat()
    snapshot_prefix = config.get("output_snapshot_prefix", "real_source")
    snapshot_id = f"{snapshot_prefix}_{now.strftime('%Y%m%dT%H%M%SZ')}"

    if output_root is not None:
        root = Path(output_root)
    else:
        # Relative roots in the config file are relative to the project, not the CWD.
        configured_root = Path(config["output"]["root"])
        root = configured_root if configured_root.is_absolute() else PROJECT_ROOT / configured_root
    snapshot_path = root / f"snapshot_date={snapshot_date}" / f"snapshot_id={snapshot_id}"
    snapshot_path.mkdir(parents=True, exist_ok=False)

    features_path = snapshot_path / "features.parquet"
    metadata_path = snapshot_path / "metadata.json"

    validation = validate_real_source_features(dataframe, MODEL_FEATURE_COLUMNS)
    data_quality_status = "passed" if validation.is_valid else "failed"
    if validation.is_valid:
        dataframe.to_parquet(features_path, index=False)
    else:
        LOGGER.error(
            "Snapshot %s failed validation and is not trainable: %s",
            snapshot_id,
            "; ".join(validation.errors),
        )

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
        "feature_set_fingerprint": feature_set_fingerprint(MODEL_FEATURE_COLUMNS),
        "data_quality_status": data_quality_status,
        "trainable": validation.is_valid,
        "validation_errors": list(validation.errors),
        "days_since_last_order_never_ordered_fill": DAYS_SINCE_LAST_ORDER_NEVER_ORDERED,
        **(extraction_report or ExtractionReport()).to_metadata(),
        "notes": (
            "Real-source extract from Project 1 PostgreSQL user_features and "
            "Project 2/3 PostgreSQL warehouse/marts."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    LOGGER.info("Wrote real-source snapshot (%s): %s", data_quality_status, snapshot_path)

    return SnapshotWriteResult(
        snapshot_id=snapshot_id,
        snapshot_path=snapshot_path,
        features_path=features_path,
        metadata_path=metadata_path,
        row_count=len(dataframe),
        feature_schema_version=config["feature_schema_version"],
        snapshot_type=config["snapshot_type"],
        data_quality_status=data_quality_status,
    )


def run_source_extract(
    config_path: str | Path,
    output_root: str | Path | None = None,
) -> SnapshotWriteResult:
    """Run end-to-end source extraction and snapshot writing.

    Raises SourceExtractionError when the snapshot fails validation (its
    metadata.json with the reasons is still written).
    """

    config = load_config(config_path)
    report = ExtractionReport()
    dataframe = build_unified_feature_dataframe(config, report=report)
    result = write_snapshot(dataframe, config, output_root=output_root, extraction_report=report)
    if result.data_quality_status != "passed":
        raise SourceExtractionError(
            f"Snapshot {result.snapshot_id} failed validation and is not trainable; "
            f"see {result.metadata_path}"
        )
    return result


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
    print(f"data_quality_status={result.data_quality_status}")
    print(f"row_count={result.row_count}")
    print(f"features_path={result.features_path}")
    print(f"metadata_path={result.metadata_path}")


if __name__ == "__main__":
    main()
