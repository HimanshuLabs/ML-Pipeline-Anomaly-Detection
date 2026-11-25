"""Extract real Project 1 and Project 2/3 features into a Project 4 training snapshot.

This module intentionally reads from local PostgreSQL sources instead of Kafka/Spark.
For retraining, Project 4 needs stable historical feature snapshots, not a live replay dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np
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
    content_sha256: str | None = None
    training_dataset_id: str | None = None


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


# Every session on a Project 1 / Project 2/3 source is read-only on the server side
# and bounded in time. Only the source engines get these; Project 4's own database
# engine (database.get_engine) is unchanged. The bound is the source's
# statement_timeout_ms in configs/source_extract.yaml, 120 s when it is absent.
DEFAULT_SOURCE_STATEMENT_TIMEOUT_MS = 120_000


def source_session_options(source: dict[str, Any]) -> str:
    """libpq options of one source session: read-only, every statement bounded in time."""

    value = source.get("statement_timeout_ms", DEFAULT_SOURCE_STATEMENT_TIMEOUT_MS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceExtractionError(
            f"statement_timeout_ms must be a whole number of milliseconds, at least 1; "
            f"got {value!r}"
        )
    return f"-c default_transaction_read_only=on -c statement_timeout={value}"


def build_postgres_engine(source: dict[str, Any]) -> Engine:
    """Create a read-only SQLAlchemy engine for one configured upstream source."""

    options = source_session_options(source)
    password = quote_plus(_password_from_env(source["password_env_var"]))
    username = quote_plus(str(source["username"]))
    host = source["host"]
    port = source["port"]
    database = source["database"]

    url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    engine = create_engine(
        url,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"options": options},
    )
    return engine.execution_options(postgresql_readonly=True)


def _require_read_only_session(connection: Any) -> None:
    """Refuse to read from a session the server does not report as read-only."""

    value = connection.execute(text("SHOW transaction_read_only")).scalar_one()
    if str(value) != "on":
        raise SourceExtractionError(
            f"source session is not read-only (transaction_read_only={value!r}); nothing was read"
        )


def read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    """Read a PostgreSQL table with quoted names, on a session proven read-only first."""

    query = text(f'SELECT * FROM "{schema}"."{table}"')
    with engine.connect() as connection:
        _require_read_only_session(connection)
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

WEB_EVENT_ORDER_COLUMNS = (
    "event_id",
    "customer_sk",
    "session_id",
    "event_type",
    "event_timestamp",
    *WEB_EVENT_MEASURES,
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

    # Averages are float sums, so they depend on the order of a customer's events.
    # A stable sort on the event key (then every other column read) fixes that order,
    # whatever order the database, or a bucketed read, returned the rows in.
    working = working.sort_values(
        by=list(WEB_EVENT_ORDER_COLUMNS), kind="mergesort", na_position="last"
    ).reset_index(drop=True)

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


# --- fact_web_events in buckets of business customers ---------------------------------
#
# The whole table does not fit in memory at scale (79 lakh rows x 35 columns), so it
# is read in buckets of BUSINESS customers (dim_customer.customer_id): every SCD2
# version key (customer_sk) of a customer is in the same bucket, and the sentinel
# members (customer_sk <= 0, UNKNOWN, INVALID) are all in the first one. Each bucket
# goes through the unchanged transform_web_events, so the result is the same frame,
# the same statistics and the same hash as one read of the whole table.

WEB_EVENT_BUCKET_CUSTOMERS = 50_000
WEB_EVENT_BUCKETS_ENV = "P4_WEB_EVENT_BUCKETS"
WEB_EVENT_READ_COLUMNS = REQUIRED_SOURCE_COLUMNS[WEB_EVENTS_TABLE]
CUSTOMER_DIMENSION_READ_COLUMNS = REQUIRED_SOURCE_COLUMNS[CUSTOMER_DIMENSION_TABLE]


def _business_keys(customer_dimension: pd.DataFrame) -> pd.DataFrame:
    return customer_dimension[["customer_sk", "customer_id"]].drop_duplicates(
        subset=["customer_sk"]
    )


def _sentinel_customer_ids(keys: pd.DataFrame) -> set[Any]:
    """Business ids read with the sentinel members: UNKNOWN, INVALID, NULL, or any sk <= 0."""

    ids = keys["customer_id"]
    is_sentinel = (
        ids.isna()
        | ids.astype("string").isin(SENTINEL_CUSTOMER_IDS)
        | (pd.to_numeric(keys["customer_sk"], errors="coerce") <= 0)
    )
    return set(ids[is_sentinel].tolist())


def web_event_bucket_count(customer_dimension: pd.DataFrame, setting: str | None) -> int:
    """Buckets for fact_web_events: P4_WEB_EVENT_BUCKETS, else one per 50,000 customers."""

    if setting:
        try:
            count = int(setting)
        except ValueError:
            raise SourceExtractionError(
                f"{WEB_EVENT_BUCKETS_ENV} must be a whole number, got {setting!r}"
            ) from None
        if count < 1:
            raise SourceExtractionError(f"{WEB_EVENT_BUCKETS_ENV} must be at least 1, got {count}")
        return count

    keys = _business_keys(customer_dimension)
    sentinels = _sentinel_customer_ids(keys)
    customers = keys.loc[~keys["customer_id"].isin(sentinels), "customer_id"].nunique()
    return max(1, -(-int(customers) // WEB_EVENT_BUCKET_CUSTOMERS))


def web_event_buckets(customer_dimension: pd.DataFrame, bucket_count: int) -> list[list[int]]:
    """customer_sk lists, one per non-empty bucket; together every dimension sk once."""

    keys = _business_keys(customer_dimension)
    sentinels = _sentinel_customer_ids(keys)
    is_sentinel = keys["customer_id"].isin(sentinels)
    customers = sorted(keys.loc[~is_sentinel, "customer_id"].unique().tolist())
    ranges = [part for part in np.array_split(np.asarray(customers, dtype=object), bucket_count)
              if len(part)] or [np.asarray([], dtype=object)]

    bucket_of = {customer: index for index, part in enumerate(ranges) for customer in part}
    buckets: list[list[int]] = [[] for _ in ranges]
    for sk, customer in zip(keys["customer_sk"], keys["customer_id"], strict=True):
        buckets[0 if customer in sentinels or pd.isna(customer) else bucket_of[customer]].append(
            int(sk)
        )
    return [bucket for bucket in buckets if bucket]


def _unmapped_events_error(unmapped: int) -> SourceExtractionError:
    return SourceExtractionError(
        f"{WEB_EVENTS_TABLE} has {unmapped} event(s) whose customer_sk is not in "
        f"{CUSTOMER_DIMENSION_TABLE}"
    )


def transform_web_events_in_buckets(
    read_events: Callable[[list[int] | None], pd.DataFrame],
    count_unmapped: Callable[[], int],
    customer_dimension: pd.DataFrame,
    *,
    report: ExtractionReport | None = None,
    bucket_count: int = 1,
) -> pd.DataFrame:
    """transform_web_events over fact_web_events read whole (one bucket) or in buckets.

    read_events(None) returns every event, read_events(sks) the events of those
    customer_sk values. With buckets, NULL and unmapped customer_sk values are
    counted first and refused as transform_web_events refuses them.
    """

    report = report if report is not None else ExtractionReport()
    if bucket_count <= 1:
        events = read_events(None)
        if events.empty:
            raise SourceExtractionError(f"Source table returned zero rows: {WEB_EVENTS_TABLE}")
        return transform_web_events(events, customer_dimension, report=report)

    unmapped = count_unmapped()
    if unmapped:
        raise _unmapped_events_error(unmapped)

    frames = []
    rows = 0
    for customer_sks in web_event_buckets(customer_dimension, bucket_count):
        events = read_events(customer_sks)
        rows += len(events)
        frames.append(transform_web_events(events, customer_dimension, report=report))
        LOGGER.info(
            "Read %s web event(s) of %s customer_sk value(s)", len(events), len(customer_sks)
        )
    if not rows:
        raise SourceExtractionError(f"Source table returned zero rows: {WEB_EVENTS_TABLE}")
    return pd.concat(frames, ignore_index=True)


def read_web_events(
    engine: Engine,
    schema: str,
    table: str,
    customer_sks: list[int] | None = None,
) -> pd.DataFrame:
    """The columns transform_web_events reads, of every event or of these customer_sk values."""

    with engine.connect() as connection:
        _require_read_only_session(connection)
        available = pd.read_sql_query(
            text(f'SELECT * FROM "{schema}"."{table}" LIMIT 0'), connection
        ).columns
        _require_columns(pd.DataFrame(columns=available), WEB_EVENTS_TABLE, WEB_EVENT_READ_COLUMNS)

        columns = ", ".join(f'"{column}"' for column in WEB_EVENT_READ_COLUMNS)
        query = f'SELECT {columns} FROM "{schema}"."{table}"'
        params: dict[str, Any] = {}
        if customer_sks is not None:
            query += ' WHERE "customer_sk" = ANY(:customer_sks)'
            params["customer_sks"] = [int(sk) for sk in customer_sks]
        return pd.read_sql_query(text(query), connection, params=params)


def read_customer_dimension(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    """customer_sk and customer_id of the customer dimension: the only columns used."""

    with engine.connect() as connection:
        _require_read_only_session(connection)
        available = pd.read_sql_query(
            text(f'SELECT * FROM "{schema}"."{table}" LIMIT 0'), connection
        ).columns
        _require_columns(
            pd.DataFrame(columns=available),
            CUSTOMER_DIMENSION_TABLE,
            CUSTOMER_DIMENSION_READ_COLUMNS,
        )
        columns = ", ".join(f'"{column}"' for column in CUSTOMER_DIMENSION_READ_COLUMNS)
        query = text(f'SELECT {columns} FROM "{schema}"."{table}"')
        dataframe = pd.read_sql_query(query, connection)

    if dataframe.empty:
        raise SourceExtractionError(f"Source table returned zero rows: {schema}.{table}")

    return dataframe


def count_unmapped_web_events(engine: Engine, schema: str, table: str, dimension: str) -> int:
    """Events whose customer_sk is NULL or not in the customer dimension."""

    query = text(
        f'SELECT count(*) FROM "{schema}"."{table}" AS f '
        f'WHERE f."customer_sk" IS NULL OR NOT EXISTS ('
        f'SELECT 1 FROM "{schema}"."{dimension}" AS d WHERE d."customer_sk" = f."customer_sk")'
    )
    with engine.connect() as connection:
        _require_read_only_session(connection)
        return int(connection.execute(query).scalar_one())


def extract_web_event_features(
    engine: Engine,
    web_config: dict[str, Any],
    customer_dimension: pd.DataFrame,
    report: ExtractionReport | None = None,
) -> pd.DataFrame:
    """Per business customer web features from fact_web_events, read in buckets at scale."""

    schema, table = web_config["schema"], web_config["table"]
    dimension = web_config.get("customer_dimension_table", DEFAULT_CUSTOMER_DIMENSION)
    bucket_count = web_event_bucket_count(customer_dimension, os.getenv(WEB_EVENT_BUCKETS_ENV))
    LOGGER.info("Reading %s.%s in %s bucket(s)", schema, table, bucket_count)
    return transform_web_events_in_buckets(
        lambda customer_sks: read_web_events(engine, schema, table, customer_sks),
        lambda: count_unmapped_web_events(engine, schema, table, dimension),
        customer_dimension,
        report=report,
        bucket_count=bucket_count,
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
    customer_dimension = read_customer_dimension(
        engine,
        web_config["schema"],
        web_config.get("customer_dimension_table", DEFAULT_CUSTOMER_DIMENSION),
    )

    frames = [
        transform_customer_360(_read("customer_360"), report=report),
        transform_campaign_performance(_read("campaign_performance"), report=report),
        transform_marketing_funnel(_read("marketing_funnel"), report=report),
        transform_product_sales(_read("product_sales"), report=report),
        extract_web_event_features(engine, web_config, customer_dimension, report=report),
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


def _source_tables(config: dict[str, Any]) -> list[tuple[str, str]]:
    """(config project key, "schema.table") for every table the extract reads, in order.

    Both source_tables and source_paths come from this list, so they always agree.
    """
    tables = []

    project1 = config["project_1"]
    if project1.get("enabled", True):
        tables.append(("project_1", f'{project1["schema"]}.{project1["table"]}'))

    project23 = config["project_2_3"]
    if project23.get("enabled", True):
        for name, table_config in project23["tables"].items():
            tables.append(("project_2_3", f'{table_config["schema"]}.{table_config["table"]}'))
            if name == "web_events":
                # extract_project23_features reads the customer dimension for web events too.
                dimension = table_config.get("customer_dimension_table", DEFAULT_CUSTOMER_DIMENSION)
                tables.append(("project_2_3", f'{table_config["schema"]}.{dimension}'))

    return tables


# --- deterministic snapshot content -------------------------------------------------
#
# Only the web_customer rows are provably unique per (entity_type, entity_id) (a
# groupby); the other types rely on upstream keys. So rows are ordered by every
# column, in this fixed order, which makes the order total whatever the sources do.
SNAPSHOT_LEADING_COLUMNS = (
    "entity_type",
    "entity_id",
    "source_project",
    "source_table",
    "source_event_timestamp",
)
# When this run happened, not what was extracted: kept in the file, not in the hash.
SNAPSHOT_RUN_COLUMNS = ("extracted_at",)
TRAINING_DATASET_ID_NAMESPACE = "project4.training_dataset"


def _snapshot_column_order(frame: pd.DataFrame) -> list[str]:
    known = [*SNAPSHOT_LEADING_COLUMNS, *MODEL_FEATURE_COLUMNS, *SNAPSHOT_RUN_COLUMNS]
    present = [column for column in known if column in frame.columns]
    return present + sorted(str(column) for column in frame.columns if column not in known)


def sort_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame in a deterministic row order (every column, fixed order, stable sort)."""

    return frame.sort_values(
        by=_snapshot_column_order(frame), kind="mergesort", na_position="last"
    ).reset_index(drop=True)


CONTENT_HASH_CHUNK_ROWS = 100_000


def content_sha256(frame: pd.DataFrame, *, chunk_rows: int = CONTENT_HASH_CHUNK_ROWS) -> str:
    """sha256 of the data: rows sorted, fixed column order, stable CSV, run columns left out.

    The CSV is hashed in chunks of rows, never held whole. The bytes are those of
    one to_csv of the whole frame because every value is formatted on its own;
    columns pandas formats per column (datetime, timedelta, categorical) are
    refused rather than hashed differently.
    """

    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be at least 1, got {chunk_rows}")
    columns = [c for c in _snapshot_column_order(frame) if c not in SNAPSHOT_RUN_COLUMNS]
    formatted_per_column = [
        str(column)
        for column in columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
        or pd.api.types.is_timedelta64_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]
    if formatted_per_column:
        raise SourceExtractionError(
            "content_sha256 cannot stream column(s) pandas formats per column: "
            + ", ".join(formatted_per_column)
        )

    ordered = sort_feature_frame(frame).loc[:, columns]
    digest = hashlib.sha256()
    digest.update(ordered.iloc[:0].to_csv(index=False, lineterminator="\n").encode("utf-8"))
    for start in range(0, len(ordered), chunk_rows):
        chunk = ordered.iloc[start : start + chunk_rows]
        digest.update(chunk.to_csv(index=False, header=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def training_dataset_id_for(snapshot_id: str, sha256: str, fingerprint: str) -> str:
    """Deterministic training_dataset_id of one snapshot's data and feature set."""

    key = "|".join([TRAINING_DATASET_ID_NAMESPACE, snapshot_id, sha256, fingerprint])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


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
    from anomaly_detection.registry import _write_bytes_atomic
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

    # One row order for the hash and the parquet file, whatever order the sources returned.
    # Before the directory exists, so a sort or hash failure leaves nothing behind.
    dataframe = sort_feature_frame(dataframe)
    sha256 = content_sha256(dataframe)
    fingerprint = feature_set_fingerprint(MODEL_FEATURE_COLUMNS)

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

    training_dataset_id = (
        training_dataset_id_for(snapshot_id, sha256, fingerprint) if validation.is_valid else None
    )

    source_tables = _source_tables(config)
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
        "source_tables": [table for _project, table in source_tables],
        # "<config project key>/<schema>.<table>": no host, port, database, user or home path.
        "source_paths": [f"{project}/{table}" for project, table in source_tables],
        "entity_type_counts": dataframe["entity_type"].value_counts().to_dict(),
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_set_fingerprint": fingerprint,
        "content_sha256": sha256,
        # Only a trainable snapshot is a training dataset.
        "training_dataset_id": training_dataset_id,
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

    # No trailing newline: the bytes stay those of every earlier snapshot.
    _write_bytes_atomic(
        metadata_path, json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
    )

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
        content_sha256=sha256,
        training_dataset_id=training_dataset_id,
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
    print(f"content_sha256={result.content_sha256}")
    print(f"training_dataset_id={result.training_dataset_id}")
    print(f"features_path={result.features_path}")
    print(f"metadata_path={result.metadata_path}")


if __name__ == "__main__":
    main()
