# Project 4 Data Contract — Anomaly Detection Feature Dataset

## Purpose

This document defines the feature input contract for Project 4: ML Pipeline & Anomaly Detection.

Project 4 does not train directly on raw Project 1 events or raw Project 2/3 warehouse tables. It first builds a controlled, model-ready feature dataset. This protects the anomaly model from schema drift, null spikes, bad types, impossible values, and silent upstream contract changes.

The current data contract supports:

- feature dataset generation
- feature validation
- schema version enforcement
- numeric model-ready features
- future training snapshots
- future batch inference
- future online inference
- future drift checks

This checkpoint is the gate before Isolation Forest training.

---

## Source systems

### Project 1 — real-time behavioral source

Project 1 provides event-level behavioral, commerce, recommendation, engagement, device, latency, and fraud/risk proxy fields.

Relevant raw Project 1 fields include:

| Raw field | Project 4 usage |
|---|---|
| `event_id` | rejected from model features; useful for traceability only |
| `session_id` | rejected from current model features; useful for future session-level aggregation |
| `user_id` | mapped to `entity_id` |
| `event_time` | accepted as source timestamp if `event_timestamp` is absent |
| `event_timestamp` | preferred source timestamp for feature windows |
| `event_type` | rejected from v001 numeric model features; future aggregation candidate |
| `cart_value` | used to calculate `avg_cart_value_7d` |
| `api_latency_ms` | used to calculate `avg_api_latency_ms` |
| `page_load_time_ms` | used to calculate `page_load_p95_ms` |
| `fraud_score` | used to calculate `fraud_score_avg` |
| `purchase_probability` | used to calculate `purchase_probability_delta` |
| `cart_abandonment_probability` | used to calculate `cart_abandonment_rate` |
| `discount_percent` | used to calculate `discount_sensitivity` |
| `schema_version` | source-side contract indicator; Project 4 emits its own feature schema version |
| `source` | rejected from model features; useful for lineage only |

Project 1 fields are not used blindly. Raw identifiers, personal fields, categorical strings, and high-cardinality text fields are intentionally excluded from the current numeric Isolation Forest feature contract.

### Project 2/3 — trusted historical warehouse and mart source

Project 2 and Project 3 are treated as the trusted historical analytics source. Project 3 is merged into Project 2 as the PostgreSQL warehouse, SCD2, marts, dbt, reconciliation, and reporting layer.

Relevant Project 2/3 signals include:

| Warehouse or mart signal | Project 4 usage |
|---|---|
| `campaign_roas` | used directly as a model feature |
| `conversion_rate` | used directly as a model feature |
| `customer_lifetime_value` | used directly as a model feature |
| warehouse facts and dimensions | planned source for richer training snapshots |
| dbt validation evidence | planned quality evidence for feature readiness |
| BI and funnel marts | planned source for delayed-truth analysis and monitoring context |

The current implementation accepts Project 2/3 signals as a dataframe input. Direct database extraction comes later.

---

## Current feature entity grain

The current feature dataset grain is:

`one row per entity_id`

Current entity mapping:

| Source column | Output column |
|---|---|
| `entity_id` | `entity_id` |
| `user_id` | `entity_id` |
| `customer_id` | `entity_id` |

The feature builder accepts these entity column names:

- `entity_id`
- `user_id`
- `customer_id`

The output always uses:

- `entity_id`

This keeps training, batch scoring, online inference, prediction logging, and drift checks aligned around one stable entity key.

---

## Current feature schema version

The current Project 4 feature schema version is:

`feature_schema_v001`

Every generated feature row must include:

`schema_version = feature_schema_v001`

The schema version is controlled by:

`configs/model_config.yaml`

The validation layer rejects rows where the feature dataframe schema version does not match the expected schema version.

This matters because model versions must be tied to the feature schema used during training. A model trained on `feature_schema_v001` should not silently score a different schema.

---

## Required output columns

Every model-ready feature dataframe must contain these columns:

| Column | Required | Type expectation | Purpose |
|---|---:|---|---|
| `entity_id` | yes | string-compatible | model entity key |
| `feature_timestamp` | yes | timestamp-compatible | point-in-time feature generation timestamp |
| `schema_version` | yes | string | Project 4 feature schema version |
| `avg_cart_value_7d` | yes | numeric | commerce behavior |
| `event_count_1h` | yes | numeric | recent behavioral burst signal |
| `avg_api_latency_ms` | yes | numeric | backend/system latency signal |
| `fraud_score_avg` | yes | numeric | fraud/risk proxy |
| `purchase_probability_delta` | yes | numeric | purchase-intent movement |
| `cart_abandonment_rate` | yes | numeric | funnel friction signal |
| `campaign_roas` | yes | numeric | campaign performance context |
| `conversion_rate` | yes | numeric | funnel conversion context |
| `customer_lifetime_value` | yes | numeric | customer value context |
| `discount_sensitivity` | yes | numeric | pricing/promotion behavior |
| `page_load_p95_ms` | yes | numeric | frontend/system latency signal |

The required numeric feature list is configured in:

`configs/model_config.yaml`

---

## Required numeric features

Current required numeric features:

| Feature | Source family | Feature domain |
|---|---|---|
| `avg_cart_value_7d` | Project 1 | commerce/cart |
| `event_count_1h` | Project 1 | behavioral/session |
| `avg_api_latency_ms` | Project 1 | system performance latency |
| `fraud_score_avg` | Project 1 | fraud/risk proxy |
| `purchase_probability_delta` | Project 1 | engagement/commerce proxy |
| `cart_abandonment_rate` | Project 1 | funnel risk |
| `campaign_roas` | Project 2/3 | campaign/warehouse signal |
| `conversion_rate` | Project 2/3 | funnel/warehouse signal |
| `customer_lifetime_value` | Project 2/3 | customer value signal |
| `discount_sensitivity` | Project 1 | pricing/promotion behavior |
| `page_load_p95_ms` | Project 1 | system performance latency |

This feature set is deliberately compact. The goal is stable production ML plumbing, not dumping every raw field into a model.

---

## Feature domains covered

| Domain | Current coverage |
|---|---|
| Behavioral/session | `event_count_1h` |
| Commerce/cart | `avg_cart_value_7d` |
| Recommendation interaction | not directly modeled in v001; planned after baseline stability |
| Engagement | indirectly represented by purchase probability and abandonment behavior |
| Fraud/risk proxy | `fraud_score_avg` |
| System performance latency | `avg_api_latency_ms`, `page_load_p95_ms` |
| Campaign/funnel/warehouse signals | `campaign_roas`, `conversion_rate`, `customer_lifetime_value` |
| Pricing/promotion | `discount_sensitivity` |

---

## Rejected raw fields

The following fields are intentionally rejected from the current model feature matrix:

| Raw field | Reason rejected from v001 model features |
|---|---|
| `event_id` | raw identifier; not a stable behavioral signal |
| `session_id` | raw identifier; useful for aggregation but not direct modeling |
| `user_name` | personal text field; not model-safe |
| `email` | personal identifier; excluded |
| `ip_address` | sensitive and high-cardinality; excluded |
| `product_id` | identifier; can be aggregated later |
| `product_name` | high-cardinality text; requires encoding strategy |
| `category` | categorical; not included until encoding is designed |
| `search_query` | free text; requires NLP/vector strategy later |
| `recommendation_algorithm` | categorical; not included until encoding is designed |
| `ab_test_group` | experiment metadata; can bias model behavior |
| `payment_method` | categorical; not included until encoding is designed |
| `device_type` | categorical; not included until encoding is designed |
| `operating_system` | categorical; not included until encoding is designed |
| `browser` | categorical; not included until encoding is designed |
| `network_type` | categorical; not included until encoding is designed |
| `app_version` | version string; useful for monitoring, not v001 model feature |
| `country` | location field; excluded until leakage/fairness review |
| `city` | location field; excluded until leakage/fairness review |
| `source` | lineage metadata, not model signal |

Rejected does not mean useless. It means not safe, stable, or necessary for the first numeric Isolation Forest contract.

---

## Validation rules

The implemented validation layer checks the following:

| Rule | Failure condition |
|---|---|
| dataframe type | input is not a pandas dataframe |
| empty input | dataframe has no rows |
| required columns | required entity, timestamp, schema, or feature columns are missing |
| schema version | `schema_version` does not match expected feature schema |
| entity key | `entity_id` contains nulls |
| timestamp parsing | `feature_timestamp` contains invalid timestamp values |
| numeric features | required model features contain non-numeric values |
| null fraction | a feature exceeds the allowed null threshold |
| finite values | a feature contains infinite values |
| non-negative features | a feature that cannot be negative contains negative values |
| probability features | a probability-style feature falls outside `[0, 1]` |

The current null threshold is:

`max_null_fraction = 0.05`

This means a feature fails validation if more than 5% of its values are null.

---

## Non-negative feature rules

The following fields must not contain negative values:

- `avg_cart_value_7d`
- `event_count_1h`
- `avg_api_latency_ms`
- `fraud_score_avg`
- `cart_abandonment_rate`
- `campaign_roas`
- `conversion_rate`
- `customer_lifetime_value`
- `discount_sensitivity`
- `page_load_p95_ms`
- `session_duration_sec`
- `items_viewed_in_session`
- `time_on_page_sec`
- `scroll_depth_percent`
- `hover_duration_ms`
- `api_latency_ms`
- `page_load_time_ms`
- `cart_value`
- `quantity`
- `discount_percent`
- `discounted_price`

Negative latency, negative event counts, negative cart values, and negative conversion rates are treated as invalid data.

---

## Probability-style feature rules

The following features are treated as probability-style fields and must remain in the range `[0, 1]`:

- `purchase_probability_delta`
- `cart_abandonment_rate`
- `conversion_rate`
- `fraud_score_avg`
- `recommendation_clicked_rate`

This strict range check prevents malformed upstream values from distorting anomaly scores and baseline statistics.

---

## Feature timestamp rule

The output feature timestamp is controlled by the feature builder.

When a reference timestamp is supplied, it is used as the point-in-time anchor.

When no reference timestamp is supplied, the builder uses the maximum event timestamp found in the input event dataframe.

All timestamps are parsed with explicit mixed-format support. This avoids pandas datetime inference warnings and makes the parser safer for mixed ISO timestamp inputs.

---

## Warehouse fallback behavior

If warehouse or mart signals are missing, the current builder fills the following warehouse-backed features with `0.0`:

- `campaign_roas`
- `conversion_rate`
- `customer_lifetime_value`

This is acceptable for local validation and controlled tests.

For production-like training snapshots, missing warehouse signals should be measured as a data quality issue before model training.

---

## Operational value

Isolation Forest can detect unusual patterns, but it cannot tell whether the input data itself is broken.

A schema mismatch, null spike, type drift, impossible value, or timestamp parsing issue can look like a real anomaly. This contract prevents broken pipeline behavior from being mistaken for business or system anomalies.

The model should detect anomalies in behavior and operations, not become a mirror for corrupted input data.
