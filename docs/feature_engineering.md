# Project 4 Feature Engineering — Anomaly Detection

## Purpose

This document explains how Project 4 converts source signals from the real-time pipeline and warehouse layer into a validated anomaly detection feature dataset.

Project 4 sits above the earlier data engineering projects:

- Project 1 contributes behavioral, commerce, latency, engagement, recommendation, and risk-related event signals.
- Project 2/3 contributes trusted historical warehouse, mart, campaign, funnel, and customer-value signals.

The feature engineering layer does not pass raw source tables directly into the model. It builds a stable numeric feature contract first, validates it, and only then allows the data to move toward model training or inference.

Main implementation files:

- `src/anomaly_detection/feature_builder.py`
- `src/anomaly_detection/feature_validation.py`
- `configs/model_config.yaml`

---

## Design principle

The feature layer follows this flow:

    Project 1 events
    + Project 2/3 warehouse or mart signals
            ↓
    entity-level aggregation
            ↓
    model-ready numeric features
            ↓
    schema and value validation
            ↓
    training / batch scoring / online scoring input

The purpose is not to create the largest possible feature set. The purpose is to create a compact, explainable, validated feature surface that can support a production-style anomaly detection lifecycle.

This protects the model from noisy raw identifiers, unstable categorical strings, personal fields, high-cardinality text, and silent upstream schema changes.

---

## Feature grain

The current feature output grain is:

    one row per entity_id

The builder accepts the following source entity columns:

- `entity_id`
- `user_id`
- `customer_id`

All accepted entity columns are normalized into:

- `entity_id`

This keeps the downstream model, prediction logs, drift metrics, and rollback evidence aligned around one stable entity key.

---

## Timestamp handling

The builder accepts the following source timestamp columns:

- `event_timestamp`
- `event_time`

The output timestamp column is:

- `feature_timestamp`

The feature builder supports an explicit reference timestamp. This timestamp acts as the point-in-time anchor for lookback windows.

Example call pattern:

    build_anomaly_features(
        event_frame,
        warehouse_frame,
        reference_timestamp="2026-06-08T05:00:00+05:30",
    )

When no reference timestamp is provided, the builder uses the maximum source event timestamp.

Timestamps are parsed with explicit mixed-format support so pandas does not rely on fragile inference behavior.

---

## Feature schema

The current feature schema version is:

    feature_schema_v001

Every generated feature row includes:

    schema_version = feature_schema_v001

The schema version and required feature list are controlled through:

- `configs/model_config.yaml`

This allows future model versions to be tied to the exact feature schema used during training.

---

## Output columns

The generated dataframe follows this column order:

| Column | Purpose |
|---|---|
| `entity_id` | entity key used for training, scoring, and prediction logs |
| `feature_timestamp` | point-in-time feature generation timestamp |
| `schema_version` | Project 4 feature schema version |
| `avg_cart_value_7d` | commerce/cart behavior |
| `event_count_1h` | recent activity burst signal |
| `avg_api_latency_ms` | backend/system latency signal |
| `fraud_score_avg` | fraud/risk proxy |
| `purchase_probability_delta` | purchase intent movement |
| `cart_abandonment_rate` | funnel friction signal |
| `campaign_roas` | campaign performance signal |
| `conversion_rate` | funnel conversion signal |
| `customer_lifetime_value` | customer value signal |
| `discount_sensitivity` | pricing/promotion behavior |
| `page_load_p95_ms` | frontend/page-load latency signal |

All model features after `schema_version` are numeric.

---

## Feature definitions

### avg_cart_value_7d

Source:

- Project 1: `cart_value`

Logic:

    Average cart_value per entity_id over the last 7 days.

Reason:

This captures unusual spending or cart-size behavior. A sharp increase or collapse in cart value can be meaningful for anomaly detection.

Validation expectations:

- numeric
- finite
- non-negative
- null fraction within threshold

---

### event_count_1h

Source:

- Project 1: `event_timestamp` or `event_time`

Logic:

    Count events per entity_id in the last 1 hour from the reference timestamp.

Reason:

This captures recent behavioral bursts. A sudden spike in user activity can indicate abnormal behavior, replay behavior, automation, campaign impact, or a system-side event storm.

Validation expectations:

- numeric
- finite
- non-negative

---

### avg_api_latency_ms

Source:

- Project 1: `api_latency_ms`

Logic:

    Average API latency per entity_id.

Reason:

This connects anomaly detection with system reliability. If anomalous users or sessions are associated with high API latency, the model can support both business and platform monitoring narratives.

Validation expectations:

- numeric
- finite
- non-negative

---

### fraud_score_avg

Source:

- Project 1: `fraud_score`

Logic:

    Average fraud_score per entity_id.

Reason:

This is a risk proxy. It does not claim confirmed fraud. It gives the anomaly model a signal for suspicious behavior intensity.

Validation expectations:

- numeric
- finite
- non-negative
- range: 0 to 1

---

### purchase_probability_delta

Source:

- Project 1: `purchase_probability`

Logic:

    max(purchase_probability) - min(purchase_probability) per entity_id

Reason:

This captures movement in purchase intent. A sharp swing can be more useful than a static probability value.

Validation expectations:

- numeric
- finite
- range: 0 to 1

This is a proxy feature. It does not claim true customer intent. It measures movement in the upstream signal.

---

### cart_abandonment_rate

Source:

- Project 1: `cart_abandonment_probability`

Logic:

    Average cart_abandonment_probability per entity_id.

Reason:

This captures funnel friction and abandoned purchase behavior.

Validation expectations:

- numeric
- finite
- non-negative
- range: 0 to 1

---

### campaign_roas

Source:

- Project 2/3 warehouse or mart layer

Logic:

    Average campaign_roas per entity_id from warehouse or mart input.

Reason:

This connects behavioral anomalies with campaign performance. It helps the model account for business context beyond raw clickstream behavior.

Validation expectations:

- numeric
- finite
- non-negative

---

### conversion_rate

Source:

- Project 2/3 warehouse or mart layer

Logic:

    Average conversion_rate per entity_id from warehouse or mart input.

Reason:

This captures funnel health and helps detect unusual conversion behavior.

Validation expectations:

- numeric
- finite
- non-negative
- range: 0 to 1

---

### customer_lifetime_value

Source:

- Project 2/3 warehouse or mart layer

Logic:

    Average customer_lifetime_value per entity_id.

Reason:

This gives business weight to the anomaly surface. An anomaly on a high-value customer is operationally different from an anomaly on a low-value customer.

Validation expectations:

- numeric
- finite
- non-negative

---

### discount_sensitivity

Source:

- Project 1: `discount_percent`

Logic:

    Average discount_percent per entity_id.
    If the value is greater than 1.0, convert percent-style values to ratio form.

Examples:

    20.0 -> 0.20
    0.20 -> 0.20

Reason:

This captures how strongly behavior is associated with pricing or promotion changes.

Validation expectations:

- numeric
- finite
- non-negative

---

### page_load_p95_ms

Source:

- Project 1: `page_load_time_ms`

Logic:

    95th percentile page load time per entity_id.

Reason:

This captures frontend or page-experience degradation. It also strengthens the production monitoring story because the model can see user-level latency stress.

Validation expectations:

- numeric
- finite
- non-negative

---

## Warehouse signal behavior

The current builder accepts warehouse or mart signals as an optional dataframe.

Expected warehouse columns:

- `entity_id`
- `campaign_roas`
- `conversion_rate`
- `customer_lifetime_value`

If the optional warehouse dataframe is absent or a warehouse-backed feature is missing, the builder fills the feature with:

    0.0

This is acceptable for local test data and controlled validation. In a production-like run, missing warehouse coverage should be measured and reported as a data quality issue.

---

## Validation flow

Feature generation and validation are coupled.

The builder performs this sequence:

    copy source events
    resolve entity column
    resolve timestamp column
    parse source event timestamp
    prepare optional warehouse signals
    calculate event lookback windows
    aggregate numeric features
    add feature_timestamp
    add schema_version
    order output columns
    validate generated dataframe
    return model-ready features

Validation is strict by default when the builder returns its output.

If the generated dataframe violates the contract, feature generation fails.

That is intentional. A loud failure is better than training a model on corrupted features.

---

## Validation checks

The validation layer checks:

- required columns exist
- schema version matches expected version
- entity key is not null
- feature timestamp is parseable
- required model features are numeric
- null fraction does not exceed threshold
- values are finite
- non-negative features are not negative
- probability-style features remain within 0 to 1

The current null threshold is:

    max_null_fraction = 0.05

This means a feature fails validation when more than 5% of its values are null.

---

## Raw fields intentionally excluded

Many raw fields are intentionally excluded from the first feature contract.

Examples:

- `event_id`
- `session_id`
- `email`
- `ip_address`
- `product_name`
- `search_query`
- `recommendation_algorithm`
- `device_type`
- `browser`
- `city`
- `country`
- `source`

Reasons:

- raw identifiers are not stable model signals
- personal fields should not be used in this anomaly model
- free text requires a separate encoding strategy
- high-cardinality categoricals can make the model noisy
- location fields need leakage and fairness review
- lineage fields belong in metadata, not model features

This is a deliberate engineering choice. The first anomaly model should prove a controlled ML pipeline, not hide instability behind too many raw columns.

---

## Test coverage

The current test file is:

- `tests/test_feature_validation.py`

It covers:

- configuration loading
- valid feature dataframe
- missing required columns
- schema version mismatch
- null explosion
- non-numeric feature values
- infinite values
- negative impossible values
- probability out-of-range values
- invalid timestamps
- strict validation failure mode
- model feature matrix extraction
- feature building from Project 1-style events and warehouse signals
- 1-hour event-count behavior
- missing source entity column
- invalid source event timestamp

Tests are run with warnings treated as errors:

    python -m pytest tests/test_feature_validation.py -q -W error

This is intentional. Dependency warnings often become future failures, so the feature layer should stay warning-clean.

---

## Operational value

This feature layer makes Project 4 behave like a production ML system instead of a notebook.

The model receives a versioned, numeric, validated feature contract. Upstream source instability is caught before model training or inference. This improves data quality, reliability, reproducibility, and operational confidence in the ML system.

The feature contract is the steel gate before the model.
