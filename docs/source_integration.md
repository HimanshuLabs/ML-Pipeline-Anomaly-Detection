# Real-Source Feature Integration

## Purpose

Project 4 is the anomaly detection layer for the wider portfolio platform.

It now consumes real local outputs from:

- Project 1: real-time personalization / streaming feature platform
- Project 2/3: batch lakehouse, warehouse, marts, and BI reporting platform

This fixes the earlier gap where the model training path existed, but the real upstream source extraction layer was not yet implemented.

## Implemented source bridge

| File | Purpose |
|---|---|
| `configs/source_extract.yaml` | Defines Project 1 and Project 2/3 PostgreSQL source tables. |
| `src/anomaly_detection/source_extract.py` | Extracts, transforms, normalizes, and writes a real-source Project 4 training snapshot. |
| `tests/test_source_extract.py` | Validates source transforms, feature normalization, web-event aggregation, and snapshot metadata writing. |

## Source systems

### Project 1

| Field | Value |
|---|---|
| Source | PostgreSQL |
| Host/port | `localhost:5433` |
| Database | `personalization_db` |
| Table | `public.user_features` |
| Grain | user |
| Extracted rows | `8,653` |

### Project 2/3

| Source table | Grain | Extracted/transformed rows |
|---|---:|---:|
| `marts.mart_customer_360` | customer | `500` |
| `marts.mart_campaign_performance` | campaign | `39` |
| `marts.mart_marketing_funnel` | campaign/day/channel funnel row | `6,910` |
| `marts.mart_product_sales` | product | `148` |
| `warehouse.fact_web_events` | aggregated web customer | `500` |

## Real-source snapshot

| Field | Value |
|---|---|
| Snapshot ID | `real_source_20260610T060738Z` |
| Snapshot type | `real_source_extract` |
| Feature schema version | `feature_schema_v001` |
| Row count | `16,750` |
| Column count | `57` |
| Model feature count | `51` |
| Null entity IDs | `0` |

Entity breakdown:

| Entity type | Rows |
|---|---:|
| `user` | `8,653` |
| `funnel_day` | `6,910` |
| `customer` | `500` |
| `web_customer` | `500` |
| `product` | `148` |
| `campaign` | `39` |

## Design decision

The extractor does not fake a cross-project identity join.

Project 1 `user_id` and Project 2/3 `customer_id` are not confirmed to be the same business key. Joining them would create false lineage.

Instead, Project 4 builds one unified anomaly-feature corpus with explicit lineage columns:

- `source_project`
- `source_table`
- `entity_type`
- `entity_id`
- `source_event_timestamp`

The model trains on numeric behavioral, operational, marketing, customer, campaign, product, and funnel features while preserving source lineage for auditability.

## Why Kafka/Spark replay was not used for training

Kafka and Spark Structured Streaming are part of Project 1's real-time pipeline.

For training, the correct dependency is a reproducible frozen snapshot:

1. validated upstream outputs
2. source extraction
3. frozen Project 4 training snapshot
4. model training
5. registry promotion

Kafka replay is useful for streaming recovery, online samples, and future inference/drift monitoring. It is not the right primitive for deterministic model retraining.

## v002 production model

| Field | Value |
|---|---|
| Model | `isolation_forest` |
| Version | `v002` |
| Status | `production` |
| Training snapshot | `real_source_20260610T060738Z` |
| Training rows | `16,750` |
| Feature count | `51` |
| Baseline anomaly rate | `0.05002985074626866` |
| Label availability | `unlabeled_proxy_metrics` |

`v001` remains the earlier demo-trained baseline and is archived after `v002` promotion.

## Reproduction commands

Set credentials:

    export PROJECT1_POSTGRES_PASSWORD=de_password
    export PROJECT23_POSTGRES_PASSWORD=project2

Generate real-source snapshot:

    PYTHONPATH=src python -m anomaly_detection.source_extract --config configs/source_extract.yaml --output-root data/features/training

Train v002:

    PYTHONPATH=src python -m anomaly_detection.training --snapshot latest --model-version v002

Validate:

    PYTHONPATH=src pytest tests/test_training.py tests/test_registry.py tests/test_source_extract.py -q
