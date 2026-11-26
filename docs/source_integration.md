# Real-Source Feature Integration

## Purpose

ML-Pipeline-Anomaly-Detection is the anomaly detection layer for the wider project platform.

It now consumes real local outputs from:

- [Real-time-personalization-pipeline](https://github.com/HimanshuLabs/Real-time-personalization-pipeline): real-time personalization / streaming feature platform
- [Batch-Lakehouse-Marketing-Analytics](https://github.com/HimanshuLabs/Batch-Lakehouse-Marketing-Analytics): batch lakehouse, warehouse, marts, and BI reporting platform

This fixes the earlier gap where the model training path existed, but the real upstream source extraction layer was not yet implemented.

## Implemented source bridge

| File | Purpose |
|---|---|
| `configs/source_extract.yaml` | Defines the PostgreSQL source tables of the real-time pipeline and the batch lakehouse. |
| `src/anomaly_detection/source_extract.py` | Extracts, transforms, normalizes, and writes a real-source training snapshot of this pipeline. |
| `tests/test_source_extract.py` | Validates source transforms, feature normalization, web-event aggregation, and snapshot metadata writing. |

## Current snapshot (2026-11-26)

The production model v007 was trained from this real-source snapshot:

| Field | Value |
|---|---|
| Snapshot ID | `real_source_20261126T120904Z` |
| Snapshot type | `real_source_extract` |
| Extract label (`source_extract.yaml`) | `feature_schema_v002` |
| Model feature label (v007) | `feature_schema_v003` (segmented spec, `model_config.yaml`) |
| Row count | `11,85,050` |
| Column count | `58` |
| Data quality status | `passed` |
| `content_sha256` | `47ac68e6a002f158136c467bc6b6ad4d3120541415ec5d5f60999673891eef1e` |

| Entity type | Rows |
|---|---:|
| `customer` | `3,99,997` |
| `web_customer` | `3,99,784` |
| `user` | `3,83,965` |
| `funnel_day` | `1,120` |
| `product` | `147` |
| `campaign` | `37` |

## History: v004's snapshot (2026-11-15)

v004 (production until 2026-11-20) was trained from this real-source snapshot:

| Field | Value |
|---|---|
| Snapshot ID | `real_source_20261115T010701Z` |
| Snapshot type | `real_source_extract` |
| Extract label (`source_extract.yaml`) | `feature_schema_v002` |
| Model feature label (v004) | `feature_schema_v003` (segmented spec, `model_config.yaml`) |
| Row count | `2,504` |
| Column count | `58` |
| Data quality status | `passed` |

| Entity type | Rows |
|---|---:|
| `funnel_day` | `926` |
| `customer` | `497` |
| `web_customer` | `497` |
| `user` | `400` |
| `product` | `147` |
| `campaign` | `37` |

Each entity type trains its own Isolation Forest (one segment per type).

## Fresh extract for v005 (stage 15b, 2026-11-16)

The first read-only extract (sessions with `default_transaction_read_only=on`, see
"Reproduction commands") wrote `real_source_20261116T035043Z`: 2,504 rows, `passed`,
`content_sha256` `ad2c8596da09eae4b77875b3d96c698de948ac1508a42a482bb9e4f44f02f6f1`,
`training_dataset_id` `64e7a3c0-bd07-5f74-95f5-71cc9e494e1b`. That hash is identical to
the hash of v004's training snapshot above (recomputed from its directory by
`retrain --compare-with v004`): the sources had not changed between the two extracts,
the hash is deterministic across extract runs, and the batch lakehouse's recreated Postgres
container still held the same data. v005 was trained on it (stage 15c).

## Source systems

The row counts below are from the first extract (2026-06-10).

### Real-time-personalization-pipeline

| Field | Value |
|---|---|
| Source | PostgreSQL |
| Host/port | `localhost:5433` |
| Database | `personalization_db` |
| Table | `public.user_features` |
| Grain | user |
| Extracted rows | `8,653` |

### Batch-Lakehouse-Marketing-Analytics

| Source table | Grain | Extracted/transformed rows |
|---|---:|---:|
| `marts.mart_customer_360` | customer | `500` |
| `marts.mart_campaign_performance` | campaign | `39` |
| `marts.mart_marketing_funnel` | campaign/day/channel funnel row | `6,910` |
| `marts.mart_product_sales` | product | `148` |
| `warehouse.fact_web_events` | aggregated web customer | `500` |

## History: first real-source snapshot (2026-06-10)

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

The real-time pipeline's `user_id` and the batch lakehouse's `customer_id` are not confirmed to be the same business key. Joining them would create false lineage.

Instead, this pipeline builds one unified anomaly-feature corpus with explicit lineage columns:

- `source_project`
- `source_table`
- `entity_type`
- `entity_id`
- `source_event_timestamp`

The model trains on numeric behavioral, operational, marketing, customer, campaign, product, and funnel features while preserving source lineage for auditability.

## Why Kafka/Spark replay was not used for training

Kafka and Spark Structured Streaming are part of the real-time pipeline.

For training, the correct dependency is a reproducible frozen snapshot:

1. validated upstream outputs
2. source extraction
3. frozen training snapshot of this pipeline
4. model training
5. registry promotion

Kafka replay is useful for streaming recovery, online samples, and future inference/drift monitoring. It is not the right primitive for deterministic model retraining.

## History: v002, the first real-source model

v002 was production from 2026-06-10 until v004 was promoted on 2026-11-15; it is now
`archived`.

| Field | Value |
|---|---|
| Model | `isolation_forest` |
| Version | `v002` |
| Status at the time | `production` |
| Training snapshot | `real_source_20260610T060738Z` |
| Training rows | `16,750` |
| Feature count | `51` |
| Baseline anomaly rate | `0.05002985074626866` |
| Label availability | `unlabeled_proxy_metrics` |

`v001` was the earlier demo-trained baseline; it is now `retired`.

## Reproduction commands

Run from the repository root with the `tools()` and `dev()` functions from the README's
"Local runbook".

The source passwords are read from the environment variables named in
`configs/source_extract.yaml`. Enter them without echo, so they appear neither on screen
nor in the shell history:

    read -rs PROJECT1_POSTGRES_PASSWORD && export PROJECT1_POSTGRES_PASSWORD
    read -rs PROJECT23_POSTGRES_PASSWORD && export PROJECT23_POSTGRES_PASSWORD

The extract is read-only against both sources. Every source session is opened with
`-c default_transaction_read_only=on -c statement_timeout=120000` (the timeout is each
source's `statement_timeout_ms` in `configs/source_extract.yaml`; the batch lakehouse
source also sets `work_mem` (`configs/source_extract.yaml`, 128MB) for its session; only
the source engines; this pipeline's own database engine is unchanged), and before each table is read
the extract checks `SHOW transaction_read_only` is `on` and refuses otherwise, before any
row is read. It issues only `SELECT`s: `SELECT *` on the configured tables; for
`warehouse.dim_customer` (listed in the snapshot's `source_tables`), a `SELECT * ...
LIMIT 0` column check, then only `customer_sk` and `customer_id`; for
`warehouse.fact_web_events`, a `SELECT * ... LIMIT 0` column check, then only the 12
columns it uses, either whole or, at scale, in buckets of business customers
(`WHERE customer_sk = ANY(...)`, every SCD2 version of a customer in one bucket) after one
`count(*)` of events whose `customer_sk` is NULL or not in `dim_customer`. No DDL, no
temporary tables, no writes. It runs on the host network against 127.0.0.1:5433 and
127.0.0.1:5434, so the sources never listen beyond loopback. Never pass `--env-file .env`
to it: only the two source passwords, by name.

Generate a real-source snapshot. `-e NAME` without a value passes each variable from the
environment, so the passwords never appear on a command line:

    docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -e PROJECT1_POSTGRES_PASSWORD -e PROJECT23_POSTGRES_PASSWORD \
      -v "$PWD":/app -w /app --entrypoint "" project4-anomaly-tools:local \
      python -m anomaly_detection.source_extract --config configs/source_extract.yaml --output-root data/features/training

The snapshot's `metadata.json` records `content_sha256` (the data, rows sorted by every
column in a fixed order, `extracted_at` excluded; `features.parquet` is written in the same
order) and, when it passed validation, `training_dataset_id`
(`uuid5` of `project4.training_dataset|<snapshot_id>|<content_sha256>|<feature_set_fingerprint>`).
Re-extracting unchanged sources gives the same `content_sha256` under a new
`snapshot_id`, and so a new `training_dataset_id`.

Train a candidate from it (never promotes), comparing it with the active version on the
new snapshot:

    tools python -m anomaly_detection.retrain --snapshot latest --model-version v005 --compare-with v004

Validate:

    dev env -u PYTHONPATH python -m pytest -p no:cacheprovider -o addopts= -q tests/test_training.py tests/test_registry.py tests/test_source_extract.py
