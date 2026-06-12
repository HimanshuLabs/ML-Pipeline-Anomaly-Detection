

<!-- BEGIN REAL_SOURCE_ML_BRIDGE -->
## Real-source ML bridge


### Real-source ML bridge

Project 4 now has a deterministic bridge from upstream data engineering systems into the ML training pipeline.

Flow:

1. Project 1 writes user-level behavioral features to `public.user_features`.
2. Project 2/3 writes warehouse and mart outputs to PostgreSQL.
3. `configs/source_extract.yaml` defines the real source contract.
4. `src/anomaly_detection/source_extract.py` extracts and normalizes the sources.
5. Project 4 writes a frozen real-source training snapshot.
6. `src/anomaly_detection/training.py` trains Isolation Forest `v002`.
7. `src/anomaly_detection/registry.py` promotes `v002`.
8. `configs/active_model.yaml` points production traffic to `v002`.

Project 1 source:

- `public.user_features`

Project 2/3 sources:

- `marts.mart_customer_360`
- `marts.mart_campaign_performance`
- `marts.mart_marketing_funnel`
- `marts.mart_product_sales`
- `warehouse.fact_web_events`

### Architecture tradeoff

The source extractor intentionally avoids creating a fake identity join between Project 1 users and Project 2/3 customers.

The confirmed local outputs do not prove that Project 1 `user_id` and Project 2/3 `customer_id` share the same natural key. A false join would make the project less credible.

Instead, the model trains on a unified anomaly-feature corpus with explicit lineage columns:

- `source_project`
- `source_table`
- `entity_type`
- `entity_id`
- `source_event_timestamp`

This keeps the architecture auditable and honest.

### Training versus inference boundary

The implemented flow is batch/offline model training.

Training path:

1. real-source extract
2. frozen training snapshot
3. Isolation Forest training
4. baseline metric capture
5. model registry promotion
6. active model pointer update

Future online inference can consume low-latency feature samples from Project 1 or a feature API, but production training remains snapshot-based for reproducibility.

<!-- END REAL_SOURCE_ML_BRIDGE -->

<!-- BASELINE_METRICS_CLOSURE_START -->
## Baseline metrics closure

The production architecture now has a verified baseline reference for `v002`.

The active model pointer in `configs/active_model.yaml` references `isolation_forest` version `v002`, trained from real local Project 1 and Project 2/3 extracts. The model artifact folder contains `metadata.json`, `feature_schema.json`, and `baseline_stats.json`.

The baseline layer currently provides:

- baseline anomaly rate
- per-feature baseline statistics for all 51 model features
- prediction summary from offline training
- baseline metric type marked as `unsupervised_training_baseline`
- label availability marked as `unlabeled_proxy_metrics`

This keeps the monitoring design honest: drift checks and current anomaly rates will be compared against real training baselines, but the system does not claim supervised production accuracy. Precision, recall, false positive rate, and false negative rate are intentionally not claimed until verified labels, backtesting labels, or delayed-truth labels exist.

The latency values stored with the offline baseline are placeholders. Real online serving latency belongs to the later API latency-budget validation path.
<!-- BASELINE_METRICS_CLOSURE_END -->
