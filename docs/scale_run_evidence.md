# Scale run evidence

Synthetic data, one local machine (16 threads), not production. CI runs every test at the
default size (about 500 customers); it does not run this.

Upstream: [Real-time-personalization-pipeline](https://github.com/HimanshuLabs/Real-time-personalization-pipeline)
(`public.user_features`) and [Batch-Lakehouse-Marketing-Analytics](https://github.com/HimanshuLabs/Batch-Lakehouse-Marketing-Analytics)
(four marts, `warehouse.fact_web_events`, `warehouse.dim_customer`), read-only.

## Run of 2026-11-26

| Item | Value |
|---|---|
| Snapshot | `real_source_20261126T120904Z`, `real_source_extract`, `passed` |
| Rows / columns | 11,85,050 / 58 |
| `content_sha256` | `47ac68e6a002f158136c467bc6b6ad4d3120541415ec5d5f60999673891eef1e` |
| `training_dataset_id` | `01191f7a-d246-5873-b5c3-5605fc867761` |
| Web events read | 79,18,071, in 8 buckets |
| Source settings | `statement_timeout_ms` 120000 on both sources, `work_mem` 128MB on the batch lakehouse source; `marts.mart_customer_360` reads in 69.9 s, `marts.mart_marketing_funnel` in 75.5 s |
| Model | `isolation_forest` `v007`, segmented, 6 segments, 52 features, `feature_schema_v003` |
| MLflow run | `906124d6de5a4f078c377eadc933f8f0` |
| Compared with | `v006` (snapshot `content_sha256` `60ef92c0e7e20e4f3cea0f9e634eee7fdb23b2f05ccd6199d0012e3fc5763cad`) |
| Promote | `v007` production, `v006` archived, `pointer_version` 6 |
| Batch run | `0cbe4b79-5fd9-4628-bb01-b7eeaf918cf7`: 11,85,050 rows, 59,255 anomalies, one transaction |
| Time | extract 1058 s, retrain 95 s, batch 562 s |

| Entity type | Rows | Training anomaly rate | Batch anomalies |
|---|---:|---:|---:|
| `user` | 3,83,965 | 0.0500 | 19,199 |
| `customer` | 3,99,997 | 0.0500 | 20,000 |
| `web_customer` | 3,99,784 | 0.0500 | 19,990 |
| `funnel_day` | 1,120 | 0.0500 | 56 |
| `product` | 147 | 0.0544 | 8 |
| `campaign` | 37 | 0.0541 | 2 |

`funnel_day` has 1,120 rows against 1,137 on 2026-11-20: the batch lakehouse removed 17
invalid campaign-days on 2026-11-25.

## Gate: 26/26 before the promote

Read from `v007`'s `evaluation_report.json` and `metadata.json`.

1. `feature_schema_version` is `feature_schema_v003`
2. 6 segments, none missing
3. `feature_count` 52
4. `row_counts.snapshot` == `row_counts.trained`
5-22. Per entity type (`user`, `customer`, `web_customer`, `funnel_day`, `product`, `campaign`), three checks each:
   - `row_count >= 30`
   - `anomaly_rate` within 0.03-0.07
   - `row_count >=` default-size floor (400, 497, 497, 926, 147, 37)
23. extract validation passed (`data_quality_status` `passed`, `trainable`, no validation errors)
24. compared with `v006`
25. `v006` snapshot hash is `60ef92c0…5763cad`
26. new snapshot hash differs from it

## Verify: 14/14 after the batch run

| Check | Expected |
|---|---|
| registry `v007` | `production` / approved |
| registry `v006` | `archived` |
| database pointer | `v007` |
| file pointer (`configs/active_model.yaml`) | `v007` |
| batch runs of `v007` | 1 |
| latest run, by the rule of [AI-Decision-Intelligence-Layer](https://github.com/HimanshuLabs/AI-Decision-Intelligence-Layer) (`recorded_at` DESC, `batch_run_id` DESC) | `0cbe4b79-5fd9-4628-bb01-b7eeaf918cf7` |
| rows in the run | 11,85,050 |
| entity types in the run | 6 |
| distinct `recorded_at` in the run | 1 |
| latest-run indexes on `ml.batch_predictions` | 2 (`idx_batch_predictions_recorded_run`, `idx_batch_predictions_run_anomalies`) |
| integration view signatures (md5 of `column_name:data_type`) | `093e2da8861ba8a59567bcf5d0692daa`, `6ac0c9195b96abda9db79a6822b10dbd`, `36f6f02f015e3c84cdfbbc68bfc67364` |
| API `/health` `active_model_version` | `v007` |
| API `/health` `pointer_source` | `postgres` |
| gate | `GATE PASS` |

Rollback dry run `v007` → `v006`: `validated`, nothing applied.

## Drift

`v007` over the batch window, 11,85,050 rows: 6 segments `normal`, 87 drift events, 0
alerts. The read in chunks has a peak of 509579264 bytes (0.51 GB) against 5786619904
bytes (5.79 GB) for the one-piece read, with identical output, and completes inside the
CronJob's 512 MiB limit.

## Fault injection

`scripts/fault_injection.py` on `v007` and `real_source_20261126T120904Z`, seed
20261126: per segment a seeded 1% of the rows (at least 5) gets one fault per feature
family the segment has and is scored in memory; only the report is written.

| Segment | Fault | Faulted rows | Flagged | Share of synthetic faults flagged | Clean flag rate |
|---|---|---:|---:|---:|---:|
| `campaign` | `money_x10` | 5 | 1 | 0.2000 | 0.0541 |
| `customer` | `money_x10` | 4,000 | 302 | 0.0755 | 0.0500 |
| `funnel_day` | `purchases_without_events` | 12 | 0 | 0.0000 | 0.0500 |
| `product` | `money_x10` | 5 | 0 | 0.0000 | 0.0544 |
| `user` | `money_x10` | 3,840 | 619 | 0.1612 | 0.0500 |
| `user` | `purchases_without_events` | 3,840 | 332 | 0.0865 | 0.0500 |
| `web_customer` | `money_x10` | 3,998 | 1,629 | 0.4075 | 0.0500 |
| `web_customer` | `latency_x20` | 3,998 | 675 | 0.1688 | 0.0500 |
| `web_customer` | `purchases_without_events` | 3,998 | 318 | 0.0795 | 0.0500 |

Faults: `money_x10`, the segment's money feature x10; `latency_x20`,
`avg_api_latency_ms` x20; `purchases_without_events`, `total_events` 0 and
`purchase_count` at least 1. `campaign`, `product` and `funnel_day` (5 to 12 faulted
rows) are too small to show anything.

## Previous run (2026-11-20)

`v006` on `real_source_20261120T182648Z`: 11,85,067 rows, `content_sha256`
`60ef92c0e7e20e4f3cea0f9e634eee7fdb23b2f05ccd6199d0012e3fc5763cad`, gate 26/26, batch run
`a3cd6a97-d786-4345-8a65-0c4bf21ec5cd` with 59,256 anomalies, verify 14/14. Extract 616 s,
retrain 94 s, batch 543 s.
