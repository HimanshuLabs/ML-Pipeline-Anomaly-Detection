# Rollback Runbook — Project 4 ML Pipeline & Anomaly Detection

## Current implementation status

Rollback is implemented locally.

Implemented:

- Previous production models remain rollback-eligible after promotion.
- `POST /admin/rollback` supports dry-run validation.
- `POST /admin/rollback` supports applied rollback.
- Applied rollback updates `configs/active_model.yaml`.
- Applied rollback writes evidence to `logs/alerts/rollback_events.jsonl`.
- Applied rollback increments `model_rollback_total`.
- `audit.rollback_events` defines the PostgreSQL audit-table contract.

Planned hardening:

- Direct PostgreSQL insert into `audit.rollback_events`.
- Authentication/authorization for `/admin/rollback`.
- Explicit in-process model reload after rollback.
- Optional Kubernetes config/image rollback.

## Purpose

Rollback exists because a deployed anomaly model can degrade after promotion.

Project 4 should not only detect model failure through drift, alerts, metrics, and dashboards. It must also recover by reverting the active model pointer to the previous stable approved model.

Validated rollback path:

- Current production model: `isolation_forest v002`
- Previous stable rollback target: `isolation_forest v001`

## When to rollback

Use rollback when evidence shows the active model is unsafe.

Valid triggers:

- Critical drift.
- Anomaly-rate spike.
- p95 latency breach.
- Prediction error spike.
- Excessive false positives.
- Suspected missed anomalies.
- Manual operator decision after reviewing alert evidence.

Rollback should not be blind automation. Review alert events, drift output, prediction evidence, active model state, and registry state first.

## Safety rule

Always run dry-run first.

Dry-run validates the rollback target without changing `configs/active_model.yaml`.

Applied rollback changes the active model pointer and writes audit evidence.

## Dry-run rollback

Run:

    curl -X POST http://localhost:8004/admin/rollback \
      -H "Content-Type: application/json" \
      -d '{
        "rollback_reason": "manual dry-run validation",
        "triggered_by": "operator",
        "dry_run": true
      }'

Expected result:

- Response status is `validated`.
- `action_taken` is `false`.
- `from_model_version` is `v002`.
- `to_model_version` is `v001`.
- `active_model_version` remains `v002`.
- No rollback event is written.

## Applied rollback

Run:

    curl -X POST http://localhost:8004/admin/rollback \
      -H "Content-Type: application/json" \
      -d '{
        "rollback_reason": "critical drift detected",
        "triggered_by": "operator",
        "dry_run": false
      }'

Expected result:

- Response status is `applied`.
- `action_taken` is `true`.
- `from_model_version` is `v002`.
- `to_model_version` is `v001`.
- `configs/active_model.yaml` points to `v001`.
- `logs/alerts/rollback_events.jsonl` receives one rollback event.
- `model_rollback_total` increments.

## Verify active model

Run:

    python - <<'PY'
    from pathlib import Path
    import yaml

    active = yaml.safe_load(Path("configs/active_model.yaml").read_text(encoding="utf-8"))
    print(active["model_name"])
    print(active["active_model_version"])
    print(active["status"])
    print(active["artifact_path"])
    PY

Expected after applied rollback:

    isolation_forest
    v001
    production
    artifacts/models/isolation_forest/model_version=v001/model.joblib

## Verify rollback evidence

Run:

    tail -n 5 logs/alerts/rollback_events.jsonl

Each event includes:

- `rollback_id`
- `model_name`
- `from_model_version`
- `to_model_version`
- `rollback_reason`
- `triggered_by`
- `triggered_at_utc`
- `validation_status`
- `dry_run`
- `active_model_path`
- `rollback_event_path`

The local JSONL evidence aligns with the PostgreSQL table contract:

    audit.rollback_events

## Verify rollback metric

Run:

    curl http://localhost:8004/metrics | grep model_rollback_total

Expected:

    model_rollback_total

Note: Prometheus metrics are in-memory per process. If rollback is executed in one Python process and `/metrics` is checked in another, the counter will not carry over.

## Failure modes

### No previous stable model exists

Rollback returns HTTP 400.

Check:

- Previous model has `status: archived`.
- Previous model has `approved_for_prod: true`.
- Previous model artifact path exists.
- Previous model has the same `model_name`.

### Active model pointer missing

Restore `configs/active_model.yaml` from Git or re-promote the last known good model.

### API keeps old in-memory model

Current rollback updates the active pointer. If the API process keeps a loaded model object, restart Uvicorn after rollback.

Run:

    pkill -f "uvicorn api.main:app" || true
    uvicorn api.main:app --host 0.0.0.0 --port 8004

## Smoke-test recovery

For validation, restore `configs/active_model.yaml` back to `v002` after proving applied rollback.

Required smoke-test sequence:

1. Back up `configs/active_model.yaml`.
2. Apply rollback.
3. Verify `v001`.
4. Verify rollback event.
5. Verify metric in the same API process.
6. Restore `configs/active_model.yaml`.
7. Confirm `v002` is active again.

## Operational interpretation

Rollback is a recovery control, not a replacement for retraining.

After rollback:

- Investigate drift events.
- Inspect alert events.
- Review prediction evidence.
- Compare baseline and current metrics.
- Retrain or tune thresholds only after understanding the failure.
