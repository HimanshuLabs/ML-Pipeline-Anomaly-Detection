# Rollback Runbook — Project 4 ML Pipeline & Anomaly Detection

## Current state (2026-11-15)

- `POST /admin/rollback` runs a dry run or an applied rollback. It requires the header
  `X-Admin-Token`: without a configured token the API answers **503** (admin disabled),
  with a missing or wrong token **401**.
- In database mode (`PROJECT4_POSTGRES_*` set, as in compose and Kubernetes) an applied
  rollback changes the registry rows, the shared pointer `ml.active_model_pointer`, and
  the `audit.rollback_events` row in one transaction. `configs/active_model.yaml` and the
  JSON registry are only a file mirror, written when `PROJECT4_REGISTRY_FILE_MIRROR` is
  on (the default; Kubernetes sets it to `false`). Without a database the files are the
  store.
- The API loads the target before anything is written (dry runs included) and swaps the
  model in-process after the commit; if the committed pointer fails to load, the
  registry and pointer are restored and the event is marked `failed`. No restart is
  needed, and every other replica follows the shared pointer within one poll
  (`PROJECT4_POINTER_POLL_SECONDS`, default 30).
- Every applied rollback appends to `logs/alerts/rollback_events.jsonl` and increments
  `model_rollback_total`.
- **Today there is no compatible target until v005.** v001 and v003 are retired, and v002
  (single model) is never compatible with the segmented v004, so a dry run returns
  HTTP 400 `No compatible rollback target for isolation_forest: …`.

## Purpose

Rollback exists because a deployed anomaly model can degrade after promotion.

Project 4 should not only detect model failure through drift, alerts, metrics, and
dashboards. It must also recover by moving the active model pointer back to the previous
stable approved model.

## When to roll back

Use rollback when evidence shows the active model is unsafe.

Valid triggers:

- Critical drift.
- Anomaly-rate spike.
- p95 latency breach.
- Prediction error spike.
- Excessive false positives.
- Suspected missed anomalies.
- Manual operator decision after reviewing alert evidence.

Rollback should not be blind automation. Review alert events, drift output, prediction
evidence, active model state, and registry state first.

## Target rules

A version is a rollback target only if it:

- has status `archived` and `approved_for_prod: true`;
- has the same `model_name` as the active model;
- loads, and serves the same feature contract as the active model: the same ordered
  `feature_names` for single models, the same segment spec (entity types and each
  segment's ordered features) for segmented models. A single and a segmented model are
  never compatible.

Among eligible versions the most recently archived one is chosen.

## One-time setup: the admin token

Compose passes `PROJECT4_ADMIN_TOKEN` from `.env` to the API as `ADMIN_TOKEN` (at least 16
characters). Generate it once; nobody types or sees it, and nothing happens when `.env`
already has a value:

    grep -q '^PROJECT4_ADMIN_TOKEN=.' .env || {
      sed -i '/^PROJECT4_ADMIN_TOKEN=/d' .env
      printf 'PROJECT4_ADMIN_TOKEN=%s\n' "$(openssl rand -hex 32)" >> .env
    }
    docker compose up -d anomaly-api

## Safety rule

Always run a dry run first. A dry run validates the target and loads it, and changes
nothing. An applied rollback changes the shared pointer and writes audit evidence.

## Dry run

Read the token into a variable (never onto the screen), call the compose API on 8014,
and unset the variable afterwards:

    ADMIN_TOKEN="$(grep '^PROJECT4_ADMIN_TOKEN=' .env | cut -d= -f2-)"
    curl -s -X POST http://127.0.0.1:8014/admin/rollback \
      -H "X-Admin-Token: $ADMIN_TOKEN" -H 'Content-Type: application/json' \
      -d '{"rollback_reason": "manual dry-run validation", "triggered_by": "operator", "dry_run": true}'
    unset ADMIN_TOKEN

Expected result today: HTTP 400 `No compatible rollback target for isolation_forest: …`
(see "Current state").

Expected result once a compatible target exists:

- `validation_status` is `dry_run_validated` and `action_taken` is `false`.
- `from_model_version` is the active version; `to_model_version` is the newest
  compatible archived version.
- `active_model_version` is unchanged, and no rollback event is written.

## Applied rollback

    ADMIN_TOKEN="$(grep '^PROJECT4_ADMIN_TOKEN=' .env | cut -d= -f2-)"
    curl -s -X POST http://127.0.0.1:8014/admin/rollback \
      -H "X-Admin-Token: $ADMIN_TOKEN" -H 'Content-Type: application/json' \
      -d '{"rollback_reason": "critical drift detected", "triggered_by": "operator", "dry_run": false}'
    unset ADMIN_TOKEN

Expected result:

- `validation_status` is `applied` and `action_taken` is `true`.
- `active_model_version` is the target; the version left becomes `rolled_back`.
- `ml.active_model_pointer` names the target (its `pointer_version` increases by one).
- `audit.rollback_events` and `logs/alerts/rollback_events.jsonl` each get one event.
- `model_rollback_total` increments.

## Verify the active model

    curl -s http://127.0.0.1:8014/model/active    # model_version is the target
    curl -s http://127.0.0.1:8014/health          # pointer_source "postgres" in database mode

## Verify rollback evidence

    tail -n 5 logs/alerts/rollback_events.jsonl
    docker compose exec -T anomaly-postgres psql -U manshu -d project4_ml -c \
      "SELECT rollback_id, from_model_version, to_model_version, validation_status, triggered_at_utc
       FROM audit.rollback_events ORDER BY triggered_at_utc DESC LIMIT 5"

Each event includes `rollback_id`, `model_name`, `from_model_version`,
`to_model_version`, `rollback_reason`, `triggered_by`, `triggered_at_utc`,
`validation_status`, `dry_run`, `active_model_path`, and `rollback_event_path`. Since
stage 14c-4 the two paths are stored relative to the repository root; the API response
still returns `rollback_event_path` as an absolute path.

## Verify the rollback metric

    curl -s http://127.0.0.1:8014/metrics | grep model_rollback_total

Prometheus metrics are per process: with several replicas, the counter increments on the
replica that handled the rollback.

## Failure modes

### No compatible target

HTTP 400. Check, for the version you expected:

- status is `archived` (not `retired` or `rolled_back`) and `approved_for_prod` is true;
- same `model_name`;
- its bundle loads, and its feature contract matches the active model: single vs
  segmented never matches, and segmented models need the same segment spec.

### Admin disabled or token rejected

HTTP 503: the API has no `ADMIN_TOKEN` (generate `PROJECT4_ADMIN_TOKEN`, then recreate
the API container). HTTP 401: the header is missing or the token differs from the
server's.

### Target cannot serve

The rollback is refused before anything is written. If the committed pointer fails to
load, the registry and pointer are restored, the event is marked `failed`, and the API
keeps serving the current model.

## Smoke test after a real rollback

1. Run the dry run, then the applied rollback.
2. Verify `/model/active`, `/health`, the JSONL event, the `audit.rollback_events` row,
   and `model_rollback_total`.
3. Promote the previous version back if the rollback was only a test:
   `python -m anomaly_detection.registry promote --model-version <version> --notes "…"`
   (tools image).

## Operational interpretation

Rollback is a recovery control, not a replacement for retraining.

After rollback:

- Investigate drift events.
- Inspect alert events.
- Review prediction evidence.
- Compare baseline and current metrics.
- Retrain or tune thresholds only after understanding the failure.

## Kubernetes

See [kubernetes_runtime_validation.md](kubernetes_runtime_validation.md). On the
2026-11-15 live run, cross-replica propagation of the shared pointer was proven with
promotes (v004 → v002 → v004, both replicas following each change), and the rollback dry
run returned the expected HTTP 400. An applied rollback in the cluster is pending v005,
when v004 becomes a compatible archived target. Kubernetes rollout rollback (images or
configuration) is a separate deployment operation, not a model-registry operation.
