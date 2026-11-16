# Latency Budget — Project 4 ML Pipeline & Anomaly Detection

## Current state (2026-11-15)

- Online inference endpoint: `POST /predict` (and `POST /predict/batch`).
- Active production model: `v004`, segmented (one Isolation Forest per entity type, 52
  features in total), served from the shared pointer by `OnlineInferenceService`.
- Online prediction latency budget: `p95 < 200 ms`.
- Prometheus latency metric: `prediction_latency_ms` (histogram, 5 ms buckets).
- Client-side benchmark script: `scripts/benchmark_latency.py`.

## Latency target

The online inference path targets:

    p95_latency_ms < 200

This target applies to the FastAPI `/predict` endpoint.

The goal is not to claim ultra-low-latency infrastructure. The goal is to prove that the anomaly scoring path is bounded, measurable, and suitable for operational detection.

## Serving design

- The active model is loaded once and kept in memory; a pointer change swaps it in-process.
- Requests send feature payloads; a segmented model scores each request with its
  entity type's segment and needs only that segment's features.
- Feature validation checks the required model features.
- Prediction evidence (JSONL, plus `ml.online_predictions` in database mode) is written
  by a background task after the response, so it is not on the request path.
- Prometheus records request counts and the scoring latency.

The path does not fetch features from Project 1 or Project 2/3; it scores the payload it
receives.

## Measured result: v004 (live, 2026-11-15)

1,644 `/predict` requests built from real feature rows, sent through the compose API, read
from the Prometheus histogram `prediction_latency_ms` (5 ms buckets):

    model_version=v004
    requests=1644
    errors=0
    p50_latency_ms≈47.6
    p95_latency_ms≈54.4
    pass_latency_budget=true

About 29% of the requests fell in the 15–20 ms bucket, so the latency has two groups. The
percentiles are bucket estimates, not exact values.

What the histogram measures: inside the `/predict` handler, from just before the model
call to just after the request counters, i.e. model scoring, the cached drift-status
lookup, and the metric updates. It excludes request parsing, response serialization,
network time, and the JSONL and database evidence writes (a background task after the
response).

## Different methods: v004 and v002 are not directly comparable

The v002 figures below were measured client-side by `scripts/benchmark_latency.py`
(wall-clock time per request as the client sees it). The v004 figures were read from the
server-side histogram (scoring time inside the handler, 5 ms buckets). The two methods
measure different spans, so the numbers are not directly comparable.

## History: v002 benchmark (client-side, checkpoint 17)

Benchmark command used at the time (host Python, local uvicorn on 8004):

    PYTHONPATH=src python scripts/benchmark_latency.py \
      --url http://localhost:8004/predict \
      --requests 100 \
      --warmup-requests 10

Measured output:

    request_count=100
    warmup_request_count=10
    model_version=v002
    feature_count=51
    budget_ms=200.000
    p50_latency_ms=22.823
    p95_latency_ms=23.530
    max_latency_ms=24.274
    mean_latency_ms=22.800
    api_p95_latency_ms=19.994
    pass_latency_budget=true

Interpretation at the time:

- Client-observed p95 latency: `23.530 ms`
- API-reported p95 scoring latency: `19.994 ms`
- Budget: `200 ms`
- Result: passed

## Scope and limitations

These measurements cover local single-client scoring latency.

They do not prove:

- multi-client load-test performance
- latency inside Kubernetes (HPA scale-up and scale-down were observed on an earlier kind
  cluster in June 2026, and the HPA target on 2026-11-15; latency was not measured)
- cloud network latency
- Project 1 feature API lookup latency
- Project 2/3 warehouse query latency

## Failure conditions

A latency check fails if:

- `/predict` returns non-200 responses during the measurement
- `p95_latency_ms >= 200`
- the requests use an invalid feature payload or the wrong model version
- the measured result is not documented

## Operational follow-up

If p95 latency exceeds `200 ms`, investigate in this order:

1. Confirm the API still preloads the active model (`/health`, `/model/active`).
2. Confirm the active model version and its feature count.
3. Check payload validation overhead.
4. Compare the server-side histogram with client-observed latency.
5. Check `prediction_persist_failures_total`: evidence writes run after the response,
   but a slow database can still compete for the process.
6. Scale API replicas only after code-path bottlenecks are understood.

## Kubernetes autoscaling scope

`k8s/hpa.yaml` scales the `project4-anomaly-api` Deployment from 2 to 5 replicas on a
70% CPU target; the API container defines CPU and memory requests and limits in
`k8s/deployment.yaml`.

Validated live on 2026-11-15 on the dedicated kind cluster `project4-local` (Metrics
Server v0.9.0): the HPA reported `cpu 1%/70%`; see the live-validation section of
[kubernetes_runtime_validation.md](kubernetes_runtime_validation.md). The offline render
is `kubectl --context kind-project4-local kustomize k8s/`.

Not performed: a Kubernetes p95 latency benchmark, an ingress or cloud latency
benchmark, or a production-scale load test.
