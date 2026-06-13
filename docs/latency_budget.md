# Latency Budget — Project 4 ML Pipeline & Anomaly Detection

## Current implementation status

Implemented locally:

- FastAPI online inference endpoint: `POST /predict`
- Active production model loaded through `OnlineInferenceService`
- Active model pointer: `configs/active_model.yaml`
- Production model version: `v002`
- Model feature count: `51`
- Online prediction latency budget: `p95 < 200 ms`
- Prometheus latency metric: `prediction_latency_ms`
- Repeatable benchmark script: `scripts/benchmark_latency.py`

Planned hardening:

- Run the benchmark inside CI after the API service is available.
- Add concurrent-client load testing after Dockerization.
- Compare latency across future model versions.
- Add production-like latency tests only when external feature/database calls are introduced.

## Latency target

The online inference path targets:

    p95_latency_ms < 200

This target applies to the FastAPI `/predict` endpoint.

The goal is not to claim ultra-low-latency infrastructure. The goal is to prove that the anomaly scoring path is bounded, measurable, and suitable for operational detection.

## Serving design

The current online inference path is intentionally lightweight:

- The active model is loaded once by the inference service.
- Requests send model-ready feature payloads.
- Feature validation checks required model features.
- Scoring uses the in-memory active model.
- Prediction evidence is written to JSONL.
- Prometheus records request and latency metrics.

The current benchmark does not include external feature fetches from Project 1 or Project 2/3. It validates the local Project 4 model-serving path only.

## Measured benchmark result

Benchmark command used:

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

Interpretation:

- Client-observed p95 latency: `23.530 ms`
- API-reported p95 scoring latency: `19.994 ms`
- Budget: `200 ms`
- Result: passed

Checkpoint 17 passes for local single-client latency smoke validation.

## Scope and limitations

This benchmark proves local online scoring latency.

It does not prove:

- multi-client load-test performance
- Kubernetes HPA manifest behavior; live metrics-server autoscaling validation remains optional / not performed
- cloud network latency
- production database write latency
- Project 1 feature API lookup latency
- Project 2/3 warehouse query latency

Those remain planned hardening areas.

## Failure conditions

This checkpoint fails if:

- `scripts/benchmark_latency.py` cannot run
- `/predict` returns non-200 responses during the benchmark
- `p95_latency_ms >= 200`
- the benchmark uses an invalid feature payload
- the benchmark points to the wrong model version
- the measured result is not documented

## Operational follow-up

If future p95 latency exceeds `200 ms`, investigate in this order:

1. Confirm the API still preloads the active model.
2. Check whether prediction logging is blocking the request path.
3. Check payload validation overhead.
4. Confirm active model version and feature count.
5. Compare API-reported latency with client-observed latency.
6. Review Prometheus `prediction_latency_ms`.
7. Consider async or buffered logging if persistence becomes the bottleneck.
8. Scale API replicas only after code-path bottlenecks are understood.

## Completion status

Current status:

    Complete for local single-client latency smoke validation.

## Kubernetes autoscaling scope

Project 4 includes a Kubernetes HPA manifest for the anomaly inference API.

Implemented:

- `k8s/hpa.yaml` targets the `project4-anomaly-api` Deployment.
- Minimum replicas: `2`.
- Maximum replicas: `5`.
- CPU utilization target: `70%`.
- The API container defines CPU and memory requests/limits in `k8s/deployment.yaml`.

Validation performed:

- YAML parse validation.
- Cross-manifest consistency audit.
- Offline render validation with `kubectl kustomize k8s/`.

Validation not performed:

- Live HPA behavior.
- Metrics Server integration.
- Load-driven autoscaling test.
- Cluster rollout.

Reason: no Kubernetes context is configured in the current local environment.

Latency budget implication:

The implemented HPA manifest documents the intended scale-out policy, but the measured p95 latency budget remains based on the local FastAPI/Docker runtime validation. Live Kubernetes latency and autoscaling behavior must be measured separately after a local cluster is configured.
