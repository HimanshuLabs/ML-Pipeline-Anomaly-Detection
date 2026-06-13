# Kubernetes Runtime Validation

Project 4 includes Kubernetes runtime manifests for the anomaly inference API. The manifests were validated on a local kind cluster to prove that the API can run as a replicated Kubernetes workload, receive traffic through a Kubernetes Service, expose the FastAPI interface through port-forwarding, serve the active production model, publish Prometheus metrics, and participate in HorizontalPodAutoscaler decisions.

This is a local, cost-safe Kubernetes validation. It is not a cloud deployment.

## Validation environment

| Component | Value |
|---|---|
| Cluster type | Local kind cluster |
| Kubernetes namespace | `project4-ml` |
| Workload | `Deployment/project4-anomaly-api` |
| Service | `Service/project4-anomaly-api` |
| Service type | `ClusterIP` |
| API container port | `8004` |
| Local access path | `localhost:8015 -> service/project4-anomaly-api:8004` |
| Model | `isolation_forest` |
| Active model version | `v002` |
| Feature schema version | `feature_schema_v001` |
| Feature count | `51` |
| Autoscaling | Kubernetes HPA with Metrics Server |

## Runtime deployment validation

The Kubernetes deployment was validated with replicated FastAPI inference pods and service routing through the Kubernetes Service.

Validated runtime resources:

    namespace/project4-ml
    deployment/project4-anomaly-api
    service/project4-anomaly-api
    horizontalpodautoscaler/project4-anomaly-api

Observed stable state after HPA scale-down:

    project4-anomaly-api pods: 2 Running
    pod restarts: 0
    service type: ClusterIP
    service port: 8004
    HPA target: cpu 1% / 70%
    HPA replicas: 2 current / 2 desired

## API validation through Kubernetes

The API was reached through Kubernetes port-forwarding:

    localhost:8015 -> service/project4-anomaly-api:8004

Validated endpoints:

| Endpoint | Result | Evidence |
|---|---:|---|
| `GET /health` | HTTP 200 | API returned `status=ok`, `active_model_version=v002`, and `model_loaded=true`. |
| `GET /model/active` | HTTP 200 | Active production model remained `v002`. |
| `POST /predict` | HTTP 200 | Single prediction returned `prediction_status=success`. |
| `POST /predict/batch` | HTTP 200 | Batch prediction returned two prediction records with `model_version=v002`. |
| `POST /admin/rollback` | HTTP 200 | Dry-run rollback validated without changing the active model pointer. |
| `GET /metrics` | HTTP 200 | Prometheus prediction counters and latency histograms updated after live traffic. |

Rollback was tested only as a dry-run during Kubernetes validation. The active model pointer remained on `v002`.

## HPA validation

Metrics Server was installed in the local kind cluster so the HorizontalPodAutoscaler could read pod CPU utilization.

Validated HPA behavior:

| Capability | Result |
|---|---:|
| Metrics Server pod running | Passed |
| `kubectl top nodes` | Passed |
| `kubectl top pods -n project4-ml` | Passed |
| HPA CPU target changed from `<unknown>` to real utilization | Passed |
| HPA `ScalingActive=True` | Passed |
| HPA `ValidMetricFound` | Passed |
| HPA scale-up behavior | Passed |
| HPA scale-down behavior | Passed |
| Minimum replica bound | Passed |
| Maximum replica bound | Passed |

Final HPA state after stabilization:

    TARGETS: cpu: 1%/70%
    MINPODS: 2
    MAXPODS: 5
    REPLICAS: 2
    Deployment pods: 2 current / 2 desired
    ScalingActive: True
    Reason: ValidMetricFound

Observed HPA rescale events:

    SuccessfulRescale: New size: 4
    SuccessfulRescale: New size: 5
    SuccessfulRescale: New size: 3
    SuccessfulRescale: New size: 2

The earlier HPA `<unknown>` state was caused by the Metrics API not being available before Metrics Server was installed. After Metrics Server was installed and patched for local kind kubelet TLS behavior, HPA calculated CPU utilization successfully.

## Known limitations

Implemented and validated locally:

- Kubernetes manifests.
- Local kind rollout.
- Replicated FastAPI inference pods.
- ClusterIP service routing.
- Port-forwarded Swagger/API access.
- Live single prediction through Kubernetes.
- Live batch prediction through Kubernetes.
- Rollback dry-run through Kubernetes.
- Prometheus metrics after Kubernetes traffic.
- Metrics Server-backed HPA target calculation.
- HPA scale-up and scale-down behavior.

Not implemented:

- Cloud Kubernetes deployment on EKS, GKE, or AKS.
- Public ingress.
- External load balancer.
- Managed cluster autoscaler.
- Production TLS termination.
- Cloud-native secrets management.

Those cloud capabilities are optional future extensions and must not be claimed as implemented.
