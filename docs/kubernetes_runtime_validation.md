# Kubernetes Runtime Validation (live runbook)

Step-by-step procedure to run Project 4 in its own local kind cluster `project4-local`
and verify it: the database, the seed, the hardened API, rollback across replicas, the
network policies, the drift job and teardown. It is a local, cost-safe validation, not a
cloud deployment.

Rules for every step:

- Every `kubectl` command that talks to a cluster names `--context kind-project4-local`.
  Never rely on the current context: it may point at another project's cluster.
- `kind` commands always name the cluster with `--name project4-local`.
- Secrets are generated, never typed or printed.

## 0. Build the images

The cluster runs the images built from this repository (the seed job needs the tools
image's `restrict-connect` and `sync-from-files --if-empty`):

```bash
docker build -f docker/Dockerfile --target runtime -t project4-anomaly-api:local .
docker build -f docker/Dockerfile --target tools -t project4-anomaly-tools:local .
```

## 1. Create the cluster, then restore the previous context immediately

`kind create cluster` makes the new cluster the current context of the kubeconfig. Other
projects' tooling relies on whatever context was current, so save it first and switch
straight back; from here on every command names its context explicitly. If no context
was current (an empty kubeconfig), nothing is saved and nothing is restored.

```bash
PREVIOUS_CONTEXT="$(kubectl config current-context 2>/dev/null || true)"
kind create cluster --name project4-local
if [ -n "$PREVIOUS_CONTEXT" ]; then kubectl config use-context "$PREVIOUS_CONTEXT"; fi
kubectl --context kind-project4-local get nodes
```

## 2. Metrics Server (for the HPA)

Pinned version `v0.9.0` (verify on the release page before use:
https://github.com/kubernetes-sigs/metrics-server/releases). kind's kubelets use
self-signed certificates, so Metrics Server needs `--kubelet-insecure-tls`.

```bash
kubectl --context kind-project4-local apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.0/components.yaml
kubectl --context kind-project4-local -n kube-system patch deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl --context kind-project4-local -n kube-system rollout status deployment/metrics-server --timeout=180s
```

## 3. Load the images into the cluster

The Postgres image is the same `postgres:16.14` that compose's `anomaly-postgres` uses.
With Docker's containerd image store, locally built images are multi-platform indexes with
attestation manifests; loading them straight from Docker imports all platforms and fails with
`ctr: content digest … not found` (observed on Docker 29.9.3), so each image is saved for
the host platform only and loaded as an archive.

```bash
case "$(uname -m)" in x86_64) PLATFORM=linux/amd64 ;; aarch64|arm64) PLATFORM=linux/arm64 ;; esac
for IMAGE in project4-anomaly-api:local project4-anomaly-tools:local postgres:16.14; do
  ARCHIVE="/tmp/$(echo "$IMAGE" | tr ':/' '__').tar"
  docker save --platform "$PLATFORM" "$IMAGE" -o "$ARCHIVE" && kind load image-archive "$ARCHIVE" --name project4-local && rm -f "$ARCHIVE"
done
docker exec project4-local-control-plane crictl images | grep -E 'project4|postgres'
```

## 4. Namespace and secrets (generated values)

The app password goes into `project4-app-db`; the seed job gives the same value to the
`project4_app` role, so they always match.

```bash
kubectl --context kind-project4-local apply -f k8s/namespace.yaml
kubectl --context kind-project4-local -n project4-ml create secret generic project4-app-db \
  --from-literal=PROJECT4_POSTGRES_HOST=project4-postgres \
  --from-literal=PROJECT4_POSTGRES_PORT=5432 \
  --from-literal=PROJECT4_POSTGRES_DB=project4_ml \
  --from-literal=PROJECT4_POSTGRES_USER=project4_app \
  --from-literal=PROJECT4_POSTGRES_PASSWORD="$(openssl rand -hex 24)"
kubectl --context kind-project4-local -n project4-ml create secret generic project4-postgres-owner \
  --from-literal=POSTGRES_USER=project4_owner \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" \
  --from-literal=POSTGRES_DB=project4_ml
kubectl --context kind-project4-local -n project4-ml create secret generic project4-admin \
  --from-literal=admin-token="$(openssl rand -hex 32)"
```

## 5. Apply, wait for Postgres, seed

Until the seed has run, the database has no pointer (and no `project4_app` role). The
API does not crash and is not unready: it logs "Could not load the model from the shared
database pointer; starting from the file pointer" and serves the model baked into the
image, so `/health` returns 200 with `"pointer_source": "file"`. The pointer poller keeps
retrying; within one poll (30 s) after the seed every pod logs "Swapped in model … from
shared pointer version 1" and reports `"pointer_source": "postgres"`. No restart is
needed; the loop at the end checks every pod.

```bash
kubectl --context kind-project4-local apply -k k8s/
kubectl --context kind-project4-local -n project4-ml rollout status statefulset/project4-postgres --timeout=300s
kubectl --context kind-project4-local apply -f k8s/seed-job.yaml
kubectl --context kind-project4-local -n project4-ml wait --for=condition=complete job/project4-db-seed --timeout=600s
kubectl --context kind-project4-local -n project4-ml logs job/project4-db-seed
kubectl --context kind-project4-local -n project4-ml rollout status deployment/project4-anomaly-api --timeout=300s
sleep 35   # one pointer poll (30 s) after the seed
for pod in $(kubectl --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[*].metadata.name}'); do
  kubectl --context kind-project4-local -n project4-ml exec "$pod" -- python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8004/health"))["pointer_source"])'
done   # every pod: postgres
```

Expected seed log: `applied …` per schema file, `role project4_app: LOGIN`, the
registry rows inserted and `pointer_version 1`, then the role table where only
`project4_owner`, `project4_app` and `project5_reader` can connect. Running the seed
again is safe: `sync-from-files --if-empty` prints `skipped: … nothing was overwritten`.

## 6. Smoke test through port-forward 8015

```bash
kubectl --context kind-project4-local -n project4-ml port-forward svc/project4-anomaly-api 8015:8004 &
curl -s http://127.0.0.1:8015/health          # status ok, pointer_source "postgres"
curl -s http://127.0.0.1:8015/model/active
curl -s http://127.0.0.1:8015/metrics | grep -E '^(active_model_version|predictions_scored_total)'
kubectl --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- python -c '
import json, urllib.request
base = "http://127.0.0.1:8004"
active = json.load(urllib.request.urlopen(base + "/model/active"))
version = active["model_version"]
meta = json.load(open("/app/artifacts/models/isolation_forest/model_version=" + version + "/metadata.json"))
names = meta["segments"]["user"]["feature_names"] if meta.get("segments") else meta["feature_names"]
body = json.dumps({"entity_type": "user", "feature_payload": {n: 1.0 for n in names}}).encode()
request = urllib.request.Request(base + "/predict", data=body, headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(request)))'
```

## 7. A pointer change reaches every replica (database pointer)

The seeded registry has no compatible rollback target: v001 and v003 are `retired`,
and v002 (a single model) is never compatible with the segmented v004. The rollback dry
run therefore refuses (HTTP 400, `No compatible rollback target for isolation_forest …`),
which is expected. Cross-replica propagation is proven with promotes instead: one pod
changes the shared database pointer, and every replica follows within one poll. The
real `/admin/rollback` run belongs after the v005 retrain (step 15), when v004 becomes
a compatible archived target.

The admin token is read into a shell variable; it never appears on screen or in the
shell history (the history holds only the command that reads it).

```bash
ADMIN_TOKEN="$(kubectl --context kind-project4-local -n project4-ml get secret project4-admin -o jsonpath='{.data.admin-token}' | base64 -d)"
POD="$(kubectl --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[0].metadata.name}')"
kubectl --context kind-project4-local -n project4-ml port-forward "pod/$POD" 8016:8004 &
curl -s http://127.0.0.1:8016/model/active    # model_version v004
curl -s -X POST http://127.0.0.1:8016/admin/rollback -H "X-Admin-Token: $ADMIN_TOKEN" -H 'Content-Type: application/json' -d '{"dry_run": true, "rollback_reason": "runbook check", "triggered_by": "runbook"}'
# expected: 400 {"detail": "No compatible rollback target for isolation_forest: ..."}
unset ADMIN_TOKEN
```

Promote v002 from one pod (as `project4_app`, through the registry CLI), then check
every replica:

```bash
kubectl --context kind-project4-local -n project4-ml exec "$POD" -- python -m anomaly_detection.registry promote --model-version v002 --notes "runbook: cross-replica propagation"
sleep 35   # the other replicas poll the shared pointer every 30 s
for pod in $(kubectl --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[*].metadata.name}'); do
  kubectl --context kind-project4-local -n project4-ml exec "$pod" -- python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8004/model/active"))["model_version"])'
done   # every pod: v002
```

Promote v004 back and check every replica again:

```bash
kubectl --context kind-project4-local -n project4-ml exec "$POD" -- python -m anomaly_detection.registry promote --model-version v004 --notes "runbook: back to v004"
sleep 35
for pod in $(kubectl --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[*].metadata.name}'); do
  kubectl --context kind-project4-local -n project4-ml exec "$pod" -- python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8004/model/active"))["model_version"])'
done   # every pod: v004
```

## 8. Network policy deny test

A pod without an allowed label cannot reach Postgres; the API cannot reach anything
outside its allowed paths, such as the Kubernetes API (`kubernetes.default.svc:443`,
always reachable inside the cluster without a policy); the API still reaches Postgres
and DNS.

```bash
kubectl --context kind-project4-local -n project4-ml run np-probe --rm -i --restart=Never --image=project4-anomaly-tools:local --image-pull-policy=IfNotPresent --command -- python -c 'import socket; socket.create_connection(("project4-postgres", 5432), timeout=5); print("REACHED: policy not enforced")'
kubectl --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- python -c 'import socket; socket.create_connection(("kubernetes.default.svc", 443), timeout=5); print("REACHED: policy not enforced")'
kubectl --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- python -c 'import socket; print(socket.getaddrinfo("project4-postgres", 5432)[0][4]); socket.create_connection(("project4-postgres", 5432), timeout=5); print("postgres reachable")'
```

Expected: the first two fail with a timeout (never `REACHED`); the third prints an
address and `postgres reachable`. kindnet's policy engine fails open if it is not
running, so a `REACHED` line means the policies are not enforced.

## 9. One manual drift job run

Nothing has been batch-scored in the cluster, so the job reports `not_evaluated` and
exits 0.

```bash
kubectl --context kind-project4-local -n project4-ml create job --from=cronjob/project4-drift-job drift-manual
kubectl --context kind-project4-local -n project4-ml wait --for=condition=complete job/drift-manual --timeout=300s
kubectl --context kind-project4-local -n project4-ml logs job/drift-manual
```

## 10. Hardening check

```bash
kubectl --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- id          # uid=999 gid=999
kubectl --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- sh -c 'touch /app/probe' 2>&1 | tail -2   # touch: cannot touch '/app/probe': Read-only file system
kubectl --context kind-project4-local -n project4-ml exec statefulset/project4-postgres -- id        # uid=999
curl -s http://127.0.0.1:8015/metrics | grep registry_mirror_write_failures_total   # 0.0
kubectl --context kind-project4-local -n project4-ml get pods,svc,hpa,pdb,cronjob,networkpolicy
```

## 11. Teardown

Stop the background port-forwards from steps 6 and 7 first, then delete the cluster.

```bash
pkill -f "port-forward svc/project4-anomaly-api 8015:8004"
pkill -f "port-forward pod/.* 8016:8004"
kind delete cluster --name project4-local
```

Deleting the cluster also removes its entry from the kubeconfig; the context restored in
step 1 stays current.

## Live validation (2026-11-15)

The runbook above was run end to end on a fresh `project4-local` cluster.

| Step | Observed |
| --- | --- |
| 5 seed | 6 schema files applied, `role project4_app: LOGIN`, 4 registry rows, pointer v004 at `pointer_version 1` |
| 5 pointer | both pods report `pointer_source` `postgres`, model v004 |
| 6 smoke | `/predict` served by v004; `predictions_scored_total` 1; `registry_mirror_write_failures_total` 0 |
| 7 rollback dry run | HTTP 400: v002 is never compatible with v004 (expected until v005) |
| 7 propagation | promote v002 through one pod: both pods v002 (`pointer_version 2`); promote v004 back: both pods v004 (`pointer_version 3`) |
| 8 network policy | unlabelled pod to Postgres: `TimeoutError`; API to `kubernetes.default.svc:443`: `TimeoutError`; API to Postgres: reachable |
| 9 drift job | `not_evaluated`, exit 0 |
| 10 hardening | uid 999 for the API and Postgres; read-only root filesystem |
| 10 HPA | cpu 1%/70% |
| 11 teardown | the other cluster and the restored context were left untouched |

Environment: kind node image v1.36.1, metrics-server v0.9.0, Docker 29.9.3.

## Earlier validation (historical, 2026-06-13)

Before stage 13 the manifests were validated on a local kind cluster that no longer
exists: two API replicas behind a ClusterIP Service, port-forward 8015 to 8004, single
and batch predictions, a rollback dry run, Prometheus metrics, and HorizontalPodAutoscaler
scale-up and scale-down (2 to 5 replicas, 70% CPU) once Metrics Server was installed with
`--kubelet-insecure-tls`. The API then served `v002` from the file pointer, as root,
without network policies or a database; stage 13 replaced that setup with the one above.
