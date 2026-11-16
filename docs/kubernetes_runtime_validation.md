# Kubernetes Runtime Validation (live runbook)

Step-by-step procedure to run Project 4 in its own local kind cluster `project4-local`
and verify it: the database, the seed, the hardened API, a promote and a real rollback
across replicas, the network policies, the drift job and teardown. It is a local,
cost-safe validation, not a cloud deployment.

Rules for every step:

- The cluster lives in its own kubeconfig file, `$KCFG`, never in the default one: the
  current context is never changed, so nothing is switched and nothing has to be
  restored.
- Every `kubectl` command that talks to the cluster names
  `--kubeconfig "$KCFG" --context kind-project4-local`.
- `kind` commands always name the cluster with `--name project4-local`.
- Secrets are generated, never typed or printed.
- Background port-forwards run `kubectl` directly (never through a shell function), and
  their PIDs are recorded; teardown kills those PIDs (step 11).

Run everything from the repository root in one shell. In a new shell, set `STATE` to the
path step 1 printed and `KCFG="$STATE/kubeconfig-project4"` again.

## 0. Build the images

The cluster runs the images built from this repository (the seed job needs the tools
image's `restrict-connect` and `sync-from-files --if-empty`, and seeds from the registry
and pointer files baked into it, so build from a clean checkout of the commit to test):

```bash
git status --porcelain   # expect: nothing
docker build -f docker/Dockerfile --target runtime -t project4-anomaly-api:local .
docker build -f docker/Dockerfile --target tools -t project4-anomaly-tools:local .
```

Retagging `:local` does not change a running compose container (it keeps its image), but
the next `docker compose up` recreates `anomaly-api` and `mlflow` on the new images.

## 1. Create the cluster in its own kubeconfig

`kind create cluster --kubeconfig "$KCFG"` writes the cluster's entry and context into
that file only. The default kubeconfig and its current context are recorded first and
checked afterwards: both must be unchanged.

```bash
STATE="$(mktemp -d)"; echo "STATE=$STATE"
KCFG="$STATE/kubeconfig-project4"
kubectl config current-context > "$STATE/context.before" 2>/dev/null || true
sha256sum "${KUBECONFIG:-$HOME/.kube/config}" > "$STATE/kubeconfig.before"
kind create cluster --name project4-local --kubeconfig "$KCFG"
kubectl --kubeconfig "$KCFG" --context kind-project4-local get nodes
kubectl config current-context 2>/dev/null | diff - "$STATE/context.before" && echo "current context unchanged"
sha256sum -c "$STATE/kubeconfig.before"   # expect: OK
```

## 2. Metrics Server (for the HPA)

Pinned version `v0.9.0` (verify on the release page before use:
https://github.com/kubernetes-sigs/metrics-server/releases). kind's kubelets use
self-signed certificates, so Metrics Server needs `--kubelet-insecure-tls`.

```bash
kubectl --kubeconfig "$KCFG" --context kind-project4-local apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.0/components.yaml
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n kube-system patch deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n kube-system rollout status deployment/metrics-server --timeout=180s
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
  ARCHIVE="$STATE/$(echo "$IMAGE" | tr ':/' '__').tar"
  docker save --platform "$PLATFORM" "$IMAGE" -o "$ARCHIVE" && kind load image-archive "$ARCHIVE" --name project4-local && rm -f "$ARCHIVE"
done
docker exec project4-local-control-plane crictl images | grep -E 'project4|postgres'
```

## 4. Namespace and secrets (generated values)

The app password goes into `project4-app-db`; the seed job gives the same value to the
`project4_app` role, so they always match.

No value is on the command line of `kubectl`, where `ps` would show it: `printf` (a shell
builtin) writes `KEY=value` lines, reusing its `'%s=%s\n'` format for each key and value
pair, and each secret is created from stdin (`--from-env-file=/dev/stdin`). The generated
passwords and token come from `openssl rand`, whose output is never an argument either.

```bash
kubectl --kubeconfig "$KCFG" --context kind-project4-local apply -f k8s/namespace.yaml
printf '%s=%s\n' \
  PROJECT4_POSTGRES_HOST project4-postgres \
  PROJECT4_POSTGRES_PORT 5432 \
  PROJECT4_POSTGRES_DB project4_ml \
  PROJECT4_POSTGRES_USER project4_app \
  PROJECT4_POSTGRES_PASSWORD "$(openssl rand -hex 24)" \
  | kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml create secret generic project4-app-db --from-env-file=/dev/stdin
printf '%s=%s\n' \
  POSTGRES_USER project4_owner \
  POSTGRES_PASSWORD "$(openssl rand -hex 24)" \
  POSTGRES_DB project4_ml \
  | kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml create secret generic project4-postgres-owner --from-env-file=/dev/stdin
printf '%s=%s\n' \
  admin-token "$(openssl rand -hex 32)" \
  | kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml create secret generic project4-admin --from-env-file=/dev/stdin
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
kubectl --kubeconfig "$KCFG" --context kind-project4-local apply -k k8s/
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml rollout status statefulset/project4-postgres --timeout=300s
kubectl --kubeconfig "$KCFG" --context kind-project4-local apply -f k8s/seed-job.yaml
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml wait --for=condition=complete job/project4-db-seed --timeout=600s
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml logs job/project4-db-seed
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml rollout status deployment/project4-anomaly-api --timeout=300s
sleep 35   # one pointer poll (30 s) after the seed
for pod in $(kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[*].metadata.name}'); do
  kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec "$pod" -- python -c 'import json, urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:8004/health"))["pointer_source"])'
done   # every pod: postgres
```

The seed copies the registry and pointer files baked into the tools image. Expected seed
log: `applied …` per schema file (6), `role project4_app: LOGIN`, then

```text
snapshot_rows_inserted 6
registry_inserted 5
registry_updated 0
registry_unchanged 0
pointer_model isolation_forest
pointer_model_version v004
pointer_version 1
pointer_changed True
```

then the role table where only `project4_owner`, `project4_app` and `project5_reader` can
connect. The 6 snapshot rows are 4 `ml.feature_snapshots` rows (v003 and v004 share one
snapshot) and 2 `ml.training_datasets` rows: v001's, and v005's, built from v005's bundle
metadata. Running the seed again is safe: `sync-from-files --if-empty` prints
`skipped: … nothing was overwritten`. Check the seeded rows:

```bash
OWNER="$(kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get secret project4-postgres-owner -o jsonpath='{.data.POSTGRES_USER}' | base64 -d)"
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec statefulset/project4-postgres -- psql -U "$OWNER" -d project4_ml -v ON_ERROR_STOP=1 \
  -c "SELECT model_version, status, training_dataset_id FROM ml.model_registry ORDER BY model_version" \
  -c "SELECT model_version, pointer_version, updated_by FROM ml.active_model_pointer" \
  -c "SELECT (SELECT count(*) FROM ml.feature_snapshots) AS snapshots, (SELECT count(*) FROM ml.training_datasets) AS datasets, (SELECT count(*) FROM audit.rollback_events) AS rollback_events"
```

Expected: 5 registry rows (v001 retired, v002 archived, v003 retired, v004 production,
v005 rolled_back with `training_dataset_id 64e7a3c0-bd07-5f74-95f5-71cc9e494e1b`), the
pointer at v004 with `pointer_version 1` and `updated_by sync-from-files`, 4 snapshots,
2 datasets and 0 rollback events (the seed never copies events).

## 6. Smoke test through port-forward 8015

```bash
ss -ltn '( sport = :8015 )' | tail -n +2   # expect: nothing listening
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml port-forward svc/project4-anomaly-api 8015:8004 > "$STATE/pf8015.log" 2>&1 &
PF_8015_PID=$!
sleep 3
curl -s http://127.0.0.1:8015/health          # status ok, pointer_source "postgres"
curl -s http://127.0.0.1:8015/model/active
curl -s http://127.0.0.1:8015/metrics | grep -E '^(active_model_version|predictions_scored_total)'
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- python -c '
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

## 7. Promote v005, then a real rollback to v004, on every replica

The seeded registry has v004 active and v005 `rolled_back`. `registry promote` accepts a
`rolled_back` version (only a `retired` one needs `--allow-retired`), after the same
serving check as always; it makes v005 production and archives v004. v004 is then the
only compatible rollback target: v001 and v003 are retired, and v002 (a single model) is
never compatible with the segmented v005. One pod changes the shared pointer; every
replica follows within one poll. The pointer goes 1 (seed) → 2 (promote) → 3 (rollback).

The admin token is read into a shell variable; it never appears on screen or in the
shell history (the history holds only the command that reads it), nor in any process's
arguments: `printf` is a shell builtin and curl reads the header from stdin (`-H @-`).

```bash
POD="$(kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[0].metadata.name}')"
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec "$POD" -- python -m anomaly_detection.registry promote --model-version v005 --notes "runbook: promote v005"
# expected: OK: isolation_forest v005 status=production ... registry=postgres active_model_version=v005 pointer_version=2
sleep 35   # the other replicas poll the shared pointer every 30 s
for pod in $(kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[*].metadata.name}'); do
  kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec "$pod" -- python -c 'import json, urllib.request; a = json.load(urllib.request.urlopen("http://127.0.0.1:8004/model/active")); print(a["model_version"], a["training_dataset_id"])'
done   # every pod: v005 64e7a3c0-bd07-5f74-95f5-71cc9e494e1b
```

Dry run, then the applied rollback, through one pod:

```bash
ss -ltn '( sport = :8016 )' | tail -n +2   # expect: nothing listening
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml port-forward "pod/$POD" 8016:8004 > "$STATE/pf8016.log" 2>&1 &
PF_8016_PID=$!
sleep 3
ADMIN_TOKEN="$(kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get secret project4-admin -o jsonpath='{.data.admin-token}' | base64 -d)"
printf 'X-Admin-Token: %s\n' "$ADMIN_TOKEN" | curl -s -X POST http://127.0.0.1:8016/admin/rollback -H @- -H 'Content-Type: application/json' -d '{"dry_run": true, "rollback_reason": "runbook dry run", "triggered_by": "runbook"}'
# expected: "status":"validated", "from_model_version":"v005", "to_model_version":"v004", "validation_status":"dry_run_validated"
printf 'X-Admin-Token: %s\n' "$ADMIN_TOKEN" | curl -s -X POST http://127.0.0.1:8016/admin/rollback -H @- -H 'Content-Type: application/json' -d '{"dry_run": false, "rollback_reason": "runbook: real rollback", "triggered_by": "runbook"}'
# expected: "status":"applied", "action_taken":true, "active_model_version":"v004"
unset ADMIN_TOKEN
sleep 35
for pod in $(kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get pods -l app.kubernetes.io/name=project4-anomaly-api -o jsonpath='{.items[*].metadata.name}'); do
  kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec "$pod" -- python -c 'import json, urllib.request; a = json.load(urllib.request.urlopen("http://127.0.0.1:8004/model/active")); print(a["model_version"], a["training_dataset_id"])'
done   # every pod: v004 None
curl -s http://127.0.0.1:8016/metrics | grep '^model_rollback_total'   # 1.0 on this pod (per process)
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec statefulset/project4-postgres -- psql -U "$OWNER" -d project4_ml \
  -c "SELECT from_model_version, to_model_version, validation_status, triggered_by FROM audit.rollback_events" \
  -c "SELECT model_version, pointer_version FROM ml.active_model_pointer"
# expected: one event v005 -> v004 applied; pointer v004 at pointer_version 3
```

The integration views, as Project 5 reads them (`project5_reader` cannot log in inside
the cluster, so the owner switches to it):

```bash
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec statefulset/project4-postgres -- psql -U "$OWNER" -d project4_ml \
  -c "SET ROLE project5_reader" \
  -c "SELECT source_entity, count(*) AS n, count(*) FILTER (WHERE source_system <> 'postgresql' OR schema_version <> 'project4.integration.v1') AS bad FROM (SELECT source_entity, source_system, schema_version FROM integration.model_metadata UNION ALL SELECT source_entity, source_system, schema_version FROM integration.prediction_outputs UNION ALL SELECT source_entity, source_system, schema_version FROM integration.drift_validation_rollback) v GROUP BY 1 ORDER BY 1"
```

Expected: `bad` is 0 for every view.

Then the column signatures, once in the cluster and once in compose, both as
`project5_reader` and with the same query and output format. In compose
`project5_reader` logs in itself (read-only; its password is read singly from `.env`):

```bash
SIGNATURE_QUERY="SELECT table_name, count(*) AS columns, md5(string_agg(column_name || ':' || data_type, ',' ORDER BY ordinal_position)) AS signature FROM information_schema.columns WHERE table_schema = 'integration' GROUP BY 1 ORDER BY 1"
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec statefulset/project4-postgres -- psql -U "$OWNER" -d project4_ml -At -F ' | ' \
  -c "SET ROLE project5_reader" -c "$SIGNATURE_QUERY" > "$STATE/signatures.cluster"
PGPASSWORD="$(grep '^PROJECT4_P5_READER_PASSWORD=' .env | cut -d= -f2-)" \
  docker compose exec -T -e PGPASSWORD anomaly-postgres psql -h 127.0.0.1 -U project5_reader -d project4_ml -At -F ' | ' \
  -c "$SIGNATURE_QUERY" > "$STATE/signatures.compose"
grep -v '^SET$' "$STATE/signatures.cluster" | diff - "$STATE/signatures.compose" && echo "integration column signatures identical"
```

Expected: three lines each (22, 26 and 23 columns for `drift_validation_rollback`,
`model_metadata` and `prediction_outputs`) and `integration column signatures identical`.
The cluster output starts with psql's `SET` line from `SET ROLE`; the `grep -v` drops it
before the diff.

## 8. Network policy deny test

A pod without an allowed label cannot reach Postgres; the API cannot reach anything
outside its allowed paths, such as the Kubernetes API (`kubernetes.default.svc:443`,
always reachable inside the cluster without a policy); the API still reaches Postgres
and DNS.

```bash
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml run np-probe --rm -i --restart=Never --image=project4-anomaly-tools:local --image-pull-policy=IfNotPresent --command -- python -c 'import socket; socket.create_connection(("project4-postgres", 5432), timeout=5); print("REACHED: policy not enforced")'
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- python -c 'import socket; socket.create_connection(("kubernetes.default.svc", 443), timeout=5); print("REACHED: policy not enforced")'
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- python -c 'import socket; print(socket.getaddrinfo("project4-postgres", 5432)[0][4]); socket.create_connection(("project4-postgres", 5432), timeout=5); print("postgres reachable")'
```

Expected: the first two fail with a timeout (never `REACHED`); the third prints an
address and `postgres reachable`. kindnet's policy engine fails open if it is not
running, so a `REACHED` line means the policies are not enforced.

## 9. One manual drift job run

Nothing has been batch-scored in the cluster, so the job reports `not_evaluated` and
exits 0.

```bash
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml create job --from=cronjob/project4-drift-job drift-manual
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml wait --for=condition=complete job/drift-manual --timeout=300s
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml logs job/drift-manual
```

## 10. Hardening check

```bash
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- id          # uid=999 gid=999
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec deploy/project4-anomaly-api -- sh -c 'touch /app/probe' 2>&1 | tail -2   # touch: cannot touch '/app/probe': Read-only file system
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml exec statefulset/project4-postgres -- id        # uid=999
curl -s http://127.0.0.1:8015/metrics | grep registry_mirror_write_failures_total   # 0.0
kubectl --kubeconfig "$KCFG" --context kind-project4-local -n project4-ml get pods,svc,hpa,pdb,cronjob,networkpolicy
```

## 11. Teardown

Stop the port-forwards from steps 6 and 7 by the PIDs recorded when they started, and
check both ports are free, then delete the cluster and check the default kubeconfig is
byte-identical to step 1. The port-forwards are killed by PID, not by matching command
lines (a pattern can match another project's forwards), and they are started from
`kubectl` itself: when a shell function is put in the background, `$!` is the PID of the
subshell running the function, so killing it leaves `kubectl` running as an orphan that
still holds the port.

```bash
kill "$PF_8015_PID" "$PF_8016_PID"
sleep 1
ss -ltn '( sport = :8015 or sport = :8016 )' | tail -n +2   # expect: nothing listening
kind delete cluster --name project4-local --kubeconfig "$KCFG"
kind get clusters                                           # project4-local is gone
kubectl config current-context 2>/dev/null | diff - "$STATE/context.before" && echo "current context unchanged"
sha256sum -c "$STATE/kubeconfig.before"                     # expect: OK
rm -rf "$STATE"
```

## 2026-11-16 live record (stage 15e)

The runbook above was run end to end on a fresh `project4-local` cluster, with images
built from commit `1f855c5` (compose then served v004, with v005 `rolled_back` after the
stage 15d compose rollback).

| Step | Observed |
| --- | --- |
| 0 images | built from `1f855c5`; the tools image's `model_registry.json` sha256 equal to the repository's (`02f0962f4224b0ff447c2421256d1f9e216f56d58fc165378878edcf74cd6858`), and its `configs/active_model.yaml` on v004 |
| 1 kubeconfig | cluster created in its own kubeconfig file; the default kubeconfig's current context unchanged and the default kubeconfig byte-identical (`sha256sum -c` OK) after create |
| 5 seed | the committed registry state reproduced exactly: 5 registry rows, pointer v004 at `pointer_version 1`, 4 snapshot rows, 2 dataset rows (v001's, and v005's `64e7a3c0-bd07-5f74-95f5-71cc9e494e1b` from its bundle metadata), 0 rollback events |
| 7 promote | `registry promote --model-version v005` through one pod: both pods on v005 within one poll (`pointer_version 2`), `model_reload_total{result="success"}` 1 → 2 on each pod; a v005 `/predict` succeeded |
| 7 rollback | dry run validated v005 → v004; applied: both pods on v004 within one poll, `pointer_version 3`, one `audit.rollback_events` row |
| 7 views | as `project5_reader`: `bad` 0; column signatures `drift_validation_rollback` 22 columns `093e2da8…`, `model_metadata` 26 columns `6ac0c919…`, `prediction_outputs` 23 columns `36f6f02f…`, identical to the compose database |
| 11 port 8016 | this run still started the 8016 port-forward through a shell function: `kill $!` ended only the subshell, and `ss` one second later (before `kind delete`) still showed 8016 held by the orphaned port-forward process. It exited on its own once the cluster was deleted (`pgrep` for the temporary kubeconfig path found nothing) and `ss` then showed 8016 free. Against a cluster that keeps running it would have held the port indefinitely, which is why the runbook now runs `kubectl` directly and kills its own PID |
| 11 teardown | the other kind cluster's nodes Ready and its pods unchanged (UIDs, restart counts); the default kubeconfig's current context unchanged and the default kubeconfig byte-identical (`sha256sum -c` OK) |

## Live validation (2026-11-15)

The runbook as it was then (the cluster in the default kubeconfig, its previous context
restored right after creation) was run end to end on a fresh `project4-local` cluster.

| Step | Observed |
| --- | --- |
| 5 seed | 6 schema files applied, `role project4_app: LOGIN`, 4 registry rows, pointer v004 at `pointer_version 1` |
| 5 pointer | both pods report `pointer_source` `postgres`, model v004 |
| 6 smoke | `/predict` served by v004; `predictions_scored_total` 1; `registry_mirror_write_failures_total` 0 |
| 7 rollback dry run | HTTP 400: v002 is never compatible with v004, and v005 did not exist yet |
| 7 propagation | promote v002 through one pod: both pods v002 (`pointer_version 2`); promote v004 back: both pods v004 (`pointer_version 3`) |
| 8 network policy | unlabelled pod to Postgres: `TimeoutError`; API to `kubernetes.default.svc:443`: `TimeoutError`; API to Postgres: reachable |
| 9 drift job | `not_evaluated`, exit 0 |
| 10 hardening | uid 999 for the API and Postgres; read-only root filesystem |
| 10 HPA | cpu 1%/70% |
| 11 teardown | the other cluster and the previous context were left untouched |

Environment: kind node image v1.36.1, metrics-server v0.9.0, Docker 29.9.3.

## Earlier validation (historical, 2026-06-13)

Before stage 13 the manifests were validated on a local kind cluster that no longer
exists: two API replicas behind a ClusterIP Service, port-forward 8015 to 8004, single
and batch predictions, a rollback dry run, Prometheus metrics, and HorizontalPodAutoscaler
scale-up and scale-down (2 to 5 replicas, 70% CPU) once Metrics Server was installed with
`--kubelet-insecure-tls`. The API then served `v002` from the file pointer, as root,
without network policies or a database; stage 13 replaced that setup with the one above.
