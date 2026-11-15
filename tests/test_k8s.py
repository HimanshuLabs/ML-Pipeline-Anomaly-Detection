"""Kubernetes manifests (parsed only, no cluster): hardened pods, least privilege, docs.

The cluster for Project 4 is the dedicated kind cluster `project4-local`; every kubectl
command in the docs names its context explicitly so nothing lands in another cluster.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
K8S = REPO_ROOT / "k8s"
CONTEXT_FLAG = "--context kind-project4-local"
NAMESPACE = "project4-ml"
CONFIGMAP = "project4-anomaly-api-config"
APP_DB_SECRET = "project4-app-db"
OWNER_SECRET = "project4-postgres-owner"
ADMIN_SECRET = "project4-admin"
IMAGES = {"project4-anomaly-api:local", "project4-anomaly-tools:local"}
UID = 999
PLACEHOLDER = "change-me"
# Keys whose values are not secrets: in-cluster addresses and names.
NON_SECRET_VALUES = {
    "PROJECT4_POSTGRES_HOST": "project4-postgres",
    "PROJECT4_POSTGRES_PORT": "5432",
    "PROJECT4_POSTGRES_DB": "project4_ml",
    "PROJECT4_POSTGRES_USER": "project4_app",
    "POSTGRES_USER": "project4_owner",
    "POSTGRES_DB": "project4_ml",
}


def _load(name: str) -> dict:
    return yaml.safe_load((K8S / name).read_text(encoding="utf-8"))


def _pod_specs() -> dict[str, dict]:
    deployment = _load("deployment.yaml")
    cronjob = _load("cronjob-drift.yaml")
    return {
        "deployment": deployment["spec"]["template"]["spec"],
        "cronjob": cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"],
    }


def _docs(name: str) -> dict[str, dict]:
    """{kind: document} of a multi-document manifest."""

    text = (K8S / name).read_text(encoding="utf-8")
    return {doc["kind"]: doc for doc in yaml.safe_load_all(text) if doc}


def _seed_job() -> dict:
    return _load("seed-job.yaml")


def _all_pod_specs() -> dict[str, dict]:
    """Every Project 4 pod: API, drift job, seed job, Postgres."""

    return {
        **_pod_specs(),
        "seed": _seed_job()["spec"]["template"]["spec"],
        "postgres": _docs("postgres.yaml")["StatefulSet"]["spec"]["template"]["spec"],
    }


# Writable paths per pod; everything else is the read-only root filesystem.
WRITABLE_MOUNTS = {
    "deployment": {"/tmp", "/app/logs"},
    "cronjob": {"/tmp", "/app/logs"},
    "seed": {"/tmp"},
    "postgres": {"/tmp", "/var/run/postgresql", "/var/lib/postgresql/data"},
}


def test_every_pod_is_hardened() -> None:
    for name, spec in _all_pod_specs().items():
        assert spec["automountServiceAccountToken"] is False, name
        assert spec["securityContext"] == {
            "runAsNonRoot": True,
            "runAsUser": UID,
            "runAsGroup": UID,
            "fsGroup": UID,
            "seccompProfile": {"type": "RuntimeDefault"},
        }, name
        empty_dirs = {v["name"] for v in spec["volumes"] if v.get("emptyDir") == {}}
        for container in spec["containers"]:
            context = container["securityContext"]
            assert context == {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            }, (name, container["name"])
            mounts = {m["mountPath"]: m["name"] for m in container["volumeMounts"]}
            assert set(mounts) == WRITABLE_MOUNTS[name], (name, mounts)
            scratch = {path: volume for path, volume in mounts.items()
                       if path != "/var/lib/postgresql/data"}
            assert set(scratch.values()) <= empty_dirs, (name, mounts)


def test_no_container_runs_as_root_or_privileged() -> None:
    for name, spec in _all_pod_specs().items():
        assert spec["securityContext"]["runAsUser"] != 0, name
        for container in spec["containers"]:
            context = container.get("securityContext", {})
            assert context.get("runAsUser", UID) != 0, (name, container["name"])
            assert not context.get("privileged", False), (name, container["name"])


def test_images_are_the_local_project_images_pulled_if_not_present() -> None:
    specs = _pod_specs()
    images = {}
    for name, spec in specs.items():
        for container in spec["containers"]:
            assert container["imagePullPolicy"] == "IfNotPresent", name
            images[name] = container["image"]
    assert images == {
        "deployment": "project4-anomaly-api:local",
        "cronjob": "project4-anomaly-tools:local",
    }
    assert set(images.values()) <= IMAGES


def test_api_and_drift_job_use_the_app_role_secret_never_the_owner() -> None:
    for name, spec in _pod_specs().items():
        (container,) = spec["containers"]
        sources = {
            ("configMap", s["configMapRef"]["name"]) if "configMapRef" in s
            else ("secret", s["secretRef"]["name"])
            for s in container["envFrom"]
        }
        assert sources == {("configMap", CONFIGMAP), ("secret", APP_DB_SECRET)}, name
    for file_name in ("deployment.yaml", "cronjob-drift.yaml"):
        assert OWNER_SECRET not in (K8S / file_name).read_text(encoding="utf-8"), file_name

    (api,) = _pod_specs()["deployment"]["containers"]
    admin = {e["name"]: e for e in api["env"]}["ADMIN_TOKEN"]["valueFrom"]["secretKeyRef"]
    assert admin == {"name": ADMIN_SECRET, "key": "admin-token", "optional": True}


def test_api_keeps_its_replicas_port_probes_and_resources() -> None:
    deployment = _load("deployment.yaml")
    (api,) = deployment["spec"]["template"]["spec"]["containers"]

    assert deployment["spec"]["replicas"] == 2
    assert api["ports"] == [{"name": "http", "containerPort": 8004, "protocol": "TCP"}]
    assert api["readinessProbe"]["httpGet"] == {"path": "/health", "port": 8004}
    assert api["livenessProbe"]["httpGet"] == {"path": "/livez", "port": 8004}
    assert set(api["resources"]) == {"requests", "limits"}


def test_drift_cronjob_is_bounded() -> None:
    cronjob = _load("cronjob-drift.yaml")
    job = cronjob["spec"]["jobTemplate"]["spec"]
    (container,) = job["template"]["spec"]["containers"]

    assert cronjob["spec"]["concurrencyPolicy"] == "Forbid"
    assert cronjob["spec"]["successfulJobsHistoryLimit"] >= 1
    assert cronjob["spec"]["failedJobsHistoryLimit"] >= 1
    assert job["activeDeadlineSeconds"] > 0
    assert job["template"]["spec"]["restartPolicy"] == "Never"
    assert set(container["resources"]) == {"requests", "limits"}
    assert container["command"][:3] == ["python", "-m", "anomaly_detection.drift_job"]


def test_configmap_holds_exactly_log_level_and_the_mirror_switch() -> None:
    assert _load("configmap.yaml")["data"] == {
        "LOG_LEVEL": "INFO",
        "PROJECT4_REGISTRY_FILE_MIRROR": "false",
    }


def test_pdb_keeps_one_api_pod() -> None:
    pdb = _load("pdb.yaml")
    selector = _load("deployment.yaml")["spec"]["selector"]

    assert pdb["kind"] == "PodDisruptionBudget"
    assert pdb["metadata"]["namespace"] == NAMESPACE
    assert pdb["spec"] == {"minAvailable": 1, "selector": selector}


def test_kustomization_lists_the_runtime_resources_and_not_the_secret_example() -> None:
    resources = _load("kustomization.yaml")["resources"]

    assert sorted(resources) == sorted([
        "namespace.yaml", "configmap.yaml", "deployment.yaml", "service.yaml",
        "hpa.yaml", "pdb.yaml", "cronjob-drift.yaml", "postgres.yaml", "networkpolicy.yaml",
    ])
    assert "secret.example.yaml" not in resources
    assert "seed-job.yaml" not in resources  # applied by the runbook once Postgres is ready


def test_secret_example_holds_only_placeholders() -> None:
    documents = [
        doc for doc in yaml.safe_load_all((K8S / "secret.example.yaml").read_text())
        if doc
    ]
    by_name = {doc["metadata"]["name"]: doc for doc in documents}

    assert set(by_name) == {APP_DB_SECRET, OWNER_SECRET, ADMIN_SECRET}
    for name, doc in by_name.items():
        assert doc["kind"] == "Secret" and doc["metadata"]["namespace"] == NAMESPACE
        assert "data" not in doc, name  # stringData placeholders only, never base64 values
        for key, value in doc["stringData"].items():
            if key in NON_SECRET_VALUES:
                assert value == NON_SECRET_VALUES[key], (name, key)
            else:
                assert value == PLACEHOLDER, (name, key)  # passwords and the admin token
    assert set(by_name[APP_DB_SECRET]["stringData"]) == {
        "PROJECT4_POSTGRES_HOST", "PROJECT4_POSTGRES_PORT", "PROJECT4_POSTGRES_DB",
        "PROJECT4_POSTGRES_USER", "PROJECT4_POSTGRES_PASSWORD",
    }
    assert set(by_name[OWNER_SECRET]["stringData"]) == {
        "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
    }


def _doc_files() -> list[Path]:
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]


def test_every_kubectl_command_in_the_docs_names_the_project4_context() -> None:
    command = re.compile(r"\bkubectl\s+\S")
    # `kubectl config ...` only edits the local kubeconfig (saving and restoring the
    # current context); every command that talks to a cluster names its context.
    kubeconfig_only = re.compile(r"\bkubectl\s+config\s")
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in _doc_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if command.search(line) and CONTEXT_FLAG not in line
        and not kubeconfig_only.search(line)
    ]
    assert not offenders
    # The commands exist (the check above is not vacuous).
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(f"kubectl {CONTEXT_FLAG}") >= 3


def test_no_file_refers_to_another_projects_cluster() -> None:
    for path in [*_doc_files(), *sorted(K8S.glob("*.yaml"))]:
        assert "project5-local" not in path.read_text(encoding="utf-8"), path


# --- stage 13e3: in-cluster Postgres, seed job, network policies, runbook ----------------

LABEL = "app.kubernetes.io/name"
API_LABEL = "project4-anomaly-api"
DRIFT_LABEL = "project4-drift-job"
SEED_LABEL = "project4-db-seed"
POSTGRES_LABEL = "project4-postgres"
DB_CLIENT_LABELS = {API_LABEL, DRIFT_LABEL, SEED_LABEL}


def _env(container: dict) -> dict[str, dict]:
    return {item["name"]: item for item in container.get("env", [])}


def _compose_postgres_image() -> str:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["anomaly-postgres"]["image"]


def test_postgres_is_a_hardened_statefulset_on_the_compose_image() -> None:
    docs = _docs("postgres.yaml")
    service, statefulset = docs["Service"], docs["StatefulSet"]
    spec = statefulset["spec"]
    (container,) = spec["template"]["spec"]["containers"]

    assert service["metadata"]["name"] == POSTGRES_LABEL
    assert service["spec"]["clusterIP"] == "None"  # headless
    assert service["spec"]["ports"] == [{"name": "postgres", "port": 5432, "targetPort": 5432}]
    assert service["spec"]["selector"] == {LABEL: POSTGRES_LABEL}
    assert (spec["replicas"], spec["serviceName"]) == (1, POSTGRES_LABEL)
    assert spec["template"]["metadata"]["labels"][LABEL] == POSTGRES_LABEL
    assert container["image"] == _compose_postgres_image()
    assert container["envFrom"] == [{"secretRef": {"name": OWNER_SECRET}}]
    pgdata = _env(container)["PGDATA"]["value"]
    assert pgdata.startswith("/var/lib/postgresql/data/") and pgdata != "/var/lib/postgresql/data/"
    (claim,) = spec["volumeClaimTemplates"]
    assert claim["spec"]["resources"]["requests"]["storage"] == "1Gi"
    assert claim["spec"]["storageClassName"] == "standard"  # kind's default (local-path)
    for probe in ("readinessProbe", "livenessProbe"):
        assert "pg_isready" in " ".join(container[probe]["exec"]["command"]), probe
    assert set(container["resources"]) == {"requests", "limits"}


def test_seed_job_runs_as_owner_and_sets_the_app_password_from_the_app_secret() -> None:
    job = _seed_job()
    spec = job["spec"]
    (container,) = spec["template"]["spec"]["containers"]
    env = _env(container)
    script = " ".join(container["command"])

    assert (job["kind"], job["metadata"]["name"]) == ("Job", "project4-db-seed")
    assert spec["template"]["metadata"]["labels"][LABEL] == SEED_LABEL
    assert container["image"] == "project4-anomaly-tools:local"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert spec["template"]["spec"]["restartPolicy"] == "Never"
    assert spec["backoffLimit"] >= 0 and spec["activeDeadlineSeconds"] > 0
    assert spec["ttlSecondsAfterFinished"] > 0
    assert {key: env[key]["value"] for key in (
        "PROJECT4_POSTGRES_HOST", "PROJECT4_POSTGRES_PORT", "PROJECT4_POSTGRES_DB"
    )} == {"PROJECT4_POSTGRES_HOST": POSTGRES_LABEL, "PROJECT4_POSTGRES_PORT": "5432",
           "PROJECT4_POSTGRES_DB": "project4_ml"}
    assert env["PROJECT4_POSTGRES_USER"]["valueFrom"]["secretKeyRef"] == {
        "name": OWNER_SECRET, "key": "POSTGRES_USER"}
    assert env["PROJECT4_POSTGRES_PASSWORD"]["valueFrom"]["secretKeyRef"] == {
        "name": OWNER_SECRET, "key": "POSTGRES_PASSWORD"}
    assert env["PROJECT4_APP_PASSWORD"]["valueFrom"]["secretKeyRef"] == {
        "name": APP_DB_SECRET, "key": "PROJECT4_POSTGRES_PASSWORD"}
    steps = [
        "anomaly_detection.database apply-schema",
        "anomaly_detection.database sync-from-files --if-empty",
        "anomaly_detection.database restrict-connect",
    ]
    positions = [script.index(step) for step in steps]
    assert positions == sorted(positions)
    assert re.search(r"-ge \d+", script)  # the wait for Postgres is bounded


def test_owner_secret_is_used_only_by_postgres_and_the_seed_job() -> None:
    users = sorted(
        path.name for path in K8S.glob("*.yaml")
        if path.name != "secret.example.yaml" and OWNER_SECRET in path.read_text()
    )
    assert users == ["postgres.yaml", "seed-job.yaml"]


def test_every_project4_pod_sets_home_to_tmp() -> None:
    for name in ("deployment", "cronjob", "seed"):
        (container,) = _all_pod_specs()[name]["containers"]
        assert _env(container)["HOME"] == {"name": "HOME", "value": "/tmp"}, name


def _policies() -> dict[str, dict]:
    text = (K8S / "networkpolicy.yaml").read_text(encoding="utf-8")
    return {doc["metadata"]["name"]: doc for doc in yaml.safe_load_all(text) if doc}


def _selected_names(selector: dict) -> set[str]:
    if "matchLabels" in selector:
        return {selector["matchLabels"][LABEL]}
    (expression,) = selector["matchExpressions"]
    assert (expression["key"], expression["operator"]) == (LABEL, "In")
    return set(expression["values"])


def test_network_policies_deny_by_default_and_allow_only_the_needed_paths() -> None:
    policies = _policies()
    for policy in policies.values():
        assert policy["kind"] == "NetworkPolicy"
        assert policy["metadata"]["namespace"] == NAMESPACE

    deny = policies["default-deny-all"]["spec"]
    assert deny == {"podSelector": {}, "policyTypes": ["Ingress", "Egress"]}

    dns = policies["allow-dns-egress"]["spec"]
    assert dns["podSelector"] == {} and dns["policyTypes"] == ["Egress"]
    (rule,) = dns["egress"]
    (target,) = rule["to"]
    assert target["namespaceSelector"] == {
        "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}}
    assert target["podSelector"] == {"matchLabels": {"k8s-app": "kube-dns"}}
    assert sorted((p["protocol"], p["port"]) for p in rule["ports"]) == [
        ("TCP", 53), ("UDP", 53)]

    api = policies["allow-api-ingress"]["spec"]
    assert _selected_names(api["podSelector"]) == {API_LABEL}
    assert api["policyTypes"] == ["Ingress"]
    (rule,) = api["ingress"]
    assert rule["from"] == [{"podSelector": {}}]  # any pod in project4-ml only
    assert rule["ports"] == [{"protocol": "TCP", "port": 8004}]

    postgres_in = policies["allow-postgres-ingress"]["spec"]
    assert _selected_names(postgres_in["podSelector"]) == {POSTGRES_LABEL}
    (rule,) = postgres_in["ingress"]
    (source,) = rule["from"]
    assert _selected_names(source["podSelector"]) == DB_CLIENT_LABELS
    assert rule["ports"] == [{"protocol": "TCP", "port": 5432}]

    postgres_out = policies["allow-postgres-egress"]["spec"]
    assert _selected_names(postgres_out["podSelector"]) == DB_CLIENT_LABELS
    assert postgres_out["policyTypes"] == ["Egress"]
    (rule,) = postgres_out["egress"]
    (target,) = rule["to"]
    assert _selected_names(target["podSelector"]) == {POSTGRES_LABEL}
    assert rule["ports"] == [{"protocol": "TCP", "port": 5432}]
    # Nothing else opens a port.
    assert set(policies) == {
        "default-deny-all", "allow-dns-egress", "allow-api-ingress",
        "allow-postgres-ingress", "allow-postgres-egress",
    }


def test_labels_used_by_the_policies_exist_on_the_workloads() -> None:
    pod_labels = {
        name: spec_owner["metadata"]["labels"][LABEL]
        for name, spec_owner in {
            "deployment": _load("deployment.yaml")["spec"]["template"],
            "cronjob": _load("cronjob-drift.yaml")["spec"]["jobTemplate"]["spec"]["template"],
            "seed": _seed_job()["spec"]["template"],
            "postgres": _docs("postgres.yaml")["StatefulSet"]["spec"]["template"],
        }.items()
    }
    assert pod_labels == {
        "deployment": API_LABEL, "cronjob": DRIFT_LABEL,
        "seed": SEED_LABEL, "postgres": POSTGRES_LABEL,
    }


RUNBOOK = REPO_ROOT / "docs" / "kubernetes_runtime_validation.md"


def test_runbook_creates_its_own_cluster_restores_the_context_and_tears_down() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    lines = text.splitlines()

    create = next(i for i, line in enumerate(lines)
                  if "kind create cluster --name project4-local" in line)
    # Saved only when a context exists; restored only when one was saved.
    save = next(i for i, line in enumerate(lines)
                if 'PREVIOUS_CONTEXT="$(kubectl config current-context 2>/dev/null || true)"'
                in line)
    restore = next(i for i, line in enumerate(lines)
                   if 'if [ -n "$PREVIOUS_CONTEXT" ]; then kubectl config use-context '
                   '"$PREVIOUS_CONTEXT"; fi' in line)
    assert save < create < restore
    # The restore is the very next command after creating the cluster.
    between = [
        line for line in lines[create + 1:restore]
        if line.strip().startswith(("kubectl", "kind", "docker"))
    ]
    assert between == []
    for step in (
        "kind load docker-image project4-anomaly-api:local project4-anomaly-tools:local "
        + _compose_postgres_image() + " --name project4-local",
        "kind delete cluster --name project4-local",
        "apply -f k8s/seed-job.yaml",
        "--kubelet-insecure-tls",
        "port-forward svc/project4-anomaly-api 8015:8004",
        "create job --from=cronjob/project4-drift-job",
        "registry_mirror_write_failures_total",
    ):
        assert step in text, step
    assert "kind create cluster --name project4-local" in text


def _section(text: str, number: int) -> str:
    return text.split(f"\n## {number}. ", 1)[1].split("\n## ", 1)[0]


def test_runbook_metrics_server_is_pinned_to_v0_9_0() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "metrics-server/releases/download/v0.9.0/components.yaml" in text
    assert "v0.8.0" not in text
    assert "verify on the release page" in text


def test_runbook_waits_for_every_pod_to_switch_to_the_database_pointer() -> None:
    # Before the seed the API serves the image's file pointer (healthy, pointer_source
    # file) and switches within one poll after it; it does not crash or stay unready.
    seed = _section(RUNBOOK.read_text(encoding="utf-8"), 5)

    assert '"pointer_source": "file"' in seed or "pointer_source `file`" in seed
    assert "sleep 35" in seed
    assert seed.index("apply -f k8s/seed-job.yaml") < seed.index("sleep 35")
    assert "/health" in seed and "pointer_source" in seed.split("sleep 35", 1)[1]
    assert "stay unready" not in seed


def test_runbook_proves_cross_replica_propagation_with_promote_not_a_rollback() -> None:
    step = _section(RUNBOOK.read_text(encoding="utf-8"), 7)

    assert '"dry_run": true' in step and '"dry_run": false' not in step
    assert "No compatible rollback target" in step  # the documented expected refusal
    v002 = step.index("registry promote --model-version v002")
    v004 = step.index("registry promote --model-version v004")
    assert v002 < v004
    assert step[v002:v004].count("sleep 35") == 1 and step[v004:].count("sleep 35") == 1
    assert step.count("/model/active") >= 3  # noted, then every pod twice
    assert "v005" in step  # the real /admin/rollback run comes after the v005 retrain


def test_runbook_egress_check_targets_the_kubernetes_api_not_the_internet() -> None:
    step = _section(RUNBOOK.read_text(encoding="utf-8"), 8)

    assert 'create_connection(("kubernetes.default.svc", 443)' in step
    assert "1.1.1.1" not in step
    assert 'create_connection(("project4-postgres", 5432)' in step  # unlabelled pod + API


def test_runbook_stops_the_port_forwards_before_teardown() -> None:
    step = _section(RUNBOOK.read_text(encoding="utf-8"), 11)

    stop = step.index("pkill -f")
    assert "port-forward svc/project4-anomaly-api 8015:8004" in step
    assert "8016:8004" in step
    assert stop < step.index("kind delete cluster --name project4-local")
