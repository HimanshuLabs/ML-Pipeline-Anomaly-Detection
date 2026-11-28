"""Kubernetes manifests (parsed only, no cluster): hardened pods, least privilege, docs.

The cluster for this pipeline is the dedicated kind cluster `project4-local`; every kubectl
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
    """Every pod of this pipeline: API, drift job, seed job, Postgres."""

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
    assert sum(
        1 for line in readme.splitlines() if command.search(line) and CONTEXT_FLAG in line
    ) >= 3


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


KCFG_FLAG = '--kubeconfig "$KCFG"'


def test_runbook_creates_the_cluster_in_its_own_kubeconfig_and_never_switches_context() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    create_step = _section(text, 1)
    teardown = _section(text, 11)

    # The default kubeconfig is never written, so there is no context to switch back.
    assert "use-context" not in text and "PREVIOUS_CONTEXT" not in text
    assert 'STATE="$(mktemp -d)"' in create_step
    assert 'KCFG="$STATE/kubeconfig-project4"' in create_step
    create = create_step.index('kind create cluster --name project4-local --kubeconfig "$KCFG"')
    assert create_step.index('> "$STATE/context.before"') < create
    assert create_step.index('> "$STATE/kubeconfig.before"') < create
    for step in (create_step, teardown):
        assert 'sha256sum -c "$STATE/kubeconfig.before"' in step
        assert 'diff - "$STATE/context.before"' in step
    assert create_step.index('sha256sum -c "$STATE/kubeconfig.before"') > create
    delete = teardown.index('kind delete cluster --name project4-local --kubeconfig "$KCFG"')
    assert delete < teardown.index('sha256sum -c "$STATE/kubeconfig.before"')
    assert teardown.index('rm -rf "$STATE"') > teardown.index('sha256sum -c')
    # Every command that talks to the cluster uses the cluster's own kubeconfig file.
    cluster_commands = [
        line for line in text.splitlines()
        if re.search(r"\bkubectl\s+\S", line) and not re.search(r"\bkubectl\s+config\s", line)
    ]
    assert len(cluster_commands) >= 20
    for line in cluster_commands:
        assert f"{KCFG_FLAG} {CONTEXT_FLAG}" in line, line
    for step in (
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


# Each Secret's keys, in the order the runbook writes them.
SECRET_KEYS = {
    APP_DB_SECRET: (
        "PROJECT4_POSTGRES_HOST", "PROJECT4_POSTGRES_PORT", "PROJECT4_POSTGRES_DB",
        "PROJECT4_POSTGRES_USER", "PROJECT4_POSTGRES_PASSWORD",
    ),
    OWNER_SECRET: ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"),
    ADMIN_SECRET: ("admin-token",),
}
GENERATED_VALUE = re.compile(r"\$\(openssl rand -hex \d+\)")


def test_runbook_creates_every_secret_from_stdin_never_from_arguments() -> None:
    import shlex

    text = RUNBOOK.read_text(encoding="utf-8")
    commands = [
        line.strip()
        for line in re.sub(r"\\\n\s*", " ", _section(text, 4)).splitlines()
        if "create secret generic" in line
    ]

    assert len(commands) == len(SECRET_KEYS)
    for name, keys in SECRET_KEYS.items():
        command = next(c for c in commands if f"create secret generic {name} " in c)
        # printf is a builtin: neither the keys nor the generated values reach a process's
        # arguments; kubectl reads KEY=value lines from stdin.
        assert command.startswith("printf ") and " | kubectl " in command, command
        printf, kubectl = command.split(" | ", 1)
        assert kubectl.startswith(f"kubectl {KCFG_FLAG} {CONTEXT_FLAG} -n {NAMESPACE} "), command
        assert kubectl.endswith(f"create secret generic {name} --from-env-file=/dev/stdin")
        words = shlex.split(printf, posix=False)
        assert words[:2] == ["printf", "'%s=%s\\n'"], command
        pairs = dict(zip(words[2::2], words[3::2], strict=True))
        assert tuple(pairs) == keys, command
        for key, value in pairs.items():
            if key in NON_SECRET_VALUES:
                assert value == NON_SECRET_VALUES[key], (name, key)
            else:  # passwords and the admin token are generated, never typed
                assert GENERATED_VALUE.fullmatch(value.strip('"')), (name, key, value)
    assert "--from-literal" not in _section(text, 4)
    assert not re.search(r"--from-literal=\S*(PASSWORD|token)", text, re.IGNORECASE)


def test_runbook_loads_single_platform_image_archives_not_docker_images() -> None:
    # `kind load docker-image` imports every platform of Docker's multi-platform image
    # index and fails ("ctr: content digest … not found"); archives of one platform work.
    text = RUNBOOK.read_text(encoding="utf-8")
    step = _section(text, 3)

    assert "kind load docker-image" not in text
    assert 'for IMAGE in project4-anomaly-api:local project4-anomaly-tools:local ' \
        f'{_compose_postgres_image()}; do' in step
    assert 'docker save --platform "$PLATFORM" "$IMAGE" -o "$ARCHIVE"' in step
    assert 'kind load image-archive "$ARCHIVE" --name project4-local' in step
    assert 'rm -f "$ARCHIVE"' in step
    assert "uname -m" in step and "linux/amd64" in step and "linux/arm64" in step
    assert "docker exec project4-local-control-plane crictl images | grep -E 'project4|postgres'" \
        in step
    assert "content digest" in step  # why: the observed failure


def test_runbook_records_the_live_validation_above_the_historical_note() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    live = text.split("\n## Live validation (2026-11-15)\n", 1)[1].split("\n## ", 1)[0]

    assert text.index("## Live validation (2026-11-15)") < text.index("## Earlier validation")
    for observed in (
        "pointer_source", "predictions_scored_total", "registry_mirror_write_failures_total",
        "never compatible", "v002", "pointer_version 3", "TimeoutError",
        "kubernetes.default.svc:443", "not_evaluated", "uid 999", "1%/70%",
        "v1.36.1", "v0.9.0", "29.9.3",
    ):
        assert observed in live, observed
    assert "'/app/probe': Read-only file system" in _section(text, 10)


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


def test_runbook_promotes_v005_then_really_rolls_back_to_v004_on_every_replica() -> None:
    step = _section(RUNBOOK.read_text(encoding="utf-8"), 7)

    promote = step.index("registry promote --model-version v005")
    dry_run = step.index('"dry_run": true')
    applied = step.index('"dry_run": false')
    assert promote < dry_run < applied
    # Each pointer change is followed by one poll and a check of every pod.
    assert step[promote:dry_run].count("sleep 35") == 1
    assert step[applied:].count("sleep 35") == 1
    assert step.count("for pod in $(") == 2 and step.count("/model/active") >= 2
    assert "every pod: v005" in step and "every pod: v004" in step
    assert "1 (seed) → 2 (promote) → 3 (rollback)" in step
    assert '"to_model_version":"v004"' in step and '"status":"applied"' in step
    assert "No compatible rollback target" not in step  # v004 is a target once v005 is active
    assert "audit.rollback_events" in step and "model_rollback_total" in step
    # The decision-intelligence layer's view of the result, without logging in as
    # project5_reader.
    assert '"SET ROLE project5_reader"' in step
    assert "bad" in step and "md5(string_agg(" in step
    # The same signature query in compose, as project5_reader itself, diffed against the
    # cluster's output without psql's SET line.
    assert step.count('-c "$SIGNATURE_QUERY"') == 2
    assert step.count("-At -F ' | '") == 2
    assert (
        "PGPASSWORD=\"$(grep '^PROJECT4_P5_READER_PASSWORD=' .env | cut -d= -f2-)\"" in step
    )
    assert (
        "docker compose exec -T -e PGPASSWORD anomaly-postgres psql -h 127.0.0.1 "
        "-U project5_reader -d project4_ml -At -F ' | '" in step
    )
    assert (
        "grep -v '^SET$' \"$STATE/signatures.cluster\" | diff - \"$STATE/signatures.compose\""
        in step
    )
    assert "--env-file" not in step


def test_runbook_seed_expectations_match_the_committed_registry() -> None:
    seed = _section(RUNBOOK.read_text(encoding="utf-8"), 5)

    for expected in (
        "snapshot_rows_inserted 6",
        "registry_inserted 5",
        "pointer_model_version v004",
        "pointer_version 1",
        "5 registry rows",
        "4 snapshots",
        "2 datasets",
        "0 rollback events",
        "64e7a3c0-bd07-5f74-95f5-71cc9e494e1b",
    ):
        assert expected in seed, expected
    assert "4 registry rows" not in seed


def test_runbook_port_forwards_run_kubectl_directly_and_are_killed_by_pid() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    lines = text.splitlines()

    # No shell function anywhere: a backgrounded function orphans its kubectl.
    assert not re.search(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", text, re.MULTILINE)
    forwards = [i for i, line in enumerate(lines) if "port-forward" in line and ":8004" in line]
    assert len(forwards) == 2
    for index in forwards:
        line = lines[index].strip()
        assert line.startswith("kubectl "), line
        assert line.endswith("&"), line
        assert re.fullmatch(r"PF_80\d\d_PID=\$!", lines[index + 1].strip()), lines[index + 1]
        before = "\n".join(lines[max(0, index - 3):index])
        assert "ss -ltn" in before  # the port is checked free before forwarding


def test_runbook_records_the_2026_11_16_live_run_above_the_earlier_ones() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    heading = "## 2026-11-16 live record (stage 15e)"
    record = text.split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]

    assert text.index(heading) < text.index("## Live validation (2026-11-15)")
    for observed in (
        "a3a240c", "5 registry rows", "pointer_version 1", "4 snapshot rows",
        "2 dataset rows", "0 rollback events", "pointer_version 2", "pointer_version 3",
        "1 → 2", "093e2da8", "6ac0c919", "36f6f02f", "22 columns", "26 columns",
        "23 columns", "byte-identical", "UIDs",
        "02f0962f4224b0ff447c2421256d1f9e216f56d58fc165378878edcf74cd6858",
        "context unchanged", "`sha256sum -c` OK", "orphaned port-forward", "8016 free",
    ):
        assert observed in record, observed
    assert "project5-local" not in record  # the other cluster is never named


def test_runbook_egress_check_targets_the_kubernetes_api_not_the_internet() -> None:
    step = _section(RUNBOOK.read_text(encoding="utf-8"), 8)

    assert 'create_connection(("kubernetes.default.svc", 443)' in step
    assert "1.1.1.1" not in step
    assert 'create_connection(("project4-postgres", 5432)' in step  # unlabelled pod + API


def test_runbook_stops_the_port_forwards_before_teardown() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    step = _section(text, 11)

    assert "pkill" not in text
    stop = step.index('kill "$PF_8015_PID" "$PF_8016_PID"')
    free = step.index("ss -ltn '( sport = :8015 or sport = :8016 )'")
    assert stop < free < step.index("kind delete cluster --name project4-local")
    assert "orphan" in step  # why the forwards are started and stopped this way
