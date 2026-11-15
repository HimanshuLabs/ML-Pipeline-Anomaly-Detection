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


def test_every_pod_is_hardened() -> None:
    for name, spec in _pod_specs().items():
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
            assert set(mounts) == {"/tmp", "/app/logs"}, (name, mounts)
            assert set(mounts.values()) <= empty_dirs, (name, mounts)


def test_no_container_runs_as_root_or_privileged() -> None:
    for name, spec in _pod_specs().items():
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
    admin = {e["name"]: e["valueFrom"]["secretKeyRef"] for e in api["env"]}["ADMIN_TOKEN"]
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
        "hpa.yaml", "pdb.yaml", "cronjob-drift.yaml",
    ])
    assert "secret.example.yaml" not in resources


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
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in _doc_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if command.search(line) and CONTEXT_FLAG not in line
    ]
    assert not offenders
    # The commands exist (the check above is not vacuous).
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.count(f"kubectl {CONTEXT_FLAG}") >= 3


def test_no_file_refers_to_another_projects_cluster() -> None:
    for path in [*_doc_files(), *sorted(K8S.glob("*.yaml"))]:
        assert "project5-local" not in path.read_text(encoding="utf-8"), path
