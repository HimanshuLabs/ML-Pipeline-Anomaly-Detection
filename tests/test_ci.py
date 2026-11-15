"""Static checks of the GitHub Actions workflow (.github/workflows/ci.yml).

GitHub Actions cannot run locally, so these tests pin what the workflow must do:
install the dev requirements, run the full suite against a required Postgres on the
compose image, prove the runtime image's contents, and validate the Kubernetes
manifests with a checksum-verified kubeconform.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from tests.test_k8s import REPO_ROOT, RUNBOOK, _compose_postgres_image

CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
KUBECONFORM_URL = (
    "https://github.com/yannh/kubeconform/releases/download/v0.8.0/kubeconform-linux-amd64.tar.gz"
)
KUBECONFORM_SHA256 = "9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883"
REAL_POSTGRES_PORTS = {"5433", "5434", "5435"}


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _steps(job: str) -> list[dict[str, Any]]:
    return _workflow()["jobs"][job]["steps"]


def _step_running(job: str, needle: str) -> dict[str, Any]:
    matches = [step for step in _steps(job) if needle in step.get("run", "")]
    assert len(matches) == 1, f"expected one step in {job} running {needle!r}, got {len(matches)}"
    return matches[0]


def _runbook_kubernetes_version() -> str:
    text = RUNBOOK.read_text(encoding="utf-8")
    live = text.split("\n## Live validation (2026-11-15)\n", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"kind node image v(\d+\.\d+\.\d+)", live)
    assert match, "runbook live validation records no kind node version"
    return match.group(1)


def test_ci_installs_dev_requirements_with_constraints_on_python_3_12() -> None:
    steps = _steps("validate")
    setup = [step for step in steps if str(step.get("uses", "")).startswith("actions/setup-python")]
    install = _step_running("validate", "pip install -r")["run"]

    assert setup[0]["with"]["python-version"] == "3.12"
    assert "python -m pip install -r requirements-dev.txt -c constraints.txt" in install
    assert "-r requirements.txt" not in CI.read_text(encoding="utf-8")


def test_ci_postgres_service_is_the_compose_image_on_a_disposable_port() -> None:
    service = _workflow()["jobs"]["validate"]["services"]["postgres"]
    host_port = service["ports"][0].split(":")[0]

    assert service["image"] == _compose_postgres_image()
    assert host_port not in REAL_POSTGRES_PORTS  # never the real P1/P2/P4 database ports
    assert host_port == "5547"


def test_full_suite_step_requires_the_ci_database_so_postgres_tests_never_skip() -> None:
    service = _workflow()["jobs"]["validate"]["services"]["postgres"]
    env = _step_running("validate", "--junitxml")["env"]
    password = service["env"]["POSTGRES_PASSWORD"]

    assert env["PROJECT4_TEST_DATABASE_URL"] == (
        f"postgresql+psycopg2://{service['env']['POSTGRES_USER']}:{password}@127.0.0.1:5547/"
        f"{service['env']['POSTGRES_DB']}"
    )
    assert env["PROJECT4_REQUIRE_TEST_DATABASE"] == "1"


def test_ci_builds_all_three_targets_and_checks_the_image_contents() -> None:
    text = CI.read_text(encoding="utf-8")
    check = _step_running(
        "validate", "docker run --rm --network none project4-anomaly-api:ci"
    )["run"]

    for target, tag in (("runtime", "api"), ("tools", "tools"), ("dev", "dev")):
        assert (
            f"docker build -f docker/Dockerfile --target {target} -t project4-anomaly-{tag}:ci ."
        ) in text
    for forbidden in ("mlflow", "pyarrow", "pytest", "ruff", "httpx"):
        assert f'"{forbidden}"' in check, forbidden
    assert 'Path("/app/sql").exists()' in check
    assert "import api.main" in check
    assert "docker run --rm --network none project4-anomaly-tools:ci test -d /app/sql" in check


def test_kubeconform_is_pinned_and_checksum_verified() -> None:
    run = _step_running("k8s-manifests", "kubeconform-linux-amd64.tar.gz")["run"]

    assert f"curl -fsSL -o kubeconform-linux-amd64.tar.gz {KUBECONFORM_URL}" in run
    assert f'echo "{KUBECONFORM_SHA256}  kubeconform-linux-amd64.tar.gz" | sha256sum -c -' in run
    assert run.index("sha256sum -c") < run.index("tar -xzf")


def test_kubectl_client_is_required_and_printed() -> None:
    run = _step_running("k8s-manifests", "kubectl version --client")["run"]

    assert "command -v kubectl" in run and "exit 1" in run


def test_kubeconform_validates_every_manifest_strictly_for_the_live_kubernetes_version() -> None:
    run = _step_running("k8s-manifests", "kubectl kustomize k8s/")["run"]
    version = _runbook_kubernetes_version()

    assert version == "1.36.1"
    assert "set -euo pipefail" in run  # a failing kustomize must fail the pipe
    assert (
        f"kubectl kustomize k8s/ | kubeconform -strict -summary -kubernetes-version {version} -"
    ) in run
    assert (
        f"kubeconform -strict -summary -kubernetes-version {version} "
        "k8s/seed-job.yaml k8s/secret.example.yaml"
    ) in run
    assert "-ignore-missing-schemas" not in CI.read_text(encoding="utf-8")
