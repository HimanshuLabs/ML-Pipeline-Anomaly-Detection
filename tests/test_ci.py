"""Static checks of the GitHub Actions workflow (.github/workflows/ci.yml).

GitHub Actions cannot run locally, so these tests pin what the workflow must do:
install the dev requirements, run the full suite against a required Postgres on the
compose image, prove the runtime image's contents, and validate the Kubernetes
manifests with a checksum-verified kubeconform.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
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
    # Never the real database ports of the real-time pipeline, the batch lakehouse or this
    # pipeline.
    assert host_port not in REAL_POSTGRES_PORTS
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


# --- registry and baseline audit (the exact heredoc from ci.yml) ----------------------

MODELS = REPO_ROOT / "artifacts" / "models" / "isolation_forest"
BUNDLE_JSON = ("metadata.json", "baseline_stats.json", "feature_schema.json")
V004_SEGMENTS = ("campaign", "customer", "funnel_day", "product", "user", "web_customer")


def _audit_script() -> str:
    run = _step_running("validate", "Baseline audit failed")["run"]
    match = re.search(r"python - <<'PY'\n(.*?)\nPY\n", run, re.DOTALL)
    assert match, "audit heredoc not found in ci.yml"
    return match.group(1) + "\n"


def _bundle_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for version in ("v002", "v004")
        for path in sorted((MODELS / f"model_version={version}").iterdir())
    }


@pytest.fixture()
def real_bundles_unchanged() -> Any:
    before = _bundle_hashes()
    yield
    assert _bundle_hashes() == before


def _run_audit(
    root: Path,
    version: str,
    corrupt: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the CI audit in root against copies of a real bundle's JSON files.

    The audit reads only the pointer and the three JSON files, never model.joblib.
    """
    relative_dir = Path("artifacts/models/isolation_forest") / f"model_version={version}"
    bundle = root / relative_dir
    bundle.mkdir(parents=True)
    files = {
        name: json.loads((MODELS / f"model_version={version}" / name).read_text("utf-8"))
        for name in BUNDLE_JSON
    }
    if corrupt is not None:
        corrupt(files)
    for name, payload in files.items():
        (bundle / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "active_model.yaml").write_text(
        yaml.safe_dump(
            {
                "active_model_version": version,
                "artifact_path": str(relative_dir / "model.joblib"),
            }
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, "-"],
        input=_audit_script(),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_audit_passes_a_single_model_without_segment_checks(
    tmp_path: Path, real_bundles_unchanged: None
) -> None:
    result = _run_audit(tmp_path, "v002")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: baseline metrics audit passed" in result.stdout
    assert "segment" not in result.stdout.lower()
    assert result.stdout.count("OK: ") == 3 + 11 + 1  # files, the 11 checks, verdict


def test_audit_checks_every_segment_of_a_segmented_model(
    tmp_path: Path, real_bundles_unchanged: None
) -> None:
    result = _run_audit(tmp_path, "v004")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "OK: segments_match_across_metadata_baseline_schema" in result.stdout
    for segment in V004_SEGMENTS:
        for check in (
            "feature_baselines_equal_feature_names",
            "feature_names_within_model",
            "anomaly_rate_in_0_1",
            "row_count_matches_baseline",
        ):
            assert f"OK: segment[{segment}].{check}" in result.stdout


def _drop_a_feature_baseline(files: dict[str, dict[str, Any]]) -> None:
    baselines = files["baseline_stats.json"]["segments"]["campaign"]["feature_baselines"]
    del baselines[next(iter(baselines))]


def _drop_an_anomaly_rate(files: dict[str, dict[str, Any]]) -> None:
    del files["baseline_stats.json"]["segments"]["product"]["anomaly_rate"]


def _drop_a_baseline_segment(files: dict[str, dict[str, Any]]) -> None:
    del files["baseline_stats.json"]["segments"]["user"]


@pytest.mark.parametrize(
    ("corrupt", "expected_fails"),
    [
        (_drop_a_feature_baseline, ["segment[campaign].feature_baselines_equal_feature_names"]),
        (_drop_an_anomaly_rate, ["segment[product].anomaly_rate_in_0_1"]),
        (
            _drop_a_baseline_segment,
            [
                "segments_match_across_metadata_baseline_schema",
                "segment[user].feature_baselines_equal_feature_names",
                "segment[user].anomaly_rate_in_0_1",
                "segment[user].row_count_matches_baseline",
            ],
        ),
    ],
    ids=["feature-baseline-removed", "anomaly-rate-removed", "baseline-segment-missing"],
)
def test_audit_fails_a_corrupted_segment_and_names_it(
    tmp_path: Path,
    real_bundles_unchanged: None,
    corrupt: Callable[[dict[str, dict[str, Any]]], None],
    expected_fails: list[str],
) -> None:
    result = _run_audit(tmp_path, "v004", corrupt)

    assert result.returncode != 0
    assert "Baseline audit failed" in result.stderr
    fails = sorted(line[len("FAIL: "):] for line in result.stdout.splitlines()
                   if line.startswith("FAIL: "))
    assert fails == sorted(expected_fails)
