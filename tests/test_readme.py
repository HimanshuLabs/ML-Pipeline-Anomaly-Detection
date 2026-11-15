"""The README must match the code: active model, metrics, admin auth, links."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys

import yaml

from tests.conftest import REAL_ACTIVE_MODEL_PATH, REPO_ROOT

README = REPO_ROOT / "README.md"
METRICS = REPO_ROOT / "src" / "anomaly_detection" / "metrics.py"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
# Setup steps every job has; the gate list documents the checks, not these.
CI_SETUP_STEPS = {"Check out repository", "Set up Python"}

STALE_PHRASES = (
    "tests=112",
    "Restart Uvicorn",
    "restart Uvicorn",
    "Direct PostgreSQL inserts are planned",
    "| `v002` | Active production model",
    "current production model: v002",
    "active model metadata points to `v002`",
    "against the approved `v002` training baseline",
    "no authentication on local administrative API endpoints",
    "JSONL persistence before direct database insert wiring",
    "including the database evidence write",  # the histogram excludes it (background task)
    "Dockerized API and batch runtimes",
    "Docker API and batch images build",
    "Python import validation",  # not a CI step
)


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _commands(text: str) -> list[str]:
    """Shell lines with backslash continuations joined."""
    return [line.strip() for line in re.sub(r"\\\n\s*", " ", text).splitlines()]


def test_readme_active_model_matches_the_pointer() -> None:
    pointer = yaml.safe_load(REAL_ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    rows = re.findall(r"^\| Active version \| `(v\d+)` \|", _text(), re.MULTILINE)

    assert rows == [pointer["active_model_version"]]


def test_readme_lists_exactly_the_metric_families_of_metrics_py() -> None:
    block = _text().split("Core metric families:", 1)[1].split("```text\n", 1)[1]
    listed = {line.strip() for line in block.split("```", 1)[0].splitlines() if line.strip()}
    registered = set(
        re.findall(
            r'(?:Counter|Gauge|Histogram)\(\s*"([a-z_]+)"',
            METRICS.read_text(encoding="utf-8"),
        )
    )

    assert registered, "no metric families found in metrics.py"
    assert listed == registered


def test_every_admin_curl_in_the_readme_sends_the_admin_token() -> None:
    admin_curls = [c for c in _commands(_text()) if "curl" in c and "/admin/" in c]

    assert admin_curls, "the README shows how to call /admin/rollback"
    for command in admin_curls:
        assert '-H "X-Admin-Token: $ADMIN_TOKEN"' in command, command


def test_readme_has_none_of_the_stale_phrases() -> None:
    text = _text()

    for phrase in STALE_PHRASES:
        assert phrase not in text, phrase


def test_every_relative_link_and_doc_path_in_the_readme_exists() -> None:
    text = _text()
    links = [
        target.split("#", 1)[0]
        for target in re.findall(r"\]\(([^)\s]+)\)", text)
        if not target.startswith(("http://", "https://", "#", "mailto:"))
    ]
    doc_paths = re.findall(r"`(docs/[A-Za-z0-9_./-]+\.md)`", text)

    assert links and doc_paths
    for path in links + doc_paths:
        assert (REPO_ROOT / path).exists(), path


def _section(heading: str) -> str:
    return _text().split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def _module_commands() -> list[list[str]]:
    """Every `python -m anomaly_detection.<module> ...` in the README, as argv tokens."""
    commands = []
    for line in _commands(_text()):
        start = line.find("python -m anomaly_detection.")
        if start != -1:
            commands.append(shlex.split(line[start:].split("#", 1)[0]))
    return commands


def test_every_readme_module_command_uses_only_flags_its_parser_accepts() -> None:
    commands = _module_commands()
    env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"}

    assert len(commands) >= 5
    for argv in commands:
        module = argv[2]
        rest = argv[3:]
        subcommand = [rest[0]] if rest and not rest[0].startswith("-") else []
        result = subprocess.run(
            [sys.executable, "-m", module, *subcommand, "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (argv, result.stderr[-500:])
        for flag in (token.split("=", 1)[0] for token in rest if token.startswith("--")):
            assert flag in result.stdout, (" ".join(argv), flag)


def test_ci_gate_list_mirrors_the_workflow_steps() -> None:
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    steps = {
        step["name"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
    } - CI_SETUP_STEPS
    gates = _text().split("Implemented CI gates:\n\n", 1)[1].split("\n\n", 1)[0]
    bullets = [line for line in gates.splitlines() if line.startswith("- ")]
    named = [re.match(r"- \*\*(.+?)\*\*", line) for line in bullets]

    assert bullets and all(named), "each gate bullet starts with its **step name**"
    # Backticks only format the name (they keep `kubectl` prose out of the kubectl check).
    assert sorted(match.group(1).replace("`", "") for match in named) == sorted(steps)


def test_local_runbook_wraps_docker_run_in_shell_functions() -> None:
    runbook = _section("Local runbook")

    assert "='docker run" not in runbook  # a quoted command string breaks on expansion
    assert "dev() { docker run" in runbook and "tools() { docker run" in runbook
    assert "dev env -u PYTHONPATH python -m pytest" in runbook
    assert "dev ruff check --no-cache ." in runbook


def test_kubernetes_section_follows_docker_and_lists_every_manifest() -> None:
    text = _text()
    after_docker = text.split("\n## Dockerized runtimes\n", 1)[1].split("\n## ", 2)[1]
    listed = set(re.findall(r"^\| `(k8s/[a-z.-]+\.yaml)` \|", _section(
        "Kubernetes runtime manifests"), re.MULTILINE))

    assert after_docker.startswith("Kubernetes runtime manifests\n")
    assert listed == {f"k8s/{path.name}" for path in (REPO_ROOT / "k8s").glob("*.yaml")}
    assert "docs/kubernetes_runtime_validation.md" in _section("Kubernetes runtime manifests")
