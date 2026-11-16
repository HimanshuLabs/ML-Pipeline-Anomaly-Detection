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


# --- docs/ (stage 14c-7) ----------------------------------------------------------

DOCS = REPO_ROOT / "docs"
# Exact stale strings replaced in 14c-7, per file (history sections keep other old text).
DOCS_STALE_PHRASES = {
    "rollback_runbook.md": (
        "restart Uvicorn",
        "Authentication/authorization for `/admin/rollback`.",
        "no Kubernetes context is configured",
        "Previous stable rollback target: `isolation_forest v001`",
        "Direct PostgreSQL insert into `audit.rollback_events`.",
        "Applied rollback updates `configs/active_model.yaml`.",
    ),
    "latency_budget.md": (
        "no Kubernetes context is configured",
        "- Production model version: `v002`",
    ),
    "model_lifecycle.md": (
        "The registry is generated runtime metadata and is intentionally ignored by Git.",
        "| Database metadata write | Planned |",
        "The future online inference service will read this pointer",
    ),
    "drift_monitoring.md": (
        "Loads the active v002 baseline statistics",
        "It does not persist directly to PostgreSQL yet.",
        "currently `v002`",
        "Direct PostgreSQL insert wiring is planned",
    ),
    "inference_design.md": (
        "| Direct PostgreSQL insert from Python | Planned |",
        "project4_prediction_latency_seconds",
        "Direct PostgreSQL inserts remain planned hardening",
        "| Authentication | Not implemented locally.",
        "Direct PostgreSQL insert wiring is planned",
    ),
    "source_integration.md": (
        "export PROJECT1_POSTGRES_PASSWORD=",
        "export PROJECT23_POSTGRES_PASSWORD=",
        "## v002 production model",
    ),
    "data_contract.md": ("The current Project 4 feature schema version is:",),
    "feature_engineering.md": ("The current feature schema version is:",),
    "false_positive_false_negative_analysis.md": (
        "Current production model state:",
        "| Active version        | `v002`",
        "| Feature count         | `51`",
        "The current baseline tells us that the model flagged approximately 5.0%",
        "The current active model baseline anomaly rate is:",
        "That is approximately 5.0%.",
        "* The active model is versioned as `v002`.",
    ),
}


def test_docs_have_none_of_the_replaced_stale_phrases() -> None:
    for name, phrases in DOCS_STALE_PHRASES.items():
        text = (DOCS / name).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase not in text, f"{name}: {phrase}"


def test_every_admin_curl_in_the_docs_sends_the_admin_token() -> None:
    admin_curls = [
        f"{path.name}: {command}"
        for path in sorted(DOCS.glob("*.md"))
        for command in _commands(path.read_text(encoding="utf-8"))
        if "curl" in command and "/admin/" in command
    ]

    assert len(admin_curls) >= 3  # rollback runbook (dry run, applied) and the k8s runbook
    for command in admin_curls:
        assert '-H "X-Admin-Token: $ADMIN_TOKEN"' in command, command


def test_readme_has_no_home_directory_paths() -> None:
    assert "~/Desktop" not in _text()


def test_local_runbook_shows_the_disposable_test_database_recipe() -> None:
    runbook = _section("Local runbook")

    assert "--name project4_pg_test -p 127.0.0.1:5547:5432" in runbook
    assert "postgres:16.14" in runbook
    assert "dev_db() { docker run --rm --network host" in runbook
    assert (
        "-e PROJECT4_TEST_DATABASE_URL=postgresql+psycopg2://manshu:test_only@127.0.0.1:5547/"
        "project4_ml -e PROJECT4_REQUIRE_TEST_DATABASE=1" in runbook
    )
    assert "dev_db env -u PYTHONPATH python -m pytest" in runbook
    assert "docker stop project4_pg_test" in runbook
    assert "`--network none`, so it cannot reach any database" in runbook


def test_latency_docs_say_the_two_methods_are_not_comparable() -> None:
    assert "not directly comparable" in _section("Latency budget")
    assert "not directly comparable" in (DOCS / "latency_budget.md").read_text(encoding="utf-8")


def test_rollback_module_docstring_describes_the_database_path() -> None:
    source = (REPO_ROOT / "src" / "anomaly_detection" / "rollback.py").read_text(encoding="utf-8")
    docstring = source.split('"""', 2)[1]

    assert "Database support is planned" not in docstring
    assert "audit.rollback_events row are committed in one" in docstring


# --- stage 15a: register, mirror, read-only extraction ---------------------------------------

SOURCE_INTEGRATION = DOCS / "source_integration.md"
HOME_OR_SCRATCH_PATH = re.compile(r"/home/[A-Za-z0-9_]|~/Desktop|scratchpad|/tmp/claude")


def _extract_commands(text: str) -> list[str]:
    return [c for c in _commands(text) if "python -m anomaly_detection.source_extract" in c]


def test_readme_documents_register_and_mirror() -> None:
    lifecycle = _section("Model lifecycle")

    assert "python -m anomaly_detection.registry register --metadata-path" in lifecycle
    assert "python -m anomaly_detection.registry mirror" in lifecycle
    assert "--compare-with" in lifecycle


def test_every_documented_extract_runs_on_the_host_network_with_passwords_by_name() -> None:
    commands = _extract_commands(_text()) + _extract_commands(
        SOURCE_INTEGRATION.read_text(encoding="utf-8")
    )

    assert len(commands) >= 2  # README and docs/source_integration.md
    for command in commands:
        assert "docker run" in command and "--network host" in command, command
        assert "--env-file" not in command, command
        for name in ("PROJECT1_POSTGRES_PASSWORD", "PROJECT23_POSTGRES_PASSWORD"):
            assert f"-e {name} " in command, (name, command)
            assert f"-e {name}=" not in command, (name, command)


def test_readme_and_source_docs_state_the_extract_is_read_only() -> None:
    for text in (_section("Training snapshots"), SOURCE_INTEGRATION.read_text(encoding="utf-8")):
        assert "default_transaction_read_only=on" in text
        assert "statement_timeout" in text
        assert "SHOW transaction_read_only" in text


def test_readme_and_docs_have_no_home_or_scratch_paths() -> None:
    for path in (README, *sorted(DOCS.glob("*.md"))):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not HOME_OR_SCRATCH_PATH.search(line), f"{path.name}:{number}: {line}"
