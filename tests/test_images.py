"""Docker images and dependency sets: runtime (API), tools, dev from one Dockerfile.

The runtime image itself proves at build time that it lacks the tools/dev packages
and can import and load the served model (see docker/Dockerfile); these tests guard
the files that produce it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
BASE_IMAGE = (
    "python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
)
IMAGES = {
    "runtime": "project4-anomaly-api:local",
    "tools": "project4-anomaly-tools:local",
    "dev": "project4-anomaly-dev:local",
}
# Declared once but never imported anywhere (psycopg2 is the driver in use).
REMOVED_PACKAGES = ("psycopg", "psycopg-binary", "python-dotenv", "pydantic-settings")
# Still installed (and pinned) as a dependency of uvicorn[standard], just not declared.
TRANSITIVE_ONLY = {"python-dotenv"}
REMOVED_IMPORTS = {
    "psycopg": r"^\s*(import|from)\s+psycopg(\s|\.|$)",
    "dotenv": r"^\s*(import|from)\s+dotenv\b",
    "pydantic_settings": r"^\s*(import|from)\s+pydantic_settings\b",
}
NOT_IN_RUNTIME = ("mlflow", "pyarrow", "pytest", "ruff", "httpx")


def _requirements(name: str) -> dict[str, str]:
    """{package name: line} of a requirements file, '-r' includes as '-r <file>'."""

    entries: dict[str, str] = {}
    for line in (REPO_ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            entries[line] = line
            continue
        package = re.split(r"[\[<>=!~ ;]", line, maxsplit=1)[0].lower().replace("_", "-")
        entries[package] = line
    return entries


def _constraints() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (REPO_ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name.lower().replace("_", "-")] = version
    return pins


def test_one_dockerfile_builds_runtime_tools_and_dev_from_a_pinned_base() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    stages = re.findall(r"^FROM\s+(\S+)\s+AS\s+(\S+)", text, flags=re.MULTILINE)

    assert stages[0] == (BASE_IMAGE, "base")
    assert {"runtime", "tools", "dev"} <= {name for _, name in stages}
    assert not (REPO_ROOT / "docker" / "Dockerfile.api").exists()
    assert not (REPO_ROOT / "docker" / "Dockerfile.batch").exists()
    # The image user stays uid/gid 999 (owner of existing volumes such as mlflow_artifacts).
    assert "--uid 999" in text and "--gid 999" in text


def test_runtime_stage_refuses_tools_and_dev_packages_and_must_serve() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    runtime = re.split(r"^FROM \S+ AS runtime$", text, flags=re.MULTILINE)[1]
    runtime = runtime.split("\nFROM ", 1)[0]

    for package in NOT_IN_RUNTIME:
        assert f'"{package}"' in runtime, package  # build-time absence check
    assert "import api.main" in runtime
    assert "anomaly_detection.bundle_check" in runtime
    assert "requirements-tools.txt" not in runtime and "requirements-dev.txt" not in runtime


def test_requirements_split_runtime_tools_dev() -> None:
    runtime = _requirements("requirements-runtime.txt")
    tools = _requirements("requirements-tools.txt")
    dev = _requirements("requirements-dev.txt")

    for package in NOT_IN_RUNTIME:
        assert package not in runtime, package
    assert {"fastapi", "uvicorn", "scikit-learn", "joblib", "sqlalchemy",
            "psycopg2-binary", "prometheus-client", "pandas", "numpy"} <= set(runtime)
    assert "-r requirements-runtime.txt" in tools and {"mlflow", "pyarrow"} <= set(tools)
    assert "-r requirements-tools.txt" in dev and {"pytest", "ruff", "httpx"} <= set(dev)
    # 13a: NumPy 2.5 deprecations (pandas 2.3 timedelta ops) and joblib's shape setter.
    assert runtime["numpy"] == "numpy>=1.26,<2.5"
    assert runtime["joblib"] == "joblib>=1.6,<2.0"
    # requirements.txt stays the full (dev) set for anything that installs it directly.
    assert set(_requirements("requirements.txt")) == {"-r requirements-dev.txt"}


def test_unused_packages_are_gone_and_never_imported() -> None:
    every_requirement = {
        package
        for name in ("requirements-runtime.txt", "requirements-tools.txt",
                     "requirements-dev.txt")
        for package in _requirements(name)
    }
    constraints = _constraints()
    for package in REMOVED_PACKAGES:
        assert package not in every_requirement, package
        if package not in TRANSITIVE_ONLY:
            assert package not in constraints, package

    offenders = []
    for root in ("src", "api", "scripts", "tests"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for module, pattern in REMOVED_IMPORTS.items():
                    if re.search(pattern, line):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{number} {module}")
    assert not offenders


def test_constraints_pin_the_13a_choice() -> None:
    constraints = _constraints()
    text = (REPO_ROOT / "constraints.txt").read_text(encoding="utf-8")

    assert constraints["numpy"] == "2.4.6"
    assert constraints["joblib"] == "1.6.0"
    assert "project4-anomaly-dev:local" in text.splitlines()[0]  # frozen from the dev image
    assert "3.12.16" in text
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any("numpy<2.5" in line for line in header)
    assert any("joblib>=1.6" in line for line in header)


def test_compose_and_k8s_use_the_three_image_names() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["anomaly-api"]["image"] == IMAGES["runtime"]
    assert services["anomaly-api"]["build"] == {
        "context": ".", "dockerfile": "docker/Dockerfile", "target": "runtime"
    }
    assert services["anomaly-batch"]["image"] == IMAGES["tools"]
    assert services["anomaly-batch"]["build"]["target"] == "tools"
    assert services["mlflow"]["image"] == IMAGES["tools"]  # mlflow is not in the runtime image
    cronjob = yaml.safe_load((REPO_ROOT / "k8s" / "cronjob-drift.yaml").read_text())
    container = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == IMAGES["tools"]
    deployment = yaml.safe_load((REPO_ROOT / "k8s" / "deployment.yaml").read_text())
    assert deployment["spec"]["template"]["spec"]["containers"][0]["image"] == IMAGES["runtime"]


def _is_secrets_file(relative: Path) -> bool:
    """.env and .env.* (except the committed .env.example): never opened by a test."""

    name = relative.name
    return (name == ".env" or name.startswith(".env.")) and name != ".env.example"


def _scanned_repo_files() -> list[Path]:
    """Repository text files to scan: no hidden directories (except .github), no
    runtime data, no secrets files."""

    files = []
    for path in sorted(REPO_ROOT.rglob("*")):
        relative = path.relative_to(REPO_ROOT)
        top = relative.parts[0]
        hidden = top.startswith(".") and top != ".github" and len(relative.parts) > 1
        if not path.is_file() or hidden or _is_secrets_file(relative) or top in {
            "data", "logs", "artifacts", "mlruns", "reports",
        } or path.suffix in {".pyc", ".joblib", ".parquet"}:
            continue
        if relative == Path("tests/test_images.py"):
            continue
        files.append(path)
    return files


def test_repo_scan_never_opens_a_secrets_file() -> None:
    scanned = {path.relative_to(REPO_ROOT) for path in _scanned_repo_files()}

    assert Path(".env.example") in scanned
    assert not any(_is_secrets_file(relative) for relative in scanned)
    for name in (".env", ".env.local", ".env.production"):
        assert _is_secrets_file(Path(name)), name
    assert not _is_secrets_file(Path(".env.example"))


def test_no_reference_to_the_old_image_name_or_dockerfiles_remains() -> None:
    stale = re.compile(r"project4-anomaly-batch|Dockerfile\.api|Dockerfile\.batch")
    offenders = []
    for path in _scanned_repo_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(REPO_ROOT)
        offenders += [f"{relative}:{n}" for n, line in enumerate(text.splitlines(), 1)
                      if stale.search(line)]
    assert not offenders


def test_tools_image_owns_the_mlflow_volume_path() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    tools = re.split(r"^FROM \S+ AS tools$", text, flags=re.MULTILINE)[1].split("\nFROM ", 1)[0]

    assert "mkdir -p" in tools and "/var/lib/mlflow" in tools
    assert "chown -R app:app /app /var/lib/mlflow" in tools


def test_every_published_compose_port_is_bound_to_localhost() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    published = {
        name: service.get("ports", []) for name, service in compose["services"].items()
    }

    assert any(published.values())
    for name, ports in published.items():
        for port in ports:
            assert str(port).startswith("127.0.0.1:"), (name, port)
    assert published["anomaly-api"] == ["127.0.0.1:${PROJECT4_API_HOST_PORT:-8014}:8004"]


def test_ci_builds_all_three_targets() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for target, image in IMAGES.items():
        tag = image.replace(":local", ":ci")
        assert f"docker build -f docker/Dockerfile --target {target} -t {tag} ." in ci
