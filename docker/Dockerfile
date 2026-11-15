# Project 4 images, one Dockerfile, built with --target:
#   runtime -> project4-anomaly-api:local    API serving only (no mlflow, pyarrow or test tools)
#   tools   -> project4-anomaly-tools:local  batch scoring, source extraction, training,
#                                            retrain, drift job, MLflow tracking and server
#   dev     -> project4-anomaly-dev:local    tools + pytest, ruff, httpx; runs the test suite
#                                            with the repository mounted at /app
# Base image pinned by its multi-arch index digest (python:3.12.16-slim, Debian trixie).

FROM python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# uid/gid 999 as in the earlier images: existing volumes (mlflow_artifacts) belong to it.
RUN groupadd --system --gid 999 app \
    && useradd --system --uid 999 --gid app --home-dir /app app \
    && python -m pip install --root-user-action=ignore --upgrade "pip>=25,<26"

COPY constraints.txt requirements-runtime.txt /app/


FROM base AS runtime-deps

RUN python -m pip install --root-user-action=ignore \
    -r /app/requirements-runtime.txt -c /app/constraints.txt


FROM runtime-deps AS tools-deps

COPY requirements-tools.txt /app/
RUN python -m pip install --root-user-action=ignore \
    -r /app/requirements-tools.txt -c /app/constraints.txt


FROM runtime-deps AS runtime

# The serving image must not carry tools or test packages.
RUN python -c 'import importlib.util, sys; present = [m for m in ("mlflow", "pyarrow", "pytest", "ruff", "httpx") if importlib.util.find_spec(m)]; sys.exit(f"runtime image must not contain {present}" if present else 0)'

COPY api /app/api
COPY src /app/src
COPY configs /app/configs
COPY artifacts/models /app/artifacts/models

# Every tracked model bundle and the registry are copied (see .dockerignore), so the
# image can serve whatever version the shared pointer selects. The build imports the
# API and checks the bundle of the pointer baked into the image (never a hardcoded
# version) with the serving loader.
RUN mkdir -p /app/logs/predictions /app/logs/alerts \
    && python -c "import api.main" \
    && python -m anomaly_detection.bundle_check --active-model configs/active_model.yaml \
    && chown -R app:app /app

USER app

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8004/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8004"]


FROM tools-deps AS tools

COPY src /app/src
COPY configs /app/configs
COPY artifacts/models /app/artifacts/models

RUN mkdir -p /app/data/features/batch_scoring /app/logs/predictions /app/logs/alerts \
    && python -m anomaly_detection.bundle_check --active-model configs/active_model.yaml \
    && chown -R app:app /app

USER app

CMD ["python", "-m", "anomaly_detection.batch_inference", "--help"]


FROM tools-deps AS dev

COPY requirements-dev.txt /app/
RUN python -m pip install --root-user-action=ignore \
    -r /app/requirements-dev.txt -c /app/constraints.txt

USER app

CMD ["python", "-m", "pytest", "-q"]
