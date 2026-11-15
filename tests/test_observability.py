"""Logging, online drift status, metrics, and the monitoring/ configuration."""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import app, state
from tests.test_database import (  # noqa: F401 - pytest fixtures
    database_mode,
    initialized_database,
    postgres_engine,
)
from tests.test_drift_job import SHIFTED_TYPE, _current_frame, _serve_segmented
from tests.test_segmented_serving import _payload, _segments

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITORING = REPO_ROOT / "monitoring"


# --- logging --------------------------------------------------------------------------


@pytest.fixture()
def fresh_logging(monkeypatch: pytest.MonkeyPatch):
    """Undo configure_logging's changes to the root logger after the test."""
    from anomaly_detection import logging_config

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    monkeypatch.setattr(logging_config, "_CONFIGURED_HANDLER", None)
    yield logging_config
    root.handlers[:] = handlers
    root.setLevel(level)


def test_logging_is_configured_once_and_info_is_visible(
    fresh_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    stream = io.StringIO()

    first = fresh_logging.configure_logging(stream=stream)
    second = fresh_logging.configure_logging(stream=io.StringIO())
    logging.getLogger("api.main").info("Swapped in model v123 from shared pointer version 7")

    assert first is second
    assert logging.getLogger().handlers.count(first) == 1
    assert logging.getLogger().level == logging.INFO
    line = stream.getvalue().strip()
    assert line.endswith("INFO api.main - Swapped in model v123 from shared pointer version 7")
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO ", line)


def test_log_level_env_is_respected(fresh_logging, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "warning")
    stream = io.StringIO()

    fresh_logging.configure_logging(stream=stream)
    logging.getLogger("anomaly_detection.drift_job").info("hidden info")
    logging.getLogger("anomaly_detection.drift_job").warning("visible warning")

    assert "hidden info" not in stream.getvalue()
    assert "WARNING anomaly_detection.drift_job - visible warning" in stream.getvalue()


def test_secrets_never_appear_in_log_output(
    fresh_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "pg-secret-value-91f3c"
    token = "admin-token-value-0123456789"
    monkeypatch.setenv("PROJECT4_POSTGRES_PASSWORD", secret)
    monkeypatch.setenv("ADMIN_TOKEN", token)
    stream = io.StringIO()
    fresh_logging.configure_logging(stream=stream)
    logger = logging.getLogger("anomaly_detection.database")

    logger.info("connecting with %s and %s", secret, token)
    try:
        raise RuntimeError(f"login failed for password {secret}")
    except RuntimeError:
        logger.exception("database error")

    output = stream.getvalue()
    assert secret not in output and token not in output
    assert output.count("[REDACTED]") >= 3


def test_api_startup_configures_logging(
    fresh_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app):
        assert fresh_logging._CONFIGURED_HANDLER in logging.getLogger().handlers


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (
            "anomaly_detection.retrain",
            ["--snapshot", "missing-snapshot", "--model-version", "v900"],
        ),
        ("anomaly_detection.registry", ["retire", "--model-version", "v900", "--notes", "x"]),
        ("anomaly_detection.drift_job", ["--input", "missing.parquet"]),
        ("anomaly_detection.database", ["check"]),
    ],
)
def test_every_cli_configures_logging(fresh_logging, module: str, argv: list[str]) -> None:
    import importlib

    importlib.import_module(module).main(argv)

    assert fresh_logging._CONFIGURED_HANDLER in logging.getLogger().handlers


def test_source_extract_and_batch_inference_clis_configure_logging() -> None:
    import inspect

    from anomaly_detection import batch_inference, source_extract, training

    for module in (source_extract, batch_inference, training):
        assert "configure_logging(" in inspect.getsource(module.main), module.__name__


# --- online drift status --------------------------------------------------------------


def test_predict_drift_status_per_segment_from_the_latest_run(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import drift
    from anomaly_detection.drift_job import run_drift_job

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    segments = _segments(artifact_dir / "metadata.json")
    frame = _current_frame(artifact_dir)
    input_path = tmp_path / "current.parquet"
    frame[frame["entity_type"] != "product"].to_parquet(input_path)  # product: no run
    run_drift_job(source="input", input_path=input_path, min_rows=30)

    reads: list[int] = []
    real_read = drift.read_latest_drift_jsonl

    def _counting_read(*args, **kwargs):
        reads.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(drift, "read_latest_drift_jsonl", _counting_read)
    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        reads_at_startup = len(reads)
        statuses = {}
        for entity_type, features in segments.items():
            response = client.post(
                "/predict",
                json={"entity_type": entity_type, "feature_payload": _payload(features)},
            )
            assert response.status_code == 200, response.text
            statuses[entity_type] = response.json()["drift_status"]
        batch = client.post(
            "/predict/batch",
            json={
                "feature_payloads": [
                    {**_payload(segments[t]), "entity_type": t} for t in sorted(segments)
                ]
            },
        ).json()["predictions"]

    assert statuses == {"campaign": "critical", "user": "normal", "product": "not_evaluated"}
    assert {p["entity_type"]: p["drift_status"] for p in batch} == statuses
    assert reads_at_startup >= 1
    assert len(reads) == reads_at_startup  # never read per request


def test_predict_drift_status_is_not_evaluated_without_a_run(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    features = _segments(artifact_dir / "metadata.json")["user"]

    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        body = client.post(
            "/predict", json={"entity_type": "user", "feature_payload": _payload(features)}
        ).json()

    assert body["drift_status"] == "not_evaluated"


@pytest.mark.postgres
def test_api_reads_drift_status_from_the_database_and_refreshes_with_the_poller(
    initialized_database,  # noqa: F811 - pytest fixture
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection import drift
    from anomaly_detection.batch_inference import run_batch_inference
    from anomaly_detection.drift_job import run_drift_job
    from anomaly_detection.registry import promote_model_version, register_model_from_artifacts
    from api.main import PointerPoller

    register_model_from_artifacts(metadata_path=segmented_metadata_paths["v202"])
    promote_model_version(model_name="isolation_forest", model_version="v202")
    artifact_dir = segmented_metadata_paths["v202"].parent
    segments = _segments(artifact_dir / "metadata.json")
    reads: list[int] = []
    real_read = drift.read_latest_drift_database

    def _counting_read(*args, **kwargs):
        reads.append(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(drift, "read_latest_drift_database", _counting_read)
    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:

        def _status(entity_type: str) -> str:
            return client.post(
                "/predict",
                json={
                    "entity_type": entity_type,
                    "feature_payload": _payload(segments[entity_type]),
                },
            ).json()["drift_status"]

        assert _status(SHIFTED_TYPE) == "not_evaluated"

        run_batch_inference(_current_frame(artifact_dir), output_path=tmp_path / "batch.jsonl")
        run_drift_job(source="batch-predictions", hours=24, min_rows=30)
        reads_before_poll = len(reads)
        assert _status(SHIFTED_TYPE) == "not_evaluated"  # cached until the next poll
        assert len(reads) == reads_before_poll

        PointerPoller(state, interval_seconds=30).poll_once()

        assert _status(SHIFTED_TYPE) == "critical"
        assert _status("user") == "normal"
        assert len(reads) == reads_before_poll + 1


# --- metrics --------------------------------------------------------------------------


def _sample(name: str, labels: dict[str, str]) -> float | None:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels)


def test_rollback_counter_has_the_labels_it_is_called_with() -> None:
    from anomaly_detection.metrics import record_model_rollback

    labels = {
        "model_name": "isolation_forest",
        "from_model_version": "v311",
        "to_model_version": "v310",
        "triggered_by": "pytest",
        "status": "success",
    }
    before = _sample("model_rollback_total", labels) or 0.0

    record_model_rollback(
        model_name="isolation_forest",
        from_model_version="v311",
        to_model_version="v310",
        triggered_by="pytest",
    )

    assert _sample("model_rollback_total", labels) == before + 1


def test_swap_clears_the_stale_active_model_version_series() -> None:
    from api.main import AppState, swap_inference_service
    from tests.test_api import _FakeService

    holder = AppState()
    swap_inference_service(holder, _FakeService("v321"), pointer_version=1, pointer_source="file")
    swap_inference_service(holder, _FakeService("v322"), pointer_version=2, pointer_source="file")

    labels = {"model_name": "isolation_forest"}
    assert _sample("active_model_version", {**labels, "model_version": "v321"}) is None
    assert _sample("active_model_version", {**labels, "model_version": "v322"}) == 1.0


def test_drift_gauges_follow_the_cached_latest_run(
    tmp_path: Path,
    segmented_metadata_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anomaly_detection.drift_job import run_drift_job

    artifact_dir = _serve_segmented(tmp_path, segmented_metadata_paths)
    input_path = tmp_path / "current.parquet"
    _current_frame(artifact_dir).to_parquet(input_path)
    result = run_drift_job(source="input", input_path=input_path, min_rows=30)
    feature = _segments(artifact_dir / "metadata.json")[SHIFTED_TYPE][0]
    labels = {"model_version": "v202", "segment": SHIFTED_TYPE, "feature_name": feature}

    monkeypatch.setattr(state, "inference_service", None)
    with TestClient(app) as client:
        text = client.get("/metrics").text

    assert _sample("feature_drift_status", labels) == 2.0
    assert _sample("feature_mean_delta", labels) == pytest.approx(5.0)
    assert _sample("feature_variance_delta", labels) is not None
    user_labels = {**labels, "segment": "user", "feature_name": _segments(
        artifact_dir / "metadata.json")["user"][0]}
    assert _sample("feature_drift_status", user_labels) == 0.0
    critical = result.segments[SHIFTED_TYPE].critical_count
    assert _sample(
        "drift_detected_total",
        {"model_version": "v202", "segment": SHIFTED_TYPE, "feature_name": feature,
         "drift_status": "critical"},
    ) >= 1.0 and critical > 0
    assert "feature_drift_status" in text


# --- monitoring/ references only emitted metrics ---------------------------------------


def _emitted_metric_names() -> set[str]:
    from prometheus_client import REGISTRY

    import api.main  # noqa: F401 - registers every metric the API process emits

    names: set[str] = set()
    for metric in REGISTRY.collect():
        names.add(metric.name)
        if metric.type == "counter":
            names.update({f"{metric.name}_total", f"{metric.name}_created"})
        if metric.type == "histogram":
            names.update(f"{metric.name}{suffix}" for suffix in ("_bucket", "_sum", "_count"))
    return names


PROMQL_WORDS = {
    "sum", "max", "min", "avg", "count", "rate", "irate", "increase", "round", "by",
    "without", "on", "ignoring", "clamp_min", "clamp_max", "histogram_quantile", "absent",
    "and", "or", "unless", "le", "bool", "offset", "time", "vector", "topk", "bottomk",
}


def _queried_metric_names(expr: str) -> set[str]:
    expr = re.sub(r"\{[^}]*\}", "", expr)  # label matchers
    expr = re.sub(r"\[[^\]]*\]", "", expr)  # range selectors
    expr = re.sub(r"\b(by|without|on|ignoring)\s*\([^)]*\)", "", expr)  # grouping labels
    expr = re.sub(r'"[^"]*"', "", expr)
    return {
        word
        for word in re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr)
        if word not in PROMQL_WORDS and not re.fullmatch(r"\d.*", word)
    }


def _monitoring_expressions() -> list[tuple[str, str]]:
    expressions: list[tuple[str, str]] = []
    for path in sorted((MONITORING / "grafana" / "dashboards").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        expressions += [(path.name, expr) for expr in re.findall(r'"expr":\s*"([^"]*)"', text)]
    for path in sorted(MONITORING.glob("*rules*.yml")):
        rules = yaml.safe_load(path.read_text(encoding="utf-8"))
        for group in rules["groups"]:
            expressions += [(path.name, rule["expr"]) for rule in group["rules"]]
    return expressions


def test_every_metric_in_dashboards_and_alert_rules_is_emitted() -> None:
    emitted = _emitted_metric_names()
    expressions = _monitoring_expressions()

    assert any(name.endswith("rules.yml") for name, _ in expressions)
    missing = {
        (source, name)
        for source, expr in expressions
        for name in _queried_metric_names(expr)
        if name not in emitted
    }
    assert not missing


def test_prometheus_scrapes_the_api_over_the_compose_network_and_loads_rules() -> None:
    config = yaml.safe_load((MONITORING / "prometheus.yml").read_text(encoding="utf-8"))
    targets = [
        target
        for job in config["scrape_configs"]
        for static in job["static_configs"]
        for target in static["targets"]
    ]

    assert targets == ["anomaly-api:8004"]
    assert config["rule_files"] == ["/etc/prometheus/alert_rules.yml"]


def test_grafana_provisioning_matches_the_dashboards_datasource() -> None:
    provisioning = MONITORING / "grafana" / "provisioning"
    datasources = yaml.safe_load(
        (provisioning / "datasources" / "prometheus.yml").read_text(encoding="utf-8")
    )["datasources"]
    providers = yaml.safe_load(
        (provisioning / "dashboards" / "project4.yml").read_text(encoding="utf-8")
    )["providers"]

    assert [(d["name"], d["url"]) for d in datasources] == [("Prometheus", "http://prometheus:9090")]
    assert providers[0]["options"]["path"] == "/var/lib/grafana/dashboards"
    for path in (MONITORING / "grafana" / "dashboards").glob("*.json"):
        dashboard = json.loads(path.read_text(encoding="utf-8"))
        variable = next(v for v in dashboard["templating"]["list"] if v["name"] == "DS_PROMETHEUS")
        assert variable["current"]["value"] == datasources[0]["name"]


def test_compose_monitoring_services_are_pinned_local_and_take_the_password_from_env() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    prometheus, grafana = services["prometheus"], services["grafana"]

    assert prometheus["image"] == "prom/prometheus:v3.5.4"
    assert grafana["image"] == "grafana/grafana-oss:13.0.2"
    assert grafana["ports"] == ["127.0.0.1:3004:3000"]
    assert prometheus["ports"] == ["127.0.0.1:9094:9090"]
    for service in (prometheus, grafana):
        assert "restart" not in service
    password = grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
    assert password == "${PROJECT4_GRAFANA_ADMIN_PASSWORD:-}"
    compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "admin/admin" not in compose_text
    assert "PROJECT4_GRAFANA_ADMIN_PASSWORD=" in (REPO_ROOT / ".env.example").read_text()
