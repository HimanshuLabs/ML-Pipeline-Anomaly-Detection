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
        ("anomaly_detection.tracking", ["backfill", "--model-version", "v900"]),
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
    """Every series name the metrics defined in anomaly_detection/metrics.py expose."""
    from prometheus_client.metrics import MetricWrapperBase

    from anomaly_detection import metrics

    names: set[str] = set()
    for value in vars(metrics).values():
        if not isinstance(value, MetricWrapperBase):
            continue
        for metric in value.describe():
            names.add(metric.name)
            if metric.type == "counter":
                names.update({f"{metric.name}_total", f"{metric.name}_created"})
            if metric.type == "histogram":
                names.update(f"{metric.name}{suffix}" for suffix in ("_bucket", "_sum", "_count"))
    return names


# Series Prometheus itself creates (alert state), allowed in dashboards.
PROMETHEUS_SERIES = {"ALERTS"}

PROMQL_WORDS = {
    "sum", "max", "min", "avg", "count", "rate", "irate", "increase", "round", "by",
    "without", "on", "ignoring", "clamp_min", "clamp_max", "histogram_quantile", "absent",
    "and", "or", "unless", "le", "bool", "offset", "time", "vector", "topk", "bottomk",
    "label_replace",
}

ACTIVE_MODEL_EXPR = (
    '(active_model_version == 1) or on() '
    'label_replace(vector(0), "model_version", "no active model", "", "")'
)


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


DASHBOARDS = MONITORING / "grafana" / "dashboards"


def _dashboards() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(DASHBOARDS.glob("*.json"))
    }


def _panels() -> list[tuple[str, dict]]:
    return [(name, panel) for name, board in _dashboards().items() for panel in board["panels"]]


def _exprs(panel: dict) -> list[str]:
    return [target["expr"] for target in panel.get("targets", []) if "expr" in target]


def _alert_rules() -> dict[str, dict]:
    rules = yaml.safe_load((MONITORING / "alert_rules.yml").read_text(encoding="utf-8"))
    return {rule["alert"]: rule for group in rules["groups"] for rule in group["rules"]}


def _monitoring_expressions() -> list[tuple[str, str]]:
    expressions = [(name, expr) for name, panel in _panels() for expr in _exprs(panel)]
    expressions += [("alert_rules.yml", rule["expr"]) for rule in _alert_rules().values()]
    return expressions


def test_monitoring_expressions_are_parsed_whole() -> None:
    expressions = _monitoring_expressions()

    assert any(name.endswith(".json") for name, _ in expressions)
    assert any(name.endswith("rules.yml") for name, _ in expressions)
    for source, expr in expressions:
        assert expr.count("(") == expr.count(")"), (source, expr)
        assert expr.count("{") == expr.count("}"), (source, expr)


def test_every_metric_in_dashboards_and_alert_rules_is_defined_in_metrics_py() -> None:
    emitted = _emitted_metric_names() | PROMETHEUS_SERIES
    missing = {
        (source, name)
        for source, expr in _monitoring_expressions()
        for name in _queried_metric_names(expr)
        if name not in emitted
    }
    assert not missing


# --- dashboard guards ------------------------------------------------------------------

# Taken from `git show HEAD:monitoring/grafana/dashboards/<file>` before stage 14a; links
# and bookmarks depend on these.
HEAD_DASHBOARD_IDENTITY = {
    "anomaly_detection_dashboard.json": (
        "project4-anomaly-monitoring",
        "ML-Pipeline-Anomaly-Detection — Monitoring",
        set(range(1, 17)),
    ),
    "anomaly_detection_executive_dashboard.json": (
        "project4-executive-ml-health",
        "ML-Pipeline-Anomaly-Detection — Executive ML Health Dashboard",
        set(range(1, 15)),
    ),
}
SPARSE_COUNTERS = ("drift_detected_total", "model_rollback_total", "prediction_errors_total")
COUNT_METRICS = (
    "prediction_requests_total",
    "anomalies_detected_total",
    "model_rollback_total",
    "drift_detected_total",
)


def _steps(panel: dict) -> list[dict]:
    return panel["fieldConfig"]["defaults"].get("thresholds", {}).get("steps", [])


def _stats() -> list[tuple[str, dict]]:
    return [(name, panel) for name, panel in _panels() if panel["type"] == "stat"]


def test_dashboard_uids_titles_and_panel_ids_are_unchanged() -> None:
    import shutil
    import subprocess

    boards = _dashboards()
    assert set(boards) == set(HEAD_DASHBOARD_IDENTITY)
    for name, (uid, title, panel_ids) in HEAD_DASHBOARD_IDENTITY.items():
        ids = [panel["id"] for panel in boards[name]["panels"]]
        assert (boards[name]["uid"], boards[name]["title"]) == (uid, title)
        assert panel_ids <= set(ids) and len(ids) == len(set(ids)), name

        if shutil.which("git"):  # also compare with HEAD where git exists (not in the image)
            head = json.loads(subprocess.run(
                ["git", "show", f"HEAD:monitoring/grafana/dashboards/{name}"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout)
            assert (head["uid"], head["title"]) == (uid, title)
            assert {panel["id"] for panel in head["panels"]} <= set(ids)


def test_dashboards_have_no_outdated_checkpoint_wording() -> None:
    for path in DASHBOARDS.glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert "checkpoint" not in text, path.name
        assert "becomes active after rollback" not in text, path.name
        assert "once rollback controls" not in text, path.name


def test_sparse_counter_and_alert_stats_show_zero_instead_of_no_data() -> None:
    checked = 0
    for name, panel in _stats():
        for expr in _exprs(panel):
            if any(metric in expr for metric in (*SPARSE_COUNTERS, "ALERTS")):
                checked += 1
                assert expr.replace(" ", "").endswith("orvector(0)"), (name, panel["title"])
    assert checked >= 6


def test_count_stats_use_one_neutral_colour_and_error_counts_go_red_at_one() -> None:
    counts, errors = 0, 0
    for name, panel in _stats():
        exprs = " ".join(_exprs(panel))
        if "/" in exprs or "histogram_quantile" in exprs:
            continue
        if "prediction_errors_total" in exprs or "ALERTS" in exprs:
            errors += 1
            colours = [(step["color"], step["value"]) for step in _steps(panel)]
            assert colours == [("green", None), ("red", 1)], (name, panel["title"])
        elif any(metric in exprs for metric in COUNT_METRICS):
            counts += 1
            assert len(_steps(panel)) == 1, (name, panel["title"])
            assert _steps(panel)[0]["color"] not in {"red", "yellow", "green"}
            assert panel["fieldConfig"]["defaults"]["color"]["mode"] == "fixed"
    assert counts >= 8 and errors >= 3


def _main_panel(panel_id: int) -> dict:
    board = _dashboards()["anomaly_detection_dashboard.json"]
    return next(panel for panel in board["panels"] if panel["id"] == panel_id)


def test_delta_tables_are_instant_tables_merged_with_drift_status() -> None:
    tables = [_main_panel(9)]
    assert tables[0]["title"] == "Feature drift detail"
    for panel in tables:
        assert panel["type"] == "table"
        assert all(t["instant"] is True and t["format"] == "table" for t in panel["targets"])
        exprs = " ".join(_exprs(panel))
        for metric in ("feature_mean_delta", "feature_variance_delta", "feature_drift_status"):
            assert metric in exprs
        kinds = [transformation["id"] for transformation in panel["transformations"]]
        assert kinds[0] == "merge" and "organize" in kinds and "sortBy" in kinds
        organize = next(t for t in panel["transformations"] if t["id"] == "organize")
        hidden = organize["options"]["excludeByName"]
        assert all(hidden.get(column) for column in ("Time", "__name__", "job", "instance"))
        renamed = organize["options"]["renameByName"]
        assert set(renamed.values()) >= {
            "segment", "feature", "status", "mean delta", "variance delta"
        }
        sort = [t for t in panel["transformations"] if t["id"] == "sortBy"][-1]
        assert sort["options"]["sort"][0] == {"field": "status", "desc": True}
        assert "own units" in panel["description"]


def _board_panel(board: str, panel_id: int) -> dict:
    return next(panel for panel in _dashboards()[board]["panels"] if panel["id"] == panel_id)


def _overrides(panel: dict) -> dict[str, dict]:
    return {
        override["matcher"]["options"]: {p["id"]: p["value"] for p in override["properties"]}
        for override in panel["fieldConfig"]["overrides"]
        if override["matcher"]["id"] == "byName"
    }


def test_count_stats_show_zero_after_an_api_restart_and_rollback_trend_shows_none() -> None:
    for board, panel_ids in (
        ("anomaly_detection_dashboard.json", (2, 3)),
        ("anomaly_detection_executive_dashboard.json", (3, 11)),
    ):
        for panel_id in panel_ids:
            (expr,) = _exprs(_board_panel(board, panel_id))
            assert expr.replace(" ", "").endswith("orvector(0)"), (board, panel_id)

    rollbacks = _main_panel(16)
    assert _exprs(rollbacks) == ["sum by (status) (increase(model_rollback_total[1h]))"]
    assert rollbacks["targets"][0]["legendFormat"] == "{{status}}"
    assert rollbacks["fieldConfig"]["defaults"]["noValue"] == "No rollbacks in range"


def test_latency_timeseries_always_show_the_200ms_sla_line() -> None:
    latency = [
        panel for _, panel in _panels()
        if panel["type"] == "timeseries" and "histogram_quantile" in " ".join(_exprs(panel))
    ]
    assert len(latency) == 3
    for panel in latency:
        custom = panel["fieldConfig"]["defaults"]["custom"]
        assert custom["axisSoftMin"] == 0, panel["title"]
        assert custom["axisSoftMax"] >= 200, panel["title"]
        assert custom["thresholdsStyle"]["mode"] == "dashed", panel["title"]
        assert [(s["color"], s["value"]) for s in _steps(panel)] == [
            ("green", None), ("yellow", 150), ("red", 200)
        ]


def test_feature_drift_detail_has_4_decimal_deltas_full_sort_and_fixed_narrow_columns() -> None:
    panel = _main_panel(9)
    overrides = _overrides(panel)

    assert overrides["mean delta"]["decimals"] == 4
    assert overrides["variance delta"]["decimals"] == 4
    # Grafana sorts by one field per sortBy step and keeps ties in order, so the last
    # step is the primary key: status desc, then segment, then feature.
    sorts = [t["options"]["sort"][0] for t in panel["transformations"] if t["id"] == "sortBy"]
    assert sorts == [
        {"field": "feature", "desc": False},
        {"field": "segment", "desc": False},
        {"field": "status", "desc": True},
    ]
    widths = {name: props.get("custom.width") for name, props in overrides.items()}
    assert widths.get("feature") is None  # takes the remaining width
    assert widths["segment"] and widths["status"]
    assert max(widths["segment"], widths["status"]) < widths["mean delta"]
    assert panel["gridPos"]["w"] == 24
    assert _main_panel(10)["gridPos"]["y"] >= panel["gridPos"]["y"] + panel["gridPos"]["h"]


def test_no_two_panels_overlap_and_all_fit_the_24_column_grid() -> None:
    def overlap(a: dict, b: dict) -> bool:
        return (
            a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
            and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]
        )

    assert overlap({"x": 0, "y": 0, "w": 12, "h": 4}, {"x": 6, "y": 3, "w": 6, "h": 4})
    for name, board in _dashboards().items():
        grids = [(panel["id"], panel["gridPos"]) for panel in board["panels"]]
        for panel_id, grid in grids:
            assert grid["x"] >= 0 and grid["x"] + grid["w"] <= 24, (name, panel_id)
        clashes = [
            (first_id, second_id)
            for index, (first_id, first) in enumerate(grids)
            for second_id, second in grids[index + 1:]
            if overlap(first, second)
        ]
        assert not clashes, name


def test_stats_use_background_colour_except_the_active_model_marker() -> None:
    for name, panel in _stats():
        expected = "value" if "active_model_version" in " ".join(_exprs(panel)) else "background"
        assert panel["options"]["colorMode"] == expected, (name, panel["title"])


def test_executive_text_panels_are_tall_enough_for_their_markdown() -> None:
    board = "anomaly_detection_executive_dashboard.json"
    for panel_id in (1, 14):
        panel = _board_panel(board, panel_id)
        lines = [line for line in panel["options"]["content"].splitlines() if line.strip()]
        # One grid unit is ~30 px; leave 2 units for the title bar and padding.
        assert panel["gridPos"]["h"] >= len(lines) + 2, panel["title"]


def test_drifting_features_by_segment_is_an_instant_bargauge() -> None:
    panel = _main_panel(10)

    assert panel["title"] == "Drifting features by segment"
    assert panel["type"] == "bargauge"
    assert _exprs(panel) == ["sum by (segment) (feature_drift_status >= bool 1)"]
    assert panel["targets"][0]["legendFormat"] == "{{segment}}"
    assert all(t.get("instant") is True for t in panel["targets"])
    assert "transformations" not in panel
    defaults = panel["fieldConfig"]["defaults"]
    assert defaults["min"] == 0
    assert [(s["color"], s["value"]) for s in _steps(panel)] == [
        ("green", None), ("yellow", 1), ("red", 5)
    ]
    assert panel["description"] == (
        "Count of features at warning or critical in the latest drift run, per segment. "
        "0 = all normal."
    )


def test_anomaly_rate_panels_show_idle_without_scored_predictions() -> None:
    rates = [panel for _, panel in _panels() if panel["title"].lower().startswith("anomaly rate")]
    stats = [panel for panel in rates if panel["type"] == "stat"]
    assert len(rates) == 4 and len(stats) == 2
    for panel in rates:
        assert panel["fieldConfig"]["defaults"]["noValue"] == "idle", panel["title"]
    for panel in stats:
        (expr,) = _exprs(panel)
        window = re.search(r"predictions_scored_total\[(\w+)\]", expr).group(1)
        guard = f"and on() (sum(increase(predictions_scored_total[{window}])) > 0)"
        assert expr.endswith(guard), panel["title"]
        # Traffic without any anomaly yet is 0 %, not idle.
        assert f"(sum(increase(anomalies_detected_total[{window}])) or vector(0))" in expr
    for panel in (panel for panel in rates if panel["type"] == "timeseries"):
        (expr,) = _exprs(panel)
        assert expr == (
            "((sum(rate(anomalies_detected_total[5m])) or vector(0)) / "
            "clamp_min(sum(rate(predictions_scored_total[5m])), 0.001)) * 100"
            " and on() (sum(rate(predictions_scored_total[5m])) > 0)"
        ), panel["title"]


def test_active_model_stats_show_the_version_label_without_sparkline_or_thresholds() -> None:
    markers = [
        panel for _, panel in _stats() if "active_model_version" in " ".join(_exprs(panel))
    ]
    assert len(markers) == 2
    for panel in markers:
        # A real series when no model is loaded: Grafana 13 renders a no-series
        # textMode "name" stat blank instead of showing noValue.
        assert _exprs(panel) == [ACTIVE_MODEL_EXPR]
        assert panel["targets"][0]["legendFormat"] == "{{model_version}}"
        mapping = panel["fieldConfig"]["defaults"]["mappings"][0]
        assert mapping["type"] == "value"
        assert mapping["options"]["0"]["color"] == "red"
        assert mapping["options"]["1"]["color"] == "text"
        assert panel["fieldConfig"]["defaults"]["noValue"] == "no active model"
        assert panel["options"]["colorMode"] == "value"
        assert panel["options"]["textMode"] == "name"
        assert panel["options"]["graphMode"] == "none"
        assert "thresholds" not in panel["fieldConfig"]["defaults"]
        assert all(t.get("instant") is True for t in panel["targets"])


def test_anomaly_rate_panels_divide_by_scored_predictions_with_contamination_thresholds() -> None:
    rates = [panel for _, panel in _panels() if panel["title"].lower().startswith("anomaly rate")]
    assert len(rates) == 4
    for panel in rates:
        (expr,) = _exprs(panel)
        assert "anomalies_detected_total" in expr and "predictions_scored_total" in expr
        assert "prediction_requests_total" not in expr and "endpoint" not in expr
        assert [(s["color"], s["value"]) for s in _steps(panel)] == [
            ("green", None), ("yellow", 7.5), ("red", 10)
        ]
        assert "contamination" in panel["description"]
        assert "configs/model_config.yaml" in panel["description"]


def test_latency_panels_use_single_predictions_and_p95_stats_show_idle() -> None:
    latency = [
        panel for _, panel in _panels() if "histogram_quantile" in " ".join(_exprs(panel))
    ]
    for panel in latency:
        for expr in _exprs(panel):
            assert 'endpoint="/predict' in expr, panel["title"]
        if panel["type"] == "stat":
            assert all(t.get("instant") is True for t in panel["targets"])
            assert panel["fieldConfig"]["defaults"]["noValue"] == "idle"
            for expr in _exprs(panel):
                assert "and on()" in expr and "> 0" in expr
        else:
            custom = panel["fieldConfig"]["defaults"]["custom"]
            assert custom["thresholdsStyle"]["mode"].startswith("dashed")
            assert (200 in [s["value"] for s in _steps(panel)])
    batch = [
        expr for _, panel in _panels() for expr in _exprs(panel)
        if 'endpoint="/predict/batch"' in expr
    ]
    assert len(batch) == 1 and "0.95" in batch[0]
    assert any(panel["type"] == "stat" for panel in latency)


def test_drift_panels_read_the_drift_status_gauge() -> None:
    by_title = {panel["title"]: panel for _, panel in _panels()}
    trend = by_title["Drift signal trend"]
    by_feature = by_title["Drift events by feature"]

    assert _exprs(trend) == ["max by (segment) (feature_drift_status)"]
    assert _exprs(by_feature) == ["max by (segment, feature_name) (feature_drift_status) > 0"]
    assert by_feature["fieldConfig"]["defaults"]["noValue"] == "No drifting features"
    for panel in (trend, by_feature):
        mapping = panel["fieldConfig"]["defaults"]["mappings"][0]["options"]
        assert {key: value["text"] for key, value in mapping.items()} == {
            "0": "normal", "1": "warning", "2": "critical"
        }


def test_latency_alert_covers_single_predictions_only() -> None:
    rule = _alert_rules()["Project4LatencyBudgetBreach"]

    assert 'endpoint="/predict"' in rule["expr"]
    assert "> 200" in rule["expr"]
    assert "single prediction" in rule["annotations"]["description"]


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


def test_compose_sets_the_grafana_root_url_to_port_3004() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    root_url = compose["services"]["grafana"]["environment"]["GF_SERVER_ROOT_URL"]

    assert root_url == "${PROJECT4_GRAFANA_ROOT_URL:-http://127.0.0.1:3004/}"
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "PROJECT4_GRAFANA_ROOT_URL=http://127.0.0.1:3004/" in env_example.splitlines()
