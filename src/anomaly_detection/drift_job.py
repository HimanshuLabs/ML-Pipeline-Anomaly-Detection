"""Drift job: compare current feature data with the ACTIVE model's training baseline.

    python -m anomaly_detection.drift_job --source batch-predictions [--hours 24]
    python -m anomaly_detection.drift_job --input current.parquet

The active model comes from the pointer (Postgres when PROJECT4_POSTGRES_* is
configured, else configs/active_model.yaml); its baseline is the
baseline_stats.json in its artifact directory. A segmented model is evaluated
per segment (entity type) against that segment's baseline and features; a
single model on its whole feature set.

Current data: the feature payloads of the last N hours of ml.batch_predictions
for the active model version, or a parquet/csv file (with an entity_type column
for a segmented model). A segment with fewer rows than the minimum is refused
instead of producing noise.

Drift events and critical-drift alerts are written with persist_drift_events /
persist_alert_events (JSONL, plus Postgres when configured). The exit code is 0
whether or not drift was found, and 1 when the job could not run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection import drift
from anomaly_detection.alerts import DEFAULT_ALERT_EVENTS_PATH, persist_alert_events
from anomaly_detection.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)

SOURCE_BATCH_PREDICTIONS = "batch-predictions"
SOURCE_INPUT = "input"
DEFAULT_HOURS = 24.0
DEFAULT_MIN_ROWS_PER_SEGMENT = 30
# The batch-predictions window is read this many rows at a time (server-side cursor).
PAYLOAD_CHUNK_ROWS = 10_000
# Whole numbers beyond this are not exact in float64; such a window is read in one piece.
_FLOAT64_EXACT_INT = 2**53


class DriftJobError(RuntimeError):
    """The drift job could not run; nothing was written."""


@dataclass(frozen=True)
class SegmentOutcome:
    """Drift of one segment (None: a single model's whole feature set)."""

    entity_type: str | None
    row_count: int
    feature_count: int
    status: str
    normal_count: int
    warning_count: int
    critical_count: int
    result: drift.DriftEvaluationResult


@dataclass(frozen=True)
class DriftJobResult:
    model_name: str
    model_version: str
    source: str
    segments: dict[str | None, SegmentOutcome]
    skipped: dict[str, str] = field(default_factory=dict)
    drift_event_count: int = 0
    alert_count: int = 0
    persisted: dict[str, Any] = field(default_factory=dict)
    # "evaluated", or drift.NOT_EVALUATED when there were no current rows at all.
    status: str = "evaluated"


def _min_rows(threshold_config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        value = override
    else:
        value = (threshold_config.get("drift") or {}).get(
            "min_rows_per_segment", DEFAULT_MIN_ROWS_PER_SEGMENT
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DriftJobError(f"min rows per segment must be a positive integer; got {value!r}")
    return value


class _OnePieceNeeded(Exception):
    """A payload value the float64 columns cannot carry exactly (bool, text, huge int)."""


def _as_float(value: Any) -> float:
    if value is None:
        return np.nan
    if type(value) is float:
        return value
    if type(value) is int and -_FLOAT64_EXACT_INT <= value <= _FLOAT64_EXACT_INT:
        return float(value)
    raise _OnePieceNeeded(f"a {type(value).__name__} value")


class _SegmentColumns:
    """One segment's payload values: one float64 array per payload key, built in chunks."""

    def __init__(self) -> None:
        self.rows = 0
        self.chunks: dict[str, list[np.ndarray]] = {}

    def add(self, payloads: list[dict[str, Any]], int_rows: dict[str, int]) -> None:
        keys = dict.fromkeys(key for payload in payloads for key in payload)
        for key in keys:
            values = [payload.get(key) for payload in payloads]
            int_rows[key] = int_rows.get(key, 0) + sum(type(v) is int for v in values)
            if key not in self.chunks:
                # A key first seen mid-way is missing (NaN) in the earlier rows.
                self.chunks[key] = [np.full(self.rows, np.nan)]
            self.chunks[key].append(
                np.fromiter(map(_as_float, values), np.float64, len(values))
            )
        for key, chunks in self.chunks.items():
            if key not in keys:
                chunks.append(np.full(len(payloads), np.nan))
        self.rows += len(payloads)

    def frame(self, int_keys: frozenset[str]) -> pd.DataFrame:
        """The columns as pd.DataFrame(all payloads) types them: int64 for a key that is
        a whole number in every row of the window, else float64. Frees the chunks."""

        columns = {}
        for key in list(self.chunks):
            values = np.concatenate(self.chunks.pop(key))
            columns[key] = values.astype(np.int64) if key in int_keys else values
        return pd.DataFrame(columns, index=pd.RangeIndex(self.rows))


@dataclass(frozen=True)
class CurrentPayloads:
    """The batch-predictions window read in chunks, by entity type (key None: one group
    for a single model)."""

    row_count: int
    null_entity_type_rows: int
    groups: dict[str | None, _SegmentColumns]
    payload_keys: frozenset[str]
    int_keys: frozenset[str]

    @property
    def empty(self) -> bool:
        return self.row_count == 0

    def __len__(self) -> int:
        return self.row_count


def payload_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Every payload (plus entity_type) in one frame: the one-piece read."""

    return pd.DataFrame(
        [{**(row["feature_payload"] or {}), "entity_type": row["entity_type"]} for row in rows]
    )


def _read_payloads_in_chunks(
    *, model_name: str, model_version: str, since: datetime, by_entity_type: bool
) -> CurrentPayloads:
    from anomaly_detection import database

    groups: dict[str | None, _SegmentColumns] = {}
    int_rows: dict[str, int] = {}
    row_count = null_rows = 0
    for rows in database.iter_batch_feature_payloads(
        model_name=model_name,
        model_version=model_version,
        since=since,
        chunk_rows=PAYLOAD_CHUNK_ROWS,
    ):
        by_group: dict[str | None, list[dict[str, Any]]] = {}
        for row in rows:
            group = row["entity_type"] if by_entity_type else None
            if by_entity_type and group is None:
                null_rows += 1
                continue
            by_group.setdefault(group, []).append(row["feature_payload"] or {})
        for group, payloads in by_group.items():
            groups.setdefault(group, _SegmentColumns()).add(payloads, int_rows)
        row_count += len(rows)
    int_keys = frozenset(key for key, count in int_rows.items() if count == row_count)
    return CurrentPayloads(row_count, null_rows, groups, frozenset(int_rows), int_keys)


def load_batch_prediction_features(
    *,
    model_name: str,
    model_version: str,
    since: datetime,
    by_entity_type: bool = True,
) -> pd.DataFrame | CurrentPayloads:
    """Feature payloads of the model's batch predictions since a time.

    Read in chunks into per-segment columns (CurrentPayloads). A window holding a value
    those columns cannot carry exactly is read in one piece into one frame, as before.
    """

    from anomaly_detection import database

    if not database.is_configured():
        raise DriftJobError(
            "--source batch-predictions needs the Project 4 database; "
            "set PROJECT4_POSTGRES_* (or use --input)"
        )
    try:
        return _read_payloads_in_chunks(
            model_name=model_name,
            model_version=model_version,
            since=since,
            by_entity_type=by_entity_type,
        )
    except _OnePieceNeeded as exc:
        LOGGER.warning(
            "Batch payloads of %s %s hold %s; reading the window in one piece",
            model_name,
            model_version,
            exc,
        )
    return payload_frame(
        database.read_batch_feature_payloads(
            model_name=model_name, model_version=model_version, since=since
        )
    )


def last_batch_prediction(*, model_name: str) -> tuple[datetime, str] | None:
    """(when, version) of the model name's last batch prediction, any version; None if never."""

    from anomaly_detection import database

    return database.read_last_batch_prediction(model_name=model_name)


def _log_nothing_to_evaluate(
    *, model_name: str, model_version: str, source: str, hours: float, since: datetime
) -> None:
    """No rows for the active version in the window: stalled batch scoring, or just quiet.

    Never batch-scored: INFO. Last batch run before the window: WARNING (stalled?).
    Last batch run inside the window under another version (e.g. right after a
    promote): INFO naming that version.
    """

    last = None
    if source == SOURCE_BATCH_PREDICTIONS:
        last = last_batch_prediction(model_name=model_name)
    if last is not None:
        last_at, last_version = last
        if last_at < since:
            LOGGER.warning(
                "No current rows for %s %s from %s; batch scoring stalled? "
                "last batch prediction at %s (model %s), none in the last %gh; drift %s",
                model_name,
                model_version,
                source,
                last_at.astimezone(UTC).isoformat(),
                last_version,
                hours,
                drift.NOT_EVALUATED,
            )
            return
        LOGGER.info(
            "No current rows for %s %s from %s: no rows for %s yet; last batch run used %s "
            "at %s; drift %s",
            model_name,
            model_version,
            source,
            model_version,
            last_version,
            last_at.astimezone(UTC).isoformat(),
            drift.NOT_EVALUATED,
        )
        return
    LOGGER.info(
        "No current rows for %s %s from %s; drift %s",
        model_name,
        model_version,
        source,
        drift.NOT_EVALUATED,
    )


def load_input_features(input_path: Path) -> pd.DataFrame:
    from anomaly_detection.batch_inference import BatchInferenceError, load_feature_batch

    try:
        return load_feature_batch(Path(input_path))
    except BatchInferenceError as exc:
        raise DriftJobError(str(exc)) from exc


def _feature_frame(frame: pd.DataFrame, feature_names: list[str], label: str) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise DriftJobError(f"current data for {label} is missing feature(s): {missing}")
    return frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")


def _evaluate(
    current: pd.DataFrame,
    *,
    baseline: dict[str, Any],
    feature_baselines: dict[str, Any],
    entity_type: str | None,
    threshold_config: dict[str, Any],
    window: tuple[datetime, datetime],
    detected_at: str,
) -> SegmentOutcome:
    try:
        result = drift.evaluate_drift(
            current_features=current,
            baseline_stats={
                "model_name": baseline["model_name"],
                "model_version": baseline["model_version"],
                "feature_schema_version": baseline["feature_schema_version"],
                "feature_baselines": feature_baselines,
            },
            threshold_config=threshold_config,
            observation_window_start=window[0],
            observation_window_end=window[1],
        )
    except drift.DriftMonitoringError as exc:
        raise DriftJobError(f"{entity_type or 'model'}: {exc}") from exc

    # One timestamp per segment run: the API's latest-run lookup groups by it.
    events = [
        replace(event, entity_type=entity_type, detected_at=detected_at)
        for event in result.drift_events
    ]
    result = replace(result, drift_events=events, entity_type=entity_type)
    return SegmentOutcome(
        entity_type=entity_type,
        row_count=len(current),
        feature_count=result.evaluated_feature_count,
        status=result.overall_drift_status,
        normal_count=result.normal_feature_count,
        warning_count=result.warning_feature_count,
        critical_count=result.critical_feature_count,
        result=result,
    )


def _checked_segment_counts(
    counts: dict[str, int], segments: dict[str, Any], min_rows: int
) -> dict[str, str]:
    """Refuse unknown or too-small segments; returns the segments with no rows."""

    unknown = sorted(set(counts) - set(segments))
    if unknown:
        raise DriftJobError(
            f"current data has entity type(s) the model has no segment for: {unknown}; "
            f"segments: {sorted(segments)}"
        )
    too_few = [f"{t} has {int(n)} rows" for t, n in sorted(counts.items()) if n < min_rows]
    if too_few:
        raise DriftJobError(
            "too few rows to evaluate drift: " + ", ".join(too_few) + f" (minimum {min_rows})"
        )
    if not counts:
        raise DriftJobError("no current rows to evaluate")
    return {t: "no rows" for t in sorted(segments) if t not in counts}


def _group_frame(current: CurrentPayloads, group: str | None, names: list[str]) -> pd.DataFrame:
    """One group's columns. A feature only other groups carry is all-NaN here, as it is
    in the rows of this group in the one-piece frame."""

    frame = current.groups.pop(group).frame(current.int_keys)
    for name in names:
        if name in current.payload_keys and name not in frame.columns:
            frame[name] = np.nan
    return frame


def _evaluate_current_payloads(
    current: CurrentPayloads,
    *,
    baseline: dict[str, Any],
    threshold_config: dict[str, Any],
    min_rows: int,
    window: tuple[datetime, datetime],
    detected_at: str,
) -> tuple[dict[str | None, SegmentOutcome], dict[str, str]]:
    segments = baseline.get("segments")
    common = {
        "baseline": baseline,
        "threshold_config": threshold_config,
        "window": window,
        "detected_at": detected_at,
    }
    if not segments:
        if len(current) < min_rows:
            raise DriftJobError(
                f"too few rows to evaluate drift: {len(current)} (minimum {min_rows})"
            )
        names = list(baseline["feature_baselines"])
        outcome = _evaluate(
            _feature_frame(_group_frame(current, None, names), names, "the model"),
            feature_baselines=baseline["feature_baselines"],
            entity_type=None,
            **common,
        )
        return {None: outcome}, {}

    if current.null_entity_type_rows:
        raise DriftJobError("current data has rows with a null entity_type")
    counts = {str(t): group.rows for t, group in current.groups.items()}
    skipped = _checked_segment_counts(counts, segments, min_rows)
    outcomes: dict[str | None, SegmentOutcome] = {}
    for entity_type in sorted(counts):
        segment_baselines = segments[entity_type]["feature_baselines"]
        names = list(segment_baselines)
        frame = _group_frame(current, entity_type, names)
        outcomes[entity_type] = _evaluate(
            _feature_frame(frame, names, f"segment {entity_type}"),
            feature_baselines=segment_baselines,
            entity_type=entity_type,
            **common,
        )
    return outcomes, skipped


def evaluate_active_model_drift(
    frame: pd.DataFrame | CurrentPayloads,
    *,
    baseline: dict[str, Any],
    threshold_config: dict[str, Any],
    min_rows: int,
    window: tuple[datetime, datetime],
    detected_at: str,
) -> tuple[dict[str | None, SegmentOutcome], dict[str, str]]:
    """Evaluate current data against the baseline; per segment for a segmented model."""

    if isinstance(frame, CurrentPayloads):
        return _evaluate_current_payloads(
            frame,
            baseline=baseline,
            threshold_config=threshold_config,
            min_rows=min_rows,
            window=window,
            detected_at=detected_at,
        )

    segments = baseline.get("segments")
    common = {
        "baseline": baseline,
        "threshold_config": threshold_config,
        "window": window,
        "detected_at": detected_at,
    }

    if not segments:
        if len(frame) < min_rows:
            raise DriftJobError(
                f"too few rows to evaluate drift: {len(frame)} (minimum {min_rows})"
            )
        names = list(baseline["feature_baselines"])
        outcome = _evaluate(
            _feature_frame(frame, names, "the model"),
            feature_baselines=baseline["feature_baselines"],
            entity_type=None,
            **common,
        )
        return {None: outcome}, {}

    if "entity_type" not in frame.columns:
        raise DriftJobError(
            "the active model is segmented; current data needs an entity_type column"
        )
    if frame["entity_type"].isna().any():
        raise DriftJobError("current data has rows with a null entity_type")

    counts = frame["entity_type"].astype(str).value_counts().sort_index()
    skipped = _checked_segment_counts(
        {str(t): int(n) for t, n in counts.items()}, segments, min_rows
    )
    entity_types = frame["entity_type"].astype(str)
    outcomes: dict[str | None, SegmentOutcome] = {}
    for entity_type in counts.index:
        segment_baselines = segments[entity_type]["feature_baselines"]
        rows = frame.loc[entity_types == entity_type]
        outcomes[str(entity_type)] = _evaluate(
            _feature_frame(rows, list(segment_baselines), f"segment {entity_type}"),
            feature_baselines=segment_baselines,
            entity_type=str(entity_type),
            **common,
        )
    return outcomes, skipped


def run_drift_job(
    *,
    source: str,
    input_path: str | Path | None = None,
    hours: float = DEFAULT_HOURS,
    min_rows: int | None = None,
    threshold_config_path: Path = drift.DEFAULT_DRIFT_THRESHOLDS_PATH,
    drift_events_path: Path = drift.DEFAULT_DRIFT_EVENTS_PATH,
    alert_events_path: Path = DEFAULT_ALERT_EVENTS_PATH,
    now: datetime | None = None,
) -> DriftJobResult:
    """Evaluate the active model's drift and persist events and alerts."""

    threshold_config = drift.load_drift_thresholds(threshold_config_path)
    minimum = _min_rows(threshold_config, min_rows)
    try:
        baseline = drift.load_baseline_stats()
    except drift.DriftMonitoringError as exc:
        raise DriftJobError(f"active model baseline: {exc}") from exc
    model_name = str(baseline["model_name"])
    model_version = str(baseline["model_version"])
    end = now or datetime.now(UTC)

    if source == SOURCE_BATCH_PREDICTIONS:
        if hours <= 0:
            raise DriftJobError("--hours must be greater than zero")
        start = end - timedelta(hours=hours)
        frame = load_batch_prediction_features(
            model_name=model_name,
            model_version=model_version,
            since=start,
            by_entity_type=bool(baseline.get("segments")),
        )
    elif source == SOURCE_INPUT:
        if input_path is None:
            raise DriftJobError("--input needs a parquet or csv path")
        start = end
        frame = load_input_features(Path(input_path))
    else:
        raise DriftJobError(f"unknown source {source!r}")

    if frame.empty:
        # Nothing scored in the window (or an empty input): not an error, and
        # nothing is written, so the last real drift run stays the latest.
        segments = baseline.get("segments") or {}
        skipped = {t: "no rows" for t in sorted(segments)} if segments else {"all": "no rows"}
        _log_nothing_to_evaluate(
            model_name=model_name,
            model_version=model_version,
            source=source,
            hours=hours,
            since=start,
        )
        return DriftJobResult(
            model_name=model_name,
            model_version=model_version,
            source=source,
            segments={},
            skipped=skipped,
            status=drift.NOT_EVALUATED,
        )

    LOGGER.info(
        "Evaluating drift of %s %s on %s current rows from %s",
        model_name,
        model_version,
        len(frame),
        source,
    )
    outcomes, skipped = evaluate_active_model_drift(
        frame,
        baseline=baseline,
        threshold_config=threshold_config,
        min_rows=minimum,
        window=(start, end),
        detected_at=end.isoformat(),
    )

    events = [event for outcome in outcomes.values() for event in outcome.result.drift_events]
    alerts = [
        alert
        for outcome in outcomes.values()
        for alert in drift.create_alerts_for_drift_result(outcome.result)
    ]
    persisted: dict[str, Any] = {
        "drift_events": drift.persist_drift_events(events, drift_events_path)
    }
    if alerts:
        persisted["alerts"] = persist_alert_events(alerts, alert_events_path)

    for outcome in outcomes.values():
        LOGGER.info(
            "Drift %s %s segment=%s rows=%s status=%s (warning=%s critical=%s)",
            model_name,
            model_version,
            outcome.entity_type or "all",
            outcome.row_count,
            outcome.status,
            outcome.warning_count,
            outcome.critical_count,
        )

    return DriftJobResult(
        model_name=model_name,
        model_version=model_version,
        source=source,
        segments=outcomes,
        skipped=skipped,
        drift_event_count=len(events),
        alert_count=len(alerts),
        persisted=persisted,
    )


def print_summary(result: DriftJobResult) -> None:
    if result.status == drift.NOT_EVALUATED:
        print(
            f"OK: nothing to evaluate for model={result.model_name} {result.model_version}: "
            f"{drift.NOT_EVALUATED} (no current rows from {result.source}; nothing recorded)"
        )
        for entity_type, reason in result.skipped.items():
            print(f"{entity_type:<14}  skipped: {reason}")
        return
    print(f"OK: drift evaluated for model={result.model_name} {result.model_version}")
    print(f"source={result.source} drift_events={result.drift_event_count} "
          f"alerts={result.alert_count} "
          f"stores={','.join(sorted(result.persisted['drift_events']))}")
    print(
        f"{'segment':<14}{'rows':>7}{'feats':>7}{'normal':>8}"
        f"{'warning':>9}{'critical':>10}  status"
    )
    for outcome in result.segments.values():
        print(
            f"{outcome.entity_type or 'all':<14}{outcome.row_count:>7}{outcome.feature_count:>7}"
            f"{outcome.normal_count:>8}{outcome.warning_count:>9}{outcome.critical_count:>10}"
            f"  {outcome.status}"
        )
    for entity_type, reason in result.skipped.items():
        print(f"{entity_type:<14}  skipped: {reason}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate feature drift of the active model (per segment when segmented)."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        choices=[SOURCE_BATCH_PREDICTIONS],
        default=None,
        help="Read current features from ml.batch_predictions of the active model version.",
    )
    source.add_argument("--input", default=None, help="parquet or csv with current features.")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Minimum rows per segment (default: drift.min_rows_per_segment in the config).",
    )
    parser.add_argument("--threshold-config", default=str(drift.DEFAULT_DRIFT_THRESHOLDS_PATH))
    args = parser.parse_args(argv)
    if args.input is None and args.source is None:
        args.source = SOURCE_BATCH_PREDICTIONS
    return args


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    from anomaly_detection.registry import RegistryError

    try:
        result = run_drift_job(
            source=SOURCE_INPUT if args.input is not None else args.source,
            input_path=args.input,
            hours=args.hours,
            min_rows=args.min_rows,
            threshold_config_path=Path(args.threshold_config),
        )
    except (DriftJobError, drift.DriftMonitoringError, RegistryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
