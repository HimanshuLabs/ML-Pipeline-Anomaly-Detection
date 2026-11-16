"""Tests for the segmented (one IsolationForest per entity type) model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from anomaly_detection import source_extract
from anomaly_detection.segmented_model import (
    SEGMENTED_MODEL_KIND,
    FeatureSpec,
    SegmentedIsolationForest,
    SegmentedModelError,
    feature_spec_from_metadata,
    fit_segmented_isolation_forest,
    segment_spec_fingerprint,
)

SEGMENTS = {
    "alpha": ["f1", "f2"],
    "beta": ["f2", "f3", "f4"],
    "gamma": ["f5"],
}
ALL_FEATURES = ["f1", "f2", "f3", "f4", "f5"]


def _build_model() -> IsolationForest:
    return IsolationForest(n_estimators=50, contamination=0.05, random_state=42)


def _frame(rows: dict[str, int], seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for entity_type, count in rows.items():
        part = pd.DataFrame(rng.normal(size=(count, len(ALL_FEATURES))), columns=ALL_FEATURES)
        part.insert(0, "entity_type", entity_type)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _fit(frame: pd.DataFrame, min_rows: int = 30) -> SegmentedIsolationForest:
    return fit_segmented_isolation_forest(
        frame,
        segment_features=SEGMENTS,
        build_model=_build_model,
        min_rows_per_segment=min_rows,
    )


# --- segment feature sets come from source_extract's per-table mappings ---------------


def test_segment_features_are_defined_by_source_extract_mappings() -> None:
    segments = source_extract.SEGMENT_FEATURE_COLUMNS

    assert set(segments) == {
        "user",
        "customer",
        "campaign",
        "funnel_day",
        "product",
        "web_customer",
    }

    def in_model_order(names) -> list[str]:
        wanted = set(names)
        return [c for c in source_extract.MODEL_FEATURE_COLUMNS if c in wanted]

    assert segments["user"] == tuple(in_model_order(source_extract.P1_PASSTHROUGH_COLUMNS))
    assert segments["customer"] == tuple(
        in_model_order(
            [
                *source_extract.CUSTOMER_360_MAPPING.values(),
                *source_extract.CUSTOMER_360_DERIVED_FEATURES,
            ]
        )
    )
    assert segments["campaign"] == tuple(in_model_order(source_extract.CAMPAIGN_MAPPING.values()))
    assert segments["funnel_day"] == tuple(in_model_order(source_extract.FUNNEL_MAPPING.values()))
    assert segments["product"] == tuple(in_model_order(source_extract.PRODUCT_MAPPING.values()))
    assert segments["web_customer"] == tuple(
        in_model_order(
            [
                *source_extract.WEB_EVENT_COUNT_FEATURES,
                *source_extract.WEB_EVENT_MEASURES.values(),
            ]
        )
    )
    # Every extracted column belongs to at least one segment.
    union = {name for names in segments.values() for name in names}
    assert union == set(source_extract.MODEL_FEATURE_COLUMNS)


# --- fitting, routing, refusal --------------------------------------------------------


def test_each_segment_model_is_fitted_on_only_its_features() -> None:
    model = _fit(_frame({"alpha": 60, "beta": 40, "gamma": 35}))

    assert model.feature_spec() == {name: list(names) for name, names in SEGMENTS.items()}
    for entity_type, names in SEGMENTS.items():
        segment = model.segments[entity_type]
        assert segment.feature_names == names
        assert segment.model.n_features_in_ == len(names)
    assert {t: s.row_count for t, s in model.segments.items()} == {
        "alpha": 60,
        "beta": 40,
        "gamma": 35,
    }


def test_each_row_is_scored_by_its_own_segment_using_only_its_features() -> None:
    model = _fit(_frame({"alpha": 60, "beta": 40, "gamma": 35}))
    scoring = _frame({"gamma": 3, "alpha": 4, "beta": 5}, seed=11).sample(frac=1, random_state=1)
    # Features outside a row's segment are garbage; routing must never read them.
    for entity_type, names in SEGMENTS.items():
        unused = [c for c in ALL_FEATURES if c not in names]
        scoring.loc[scoring["entity_type"] == entity_type, unused] = np.nan

    scores = model.decision_function(scoring)
    labels = model.predict(scoring)

    assert scores.shape == (len(scoring),)
    for position, (_, row) in enumerate(scoring.iterrows()):
        segment = model.segments[row["entity_type"]]
        own = pd.DataFrame([row[segment.feature_names].astype(float)])
        assert scores[position] == pytest.approx(segment.model.decision_function(own)[0])
        assert labels[position] == segment.model.predict(own)[0]


def test_unknown_entity_type_raises_a_clear_error() -> None:
    model = _fit(_frame({"alpha": 60, "beta": 40, "gamma": 35}))
    scoring = _frame({"alpha": 2, "delta": 1})

    with pytest.raises(SegmentedModelError, match="unknown entity_type.*'delta'"):
        model.decision_function(scoring)
    with pytest.raises(SegmentedModelError, match="entity_type"):
        model.predict(scoring.drop(columns=["entity_type"]))


def test_segment_below_min_rows_is_refused_with_type_and_count() -> None:
    with pytest.raises(SegmentedModelError, match=r"beta has 12 rows.*gamma has 5 rows.*30"):
        _fit(_frame({"alpha": 60, "beta": 12, "gamma": 5}))


def test_training_frame_with_a_type_without_segment_definition_is_refused() -> None:
    with pytest.raises(SegmentedModelError, match="no segment definition.*'delta'"):
        _fit(_frame({"alpha": 60, "delta": 40}))


def test_per_segment_anomaly_rate_is_near_contamination_where_single_model_flags_a_type() -> None:
    rng = np.random.default_rng(5)
    common = pd.DataFrame(rng.normal(size=(600, 3)), columns=["f1", "f2", "f3"])
    common.insert(0, "entity_type", "common")
    rare = pd.DataFrame(rng.normal(loc=8.0, size=(40, 3)), columns=["f1", "f2", "f3"])
    rare.insert(0, "entity_type", "rare")
    frame = pd.concat([common, rare], ignore_index=True)
    features = ["f1", "f2", "f3"]

    single = _build_model().fit(frame[features])
    single_rare_rate = float((single.predict(rare[features]) == -1).mean())

    segmented = fit_segmented_isolation_forest(
        frame,
        segment_features={"common": features, "rare": features},
        build_model=_build_model,
        min_rows_per_segment=30,
    )
    labels = segmented.predict(frame)
    rates = {
        entity_type: float((labels[frame["entity_type"].to_numpy() == entity_type] == -1).mean())
        for entity_type in ("common", "rare")
    }

    assert single_rare_rate > 0.5  # the single model flags the rare type as a whole
    for rate in rates.values():
        assert rate == pytest.approx(0.05, abs=0.03)


def test_joblib_round_trip(tmp_path: Path) -> None:
    model = _fit(_frame({"alpha": 60, "beta": 40, "gamma": 35}))
    scoring = _frame({"alpha": 5, "beta": 5, "gamma": 5}, seed=9)
    path = tmp_path / "model.joblib"

    joblib.dump(model, path)
    loaded = joblib.load(path)

    assert type(loaded) is SegmentedIsolationForest
    assert loaded.feature_spec() == model.feature_spec()
    np.testing.assert_array_equal(
        loaded.decision_function(scoring), model.decision_function(scoring)
    )
    np.testing.assert_array_equal(loaded.predict(scoring), model.predict(scoring))


# --- spec fingerprint and compatibility ----------------------------------------------


def test_spec_fingerprint_covers_types_and_per_type_order() -> None:
    base = segment_spec_fingerprint(SEGMENTS)

    assert base == segment_spec_fingerprint(dict(reversed(list(SEGMENTS.items()))))
    assert base != segment_spec_fingerprint({**SEGMENTS, "beta": ["f4", "f3", "f2"]})
    assert base != segment_spec_fingerprint({**SEGMENTS, "delta": ["f1"]})
    assert base != segment_spec_fingerprint({**SEGMENTS, "alpha": ["f1", "f2", "f5"]})


def _segmented_metadata(segments: dict[str, list[str]]) -> dict:
    union = [c for c in ALL_FEATURES if any(c in names for names in segments.values())]
    return {
        "model_kind": SEGMENTED_MODEL_KIND,
        "feature_names": union,
        "segments": {t: {"feature_names": names, "row_count": 40} for t, names in segments.items()},
    }


def test_segment_spec_compatibility_rules() -> None:
    segmented = feature_spec_from_metadata(_segmented_metadata(SEGMENTS))
    same = feature_spec_from_metadata(_segmented_metadata(dict(SEGMENTS)))
    reordered = feature_spec_from_metadata(
        _segmented_metadata({**SEGMENTS, "beta": ["f4", "f3", "f2"]})
    )
    other_types = feature_spec_from_metadata(_segmented_metadata({"alpha": SEGMENTS["alpha"]}))
    single_same_union = feature_spec_from_metadata({"feature_names": ALL_FEATURES})
    single = feature_spec_from_metadata({"feature_names": ["a", "b"]})

    assert segmented.difference(same) is None
    assert "segment spec differs" in segmented.difference(reordered)
    assert "segment spec differs" in segmented.difference(other_types)
    # A segmented and a single model are never compatible, even with the same union.
    assert segmented.feature_names == single_same_union.feature_names
    assert "never compatible" in segmented.difference(single_same_union)
    assert "never compatible" in single_same_union.difference(segmented)
    # Single models keep the ordered feature_names rule.
    assert single.difference(feature_spec_from_metadata({"feature_names": ["a", "b"]})) is None
    assert "feature_names differ" in single.difference(
        feature_spec_from_metadata({"feature_names": ["b", "a"]})
    )
    assert isinstance(segmented, FeatureSpec)


def test_malformed_segmented_metadata_is_rejected() -> None:
    metadata = _segmented_metadata(SEGMENTS)
    metadata["segments"]["alpha"]["feature_names"] = []

    with pytest.raises(ValueError, match="alpha"):
        feature_spec_from_metadata(metadata)
