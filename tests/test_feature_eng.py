"""Tests for app/graph/nodes/feature_eng.py.

Individual features are tested via parametrize over FEATURE_REGISTRY rather
than one test class per feature, so the suite automatically covers any
features added or removed from the registry.
"""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.cleaning import NUMERIC_FILL_COLS, TIMESTAMP_COLS
from app.graph.nodes.feature_eng import (
    CATEGORICAL_COLS,
    DROP_COLS,
    FEATURE_REGISTRY,
    DEFAULT_FEATURES,
    feature_engineering_node,
)
from app.ui.notebook_export import _FEATURE_CODE
from tests.conftest import TARGET

# Features whose notebook code declares dependencies.
_FEATURES_WITH_DEPS = [
    (name, deps) for name, (deps, _) in _FEATURE_CODE.items() if deps
]


@pytest.fixture()
def feature_df():
    """Minimal cleaned DataFrame with all columns the feature registry may need.

    Columns are derived from TIMESTAMP_COLS and NUMERIC_FILL_COLS so the
    fixture adapts when the schema changes.
    """
    n = 6
    data = {}

    # Timestamps — varied hours and weekdays for time-based features
    data[TIMESTAMP_COLS[0]] = pd.to_datetime([
        "2024-01-01 02:00", "2024-01-03 14:00", "2024-01-06 23:30",
        "2024-01-07 10:00", "2024-01-08 05:00", "2024-01-10 18:00",
    ])
    if len(TIMESTAMP_COLS) > 1:
        data[TIMESTAMP_COLS[1]] = pd.to_datetime([
            "2023-12-15", "2023-06-01", "2024-01-01",
            "2023-12-01", "2023-11-01", "2023-10-01",
        ])

    # Numeric columns with varied values (including a zero for division tests)
    numeric_vals = [
        [100.0, 500.0, 50.0, 2000.0, 300.0, 150.0],
        [200.0, 100.0, 0.0, 500.0, 300.0, 250.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 5.0, 0.0, 4.0, 2.0, 6.0],
    ]
    for i, col in enumerate(NUMERIC_FILL_COLS):
        data[col] = numeric_vals[i] if i < len(numeric_vals) else [1.0] * n

    data[TARGET] = [0, 1, 0, 1, 0, 1]
    return pd.DataFrame(data)


# ── Parametrized feature function tests ──────────────────────────────────

@pytest.mark.parametrize("feature_name", list(FEATURE_REGISTRY.keys()))
def test_feature_produces_column_or_skips(feature_name, feature_df):
    """Each registered feature either creates its named column or returns
    desc=None when required source columns are missing."""
    fn = FEATURE_REGISTRY[feature_name]
    df, desc = fn(feature_df.copy())
    if desc is not None:
        assert feature_name in df.columns
        assert len(df) == len(feature_df)


@pytest.mark.parametrize("feature_name", list(FEATURE_REGISTRY.keys()))
def test_feature_handles_missing_columns(feature_name):
    """Feature gracefully returns desc=None on a DataFrame with no
    recognised source columns."""
    df = pd.DataFrame({"unrelated_column": [1, 2, 3]})
    fn = FEATURE_REGISTRY[feature_name]
    df, desc = fn(df)
    assert desc is None


@pytest.mark.parametrize(
    "feature_name,deps", _FEATURES_WITH_DEPS,
    ids=[f[0] for f in _FEATURES_WITH_DEPS],
)
def test_feature_auto_creates_dependencies(feature_name, deps, feature_df):
    """Features with dependencies auto-create missing dependency columns."""
    df = feature_df.copy()
    for dep in deps:
        if dep in df.columns:
            df = df.drop(columns=[dep])
    fn = FEATURE_REGISTRY[feature_name]
    df, desc = fn(df)
    if desc is not None:
        assert feature_name in df.columns


# ── Node-level tests ─────────────────────────────────────────────────────

class TestFeatureEngineeringNode:
    def test_default_features(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        assert "feature_report" in result
        assert "default feature set" in result["feature_report"].lower()

    def test_llm_recommended_features(self, make_state):
        feats = list(FEATURE_REGISTRY.keys())[-2:]
        state = make_state(recommendations={"features_to_add": feats})
        result = feature_engineering_node(state)
        for f in feats:
            assert f in result["feature_report"]

    def test_unknown_feature_skip(self, make_state):
        known = list(FEATURE_REGISTRY.keys())[0]
        state = make_state(recommendations={
            "features_to_add": ["nonexistent_feature", known],
        })
        result = feature_engineering_node(state)
        assert "Skipped unknown feature: nonexistent_feature" in result["feature_report"]

    def test_one_hot_encoding(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        df = result["dataframe"]
        for col in CATEGORICAL_COLS:
            assert col not in df.columns

    def test_columns_dropped(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        df = result["dataframe"]
        for col in DROP_COLS:
            assert col not in df.columns

    def test_return_keys(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        assert "dataframe" in result
        assert "feature_report" in result
