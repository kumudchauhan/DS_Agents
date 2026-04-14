"""Tests for app/graph/nodes/eda.py — _data_quality_report only (no file I/O).

These tests use locally-constructed DataFrames with arbitrary column names
so they are not tied to any specific dataset schema.
"""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.eda import _data_quality_report

_TARGET = "label"


@pytest.fixture()
def basic_df():
    """Small DataFrame for quality report tests."""
    return pd.DataFrame({
        "col_a": [100.0, 200.0, 300.0, 400.0, 500.0],
        "col_b": [150.0, 250.0, 350.0, np.nan, 550.0],
        "col_c": [1, 2, 3, 4, 5],
        "cat_col": ["x", "y", "x", None, "y"],
        _TARGET: [0.0, 1.0, 0.0, 1.0, 0.0],
    })


class TestDataQualityReport:
    def test_basic_stats(self, basic_df):
        report = _data_quality_report(basic_df, _TARGET)
        assert f"Total rows: {len(basic_df)}" in report
        assert f"Total columns: {basic_df.shape[1]}" in report

    def test_missing_values(self, basic_df):
        report = _data_quality_report(basic_df, _TARGET)
        assert "col_b" in report
        assert "1 (" in report  # 1 missing value

    def test_no_missing(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            _TARGET: [0, 1, 0],
        })
        report = _data_quality_report(df, _TARGET)
        assert "No missing values" in report

    def test_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": [3, 3, 4],
            _TARGET: [0, 0, 1],
        })
        report = _data_quality_report(df, _TARGET)
        assert "Duplicate rows: 1" in report

    def test_target_quality_warnings(self):
        df = pd.DataFrame({
            "a": [100, 200, 300],
            _TARGET: [0.0, 1.0, 2.0],
        })
        report = _data_quality_report(df, _TARGET)
        assert "WARNING" in report
        assert "unexpected values" in report
