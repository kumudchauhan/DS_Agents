"""Tests for app/graph/nodes/cleaning.py.

Column lists are imported from the module under test so the tests adapt
automatically when the schema changes.
"""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.cleaning import (
    cleaning_node,
    STRING_COLS,
    NUMERIC_FILL_COLS,
    TIMESTAMP_COLS,
)
from tests.conftest import TARGET


def _run_cleaning(df, target=TARGET):
    state = {"dataframe": df, "target_column": target, "iteration": 0}
    return cleaning_node(state)


class TestCleaningNode:
    def test_numeric_coercion(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        for col in NUMERIC_FILL_COLS:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])

    def test_string_normalization(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        for col in STRING_COLS:
            if col in df.columns:
                non_null = df[col].dropna()
                for val in non_null:
                    assert val == val.strip().lower()

    def test_nan_median_fill(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        for col in NUMERIC_FILL_COLS:
            if col in df.columns:
                assert df[col].isnull().sum() == 0

    def test_nan_mode_fill(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        for col in STRING_COLS:
            if col in df.columns:
                assert df[col].isnull().sum() == 0

    def test_invalid_target_rows_removed(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        assert set(df[TARGET].unique()) <= {0.0, 1.0}
        assert len(df) < len(raw_fraud_df)

    def test_timestamp_parsing(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        for col in TIMESTAMP_COLS:
            if col in df.columns:
                assert pd.api.types.is_datetime64_any_dtype(df[col])

    def test_return_keys(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        assert "dataframe" in result
        assert "cleaned_dataframe" in result
        assert "cleaning_report" in result

    def test_report_content(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        report = result["cleaning_report"]
        assert "Coerced numeric" in report
        assert "Normalised string" in report

    def test_row_preservation_valid_rows(self):
        """All rows kept when target has only valid 0/1 values."""
        data = {}
        for col in NUMERIC_FILL_COLS:
            data[col] = [100.0, 200.0, 300.0]
        for col in STRING_COLS:
            data[col] = ["a", "b", "c"]
        for col in TIMESTAMP_COLS:
            data[col] = ["2024-01-01", "2024-01-02", "2024-01-03"]
        data[TARGET] = [0, 1, 0]
        df = pd.DataFrame(data)
        result = _run_cleaning(df)
        assert len(result["dataframe"]) == 3

    def test_independent_copy(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        assert result["dataframe"] is not result["cleaned_dataframe"]
        result["dataframe"].iloc[0, 0] = "MUTATED"
        assert result["cleaned_dataframe"].iloc[0, 0] != "MUTATED"
