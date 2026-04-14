"""Tests for app/graph/nodes/cleaning.py."""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.cleaning import cleaning_node


def _run_cleaning(df, target="is_fraud"):
    """Helper to run cleaning_node and return result dict."""
    state = {"dataframe": df, "target_column": target, "iteration": 0}
    return cleaning_node(state)


class TestCleaningNode:
    def test_numeric_coercion(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        assert pd.api.types.is_numeric_dtype(df["amount"])
        assert pd.api.types.is_numeric_dtype(df["avg_amount_7d"])

    def test_string_normalization(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        # All non-NaN string values should be lowercase and stripped
        for col in ["merchant_category", "transaction_type", "device_type"]:
            non_null = df[col].dropna()
            for val in non_null:
                assert val == val.strip().lower()

    def test_nan_median_fill(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        # Numeric fill columns should have no NaNs
        for col in ["amount", "avg_amount_7d", "prior_transactions_24h"]:
            assert df[col].isnull().sum() == 0

    def test_nan_mode_fill(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        # After cleaning, categorical columns should have fewer NaNs
        # (mode fill handles NaN values that were created from normalization)
        for col in ["merchant_category", "transaction_type", "device_type"]:
            assert df[col].isnull().sum() == 0

    def test_invalid_target_rows_removed(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        # Rows with is_fraud=2 and is_fraud=NaN should be dropped
        assert set(df["is_fraud"].unique()) <= {0.0, 1.0}
        assert len(df) < len(raw_fraud_df)

    def test_timestamp_parsing(self, raw_fraud_df):
        result = _run_cleaning(raw_fraud_df)
        df = result["dataframe"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(df["user_signup_date"])

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
        """All rows kept when target column has only valid 0/1 values."""
        df = pd.DataFrame({
            "amount": [100.0, 200.0, 300.0],
            "avg_amount_7d": [150.0, 250.0, 350.0],
            "is_international": [0, 1, 0],
            "prior_transactions_24h": [1, 2, 3],
            "merchant_category": ["food", "travel", "food"],
            "transaction_type": ["online", "in_store", "online"],
            "device_type": ["mobile", "desktop", "mobile"],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "user_signup_date": ["2023-06-01", "2023-07-01", "2023-08-01"],
            "is_fraud": [0, 1, 0],
        })
        result = _run_cleaning(df)
        assert len(result["dataframe"]) == 3

    def test_independent_copy(self, raw_fraud_df):
        """cleaned_dataframe should be an independent copy of dataframe."""
        result = _run_cleaning(raw_fraud_df)
        assert result["dataframe"] is not result["cleaned_dataframe"]
        # Mutating one should not affect the other
        result["dataframe"].iloc[0, 0] = "MUTATED"
        assert result["cleaned_dataframe"].iloc[0, 0] != "MUTATED"
