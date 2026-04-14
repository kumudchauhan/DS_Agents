"""Tests for app/graph/nodes/feature_eng.py."""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.feature_eng import (
    FEATURE_REGISTRY,
    DEFAULT_FEATURES,
    feature_engineering_node,
    _account_age_days,
    _hour_of_day,
    _day_of_week,
    _amount_to_avg_ratio,
    _is_high_amount,
    _log_amount,
    _is_new_account,
    _is_night_txn,
    _amount_deviation,
    _high_velocity,
    _amount_squared,
    _amount_x_velocity,
    _is_weekend,
    _txn_per_avg_ratio,
)


@pytest.fixture()
def feature_df():
    """Minimal cleaned DataFrame for feature function tests."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 02:00", "2024-01-06 14:00",
                                     "2024-01-07 23:30", "2024-01-08 05:00"]),
        "user_signup_date": pd.to_datetime(["2023-12-15", "2023-06-01",
                                            "2024-01-01", "2023-12-01"]),
        "amount": [100.0, 500.0, 50.0, 2000.0],
        "avg_amount_7d": [200.0, 100.0, 0.0, 500.0],
        "prior_transactions_24h": [1, 5, 0, 4],
        "is_fraud": [0, 1, 0, 1],
    })


# ── Individual feature function tests ────────────────────────────────────

class TestAccountAgeDays:
    def test_correct_values(self, feature_df):
        df, desc = _account_age_days(feature_df.copy())
        assert "account_age_days" in df.columns
        assert desc is not None
        # First row: 2024-01-01 - 2023-12-15 = 17 days
        assert df["account_age_days"].iloc[0] == 17

    def test_missing_column(self):
        df = pd.DataFrame({"amount": [1, 2]})
        df, desc = _account_age_days(df)
        assert desc is None


class TestHourOfDay:
    def test_correct_values(self, feature_df):
        df, desc = _hour_of_day(feature_df.copy())
        assert df["hour_of_day"].iloc[0] == 2
        assert df["hour_of_day"].iloc[1] == 14

    def test_missing_column(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _hour_of_day(df)
        assert desc is None


class TestDayOfWeek:
    def test_correct_values(self, feature_df):
        df, desc = _day_of_week(feature_df.copy())
        # 2024-01-01 is Monday (0), 2024-01-06 is Saturday (5)
        assert df["day_of_week"].iloc[0] == 0
        assert df["day_of_week"].iloc[1] == 5

    def test_missing_column(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _day_of_week(df)
        assert desc is None


class TestAmountToAvgRatio:
    def test_correct_values(self, feature_df):
        df, desc = _amount_to_avg_ratio(feature_df.copy())
        assert df["amount_to_avg_ratio"].iloc[0] == pytest.approx(0.5)  # 100/200
        assert df["amount_to_avg_ratio"].iloc[1] == pytest.approx(5.0)  # 500/100

    def test_zero_avg_becomes_one(self, feature_df):
        df, _ = _amount_to_avg_ratio(feature_df.copy())
        # Row 2 has avg_amount_7d=0, so ratio should fillna(1.0)
        assert df["amount_to_avg_ratio"].iloc[2] == pytest.approx(1.0)

    def test_missing_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _amount_to_avg_ratio(df)
        assert desc is None


class TestIsHighAmount:
    def test_correct_values(self, feature_df):
        df, _ = _is_high_amount(feature_df.copy())
        # 2000 is above 95th percentile
        assert df["is_high_amount"].iloc[3] == 1

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        df, desc = _is_high_amount(df)
        assert desc is None


class TestLogAmount:
    def test_correct_values(self, feature_df):
        df, _ = _log_amount(feature_df.copy())
        assert df["log_amount"].iloc[0] == pytest.approx(np.log1p(100.0))

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        df, desc = _log_amount(df)
        assert desc is None


class TestIsNewAccount:
    def test_correct_values(self, feature_df):
        df, _ = _is_new_account(feature_df.copy())
        # Row 0: age=17 days (<30) -> 1
        # Row 1: age=219 days (>=30) -> 0
        assert df["is_new_account"].iloc[0] == 1
        assert df["is_new_account"].iloc[1] == 0

    def test_auto_creates_account_age_days(self, feature_df):
        """When account_age_days is missing, it should be created from timestamps."""
        df = feature_df.copy()
        assert "account_age_days" not in df.columns
        df, desc = _is_new_account(df)
        assert "account_age_days" in df.columns
        assert desc is not None

    def test_missing_all_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _is_new_account(df)
        assert desc is None


class TestIsNightTxn:
    def test_correct_values(self, feature_df):
        df, _ = _is_night_txn(feature_df.copy())
        # hour=2 -> night, hour=14 -> not night, hour=23 -> night, hour=5 -> night
        assert df["is_night_txn"].iloc[0] == 1
        assert df["is_night_txn"].iloc[1] == 0
        assert df["is_night_txn"].iloc[2] == 1
        assert df["is_night_txn"].iloc[3] == 1

    def test_auto_creates_hour_of_day(self, feature_df):
        df = feature_df.copy()
        assert "hour_of_day" not in df.columns
        df, desc = _is_night_txn(df)
        assert "hour_of_day" in df.columns

    def test_missing_all_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _is_night_txn(df)
        assert desc is None


class TestAmountDeviation:
    def test_correct_values(self, feature_df):
        df, _ = _amount_deviation(feature_df.copy())
        assert df["amount_deviation"].iloc[0] == pytest.approx(100.0)  # |100-200|

    def test_missing_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _amount_deviation(df)
        assert desc is None


class TestHighVelocity:
    def test_correct_values(self, feature_df):
        df, _ = _high_velocity(feature_df.copy())
        # Row 1: 5 >= 4 -> 1, Row 0: 1 < 4 -> 0
        assert df["high_velocity"].iloc[0] == 0
        assert df["high_velocity"].iloc[1] == 1

    def test_missing_column(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _high_velocity(df)
        assert desc is None


class TestAmountSquared:
    def test_correct_values(self, feature_df):
        df, _ = _amount_squared(feature_df.copy())
        assert df["amount_squared"].iloc[0] == pytest.approx(10000.0)

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        df, desc = _amount_squared(df)
        assert desc is None


class TestAmountXVelocity:
    def test_correct_values(self, feature_df):
        df, _ = _amount_x_velocity(feature_df.copy())
        assert df["amount_x_velocity"].iloc[0] == pytest.approx(100.0)  # 100*1
        assert df["amount_x_velocity"].iloc[1] == pytest.approx(2500.0)  # 500*5

    def test_missing_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _amount_x_velocity(df)
        assert desc is None


class TestIsWeekend:
    def test_correct_values(self, feature_df):
        df, _ = _is_weekend(feature_df.copy())
        # 2024-01-01 Mon(0)->0, 2024-01-06 Sat(5)->1, 2024-01-07 Sun(6)->1
        assert df["is_weekend"].iloc[0] == 0
        assert df["is_weekend"].iloc[1] == 1
        assert df["is_weekend"].iloc[2] == 1

    def test_auto_creates_day_of_week(self, feature_df):
        df = feature_df.copy()
        assert "day_of_week" not in df.columns
        df, desc = _is_weekend(df)
        assert "day_of_week" in df.columns

    def test_missing_all_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _is_weekend(df)
        assert desc is None


class TestTxnPerAvgRatio:
    def test_correct_values(self, feature_df):
        df, _ = _txn_per_avg_ratio(feature_df.copy())
        assert df["txn_per_avg_ratio"].iloc[0] == pytest.approx(1 / 200.0)

    def test_zero_avg_becomes_zero(self, feature_df):
        df, _ = _txn_per_avg_ratio(feature_df.copy())
        # Row 2 has avg_amount_7d=0, so ratio should fillna(0.0)
        assert df["txn_per_avg_ratio"].iloc[2] == pytest.approx(0.0)

    def test_missing_columns(self):
        df = pd.DataFrame({"amount": [1]})
        df, desc = _txn_per_avg_ratio(df)
        assert desc is None


# ── Node-level tests ─────────────────────────────────────────────────────

class TestFeatureEngineeringNode:
    def test_default_features(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        assert "feature_report" in result
        assert "default feature set" in result["feature_report"].lower()

    def test_llm_recommended_features(self, make_state):
        state = make_state(recommendations={
            "features_to_add": ["log_amount", "high_velocity"],
        })
        result = feature_engineering_node(state)
        assert "log_amount" in result["feature_report"]
        assert "high_velocity" in result["feature_report"]

    def test_unknown_feature_skip(self, make_state):
        state = make_state(recommendations={
            "features_to_add": ["nonexistent_feature", "log_amount"],
        })
        result = feature_engineering_node(state)
        assert "Skipped unknown feature: nonexistent_feature" in result["feature_report"]

    def test_one_hot_encoding(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        df = result["dataframe"]
        # Original categorical columns should be dropped
        assert "merchant_category" not in df.columns
        assert "transaction_type" not in df.columns
        assert "device_type" not in df.columns

    def test_columns_dropped(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        df = result["dataframe"]
        for col in ["transaction_id", "user_id", "timestamp", "user_signup_date"]:
            assert col not in df.columns

    def test_return_keys(self, make_state):
        state = make_state(recommendations={})
        result = feature_engineering_node(state)
        assert "dataframe" in result
        assert "feature_report" in result
