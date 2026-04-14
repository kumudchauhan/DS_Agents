"""Shared fixtures for DS Agent test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def raw_fraud_df():
    """~20-row DataFrame matching the transactions.csv schema.

    Contains intentional NaNs, mixed-case strings, invalid target values,
    and string timestamps to exercise cleaning logic.
    """
    n = 20
    rng = np.random.RandomState(42)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    signups = pd.date_range("2023-06-01", periods=n, freq="7D")

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:04d}" for i in range(n)],
        "user_id": [f"U{i:03d}" for i in range(n)],
        "timestamp": timestamps.astype(str),          # string timestamps
        "user_signup_date": signups.astype(str),
        "amount": rng.uniform(5, 5000, n).round(2),
        "avg_amount_7d": rng.uniform(50, 500, n).round(2),
        "is_international": rng.choice([0, 1], n).astype(float),
        "prior_transactions_24h": rng.randint(0, 10, n).astype(float),
        "merchant_category": [" Electronics ", "FOOD", "  travel", "food",
                              "Electronics", np.nan, "Travel", "FOOD",
                              "electronics", "Food", "travel", "ELECTRONICS",
                              np.nan, "Food", "Travel", "food",
                              "electronics", "Travel", "FOOD", "food"],
        "transaction_type": ["ONLINE", "in_store", " Online", "IN_STORE",
                             "online", "ONLINE", np.nan, "in_store",
                             "Online", "IN_STORE", "online", "ONLINE",
                             "in_store", "Online", "IN_STORE", np.nan,
                             "online", "ONLINE", "in_store", "Online"],
        "device_type": ["mobile", "DESKTOP", " Mobile", "desktop",
                        "Mobile", "DESKTOP", "mobile", np.nan,
                        "Desktop", "mobile", "DESKTOP", "Mobile",
                        "desktop", "mobile", "DESKTOP", np.nan,
                        "Mobile", "desktop", "mobile", "DESKTOP"],
        "is_fraud": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                     1, 0, 0, 0, 1, 0, 0, 2, np.nan, 0],  # row 17=2, row 18=NaN
    })

    # Inject NaN into numeric columns
    df.loc[2, "amount"] = np.nan
    df.loc[5, "avg_amount_7d"] = np.nan
    df.loc[8, "prior_transactions_24h"] = np.nan

    # Inject a string into a numeric column to test coercion
    df["amount"] = df["amount"].astype(object)
    df.at[10, "amount"] = "bad_value"

    return df


@pytest.fixture()
def cleaned_fraud_df(raw_fraud_df):
    """The raw_fraud_df after running cleaning_node-equivalent logic.

    Numerics coerced, strings normalized, NaNs filled, timestamps parsed,
    invalid target rows (is_fraud=2, NaN) dropped.
    """
    from app.graph.nodes.cleaning import cleaning_node

    state = {
        "dataframe": raw_fraud_df,
        "target_column": "is_fraud",
        "iteration": 0,
    }
    result = cleaning_node(state)
    return result["dataframe"]


@pytest.fixture()
def make_state(cleaned_fraud_df):
    """Factory fixture returning a valid AgentState dict.

    Tests can override specific keys.
    """
    def _factory(**overrides):
        base = {
            "dataset_path": "data/transactions.csv",
            "dataframe": cleaned_fraud_df,
            "cleaned_dataframe": cleaned_fraud_df.copy(),
            "target_column": "is_fraud",
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "model": None,
            "metrics": {"accuracy": 0.75, "precision": 0.70,
                        "recall": 0.65, "f1": 0.67},
            "feedback": "",
            "iteration": 0,
            "max_iterations": 3,
            "eda_report": "Shape: 20 rows x 12 columns",
            "cleaning_report": "Coerced numeric columns to proper dtype",
            "feature_report": "Using default feature set\nCreated: account_age_days = (timestamp - signup_date).days\nCreated: hour_of_day = timestamp.hour",
            "model_report": "Model: LogisticRegression(C=1.0)\nTrain size: 14, Test size: 4",
            "evaluation_report": "Accuracy: 0.75\nPrecision: 0.70\nRecall: 0.65\nF1 Score: 0.67",
            "history": [],
            "should_continue": True,
            "recommendations": {},
        }
        base.update(overrides)
        return base
    return _factory


@pytest.fixture()
def sample_history():
    """Two-iteration history list with realistic metrics, model strings, and feature reports."""
    return [
        {
            "iteration": 0,
            "f1": 0.67,
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.65,
            "model": "Model: LogisticRegression(C=1.0)\nTrain size: 14, Test size: 4",
            "feedback": "Try adding more features.",
            "feature_report": (
                "Using default feature set\n"
                "Created: account_age_days = (timestamp - signup_date).days\n"
                "Created: hour_of_day = timestamp.hour\n"
                "Created: day_of_week = timestamp.dayofweek\n"
                "Created: amount_to_avg_ratio = amount / avg_amount_7d\n"
                "Created: is_high_amount = (amount > 1234.56)"
            ),
            "cleaning_report": "Coerced numeric columns",
            "model_obj": None,
            "feature_names": ["account_age_days", "hour_of_day"],
            "recommendations": {
                "features_to_add": ["log_amount", "is_night_txn"],
                "model_config": {"model_name": "RandomForestClassifier",
                                 "hyperparameters": {"n_estimators": 200}},
                "should_stop": False,
                "reasoning": "Add features to improve recall.",
            },
        },
        {
            "iteration": 1,
            "f1": 0.82,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.84,
            "model": "Model: RandomForestClassifier(n_estimators=200)\nTrain size: 14, Test size: 4",
            "feedback": "Good improvement. Consider GBM.",
            "feature_report": (
                "Using LLM-recommended features (4)\n"
                "Created: log_amount = log1p(amount)\n"
                "Created: is_night_txn = (hour >= 23 or hour <= 5)\n"
                "Created: account_age_days = (timestamp - signup_date).days\n"
                "Created: hour_of_day = timestamp.hour"
            ),
            "cleaning_report": "Coerced numeric columns",
            "model_obj": None,
            "feature_names": ["log_amount", "is_night_txn", "account_age_days", "hour_of_day"],
            "recommendations": {
                "features_to_add": ["log_amount", "is_night_txn", "amount_deviation"],
                "model_config": {"model_name": "GradientBoostingClassifier",
                                 "hyperparameters": {"n_estimators": 150, "learning_rate": 0.1}},
                "should_stop": False,
                "reasoning": "Switch to GBM for better performance.",
            },
        },
    ]
