"""Shared fixtures for DS Agent test suite.

All column names, feature names, and model names are derived from module
constants so these fixtures adapt automatically when the source changes.
"""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.cleaning import STRING_COLS, NUMERIC_FILL_COLS, TIMESTAMP_COLS
from app.graph.nodes.critic import VALID_FEATURES, VALID_MODELS
from app.graph.nodes.feature_eng import DEFAULT_FEATURES, DROP_COLS

# Arbitrary target column name used across all test fixtures.
TARGET = "is_fraud"

# Identifier columns from DROP_COLS that aren't timestamps or categoricals.
_ID_COLS = [c for c in DROP_COLS if c not in TIMESTAMP_COLS + STRING_COLS]


@pytest.fixture()
def raw_fraud_df():
    """DataFrame built from module constants with intentional data-quality issues.

    Includes NaNs, mixed-case strings, invalid target values, and string
    timestamps to exercise the cleaning pipeline.
    """
    n = 20
    rng = np.random.RandomState(42)
    data = {}

    # Identifier columns
    for col in _ID_COLS:
        data[col] = [f"{col}_{i:04d}" for i in range(n)]

    # Timestamp columns — stored as strings to test parsing
    for i, col in enumerate(TIMESTAMP_COLS):
        start = "2024-01-01" if i == 0 else f"2023-{max(1, 6 - i):02d}-01"
        freq = "h" if i == 0 else "7D"
        data[col] = pd.date_range(start, periods=n, freq=freq).astype(str)

    # Numeric columns
    for col in NUMERIC_FILL_COLS:
        data[col] = rng.uniform(5, 500, n).round(2).astype(float)

    # Inject NaN into first two numeric columns
    for idx, col in enumerate(NUMERIC_FILL_COLS[:2]):
        data[col] = list(data[col])
        data[col][2 + idx * 3] = np.nan

    # String columns — mixed case with NaN
    samples = ["alpha", "beta", "gamma", "delta"]
    for col in STRING_COLS:
        vals = rng.choice(samples, n).tolist()
        vals[0] = " " + vals[0].upper() + " "
        vals[3] = vals[3].title()
        vals[5] = np.nan
        vals[12] = np.nan
        data[col] = vals

    # Target column with invalid entries
    targets = rng.choice([0.0, 1.0], n).tolist()
    targets[-2] = 2.0      # invalid value
    targets[-1] = np.nan   # missing
    data[TARGET] = targets

    df = pd.DataFrame(data)

    # Make first numeric column contain a string to test coercion
    first_num = NUMERIC_FILL_COLS[0]
    df[first_num] = df[first_num].astype(object)
    df.at[10, first_num] = "bad_value"

    return df


@pytest.fixture()
def cleaned_fraud_df(raw_fraud_df):
    """The raw_fraud_df after running cleaning_node."""
    from app.graph.nodes.cleaning import cleaning_node

    state = {"dataframe": raw_fraud_df, "target_column": TARGET, "iteration": 0}
    result = cleaning_node(state)
    return result["dataframe"]


@pytest.fixture()
def make_state(cleaned_fraud_df):
    """Factory fixture returning a valid AgentState dict.

    Tests can override specific keys.
    """
    def _factory(**overrides):
        first_model = list(VALID_MODELS.keys())[0]
        first_features = VALID_FEATURES[:2]
        feature_lines = "\n".join(
            f"Created: {f} = description" for f in first_features
        )

        base = {
            "dataset_path": "data/dataset.csv",
            "dataframe": cleaned_fraud_df,
            "cleaned_dataframe": cleaned_fraud_df.copy(),
            "target_column": TARGET,
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
            "feature_report": f"Using default feature set\n{feature_lines}",
            "model_report": (
                f"Model: {first_model}(C=1.0)\n"
                "Train size: 14, Test size: 4"
            ),
            "evaluation_report": (
                "Accuracy: 0.75\nPrecision: 0.70\n"
                "Recall: 0.65\nF1 Score: 0.67"
            ),
            "history": [],
            "should_continue": True,
            "recommendations": {},
        }
        base.update(overrides)
        return base
    return _factory


@pytest.fixture()
def sample_history():
    """Two-iteration history list built from module constants."""
    model_names = list(VALID_MODELS.keys())
    first_features = VALID_FEATURES[:5]
    second_features = VALID_FEATURES[5:9]
    feature_lines_0 = "\n".join(
        f"Created: {f} = description" for f in first_features
    )
    feature_lines_1 = "\n".join(
        f"Created: {f} = description" for f in second_features
    )

    return [
        {
            "iteration": 0,
            "f1": 0.67,
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.65,
            "model": (
                f"Model: {model_names[0]}(C=1.0)\n"
                "Train size: 14, Test size: 4"
            ),
            "feedback": "Try adding more features.",
            "feature_report": f"Using default feature set\n{feature_lines_0}",
            "cleaning_report": "Coerced numeric columns",
            "model_obj": None,
            "feature_names": first_features[:2],
            "recommendations": {
                "features_to_add": VALID_FEATURES[5:7],
                "model_config": {
                    "model_name": model_names[1],
                    "hyperparameters": {"n_estimators": 200},
                },
                "should_stop": False,
                "reasoning": "Add features.",
            },
        },
        {
            "iteration": 1,
            "f1": 0.82,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.84,
            "model": (
                f"Model: {model_names[1]}(n_estimators=200)\n"
                "Train size: 14, Test size: 4"
            ),
            "feedback": "Good improvement.",
            "feature_report": f"Using LLM-recommended features\n{feature_lines_1}",
            "cleaning_report": "Coerced numeric columns",
            "model_obj": None,
            "feature_names": second_features,
            "recommendations": {
                "features_to_add": VALID_FEATURES[5:8],
                "model_config": {
                    "model_name": model_names[2],
                    "hyperparameters": {"n_estimators": 150},
                },
                "should_stop": False,
                "reasoning": "Switch model.",
            },
        },
    ]
