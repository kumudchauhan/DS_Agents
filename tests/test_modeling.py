"""Tests for app/graph/nodes/modeling.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from app.graph.nodes.modeling import _build_model, modeling_node, NEEDS_SCALING


# ── _build_model tests ───────────────────────────────────────────────────

class TestBuildModel:
    def test_logistic_regression(self):
        model = _build_model("LogisticRegression", {"C": 2.0})
        assert isinstance(model, LogisticRegression)
        assert model.C == 2.0
        assert model.max_iter == 1000

    def test_random_forest(self):
        model = _build_model("RandomForestClassifier", {"n_estimators": 200, "max_depth": 10})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 200
        assert model.max_depth == 10

    def test_gradient_boosting(self):
        model = _build_model("GradientBoostingClassifier", {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5})
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 150
        assert model.learning_rate == 0.1

    def test_svc(self):
        model = _build_model("SVC", {"C": 1.0, "kernel": "linear"})
        assert isinstance(model, SVC)
        assert model.kernel == "linear"
        assert model.probability is True

    def test_unknown_fallback(self):
        model = _build_model("XGBClassifier", {})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100

    def test_default_hyperparameters(self):
        model = _build_model("LogisticRegression", {})
        assert model.C == 1.0

    def test_random_forest_none_max_depth(self):
        model = _build_model("RandomForestClassifier", {"n_estimators": 100})
        assert model.max_depth is None


# ── modeling_node tests ──────────────────────────────────────────────────

@pytest.fixture()
def modeling_state(make_state):
    """State ready for modeling: features already engineered."""
    from app.graph.nodes.feature_eng import feature_engineering_node
    state = make_state(recommendations={})
    fe_result = feature_engineering_node(state)
    state["dataframe"] = fe_result["dataframe"]
    state["feature_report"] = fe_result["feature_report"]
    return state


class TestModelingNode:
    def test_default_config(self, modeling_state):
        result = modeling_node(modeling_state)
        assert isinstance(result["model"], LogisticRegression)

    def test_recommended_config(self, modeling_state):
        modeling_state["recommendations"] = {
            "model_config": {
                "model_name": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 150},
            }
        }
        result = modeling_node(modeling_state)
        assert isinstance(result["model"], RandomForestClassifier)

    def test_scaling_applied_for_svc(self, modeling_state):
        modeling_state["recommendations"] = {
            "model_config": {
                "model_name": "SVC",
                "hyperparameters": {"C": 1.0, "kernel": "rbf"},
            }
        }
        result = modeling_node(modeling_state)
        # X_train should be numpy array (result of StandardScaler transform)
        assert isinstance(result["X_train"], np.ndarray)

    def test_no_scaling_for_rf(self, modeling_state):
        modeling_state["recommendations"] = {
            "model_config": {
                "model_name": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 100},
            }
        }
        result = modeling_node(modeling_state)
        # X_train should be a DataFrame (no scaling applied)
        assert isinstance(result["X_train"], pd.DataFrame)

    def test_train_test_split_ratio(self, modeling_state):
        result = modeling_node(modeling_state)
        total = len(result["X_train"]) + len(result["X_test"])
        test_ratio = len(result["X_test"]) / total
        assert test_ratio == pytest.approx(0.2, abs=0.05)

    def test_model_is_fitted(self, modeling_state):
        result = modeling_node(modeling_state)
        model = result["model"]
        # A fitted sklearn model has `classes_` attribute
        assert hasattr(model, "classes_")

    def test_return_keys(self, modeling_state):
        result = modeling_node(modeling_state)
        for key in ["model", "X_train", "X_test", "y_train", "y_test", "model_report"]:
            assert key in result
