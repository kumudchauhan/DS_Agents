"""Tests for app/graph/nodes/modeling.py.

Model names and scaling rules are derived from module constants so the
tests adapt when models are added or removed.
"""

import numpy as np
import pandas as pd
import pytest

from app.graph.nodes.critic import VALID_MODELS
from app.graph.nodes.modeling import _build_model, modeling_node, NEEDS_SCALING

_SCALED_MODELS = sorted(NEEDS_SCALING)
_UNSCALED_MODELS = sorted(set(VALID_MODELS.keys()) - NEEDS_SCALING)


# ── _build_model tests ───────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", list(VALID_MODELS.keys()))
def test_build_model_returns_correct_type(model_name):
    model = _build_model(model_name, {})
    assert type(model).__name__ == model_name


def test_build_model_unknown_fallback():
    model = _build_model("NonexistentModel", {})
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_model_uses_provided_hyperparameters():
    """A range-based hyperparameter is passed through to the model."""
    for model_name, params in VALID_MODELS.items():
        for param, constraint in params.items():
            if isinstance(constraint, tuple):
                lo, hi = constraint
                val = (lo + hi) / 2
                model = _build_model(model_name, {param: val})
                if hasattr(model, param):
                    assert getattr(model, param) == pytest.approx(val, rel=0.01)
                return
    pytest.skip("No range-based param found in VALID_MODELS")


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
        assert hasattr(result["model"], "fit")

    def test_recommended_config(self, modeling_state):
        model_name = list(VALID_MODELS.keys())[1]
        modeling_state["recommendations"] = {
            "model_config": {"model_name": model_name, "hyperparameters": {}}
        }
        result = modeling_node(modeling_state)
        assert type(result["model"]).__name__ == model_name

    @pytest.mark.parametrize("model_name", _SCALED_MODELS)
    def test_scaling_applied(self, model_name, modeling_state):
        modeling_state["recommendations"] = {
            "model_config": {"model_name": model_name, "hyperparameters": {}}
        }
        result = modeling_node(modeling_state)
        assert isinstance(result["X_train"], np.ndarray)

    @pytest.mark.parametrize("model_name", _UNSCALED_MODELS)
    def test_no_scaling(self, model_name, modeling_state):
        modeling_state["recommendations"] = {
            "model_config": {"model_name": model_name, "hyperparameters": {}}
        }
        result = modeling_node(modeling_state)
        assert isinstance(result["X_train"], pd.DataFrame)

    def test_train_test_split_ratio(self, modeling_state):
        result = modeling_node(modeling_state)
        total = len(result["X_train"]) + len(result["X_test"])
        test_ratio = len(result["X_test"]) / total
        assert test_ratio == pytest.approx(0.2, abs=0.05)

    def test_model_is_fitted(self, modeling_state):
        result = modeling_node(modeling_state)
        assert hasattr(result["model"], "classes_")

    def test_return_keys(self, modeling_state):
        result = modeling_node(modeling_state)
        for key in ["model", "X_train", "X_test", "y_train", "y_test", "model_report"]:
            assert key in result
