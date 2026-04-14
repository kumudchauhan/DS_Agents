"""Tests for app/graph/nodes/critic.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.critic import (
    VALID_FEATURES,
    VALID_MODELS,
    DEFAULT_RECOMMENDATIONS,
    _parse_json_from_response,
    _validate_recommendations,
    critic_node,
)


# ── _parse_json_from_response ────────────────────────────────────────────

class TestParseJson:
    def test_plain_json(self):
        text = '{"key": "value", "num": 42}'
        result = _parse_json_from_response(text)
        assert result == {"key": "value", "num": 42}

    def test_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_from_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_from_response("not json at all")

    def test_whitespace_handling(self):
        text = '  \n  {"key": "value"}  \n  '
        result = _parse_json_from_response(text)
        assert result == {"key": "value"}


# ── _validate_recommendations ────────────────────────────────────────────

class TestValidateRecommendations:
    def test_valid_input(self):
        raw = {
            "features_to_add": ["log_amount", "is_night_txn"],
            "model_config": {
                "model_name": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 200, "max_depth": 10},
            },
            "should_stop": False,
            "reasoning": "Improve recall.",
        }
        result = _validate_recommendations(raw)
        assert result["features_to_add"] == ["log_amount", "is_night_txn"]
        assert result["model_config"]["model_name"] == "RandomForestClassifier"
        assert result["model_config"]["hyperparameters"]["n_estimators"] == 200
        assert result["should_stop"] is False

    def test_unknown_features_filtered(self):
        raw = {
            "features_to_add": ["log_amount", "FAKE_FEATURE", "is_weekend"],
            "model_config": {"model_name": "RandomForestClassifier", "hyperparameters": {}},
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert "FAKE_FEATURE" not in result["features_to_add"]
        assert "log_amount" in result["features_to_add"]
        assert "is_weekend" in result["features_to_add"]

    def test_hyperparameter_clamping_above_max(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {
                "model_name": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 9999, "max_depth": 100},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        hp = result["model_config"]["hyperparameters"]
        assert hp["n_estimators"] <= 300
        assert hp["max_depth"] <= 30

    def test_hyperparameter_clamping_below_min(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {
                "model_name": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 1, "max_depth": 1},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        hp = result["model_config"]["hyperparameters"]
        assert hp["n_estimators"] >= 50
        assert hp["max_depth"] >= 5

    def test_invalid_model_fallback(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {
                "model_name": "XGBClassifier",
                "hyperparameters": {},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        # Falls back to default model
        assert result["model_config"]["model_name"] == DEFAULT_RECOMMENDATIONS["model_config"]["model_name"]

    def test_kernel_enum_validation(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {
                "model_name": "SVC",
                "hyperparameters": {"C": 1.0, "kernel": "poly"},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        # "poly" is not in ["rbf", "linear"], so kernel should not be set
        assert "kernel" not in result["model_config"]["hyperparameters"]

    def test_valid_kernel_accepted(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {
                "model_name": "SVC",
                "hyperparameters": {"C": 1.0, "kernel": "linear"},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["model_config"]["hyperparameters"]["kernel"] == "linear"

    def test_empty_features_fallback(self):
        raw = {
            "features_to_add": ["ALL_FAKE"],
            "model_config": {"model_name": "RandomForestClassifier", "hyperparameters": {}},
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        # All features invalid -> fall back to defaults
        assert result["features_to_add"] == list(DEFAULT_RECOMMENDATIONS["features_to_add"])

    def test_should_stop_bool_check(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {"model_name": "RandomForestClassifier", "hyperparameters": {}},
            "should_stop": "yes",  # not a bool
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        # Non-bool should_stop -> defaults to False
        assert result["should_stop"] is False

    def test_should_stop_true(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {"model_name": "RandomForestClassifier", "hyperparameters": {}},
            "should_stop": True,
            "reasoning": "F1 is high enough.",
        }
        result = _validate_recommendations(raw)
        assert result["should_stop"] is True

    def test_gradient_boosting_clamping(self):
        raw = {
            "features_to_add": ["log_amount"],
            "model_config": {
                "model_name": "GradientBoostingClassifier",
                "hyperparameters": {"n_estimators": 500, "learning_rate": 0.5, "max_depth": 15},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        hp = result["model_config"]["hyperparameters"]
        assert hp["n_estimators"] <= 300
        assert hp["learning_rate"] <= 0.3
        assert hp["max_depth"] <= 10


# ── critic_node (mocked LLM) ────────────────────────────────────────────

class TestCriticNode:
    def _make_critic_state(self, make_state):
        return make_state(
            metrics={"accuracy": 0.75, "precision": 0.70, "recall": 0.65, "f1": 0.67},
            history=[],
        )

    @patch("app.graph.nodes.critic.get_llm")
    def test_successful_call(self, mock_get_llm, make_state):
        state = self._make_critic_state(make_state)

        # Mock LLM responses
        feedback_response = MagicMock()
        feedback_response.content = "Try adding more features."

        rec_response = MagicMock()
        rec_response.content = json.dumps({
            "features_to_add": ["log_amount", "is_night_txn"],
            "model_config": {"model_name": "RandomForestClassifier",
                             "hyperparameters": {"n_estimators": 200}},
            "should_stop": False,
            "reasoning": "Need more features.",
        })

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [feedback_response, rec_response]
        mock_get_llm.return_value = mock_llm

        result = critic_node(state)
        assert result["feedback"] == "Try adding more features."
        assert len(result["history"]) == 1
        assert result["recommendations"]["features_to_add"] == ["log_amount", "is_night_txn"]

    @patch("app.graph.nodes.critic.get_llm")
    def test_llm_failure_fallback(self, mock_get_llm, make_state):
        state = self._make_critic_state(make_state)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_get_llm.return_value = mock_llm

        result = critic_node(state)
        assert "LLM unavailable" in result["feedback"]
        # Recommendations should be defaults
        assert result["recommendations"] == DEFAULT_RECOMMENDATIONS

    @patch("app.graph.nodes.critic.get_llm")
    def test_history_accumulation(self, mock_get_llm, make_state):
        state = make_state(
            metrics={"accuracy": 0.80, "precision": 0.78, "recall": 0.75, "f1": 0.76},
            history=[{"iteration": 0, "f1": 0.67, "accuracy": 0.75}],
        )

        feedback_response = MagicMock()
        feedback_response.content = "Good progress."

        rec_response = MagicMock()
        rec_response.content = json.dumps({
            "features_to_add": ["log_amount"],
            "model_config": {"model_name": "RandomForestClassifier",
                             "hyperparameters": {"n_estimators": 100}},
            "should_stop": False,
            "reasoning": "Keep iterating.",
        })

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [feedback_response, rec_response]
        mock_get_llm.return_value = mock_llm

        result = critic_node(state)
        # History should now have 2 entries (previous + new)
        assert len(result["history"]) == 2
        assert result["history"][0]["iteration"] == 0
        assert result["history"][1]["f1"] == 0.76
