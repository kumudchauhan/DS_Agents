"""Tests for app/graph/nodes/critic.py.

Hyperparameter ranges, feature lists, and model names are all derived from
VALID_MODELS and VALID_FEATURES so the tests adapt when those registries change.
"""

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

# Derive parametrized test data from source constants.
_RANGE_PARAMS = [
    (model, param, lo, hi)
    for model, params in VALID_MODELS.items()
    for param, constraint in params.items()
    if isinstance(constraint, tuple) and len(constraint) == 2
    for lo, hi in [constraint]
]

_ENUM_PARAMS = [
    (model, param, valid_values)
    for model, params in VALID_MODELS.items()
    for param, constraint in params.items()
    if isinstance(constraint, list)
    for valid_values in [constraint]
]


# ── _parse_json_from_response ────────────────────────────────────────────

class TestParseJson:
    def test_plain_json(self):
        result = _parse_json_from_response('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_markdown_fences(self):
        result = _parse_json_from_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_from_response("not json at all")

    def test_whitespace_handling(self):
        result = _parse_json_from_response('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}


# ── _validate_recommendations ────────────────────────────────────────────

class TestValidateRecommendations:
    def test_valid_input(self):
        two_features = VALID_FEATURES[:2]
        model_name = list(VALID_MODELS.keys())[0]
        raw = {
            "features_to_add": two_features,
            "model_config": {"model_name": model_name, "hyperparameters": {}},
            "should_stop": False,
            "reasoning": "Test.",
        }
        result = _validate_recommendations(raw)
        assert result["features_to_add"] == two_features
        assert result["model_config"]["model_name"] == model_name
        assert result["should_stop"] is False

    def test_unknown_features_filtered(self):
        first, last = VALID_FEATURES[0], VALID_FEATURES[-1]
        raw = {
            "features_to_add": [first, "FAKE_FEATURE", last],
            "model_config": {
                "model_name": list(VALID_MODELS.keys())[0],
                "hyperparameters": {},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert "FAKE_FEATURE" not in result["features_to_add"]
        assert first in result["features_to_add"]
        assert last in result["features_to_add"]

    @pytest.mark.parametrize(
        "model_name,param,lo,hi", _RANGE_PARAMS,
        ids=[f"{m}-{p}" for m, p, _, _ in _RANGE_PARAMS],
    )
    def test_clamping_above_max(self, model_name, param, lo, hi):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": model_name,
                "hyperparameters": {param: hi * 10},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["model_config"]["hyperparameters"][param] <= hi

    @pytest.mark.parametrize(
        "model_name,param,lo,hi", _RANGE_PARAMS,
        ids=[f"{m}-{p}" for m, p, _, _ in _RANGE_PARAMS],
    )
    def test_clamping_below_min(self, model_name, param, lo, hi):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": model_name,
                "hyperparameters": {param: lo / 100},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["model_config"]["hyperparameters"][param] >= lo

    def test_invalid_model_fallback(self):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": "NonexistentModel",
                "hyperparameters": {},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["model_config"]["model_name"] == \
            DEFAULT_RECOMMENDATIONS["model_config"]["model_name"]

    @pytest.mark.parametrize(
        "model_name,param,valid_values", _ENUM_PARAMS,
        ids=[f"{m}-{p}" for m, p, _ in _ENUM_PARAMS],
    )
    def test_enum_rejects_invalid(self, model_name, param, valid_values):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": model_name,
                "hyperparameters": {param: "INVALID_VALUE"},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert param not in result["model_config"]["hyperparameters"]

    @pytest.mark.parametrize(
        "model_name,param,valid_values", _ENUM_PARAMS,
        ids=[f"{m}-{p}" for m, p, _ in _ENUM_PARAMS],
    )
    def test_enum_accepts_valid(self, model_name, param, valid_values):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": model_name,
                "hyperparameters": {param: valid_values[0]},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["model_config"]["hyperparameters"][param] == valid_values[0]

    def test_empty_features_fallback(self):
        raw = {
            "features_to_add": ["ALL_FAKE"],
            "model_config": {
                "model_name": list(VALID_MODELS.keys())[0],
                "hyperparameters": {},
            },
            "should_stop": False,
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["features_to_add"] == \
            list(DEFAULT_RECOMMENDATIONS["features_to_add"])

    def test_should_stop_bool_check(self):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": list(VALID_MODELS.keys())[0],
                "hyperparameters": {},
            },
            "should_stop": "yes",  # not a bool
            "reasoning": "",
        }
        result = _validate_recommendations(raw)
        assert result["should_stop"] is False

    def test_should_stop_true(self):
        raw = {
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {
                "model_name": list(VALID_MODELS.keys())[0],
                "hyperparameters": {},
            },
            "should_stop": True,
            "reasoning": "Done.",
        }
        result = _validate_recommendations(raw)
        assert result["should_stop"] is True


# ── critic_node (mocked LLM) ────────────────────────────────────────────

class TestCriticNode:
    def _make_critic_state(self, make_state):
        return make_state(
            metrics={"accuracy": 0.75, "precision": 0.70,
                     "recall": 0.65, "f1": 0.67},
            history=[],
        )

    @patch("app.graph.nodes.critic.get_llm")
    def test_successful_call(self, mock_get_llm, make_state):
        state = self._make_critic_state(make_state)
        model_name = list(VALID_MODELS.keys())[1]
        two_features = VALID_FEATURES[:2]

        feedback_response = MagicMock()
        feedback_response.content = "Try adding more features."

        rec_response = MagicMock()
        rec_response.content = json.dumps({
            "features_to_add": two_features,
            "model_config": {"model_name": model_name,
                             "hyperparameters": {}},
            "should_stop": False,
            "reasoning": "Need more features.",
        })

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [feedback_response, rec_response]
        mock_get_llm.return_value = mock_llm

        result = critic_node(state)
        assert result["feedback"] == "Try adding more features."
        assert len(result["history"]) == 1
        assert result["recommendations"]["features_to_add"] == two_features

    @patch("app.graph.nodes.critic.get_llm")
    def test_llm_failure_fallback(self, mock_get_llm, make_state):
        state = self._make_critic_state(make_state)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_get_llm.return_value = mock_llm

        result = critic_node(state)
        assert "LLM unavailable" in result["feedback"]
        assert result["recommendations"] == DEFAULT_RECOMMENDATIONS

    @patch("app.graph.nodes.critic.get_llm")
    def test_history_accumulation(self, mock_get_llm, make_state):
        state = make_state(
            metrics={"accuracy": 0.80, "precision": 0.78,
                     "recall": 0.75, "f1": 0.76},
            history=[{"iteration": 0, "f1": 0.67, "accuracy": 0.75}],
        )
        model_name = list(VALID_MODELS.keys())[0]

        feedback_response = MagicMock()
        feedback_response.content = "Good progress."

        rec_response = MagicMock()
        rec_response.content = json.dumps({
            "features_to_add": VALID_FEATURES[:1],
            "model_config": {"model_name": model_name,
                             "hyperparameters": {}},
            "should_stop": False,
            "reasoning": "Keep iterating.",
        })

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [feedback_response, rec_response]
        mock_get_llm.return_value = mock_llm

        result = critic_node(state)
        assert len(result["history"]) == 2
        assert result["history"][0]["iteration"] == 0
        assert result["history"][1]["f1"] == 0.76
