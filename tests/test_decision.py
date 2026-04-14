"""Tests for app/graph/nodes/decision.py."""

import pytest

from app.graph.nodes.decision import decision_node, F1_THRESHOLD


def _make_decision_state(**overrides):
    base = {
        "iteration": 0,
        "max_iterations": 3,
        "metrics": {"f1": 0.50},
        "recommendations": {},
    }
    base.update(overrides)
    return base


class TestDecisionNode:
    def test_stop_on_max_iterations(self):
        state = _make_decision_state(iteration=2, max_iterations=3, metrics={"f1": 0.50})
        result = decision_node(state)
        assert result["should_continue"] is False

    def test_stop_on_f1_threshold(self):
        state = _make_decision_state(iteration=0, metrics={"f1": 0.90})
        result = decision_node(state)
        assert result["should_continue"] is False

    def test_continue_below_threshold(self):
        state = _make_decision_state(iteration=0, metrics={"f1": 0.50})
        result = decision_node(state)
        assert result["should_continue"] is True

    def test_llm_stop_ignored_at_iter_0(self):
        state = _make_decision_state(
            iteration=0,
            metrics={"f1": 0.50},
            recommendations={"should_stop": True},
        )
        result = decision_node(state)
        # LLM stop not trusted at iteration 0
        assert result["should_continue"] is True

    def test_llm_stop_honored_at_iter_1(self):
        state = _make_decision_state(
            iteration=1,
            metrics={"f1": 0.50},
            recommendations={"should_stop": True, "reasoning": "Converged."},
        )
        result = decision_node(state)
        assert result["should_continue"] is False

    def test_iteration_always_incremented(self):
        state = _make_decision_state(iteration=0, metrics={"f1": 0.50})
        result = decision_node(state)
        assert result["iteration"] == 1

        state2 = _make_decision_state(iteration=2, max_iterations=3, metrics={"f1": 0.50})
        result2 = decision_node(state2)
        assert result2["iteration"] == 3
