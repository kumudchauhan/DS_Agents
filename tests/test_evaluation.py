"""Tests for app/graph/nodes/evaluation.py."""

import numpy as np
import pytest

from app.graph.nodes.evaluation import evaluation_node


def _make_eval_state(y_test, y_pred_source="perfect"):
    """Build a minimal state for evaluation_node.

    y_pred_source controls what predictions the model makes:
    - "perfect": model predicts y_test exactly.
    """
    from sklearn.dummy import DummyClassifier

    if y_pred_source == "perfect":
        model = DummyClassifier(strategy="most_frequent")
        X_test = np.zeros((len(y_test), 2))
        model.fit(X_test, y_test)
        model.predict = lambda X: np.array(y_test)

    return {
        "model": model,
        "X_test": np.zeros((len(y_test), 2)),
        "y_test": np.array(y_test),
        "iteration": 0,
    }


class TestEvaluationNode:
    def test_perfect_predictions(self):
        state = _make_eval_state([0, 0, 1, 1, 0, 1])
        result = evaluation_node(state)
        assert result["metrics"]["accuracy"] == 1.0
        assert result["metrics"]["f1"] == 1.0

    def test_metric_keys(self):
        state = _make_eval_state([0, 1, 0, 1])
        result = evaluation_node(state)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in result["metrics"]

    def test_rounding(self):
        state = _make_eval_state([0, 1, 0, 1])
        result = evaluation_node(state)
        for key in ["accuracy", "precision", "recall", "f1"]:
            val = result["metrics"][key]
            assert val == round(val, 4)

    def test_report_content(self):
        state = _make_eval_state([0, 1, 0, 1])
        result = evaluation_node(state)
        report = result["evaluation_report"]
        assert "Accuracy" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report
        assert "Classification Report" in report
        assert "Confusion Matrix" in report

    def test_all_same_class(self):
        state = _make_eval_state([0, 0, 0, 0])
        result = evaluation_node(state)
        assert result["metrics"]["accuracy"] == 1.0
        assert result["metrics"]["f1"] == 0.0
