"""Tests for app/ui/notebook_export.py — pure logic, no mocks."""

import json

import pytest

from app.ui.notebook_export import (
    _build_feature_code,
    _build_model_code,
    _make_cell,
    _parse_features_used,
    _parse_model_info,
    generate_notebook,
)


# ── _parse_features_used ─────────────────────────────────────────────────

class TestParseFeatures:
    def test_basic_extraction(self):
        report = (
            "Created: account_age_days = (timestamp - signup_date).days\n"
            "Created: hour_of_day = timestamp.hour\n"
        )
        assert _parse_features_used(report) == ["account_age_days", "hour_of_day"]

    def test_unknown_features_ignored(self):
        report = (
            "Created: account_age_days = ...\n"
            "Created: totally_fake_feature = ...\n"
            "Created: log_amount = log1p(amount)\n"
        )
        assert _parse_features_used(report) == ["account_age_days", "log_amount"]

    def test_empty_input(self):
        assert _parse_features_used("") == []

    def test_no_match_lines(self):
        report = "Using default feature set\nFinal feature count: 10\nShape: (20, 10)"
        assert _parse_features_used(report) == []


# ── _parse_model_info ────────────────────────────────────────────────────

class TestParseModelInfo:
    def test_logistic_regression(self):
        name, hp = _parse_model_info("Model: LogisticRegression(C=1.0)")
        assert name == "LogisticRegression"
        assert hp == {"C": 1.0}

    def test_random_forest(self):
        name, hp = _parse_model_info("Model: RandomForestClassifier(n_estimators=200, max_depth=10)")
        assert name == "RandomForestClassifier"
        assert hp == {"n_estimators": 200, "max_depth": 10}

    def test_gradient_boosting(self):
        name, hp = _parse_model_info("Model: GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5)")
        assert name == "GradientBoostingClassifier"
        assert hp == {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5}

    def test_svc(self):
        name, hp = _parse_model_info("Model: SVC(C=1.0, kernel=rbf)")
        assert name == "SVC"
        assert hp == {"C": 1.0, "kernel": "rbf"}

    def test_multiline_input(self):
        text = "Model: RandomForestClassifier(n_estimators=100)\nTrain size: 14, Test size: 4"
        name, hp = _parse_model_info(text)
        assert name == "RandomForestClassifier"
        assert hp == {"n_estimators": 100}

    def test_fallback_on_no_match(self):
        name, hp = _parse_model_info("Something unexpected")
        assert name == "LogisticRegression"
        assert hp == {"C": 1.0}


# ── _build_feature_code ──────────────────────────────────────────────────

class TestBuildFeatureCode:
    def test_no_dep_features(self):
        lines = _build_feature_code(["log_amount"])
        code = "\n".join(lines)
        assert "log_amount" in code
        assert "account_age_days" not in code

    def test_is_new_account_pulls_account_age_days(self):
        lines = _build_feature_code(["is_new_account"])
        code = "\n".join(lines)
        idx_dep = code.index("account_age_days")
        idx_feat = code.index("is_new_account")
        assert idx_dep < idx_feat

    def test_is_night_txn_pulls_hour_of_day(self):
        lines = _build_feature_code(["is_night_txn"])
        code = "\n".join(lines)
        assert "hour_of_day" in code
        idx_dep = code.index("# hour_of_day")
        idx_feat = code.index("# is_night_txn")
        assert idx_dep < idx_feat

    def test_is_weekend_pulls_day_of_week(self):
        lines = _build_feature_code(["is_weekend"])
        code = "\n".join(lines)
        assert "day_of_week" in code
        idx_dep = code.index("# day_of_week")
        idx_feat = code.index("# is_weekend")
        assert idx_dep < idx_feat

    def test_deduplication(self):
        lines = _build_feature_code(["hour_of_day", "is_night_txn", "hour_of_day"])
        code = "\n".join(lines)
        assert code.count("# hour_of_day") == 1

    def test_unknown_features_skipped(self):
        lines = _build_feature_code(["nonexistent_feat", "log_amount"])
        code = "\n".join(lines)
        assert "nonexistent_feat" not in code
        assert "log_amount" in code


# ── _build_model_code ────────────────────────────────────────────────────

class TestBuildModelCode:
    def test_logistic_regression(self):
        code = _build_model_code("LogisticRegression", {"C": 2.0})
        assert "LogisticRegression" in code
        assert "C=2.0" in code
        assert "max_iter=1000" in code

    def test_random_forest(self):
        code = _build_model_code("RandomForestClassifier", {"n_estimators": 200, "max_depth": 15})
        assert "RandomForestClassifier" in code
        assert "n_estimators=200" in code
        assert "max_depth=15" in code

    def test_gradient_boosting(self):
        code = _build_model_code("GradientBoostingClassifier", {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5})
        assert "GradientBoostingClassifier" in code
        assert "n_estimators=150" in code

    def test_svc(self):
        code = _build_model_code("SVC", {"C": 1.0, "kernel": "rbf"})
        assert "SVC" in code
        assert "kernel='rbf'" in code

    def test_unknown_fallback(self):
        code = _build_model_code("XGBClassifier", {})
        assert "RandomForestClassifier" in code


# ── _make_cell ───────────────────────────────────────────────────────────

class TestMakeCell:
    def test_code_cell_has_execution_and_outputs(self):
        cell = _make_cell("code", "x = 1", "c1")
        assert cell["cell_type"] == "code"
        assert "execution_count" in cell
        assert "outputs" in cell
        assert cell["source"] == "x = 1"
        assert cell["id"] == "c1"

    def test_markdown_cell_no_execution(self):
        cell = _make_cell("markdown", "# Title", "m1")
        assert cell["cell_type"] == "markdown"
        assert "execution_count" not in cell
        assert "outputs" not in cell


# ── generate_notebook ────────────────────────────────────────────────────

class TestGenerateNotebook:
    def test_valid_json(self, sample_history):
        nb_str = generate_notebook(sample_history, "is_fraud", "transactions.csv")
        nb = json.loads(nb_str)
        assert isinstance(nb, dict)

    def test_correct_nbformat(self, sample_history):
        nb = json.loads(generate_notebook(sample_history, "is_fraud"))
        assert nb["nbformat"] == 4
        assert nb["nbformat_minor"] == 5

    def test_best_iteration_selected(self, sample_history):
        """Best iteration is the one with highest F1 (iteration 1, F1=0.82)."""
        nb = json.loads(generate_notebook(sample_history, "is_fraud"))
        # Title cell should reference iteration 1
        title_cell = nb["cells"][0]
        assert "1" in title_cell["source"]  # best iteration = 1

    def test_scaling_included_for_svc(self):
        history = [{
            "iteration": 0,
            "f1": 0.70,
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.70,
            "model": "Model: SVC(C=1.0, kernel=rbf)\nTrain size: 14, Test size: 4",
            "feature_report": "Created: log_amount = log1p(amount)",
            "cleaning_report": "",
            "feedback": "",
            "model_obj": None,
            "feature_names": [],
            "recommendations": {},
        }]
        nb_str = generate_notebook(history, "is_fraud")
        assert "StandardScaler" in nb_str

    def test_scaling_included_for_lr(self):
        history = [{
            "iteration": 0,
            "f1": 0.70,
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.70,
            "model": "Model: LogisticRegression(C=1.0)\nTrain size: 14, Test size: 4",
            "feature_report": "Created: log_amount = log1p(amount)",
            "cleaning_report": "",
            "feedback": "",
            "model_obj": None,
            "feature_names": [],
            "recommendations": {},
        }]
        nb_str = generate_notebook(history, "is_fraud")
        assert "StandardScaler" in nb_str

    def test_no_scaling_for_rf(self):
        history = [{
            "iteration": 0,
            "f1": 0.80,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.80,
            "model": "Model: RandomForestClassifier(n_estimators=100)\nTrain size: 14, Test size: 4",
            "feature_report": "Created: log_amount = log1p(amount)",
            "cleaning_report": "",
            "feedback": "",
            "model_obj": None,
            "feature_names": [],
            "recommendations": {},
        }]
        nb_str = generate_notebook(history, "is_fraud")
        # Imports always include StandardScaler, but the actual scaling code should not appear
        assert "scaler.fit_transform" not in nb_str

    def test_cells_present(self, sample_history):
        nb = json.loads(generate_notebook(sample_history, "is_fraud"))
        assert len(nb["cells"]) >= 10  # title, imports, load, eda, cleaning, features, model, eval
