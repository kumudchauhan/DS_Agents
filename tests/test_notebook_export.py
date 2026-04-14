"""Tests for app/ui/notebook_export.py — pure logic, no mocks.

Feature names, model names, and dependency info are all derived from
module constants so the tests adapt when the registries change.
"""

import json

import pytest

from app.graph.nodes.critic import VALID_MODELS
from app.ui.notebook_export import (
    _FEATURE_CODE,
    _NEEDS_SCALING,
    _build_feature_code,
    _build_model_code,
    _make_cell,
    _parse_features_used,
    _parse_model_info,
    generate_notebook,
)

# Derive test parameters from source constants
_ALL_FEATURES = list(_FEATURE_CODE.keys())
_FEATURES_WITH_DEPS = [
    (name, deps) for name, (deps, _) in _FEATURE_CODE.items() if deps
]
_NO_SCALING_MODELS = sorted(set(VALID_MODELS.keys()) - _NEEDS_SCALING)

# Build model strings from VALID_MODELS for parse tests
_MODEL_STRINGS = {}
for _name, _params in VALID_MODELS.items():
    _parts = []
    for _param, _constraint in _params.items():
        if isinstance(_constraint, tuple):
            _parts.append(f"{_param}={_constraint[0]}")
        elif isinstance(_constraint, list):
            _parts.append(f"{_param}={_constraint[0]}")
    _MODEL_STRINGS[_name] = f"Model: {_name}({', '.join(_parts)})"


# ── _parse_features_used ─────────────────────────────────────────────────

@pytest.mark.parametrize("feature_name", _ALL_FEATURES)
def test_parse_features_recognizes_known(feature_name):
    report = f"Created: {feature_name} = some description"
    assert _parse_features_used(report) == [feature_name]


def test_parse_features_unknown_ignored():
    first, last = _ALL_FEATURES[0], _ALL_FEATURES[-1]
    report = (
        f"Created: {first} = ...\n"
        "Created: totally_fake_feature = ...\n"
        f"Created: {last} = ...\n"
    )
    result = _parse_features_used(report)
    assert "totally_fake_feature" not in result
    assert first in result
    assert last in result


def test_parse_features_empty_input():
    assert _parse_features_used("") == []


def test_parse_features_no_match_lines():
    report = "Using default feature set\nFinal feature count: 10\nShape: (20, 10)"
    assert _parse_features_used(report) == []


# ── _parse_model_info ────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", list(VALID_MODELS.keys()))
def test_parse_model_info_extracts_name(model_name):
    name, hp = _parse_model_info(_MODEL_STRINGS[model_name])
    assert name == model_name
    assert isinstance(hp, dict)


def test_parse_model_info_multiline():
    model_name = list(VALID_MODELS.keys())[0]
    text = f"{_MODEL_STRINGS[model_name]}\nTrain size: 14, Test size: 4"
    name, _ = _parse_model_info(text)
    assert name == model_name


def test_parse_model_info_fallback():
    name, hp = _parse_model_info("Something unexpected")
    assert isinstance(name, str)
    assert isinstance(hp, dict)


# ── _build_feature_code ──────────────────────────────────────────────────

@pytest.mark.parametrize("feature_name", _ALL_FEATURES)
def test_build_feature_code_includes_feature(feature_name):
    lines = _build_feature_code([feature_name])
    code = "\n".join(lines)
    assert f"# {feature_name}" in code


@pytest.mark.parametrize(
    "feature_name,deps", _FEATURES_WITH_DEPS,
    ids=[f[0] for f in _FEATURES_WITH_DEPS],
)
def test_build_feature_code_dependency_ordering(feature_name, deps):
    lines = _build_feature_code([feature_name])
    code = "\n".join(lines)
    for dep in deps:
        assert f"# {dep}" in code
        dep_idx = code.index(f"# {dep}")
        feat_idx = code.index(f"# {feature_name}")
        assert dep_idx < feat_idx


def test_build_feature_code_deduplication():
    feat = _ALL_FEATURES[0]
    lines = _build_feature_code([feat, feat, feat])
    code = "\n".join(lines)
    assert code.count(f"# {feat}") == 1


def test_build_feature_code_unknown_skipped():
    known = _ALL_FEATURES[0]
    lines = _build_feature_code(["nonexistent_feat", known])
    code = "\n".join(lines)
    assert "nonexistent_feat" not in code
    assert f"# {known}" in code


# ── _build_model_code ────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", list(VALID_MODELS.keys()))
def test_build_model_code_contains_model_name(model_name):
    code = _build_model_code(model_name, {})
    assert model_name in code


def test_build_model_code_unknown_fallback():
    code = _build_model_code("NonexistentModel", {})
    assert code.startswith("model = ")


# ── _make_cell ───────────────────────────────────────────────────────────

def test_make_cell_code_has_execution_and_outputs():
    cell = _make_cell("code", "x = 1", "c1")
    assert cell["cell_type"] == "code"
    assert "execution_count" in cell
    assert "outputs" in cell
    assert cell["source"] == "x = 1"
    assert cell["id"] == "c1"


def test_make_cell_markdown_no_execution():
    cell = _make_cell("markdown", "# Title", "m1")
    assert cell["cell_type"] == "markdown"
    assert "execution_count" not in cell
    assert "outputs" not in cell


# ── generate_notebook ────────────────────────────────────────────────────

def _make_single_history(model_name):
    """Build minimal single-iteration history for a given model."""
    feat = _ALL_FEATURES[0]
    return [{
        "iteration": 0,
        "f1": 0.70,
        "accuracy": 0.75,
        "precision": 0.70,
        "recall": 0.70,
        "model": f"Model: {model_name}(C=1.0)\nTrain size: 14, Test size: 4",
        "feature_report": f"Created: {feat} = description",
        "cleaning_report": "",
        "feedback": "",
        "model_obj": None,
        "feature_names": [],
        "recommendations": {},
    }]


def test_generate_notebook_valid_json(sample_history):
    nb_str = generate_notebook(sample_history, "target", "data.csv")
    nb = json.loads(nb_str)
    assert isinstance(nb, dict)


def test_generate_notebook_correct_nbformat(sample_history):
    nb = json.loads(generate_notebook(sample_history, "target"))
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5


def test_generate_notebook_best_iteration_selected(sample_history):
    nb = json.loads(generate_notebook(sample_history, "target"))
    best = max(sample_history, key=lambda h: h.get("f1", 0))
    title_cell = nb["cells"][0]
    assert str(best["iteration"]) in title_cell["source"]


@pytest.mark.parametrize("model_name", sorted(_NEEDS_SCALING))
def test_scaling_included_for_model(model_name):
    nb_str = generate_notebook(_make_single_history(model_name), "target")
    assert "scaler.fit_transform" in nb_str


@pytest.mark.parametrize("model_name", _NO_SCALING_MODELS)
def test_no_scaling_for_model(model_name):
    nb_str = generate_notebook(_make_single_history(model_name), "target")
    assert "scaler.fit_transform" not in nb_str


def test_generate_notebook_has_cells(sample_history):
    nb = json.loads(generate_notebook(sample_history, "target"))
    assert len(nb["cells"]) >= 10
