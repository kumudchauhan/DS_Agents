import json

from app.graph.state import AgentState
from app.llm.llm_provider import get_llm

# ---------------------------------------------------------------------------
# Registries (bounded menus the LLM can pick from)
# ---------------------------------------------------------------------------

VALID_FEATURES = [
    "account_age_days", "hour_of_day", "day_of_week", "amount_to_avg_ratio",
    "is_high_amount", "log_amount", "is_new_account", "is_night_txn",
    "amount_deviation", "high_velocity", "amount_squared",
    "amount_x_velocity", "is_weekend", "txn_per_avg_ratio",
]

VALID_MODELS = {
    "LogisticRegression": {"C": (0.01, 10.0)},
    "RandomForestClassifier": {"n_estimators": (50, 300), "max_depth": (5, 30)},
    "GradientBoostingClassifier": {
        "n_estimators": (50, 300),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
    },
    "SVC": {"C": (0.1, 10.0), "kernel": ["rbf", "linear"]},
}

DEFAULT_RECOMMENDATIONS = {
    "features_to_add": VALID_FEATURES[:7],
    "model_config": {
        "model_name": "RandomForestClassifier",
        "hyperparameters": {"n_estimators": 100},
    },
    "should_stop": False,
    "reasoning": "Fallback defaults applied.",
}


# ---------------------------------------------------------------------------
# JSON parsing & validation helpers
# ---------------------------------------------------------------------------

def _parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _clamp(val, lo, hi):
    return type(lo)(max(lo, min(hi, val)))


def _validate_recommendations(raw: dict) -> dict:
    """Validate and clamp LLM recommendations to safe values."""
    result = {
        "features_to_add": list(DEFAULT_RECOMMENDATIONS["features_to_add"]),
        "model_config": dict(DEFAULT_RECOMMENDATIONS["model_config"]),
        "should_stop": False,
        "reasoning": raw.get("reasoning", ""),
    }

    # Features — filter to known set.
    if "features_to_add" in raw and isinstance(raw["features_to_add"], list):
        valid = [f for f in raw["features_to_add"] if f in VALID_FEATURES]
        if valid:
            result["features_to_add"] = valid

    # Model config — validate name and clamp hyperparameters.
    if "model_config" in raw and isinstance(raw["model_config"], dict):
        mc = raw["model_config"]
        name = mc.get("model_name", "")
        if name in VALID_MODELS:
            result["model_config"]["model_name"] = name
            hp = mc.get("hyperparameters", {})
            if isinstance(hp, dict):
                clamped = {}
                for param, constraint in VALID_MODELS[name].items():
                    if param not in hp:
                        continue
                    val = hp[param]
                    if isinstance(constraint, tuple) and len(constraint) == 2:
                        if isinstance(val, (int, float)):
                            clamped[param] = _clamp(val, constraint[0], constraint[1])
                    elif isinstance(constraint, list):
                        if val in constraint:
                            clamped[param] = val
                result["model_config"]["hyperparameters"] = clamped

    # Should stop.
    if "should_stop" in raw and isinstance(raw["should_stop"], bool):
        result["should_stop"] = raw["should_stop"]

    return result


# ---------------------------------------------------------------------------
# Instructions threading
# ---------------------------------------------------------------------------

def _format_instructions_context(state: AgentState) -> str:
    """Build an instruction-context block for the LLM prompts."""
    instructions = state.get("instructions") or {}
    if not instructions:
        return ""

    parts: list[str] = []

    priorities = instructions.get("priorities", [])
    if priorities:
        parts.append("User priorities: " + "; ".join(priorities))

    features = instructions.get("features", {})
    must = features.get("must_include", [])
    avoid_f = features.get("avoid", [])
    if must:
        parts.append("User wants these features included: " + ", ".join(must))
    if avoid_f:
        parts.append("User wants to avoid these features: " + ", ".join(avoid_f))

    models = instructions.get("models", {})
    preferred = models.get("preferred", [])
    avoid_m = models.get("avoid", [])
    notes = models.get("notes", [])
    if preferred:
        parts.append("User prefers these models: " + ", ".join(preferred))
    if avoid_m:
        parts.append("User wants to avoid these models: " + ", ".join(avoid_m))
    if notes:
        parts.append("Model notes: " + "; ".join(notes))

    if not parts:
        return ""

    return "\n\n## User Instructions\n" + "\n".join(f"- {p}" for p in parts)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_feedback_prompt(state: AgentState) -> str:
    """Human-readable feedback prompt (displayed in UI)."""
    metrics = state["metrics"]
    eda_report = state.get("eda_report", "N/A")
    feature_report = state.get("feature_report", "N/A")
    model_report = state.get("model_report", "N/A")
    evaluation_report = state.get("evaluation_report", "N/A")
    iteration = state["iteration"]
    history = state.get("history", [])

    history_text = ""
    if history:
        for h in history:
            history_text += (
                f"\n  Iteration {h['iteration']}: "
                f"F1={h['f1']}, Accuracy={h['accuracy']}"
            )

    return f"""You are an expert data scientist reviewing a machine learning pipeline.

## Current Iteration: {iteration}
## Previous Results:{history_text if history_text else " None (first iteration)"}

## EDA Summary:
{eda_report[:500]}

## Features Created:
{feature_report[:500]}

## Model Used:
{model_report}

## Evaluation Results:
{evaluation_report}

## Current Metrics:
- Accuracy: {metrics.get('accuracy')}
- Precision: {metrics.get('precision')}
- Recall: {metrics.get('recall')}
- F1 Score: {metrics.get('f1')}

Based on these results, provide specific and actionable feedback to improve
the model's performance. Focus on:
1. Feature engineering improvements
2. Model selection or hyperparameter suggestions
3. Data quality issues to address
4. Class imbalance handling

Keep your response concise (3-5 bullet points).
{_format_instructions_context(state)}"""


def _build_recommendation_prompt(state: AgentState) -> str:
    """Structured JSON prompt for actionable recommendations."""
    metrics = state["metrics"]
    feature_report = state.get("feature_report", "N/A")
    model_report = state.get("model_report", "N/A")
    evaluation_report = state.get("evaluation_report", "N/A")
    iteration = state["iteration"]
    history = state.get("history", [])

    history_text = ""
    for h in history:
        history_text += (
            f"\n  Iteration {h['iteration']}: F1={h['f1']}, "
            f"Accuracy={h['accuracy']}, Model={h.get('model', 'N/A')}"
        )

    # Parse features already used this iteration.
    used_features = []
    for line in feature_report.split("\n"):
        if "Created:" in line:
            # Extract the feature name (before the = sign).
            part = line.split("Created:")[-1].strip()
            name = part.split("=")[0].strip()
            if name in VALID_FEATURES:
                used_features.append(name)
    available = [f for f in VALID_FEATURES if f not in used_features]

    return f"""You are an expert ML engineer. Analyze these results and recommend the next iteration's strategy.

## Current Iteration: {iteration}
## Previous Results:{history_text if history_text else " None (first iteration)"}

## Current Metrics:
- Accuracy: {metrics.get('accuracy')}
- Precision: {metrics.get('precision')}
- Recall: {metrics.get('recall')}
- F1 Score: {metrics.get('f1')}

## Features Used This Iteration:
{used_features}

## Additional Features Available:
{available}

## Current Model:
{model_report}

## Evaluation Details:
{evaluation_report[:500]}

## Available Models and Hyperparameter Ranges:
- LogisticRegression: C (0.01-10)
- RandomForestClassifier: n_estimators (50-300), max_depth (5-30 or null)
- GradientBoostingClassifier: n_estimators (50-300), learning_rate (0.01-0.3), max_depth (3-10)
- SVC: C (0.1-10), kernel (rbf or linear)

Respond with ONLY a JSON object (no markdown, no code fences). Use this exact schema:
{{
    "features_to_add": ["feature1", "feature2"],
    "model_config": {{
        "model_name": "ModelClassName",
        "hyperparameters": {{"param": value}}
    }},
    "should_stop": false,
    "reasoning": "Brief explanation of your strategy"
}}

Rules:
- features_to_add: include features from the used list that should be KEPT, plus new ones to ADD
- model_name must be one of: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC
- hyperparameters must be within the ranges given above
- should_stop: true only if F1 >= 0.85 or further improvement is unlikely
- If recall is low, prioritize features and models that improve recall
- If precision is low, consider features that reduce false positives
{_format_instructions_context(state)}"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def critic_node(state: AgentState) -> dict:
    """Use LLM to critique results and produce structured recommendations."""
    print("\n" + "=" * 60)
    print("  CRITIC NODE (LLM) - Iteration", state["iteration"])
    print("=" * 60)

    metrics = state["metrics"]
    iteration = state["iteration"]
    history = state.get("history", [])
    model_report = state.get("model_report", "N/A")
    feature_report = state.get("feature_report", "N/A")

    # ── 1) Human-readable feedback (for UI display) ──────────────────
    try:
        llm = get_llm()
        response = llm.invoke(_build_feedback_prompt(state))
        feedback = response.content
    except Exception as e:
        feedback = (
            f"[LLM unavailable: {e}]\n"
            "Fallback suggestions:\n"
            "- Try adding interaction features\n"
            "- Consider class imbalance (use SMOTE or class_weight)\n"
            "- Try a different model (GradientBoosting)\n"
            "- Add time-based features (hour, day of week)\n"
        )

    print(f"\nFeedback:\n{feedback}")

    # ── 2) Structured JSON recommendations ────────────────────────────
    recommendations = dict(DEFAULT_RECOMMENDATIONS)
    try:
        llm = get_llm()
        rec_response = llm.invoke(_build_recommendation_prompt(state))
        raw = _parse_json_from_response(rec_response.content)
        recommendations = _validate_recommendations(raw)
        print(f"\nRecommendations: {json.dumps(recommendations, indent=2)}")
    except Exception as e:
        print(f"\nFailed to parse recommendations ({e}). Using defaults.")

    # ── 3) Build history entry ────────────────────────────────────────
    model_obj = state.get("model")
    df = state.get("dataframe")
    target = state.get("target_column", "")
    feature_names = [c for c in df.columns if c != target] if df is not None else []

    new_history = list(history) + [
        {
            "iteration": iteration,
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "model": model_report,
            "feedback": feedback,
            "feature_report": feature_report,
            "cleaning_report": state.get("cleaning_report", ""),
            "model_obj": model_obj,
            "feature_names": feature_names,
            "recommendations": recommendations,
            "confusion_matrix": state.get("confusion_matrix"),
            "evaluation_report": state.get("evaluation_report", ""),
        }
    ]

    return {
        "feedback": feedback,
        "history": new_history,
        "recommendations": recommendations,
    }
