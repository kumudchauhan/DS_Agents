from app.graph.state import AgentState
from app.llm.llm_provider import get_llm


def critic_node(state: AgentState) -> dict:
    """Use a local LLM to critique pipeline results and suggest improvements."""
    print("\n" + "=" * 60)
    print("  CRITIC NODE (LLM) - Iteration", state["iteration"])
    print("=" * 60)

    metrics = state["metrics"]
    eda_report = state.get("eda_report", "N/A")
    feature_report = state.get("feature_report", "N/A")
    model_report = state.get("model_report", "N/A")
    evaluation_report = state.get("evaluation_report", "N/A")
    iteration = state["iteration"]
    history = state.get("history", [])

    # Build history context
    history_text = ""
    if history:
        for h in history:
            history_text += (
                f"\n  Iteration {h['iteration']}: "
                f"F1={h['f1']}, Accuracy={h['accuracy']}"
            )

    prompt = f"""You are an expert data scientist reviewing a machine learning pipeline.

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
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
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

    new_history = list(history) + [
        {
            "iteration": iteration,
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "model": model_report,
            "feedback": feedback,
        }
    ]

    return {"feedback": feedback, "history": new_history}
