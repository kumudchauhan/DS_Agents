"""Entry point for the autonomous data science agent."""

from app.graph.builder import build_graph
from app.graph.state import AgentState


def run_agent(
    dataset_path: str,
    target_column: str = "is_fraud",
    max_iterations: int = 3,
):
    print("=" * 60)
    print("  AUTONOMOUS DATA SCIENCE AGENT")
    print("=" * 60)
    print(f"  Dataset:        {dataset_path}")
    print(f"  Target column:  {target_column}")
    print(f"  Max iterations: {max_iterations}")
    print("=" * 60)

    graph = build_graph()

    initial_state: AgentState = {
        "dataset_path": dataset_path,
        "dataframe": None,
        "cleaned_dataframe": None,
        "target_column": target_column,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "model": None,
        "metrics": {},
        "feedback": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "eda_report": "",
        "cleaning_report": "",
        "feature_report": "",
        "model_report": "",
        "evaluation_report": "",
        "confusion_matrix": None,
        "history": [],
        "should_continue": True,
        "recommendations": {},
        "instructions": {},
        "pipeline_mode": "full",
        "da_agent_result": {},
    }

    result = graph.invoke(initial_state)

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)

    history = result.get("history", [])
    for h in history:
        print(
            f"  Iteration {h['iteration']}:  "
            f"F1={h['f1']}  Accuracy={h['accuracy']}  "
            f"Precision={h['precision']}  Recall={h['recall']}"
        )

    if history:
        best = max(history, key=lambda h: h["f1"])
        print(f"\n  Best F1: {best['f1']} (iteration {best['iteration']})")

    print("=" * 60)
    return result


if __name__ == "__main__":
    run_agent("data/transactions.csv")
