from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.graph.state import AgentState


def evaluation_node(state: AgentState) -> dict:
    """Score the trained model on the held-out test set."""
    print("\n" + "=" * 60)
    print("  EVALUATION NODE - Iteration", state["iteration"])
    print("=" * 60)

    model = state["model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    cm = confusion_matrix(y_test, y_pred)

    lines = [
        f"Accuracy:  {metrics['accuracy']}",
        f"Precision: {metrics['precision']}",
        f"Recall:    {metrics['recall']}",
        f"F1 Score:  {metrics['f1']}",
        f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}",
        f"Confusion Matrix:\n{cm}",
    ]

    report = "\n".join(lines)
    print(report)

    return {"metrics": metrics, "evaluation_report": report, "confusion_matrix": cm.tolist()}
