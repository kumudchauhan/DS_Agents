from app.graph.state import AgentState

F1_THRESHOLD = 0.85


def decision_node(state: AgentState) -> dict:
    """Decide whether to loop for another improvement iteration or stop."""
    print("\n" + "=" * 60)
    print("  DECISION NODE - Iteration", state["iteration"])
    print("=" * 60)

    iteration = state["iteration"]
    max_iterations = state.get("max_iterations", 3)
    f1 = state["metrics"].get("f1", 0)

    if f1 >= F1_THRESHOLD:
        print(f"F1 score {f1} meets threshold {F1_THRESHOLD}. Stopping.")
        should_continue = False
    elif iteration >= max_iterations - 1:
        print(f"Max iterations ({max_iterations}) reached. Stopping.")
        should_continue = False
    else:
        print(f"F1={f1} < {F1_THRESHOLD}. Continuing to iteration {iteration + 1}.")
        should_continue = True

    return {
        "should_continue": should_continue,
        "iteration": iteration + 1,
    }
