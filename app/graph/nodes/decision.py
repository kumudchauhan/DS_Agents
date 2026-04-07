from app.graph.state import AgentState

F1_THRESHOLD = 0.85


def decision_node(state: AgentState) -> dict:
    """Decide whether to loop or stop, informed by LLM recommendations."""
    print("\n" + "=" * 60)
    print("  DECISION NODE - Iteration", state["iteration"])
    print("=" * 60)

    iteration = state["iteration"]
    max_iterations = state.get("max_iterations", 3)
    f1 = state["metrics"].get("f1", 0)
    recommendations = state.get("recommendations") or {}

    # Hard stop: max iterations reached.
    if iteration >= max_iterations - 1:
        print(f"Max iterations ({max_iterations}) reached. Stopping.")
        return {"should_continue": False, "iteration": iteration + 1}

    # Hard stop: F1 threshold met.
    if f1 >= F1_THRESHOLD:
        print(f"F1={f1} meets threshold {F1_THRESHOLD}. Stopping.")
        return {"should_continue": False, "iteration": iteration + 1}

    # Soft stop: LLM recommends stopping (only trust after iteration 0).
    if recommendations.get("should_stop", False) and iteration >= 1:
        reasoning = recommendations.get("reasoning", "")
        print(f"LLM recommends stopping: {reasoning}")
        return {"should_continue": False, "iteration": iteration + 1}

    print(f"F1={f1} < {F1_THRESHOLD}. Continuing to iteration {iteration + 1}.")
    return {"should_continue": True, "iteration": iteration + 1}
