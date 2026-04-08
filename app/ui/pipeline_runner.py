"""Wrap the LangGraph pipeline with streaming progress updates for Streamlit.

Single pipeline that yields per-node updates so the UI can render analysis
results (EDA, cleaning) immediately while modeling continues in the same run.
"""

from __future__ import annotations

from app.graph.builder import build_graph, build_ds_only_graph
from app.graph.state import AgentState

# ── Node metadata ──────────────────────────────────────────────────────────

ANALYSIS_NODES = ["eda", "cleaning"]
MODELING_NODES = ["feature_engineering", "modeling", "evaluation", "critic", "decision"]
ALL_NODES = ANALYSIS_NODES + MODELING_NODES

NODE_LABELS = {
    "eda": "Exploratory Data Analysis",
    "cleaning": "Data Cleaning",
    "feature_engineering": "Feature Engineering",
    "modeling": "Model Training",
    "evaluation": "Model Evaluation",
    "critic": "LLM Critic Review",
    "decision": "Iteration Decision",
}


def _build_initial_state(
    dataset_path: str,
    target_column: str,
    max_iterations: int,
    instructions: dict | None = None,
) -> AgentState:
    """Build the initial AgentState dict with all required fields."""
    return {
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
        "instructions": instructions or {},
        "pipeline_mode": "full",
        "da_agent_result": {},
    }


def run_pipeline(
    dataset_path: str,
    target_column: str,
    max_iterations: int,
    instructions: dict | None = None,
):
    """Stream the full pipeline (analysis + modeling) node by node.

    Yields
    ------
    dict with keys:
        node        - node name (e.g. "eda", "modeling")
        label       - human-readable label
        phase       - "analysis" | "modeling"
        iteration   - current modeling iteration (0 during analysis)
        state       - accumulated state dict snapshot
    """
    graph = build_graph()

    initial_state = _build_initial_state(
        dataset_path, target_column, max_iterations, instructions
    )

    accumulated: dict = dict(initial_state)

    for chunk in graph.stream(initial_state):
        for node_name, state_update in chunk.items():
            if isinstance(state_update, dict):
                accumulated.update(state_update)

            phase = "analysis" if node_name in ANALYSIS_NODES else "modeling"

            yield {
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
                "phase": phase,
                "iteration": accumulated.get("iteration", 0),
                "state": dict(accumulated),
            }


def run_pipeline_from_da(
    da_result: dict,
    target_column: str,
    max_iterations: int,
    instructions: dict | None = None,
):
    """Run DS Agent starting from feature_engineering, using DA Agent's cleaned data.

    Builds a modified graph that starts at feature_engineering (skips EDA + cleaning).
    Seeds initial state with DA Agent's outputs.

    Yields
    ------
    dict with keys:
        node        - node name
        label       - human-readable label
        phase       - always "modeling" (analysis was done by DA Agent)
        iteration   - current modeling iteration
        state       - accumulated state dict snapshot
    """
    graph = build_ds_only_graph()

    initial_state = _build_initial_state(
        da_result.get("dataset_path", ""),
        target_column,
        max_iterations,
        instructions,
    )

    # Seed with DA Agent outputs
    initial_state["dataframe"] = da_result["cleaned_dataframe"]
    initial_state["cleaned_dataframe"] = da_result["cleaned_dataframe"]
    initial_state["eda_report"] = da_result.get("eda_report", "")
    initial_state["cleaning_report"] = da_result.get("cleaning_report", "")
    initial_state["pipeline_mode"] = "lite"
    initial_state["da_agent_result"] = da_result

    accumulated: dict = dict(initial_state)

    for chunk in graph.stream(initial_state):
        for node_name, state_update in chunk.items():
            if isinstance(state_update, dict):
                accumulated.update(state_update)

            yield {
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
                "phase": "modeling",
                "iteration": accumulated.get("iteration", 0),
                "state": dict(accumulated),
            }
