"""Wrap the LangGraph pipeline with streaming progress updates for Streamlit.

Single pipeline that yields per-node updates so the UI can render analysis
results (EDA, cleaning) immediately while modeling continues in the same run.
"""

from app.graph.builder import build_graph
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


def run_pipeline(dataset_path: str, target_column: str, max_iterations: int):
    """Stream the full pipeline (analysis + modeling) node by node.

    Yields
    ------
    dict with keys:
        node        – node name (e.g. "eda", "modeling")
        label       – human-readable label
        phase       – "analysis" | "modeling"
        iteration   – current modeling iteration (0 during analysis)
        state       – accumulated state dict snapshot
    """
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
        "history": [],
        "should_continue": True,
    }

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
