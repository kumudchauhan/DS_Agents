from langgraph.graph import END, StateGraph

from app.graph.nodes.cleaning import cleaning_node
from app.graph.nodes.critic import critic_node
from app.graph.nodes.decision import decision_node
from app.graph.nodes.eda import eda_node
from app.graph.nodes.evaluation import evaluation_node
from app.graph.nodes.feature_eng import feature_engineering_node
from app.graph.nodes.modeling import modeling_node
from app.graph.state import AgentState


def _route_decision(state: AgentState) -> str:
    """Return the next node name based on the decision node's output."""
    if state.get("should_continue", False):
        return "feature_engineering"
    return END


def build_graph():
    """Construct and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("eda", eda_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("feature_engineering", feature_engineering_node)
    graph.add_node("modeling", modeling_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("critic", critic_node)
    graph.add_node("decision", decision_node)

    # Sequential edges
    graph.set_entry_point("eda")
    graph.add_edge("eda", "cleaning")
    graph.add_edge("cleaning", "feature_engineering")
    graph.add_edge("feature_engineering", "modeling")
    graph.add_edge("modeling", "evaluation")
    graph.add_edge("evaluation", "critic")
    graph.add_edge("critic", "decision")

    # Conditional loop: improve or stop
    graph.add_conditional_edges(
        "decision",
        _route_decision,
        {
            "feature_engineering": "feature_engineering",
            END: END,
        },
    )

    return graph.compile()
