"""Streamlit UI for the Autonomous Data Science Agent."""

import tempfile

import pandas as pd
import streamlit as st

from app.ui.pipeline_runner import (
    run_pipeline,
    ANALYSIS_NODES,
    MODELING_NODES,
    NODE_LABELS,
)
from app.ui.components import (
    render_data_quality_summary,
    render_eda_visualizations,
    render_cleaning_summary,
    render_feature_summary,
    render_feature_importance,
    render_iteration_metrics_chart,
    render_key_takeaways,
)
from app.ui.qa import ask_question_about_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DS Agent", layout="wide")
st.title("Autonomous Data Science Agent")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "uploaded_df": None,
    "dataset_path": None,
    "target_column": None,
    "agent_result": None,
    "agent_history": [],
    "qa_messages": [],
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Step-status icons
# ---------------------------------------------------------------------------
_ICON_DONE = ":white_check_mark:"
_ICON_RUN = ":hourglass_flowing_sand:"
_ICON_PENDING = ":white_circle:"


def _step_line(label: str, status: str) -> str:
    icon = {"done": _ICON_DONE, "running": _ICON_RUN, "pending": _ICON_PENDING}[status]
    return f"{icon}  **{label}**"


# ===================================================================
# Section 1 — Upload Dataset
# ===================================================================
st.header("1. Upload Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.session_state.uploaded_df is None or st.session_state.dataset_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        st.session_state.dataset_path = tmp.name
        st.session_state.uploaded_df = pd.read_csv(tmp.name)
        st.session_state.agent_result = None
        st.session_state.agent_history = []

    df = st.session_state.uploaded_df

    with st.expander("Dataset preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), width="stretch")

    st.markdown(f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns")

    target_col = st.selectbox(
        "Select target column",
        options=list(df.columns),
        index=list(df.columns).index("is_fraud") if "is_fraud" in df.columns else 0,
    )
    st.session_state.target_column = target_col
else:
    st.info("Upload a CSV file to get started.")

# ===================================================================
# Section 2 — Run Pipeline
# ===================================================================
if st.session_state.dataset_path is not None:
    st.header("2. Run Pipeline")

    max_iter = st.number_input(
        "Max iterations", min_value=1, max_value=10, value=3, step=1
    )

    if st.button("Run Pipeline"):
        # -- Reserve layout slots so analysis results appear ABOVE the
        #    modeling progress tracker.
        analysis_status = st.status("Running data analysis...", expanded=True)
        analysis_steps = analysis_status.empty()

        exploration_container = st.container()

        modeling_status = st.status("Waiting for analysis to finish...", expanded=False)
        modeling_steps = modeling_status.empty()

        # -- Tracking state --
        analysis_finished: set[str] = set()
        modeling_finished_iter: set[str] = set()
        current_iter = 0
        analysis_rendered = False
        result_state = None

        for update in run_pipeline(
            dataset_path=st.session_state.dataset_path,
            target_column=st.session_state.target_column,
            max_iterations=max_iter,
        ):
            node = update["node"]
            phase = update["phase"]
            result_state = update["state"]

            # ── Analysis phase ────────────────────────────────────
            if phase == "analysis":
                analysis_finished.add(node)
                lines = []
                for n in ANALYSIS_NODES:
                    lbl = NODE_LABELS[n]
                    if n in analysis_finished:
                        lines.append(_step_line(lbl, "done"))
                    else:
                        lines.append(_step_line(lbl, "pending"))
                analysis_steps.markdown("\n\n".join(lines))

                # Once cleaning is done, render exploration results immediately
                if node == "cleaning" and not analysis_rendered:
                    analysis_status.update(
                        label="Data analysis complete",
                        state="complete",
                        expanded=False,
                    )
                    with exploration_container:
                        st.header("3. Data Exploration")
                        tab_quality, tab_viz = st.tabs(
                            ["Data Quality Report", "Visualizations"]
                        )
                        with tab_quality:
                            render_data_quality_summary(
                                st.session_state.uploaded_df,
                                st.session_state.target_column,
                            )
                        with tab_viz:
                            render_eda_visualizations()
                    analysis_rendered = True

                    # Activate modeling tracker
                    modeling_status.update(
                        label="Running model pipeline...",
                        state="running",
                        expanded=True,
                    )
                continue

            # ── Modeling phase ────────────────────────────────────
            iteration = update.get("iteration", 0)
            if iteration != current_iter:
                current_iter = iteration
                modeling_finished_iter = set()

            modeling_finished_iter.add(node)

            lines = [f"**Iteration {current_iter}**"]
            for n in MODELING_NODES:
                lbl = NODE_LABELS[n]
                if n in modeling_finished_iter:
                    lines.append(_step_line(lbl, "done"))
                else:
                    lines.append(_step_line(lbl, "pending"))
            modeling_steps.markdown("\n\n".join(lines))

        # ── Pipeline finished ─────────────────────────────────────
        modeling_status.update(
            label="Model pipeline complete",
            state="complete",
            expanded=False,
        )

        if result_state:
            st.session_state.agent_result = result_state
            st.session_state.agent_history = result_state.get("history", [])
        st.rerun()

# ===================================================================
# Section 3 — Data Exploration (persisted after pipeline run)
# ===================================================================
if st.session_state.agent_result is not None:
    st.header("3. Data Exploration")
    tab_quality, tab_viz = st.tabs(["Data Quality Report", "Visualizations"])
    with tab_quality:
        render_data_quality_summary(
            st.session_state.uploaded_df, st.session_state.target_column
        )
    with tab_viz:
        render_eda_visualizations()

# ===================================================================
# Section 4 — Pipeline Details (Cleaning, Features, Importance)
# ===================================================================
if st.session_state.agent_history:
    st.header("4. Pipeline Details")

    tab_cleaning, tab_features, tab_importance = st.tabs(
        ["Data Cleaning", "Feature Engineering", "Feature Importance"]
    )

    # Use first iteration for cleaning (same across all iterations)
    first_hist = st.session_state.agent_history[0]

    with tab_cleaning:
        render_cleaning_summary(first_hist.get("cleaning_report", ""))

    with tab_features:
        for h in st.session_state.agent_history:
            with st.expander(
                f"Iteration {h['iteration']} features", expanded=(h == first_hist)
            ):
                render_feature_summary(
                    h.get("feature_report", ""), h["iteration"]
                )

    with tab_importance:
        render_feature_importance(st.session_state.agent_history)

# ===================================================================
# Section 5 — Model Results
# ===================================================================
if st.session_state.agent_history:
    st.header("5. Model Results")

    tab_metrics, tab_takeaways = st.tabs(["Metrics Comparison", "Key Takeaways"])

    with tab_metrics:
        render_iteration_metrics_chart(st.session_state.agent_history)

    with tab_takeaways:
        render_key_takeaways(st.session_state.agent_history)

    st.subheader("LLM Critic Feedback")
    for h in st.session_state.agent_history:
        with st.expander(f"Iteration {h['iteration']} feedback"):
            st.markdown(h.get("feedback", "_No feedback recorded._"))

# ===================================================================
# Section 6 — Ask Questions (NL Q&A)
# ===================================================================
if st.session_state.uploaded_df is not None:
    st.header("6. Ask Questions About Your Data")

    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question about the dataset...")

    if user_question:
        st.session_state.qa_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question_about_data(
                    user_question, st.session_state.uploaded_df
                )
            st.markdown(answer)

        st.session_state.qa_messages.append({"role": "assistant", "content": answer})
