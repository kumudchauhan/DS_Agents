"""Streamlit UI for the Autonomous Data Science Agent."""

import tempfile

import pandas as pd
import streamlit as st

from app.ui.pipeline_runner import (
    run_pipeline,
    run_pipeline_from_da,
    ANALYSIS_NODES,
    MODELING_NODES,
    NODE_LABELS,
)
from app.ui.components import (
    render_data_profile,
    render_data_quality_summary,
    render_eda_visualizations,
    render_cleaning_summary,
    render_feature_summary,
    render_feature_importance,
    render_iteration_metrics_chart,
    render_key_takeaways,
    render_model_interpretation,
)
from app.ui.instructions_parser import parse_instructions

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
    "da_agent_result": None,
    "instructions": {},
    "pipeline_mode": "full",
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
# Sidebar — Pipeline Mode, Instructions, LLM Provider
# ===================================================================
with st.sidebar:
    st.header("Pipeline Settings")

    pipeline_mode = st.radio(
        "Pipeline Mode",
        options=["Quick Explore (DA Agent)", "End-to-End Pipeline"],
        index=0 if st.session_state.pipeline_mode == "lite" else 1,
        help="Quick Explore: fast data analysis, no modeling. "
             "End-to-End Pipeline: analysis + iterative modeling with LLM critic.",
    )
    st.session_state.pipeline_mode = "lite" if "Quick" in pipeline_mode else "full"

    st.divider()

    # Instructions input
    st.subheader("Instructions (optional)")
    instructions_method = st.radio(
        "Provide instructions via:",
        options=["Paste text", "Upload .md file"],
        label_visibility="collapsed",
    )

    instructions_text = ""
    if instructions_method == "Upload .md file":
        md_file = st.file_uploader("Upload instructions.md", type=["md", "txt"])
        if md_file is not None:
            instructions_text = md_file.getvalue().decode("utf-8")
    else:
        instructions_text = st.text_area(
            "Paste instructions markdown",
            height=150,
            placeholder=(
                "## Priorities\n"
                "- Recall matters more than precision\n\n"
                "## Features\n"
                "- Must include: log_amount, account_age_days\n"
                "- Avoid: amount_squared\n\n"
                "## Models\n"
                "- Preferred: GradientBoostingClassifier"
            ),
        )

    if instructions_text:
        st.session_state.instructions = parse_instructions(instructions_text)
        with st.expander("Parsed instructions", expanded=False):
            parsed = st.session_state.instructions
            if parsed.get("priorities"):
                st.markdown("**Priorities:** " + ", ".join(parsed["priorities"]))
            feat = parsed.get("features", {})
            if feat.get("must_include"):
                st.markdown("**Must include:** " + ", ".join(feat["must_include"]))
            if feat.get("avoid"):
                st.markdown("**Avoid features:** " + ", ".join(feat["avoid"]))
            models = parsed.get("models", {})
            if models.get("preferred"):
                st.markdown("**Preferred models:** " + ", ".join(models["preferred"]))
            if models.get("avoid"):
                st.markdown("**Avoid models:** " + ", ".join(models["avoid"]))
            if parsed.get("visualization"):
                st.markdown("**Viz prefs:** " + ", ".join(parsed["visualization"]))
    else:
        st.session_state.instructions = {}

    # LLM provider for DA Agent
    if st.session_state.pipeline_mode == "lite":
        st.divider()
        st.subheader("LLM Provider (DA Agent)")
        llm_provider = st.radio(
            "LLM Provider",
            options=["Ollama (local)", "OpenRouter (cloud)"],
            label_visibility="collapsed",
            help="Ollama runs locally. OpenRouter requires an API key.",
        )
        st.session_state["da_local_mode"] = "Ollama" in llm_provider

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
        st.session_state.da_agent_result = None

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
# Section 2 — Run Pipeline (mode-dependent)
# ===================================================================
if st.session_state.dataset_path is not None:
    st.header("2. Run Pipeline")

    # ── Quick Explore (DA Agent) ──────────────────────────────────
    if st.session_state.pipeline_mode == "lite":
        if st.button("Run Quick Explore"):
            with st.status("Running DA Agent analysis...", expanded=True) as status:
                from da_agent.agent import run_da_agent

                da_result = run_da_agent(
                    dataset_path=st.session_state.dataset_path,
                    target_column=st.session_state.target_column,
                    instructions=st.session_state.instructions or None,
                    local_mode=st.session_state.get("da_local_mode", True),
                )
                st.session_state.da_agent_result = da_result
                # Preserve dataset_path for handoff
                da_result["dataset_path"] = st.session_state.dataset_path
                status.update(label="DA Agent analysis complete", state="complete")
            st.rerun()

    # ── Full Pipeline ─────────────────────────────────────────────
    else:
        max_iter = st.number_input(
            "Max iterations", min_value=1, max_value=10, value=3, step=1
        )

        run_full = st.button("Run Pipeline")
        run_from_da = False
        if st.session_state.da_agent_result is not None:
            run_from_da = st.button("Continue from DA Agent (skip EDA/Cleaning)")

        if run_full or run_from_da:
            # -- Reserve layout slots so analysis results appear ABOVE the
            #    modeling progress tracker.
            if run_full:
                analysis_status = st.status("Running data analysis...", expanded=True)
                analysis_steps = analysis_status.empty()
            else:
                analysis_status = None
                analysis_steps = None

            exploration_container = st.container()

            modeling_status = st.status(
                "Waiting for analysis to finish..." if run_full else "Running model pipeline...",
                expanded=not run_full,
            )
            modeling_steps = modeling_status.empty()

            # -- Tracking state --
            analysis_finished: set[str] = set()
            modeling_finished_iter: set[str] = set()
            current_iter = 0
            analysis_rendered = False
            result_state = None

            # Choose the right pipeline
            if run_from_da:
                pipeline_gen = run_pipeline_from_da(
                    da_result=st.session_state.da_agent_result,
                    target_column=st.session_state.target_column,
                    max_iterations=max_iter,
                    instructions=st.session_state.instructions or None,
                )
                modeling_status.update(
                    label="Running model pipeline...",
                    state="running",
                    expanded=True,
                )
            else:
                pipeline_gen = run_pipeline(
                    dataset_path=st.session_state.dataset_path,
                    target_column=st.session_state.target_column,
                    max_iterations=max_iter,
                    instructions=st.session_state.instructions or None,
                )

            for update in pipeline_gen:
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
                    if analysis_steps is not None:
                        analysis_steps.markdown("\n\n".join(lines))

                    # Once cleaning is done, render exploration results immediately
                    if node == "cleaning" and not analysis_rendered:
                        if analysis_status is not None:
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
# Section: DA Agent Results (Quick Explore)
# ===================================================================
if st.session_state.da_agent_result is not None:
    da = st.session_state.da_agent_result

    st.header("3. Data Exploration (DA Agent)")

    tab_profile, tab_quality, tab_viz, tab_cleaning, tab_llm = st.tabs([
        "Data Profile", "Data Quality Report", "Visualizations",
        "Cleaning Summary", "AI Analysis",
    ])

    with tab_profile:
        render_data_profile(
            st.session_state.uploaded_df, st.session_state.target_column
        )

    with tab_quality:
        render_data_quality_summary(
            st.session_state.uploaded_df, st.session_state.target_column
        )

    with tab_viz:
        render_eda_visualizations()

    with tab_cleaning:
        render_cleaning_summary(da.get("cleaning_report", ""))

    with tab_llm:
        llm_analysis = da.get("llm_analysis")
        if llm_analysis:
            st.markdown("**AI-Generated Analysis Script:**")
            st.code(llm_analysis, language="python")
        else:
            st.info("No LLM was available. Deterministic analysis results are shown in other tabs.")

    # Download notebook
    notebook_bytes = da.get("notebook_bytes")
    if notebook_bytes:
        st.download_button(
            "Download Analysis (.ipynb)",
            data=notebook_bytes,
            file_name="da_agent_analysis.ipynb",
            mime="application/x-ipynb+json",
        )

    # Handoff button
    if st.session_state.pipeline_mode == "lite":
        st.divider()
        if st.button("Continue to End-to-End Pipeline ->"):
            st.session_state.pipeline_mode = "full"
            st.rerun()

# ===================================================================
# Section 3 — Data Exploration (persisted after full pipeline run)
# ===================================================================
if st.session_state.agent_result is not None and st.session_state.da_agent_result is None:
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

    tab_metrics, tab_interpretation, tab_takeaways = st.tabs([
        "Metrics Comparison", "Model Interpretation", "Key Takeaways"
    ])

    with tab_metrics:
        render_iteration_metrics_chart(st.session_state.agent_history)

    with tab_interpretation:
        render_model_interpretation(st.session_state.agent_history)

    with tab_takeaways:
        render_key_takeaways(st.session_state.agent_history)

    st.subheader("LLM Critic Feedback")
    for h in st.session_state.agent_history:
        with st.expander(f"Iteration {h['iteration']} feedback"):
            st.markdown(h.get("feedback", "_No feedback recorded._"))

