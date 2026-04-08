"""Streamlit UI for the Autonomous Data Science Agent."""

import os
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
    "pipeline_mode": None,       # None until user picks
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
# Step 1 — Upload data + instructions
# ===================================================================
st.header("1. Upload Your Data")

col_csv, col_md = st.columns(2)
with col_csv:
    uploaded_csv = st.file_uploader("Dataset (CSV)", type=["csv"])
with col_md:
    uploaded_md = st.file_uploader("Instructions (optional)", type=["md", "txt"])

# Process CSV
if uploaded_csv is not None:
    if st.session_state.uploaded_df is None or st.session_state.dataset_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_csv.getvalue())
        tmp.flush()
        st.session_state.dataset_path = tmp.name
        st.session_state.uploaded_df = pd.read_csv(tmp.name)
        st.session_state.agent_result = None
        st.session_state.agent_history = []
        st.session_state.da_agent_result = None
        st.session_state.pipeline_mode = None

    df = st.session_state.uploaded_df

    with st.expander("Dataset preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    col_shape, col_target = st.columns([1, 2])
    with col_shape:
        st.markdown(f"**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns")
    with col_target:
        target_col = st.selectbox(
            "Select target column",
            options=list(df.columns),
            index=list(df.columns).index("is_fraud") if "is_fraud" in df.columns else 0,
        )
        st.session_state.target_column = target_col
else:
    st.info("Upload a CSV file to get started.")

# Process instructions
if uploaded_md is not None:
    md_text = uploaded_md.getvalue().decode("utf-8")
    st.session_state.instructions = parse_instructions(md_text)
    parsed = st.session_state.instructions
    with st.expander("Parsed instructions", expanded=False):
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

# ===================================================================
# Step 2 — Choose mode (only shown after data is uploaded)
# ===================================================================
if st.session_state.dataset_path is not None and st.session_state.da_agent_result is None and st.session_state.agent_result is None:
    st.header("2. What would you like to do?")

    col_quick, col_full = st.columns(2)

    with col_quick:
        st.markdown(
            "**Quick Explore**\n\n"
            "Fast, local data analysis — no API key needed.\n\n"
            "Data profile, quality report, visualizations, "
            "cleaning summary, downloadable notebook."
        )
        run_quick = st.button("Run Quick Explore", use_container_width=True)

    with col_full:
        st.markdown(
            "**End-to-End DS Pipeline**\n\n"
            "Full analysis + iterative modeling with LLM critic.\n\n"
            "Feature engineering, model training, evaluation, "
            "automatic improvement loop."
        )
        run_full = st.button("Run End-to-End DS Pipeline", use_container_width=True)

    # ── Quick Explore: run immediately ────────────────────────────
    if run_quick:
        st.session_state.pipeline_mode = "lite"
        with st.status("Running Quick Explore...", expanded=True) as status:
            step_text = status.empty()

            def _on_progress(msg: str):
                step_text.markdown(f"**{msg}**")

            from da_agent.agent import run_da_agent

            da_result = run_da_agent(
                dataset_path=st.session_state.dataset_path,
                target_column=st.session_state.target_column,
                instructions=st.session_state.instructions or None,
                on_progress=_on_progress,
            )
            st.session_state.da_agent_result = da_result
            da_result["dataset_path"] = st.session_state.dataset_path
            status.update(
                label=f"Quick Explore complete ({da_result.get('elapsed', '')})",
                state="complete",
            )
        st.rerun()

    # ── End-to-End: ask for API key, then run ─────────────────────
    if run_full:
        st.session_state.pipeline_mode = "full"
        st.rerun()

# ===================================================================
# Step 2b — API key prompt (End-to-End only, before run)
# ===================================================================
if (
    st.session_state.pipeline_mode == "full"
    and st.session_state.agent_result is None
    and st.session_state.da_agent_result is None
):
    st.header("2. OpenRouter API Key")
    st.markdown(
        "The End-to-End pipeline uses an LLM critic to iteratively improve "
        "model performance. This requires an [OpenRouter](https://openrouter.ai/keys) API key."
    )

    existing_key = os.environ.get("OPENROUTER_API_KEY", "")
    if existing_key:
        st.success("API key detected from environment variable.")
        api_key = existing_key
    else:
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-...",
            help="Key is used for this request only and not stored.",
        )
        st.caption(
            "Or set it via CLI before launching: "
            "`export OPENROUTER_API_KEY='sk-or-...' && streamlit run streamlit_app.py`"
        )

    max_iter = st.number_input(
        "Max iterations", min_value=1, max_value=10, value=3, step=1
    )

    if st.button("Start Pipeline", use_container_width=True, disabled=not api_key):
        # Pass key directly into the LLM provider for this run
        os.environ["OPENROUTER_API_KEY"] = api_key

        # -- Reserve layout slots --
        analysis_status = st.status("Running data analysis...", expanded=True)
        analysis_steps = analysis_status.empty()
        exploration_container = st.container()
        modeling_status = st.status("Waiting for analysis to finish...", expanded=False)
        modeling_steps = modeling_status.empty()

        analysis_finished: set[str] = set()
        modeling_finished_iter: set[str] = set()
        current_iter = 0
        analysis_rendered = False
        result_state = None

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

                if node == "cleaning" and not analysis_rendered:
                    analysis_status.update(
                        label="Data analysis complete", state="complete", expanded=False,
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
                    modeling_status.update(
                        label="Running model pipeline...", state="running", expanded=True,
                    )
                continue

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

        modeling_status.update(
            label="Model pipeline complete", state="complete", expanded=False,
        )

        # Clean up key from env after use
        os.environ.pop("OPENROUTER_API_KEY", None)

        if result_state:
            st.session_state.agent_result = result_state
            st.session_state.agent_history = result_state.get("history", [])
        st.rerun()

# ===================================================================
# Results: DA Agent (Quick Explore)
# ===================================================================
if st.session_state.da_agent_result is not None:
    da = st.session_state.da_agent_result

    st.header("3. Data Exploration (Quick Explore)")

    tab_profile, tab_quality, tab_viz, tab_cleaning = st.tabs([
        "Data Profile", "Data Quality Report", "Visualizations",
        "Cleaning Summary",
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

    # Download notebook
    notebook_bytes = da.get("notebook_bytes")
    if notebook_bytes:
        st.download_button(
            "Download Analysis (.ipynb)",
            data=notebook_bytes,
            file_name="da_agent_analysis.ipynb",
            mime="application/x-ipynb+json",
        )

    # Handoff to full pipeline
    if st.session_state.agent_result is None:
        st.divider()
        if st.session_state.pipeline_mode != "full":
            if st.button("Continue to End-to-End DS Pipeline ->"):
                st.session_state.pipeline_mode = "full"
                st.rerun()
        else:
            # API key prompt + run (inline, right here)
            st.subheader("End-to-End DS Pipeline")
            st.markdown(
                "The pipeline uses an LLM critic to iteratively improve model performance. "
                "This requires an [OpenRouter](https://openrouter.ai/keys) API key."
            )

            existing_key = os.environ.get("OPENROUTER_API_KEY", "")
            if existing_key:
                st.success("API key detected from environment variable.")
                api_key = existing_key
            else:
                api_key = st.text_input(
                    "OpenRouter API Key",
                    type="password",
                    placeholder="sk-or-...",
                    key="handoff_api_key",
                    help="Key is used for this request only and not stored.",
                )
                st.caption(
                    "Or set it via CLI: "
                    "`export OPENROUTER_API_KEY='sk-or-...' && streamlit run streamlit_app.py`"
                )

            max_iter = st.number_input(
                "Max iterations", min_value=1, max_value=10, value=3, step=1, key="handoff_iter",
            )

            if st.button("Start Pipeline", use_container_width=True, disabled=not api_key, key="handoff_start"):
                os.environ["OPENROUTER_API_KEY"] = api_key

                modeling_status = st.status("Running model pipeline...", expanded=True)
                modeling_steps = modeling_status.empty()
                modeling_finished_iter: set[str] = set()
                current_iter = 0
                result_state = None

                pipeline_gen = run_pipeline_from_da(
                    da_result=st.session_state.da_agent_result,
                    target_column=st.session_state.target_column,
                    max_iterations=max_iter,
                    instructions=st.session_state.instructions or None,
                )

                for update in pipeline_gen:
                    node = update["node"]
                    result_state = update["state"]
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

                modeling_status.update(
                    label="Model pipeline complete", state="complete", expanded=False,
                )
                os.environ.pop("OPENROUTER_API_KEY", None)

                if result_state:
                    st.session_state.agent_result = result_state
                    st.session_state.agent_history = result_state.get("history", [])
                st.rerun()

# ===================================================================
# Results: Data Exploration (full pipeline, no prior DA Agent)
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
# Results: Pipeline Details
# ===================================================================
if st.session_state.agent_history:
    st.header("4. Pipeline Details")

    tab_cleaning, tab_features, tab_importance = st.tabs(
        ["Data Cleaning", "Feature Engineering", "Feature Importance"]
    )

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
# Results: Model Results
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
