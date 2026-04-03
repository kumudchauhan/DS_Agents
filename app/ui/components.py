"""Reusable Streamlit render helpers for each section of the UI."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Section: Data Quality Report
# ---------------------------------------------------------------------------

def render_data_quality_summary(df: pd.DataFrame, target_column: str) -> None:
    """Render a clean, executive-style data quality report from the dataframe."""
    if df is None or df.empty:
        st.info("Run the pipeline first to generate a data quality report.")
        return

    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = total_missing / total_cells * 100 if total_cells else 0
    duplicate_rows = int(df.duplicated().sum())
    target_series = pd.to_numeric(df[target_column], errors="coerce")
    valid_target = target_series.dropna()
    fraud_rate = (valid_target == 1).sum() / len(valid_target) * 100 if len(valid_target) else 0

    # ── Key metrics row ───────────────────────────────────────────────
    st.subheader("Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("Columns", total_cols)
    c3.metric("Missing Cells", f"{missing_pct:.1f}%")
    c4.metric("Duplicate Rows", duplicate_rows)
    c5.metric("Positive Class Rate", f"{fraud_rate:.1f}%")

    st.divider()

    # ── Data issues (warnings) ────────────────────────────────────────
    issues: list[str] = []

    # Missing values
    missing_by_col = df.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if not cols_with_missing.empty:
        worst_col = cols_with_missing.index[0]
        worst_pct = cols_with_missing.iloc[0] / total_rows * 100
        issues.append(
            f"**{len(cols_with_missing)}** of {total_cols} columns have missing values "
            f"(worst: `{worst_col}` at {worst_pct:.1f}%)"
        )

    # Duplicates
    if duplicate_rows > 0:
        issues.append(
            f"**{duplicate_rows}** duplicate rows detected "
            f"({duplicate_rows / total_rows * 100:.1f}% of data)"
        )

    # Target quality
    target_nan = int(df[target_column].isna().sum())
    unexpected = set(valid_target.unique()) - {0.0, 1.0}
    if target_nan > 0:
        issues.append(f"Target column `{target_column}` has **{target_nan}** missing values")
    if unexpected:
        issues.append(
            f"Target column contains unexpected values: "
            f"**{', '.join(str(v) for v in sorted(unexpected))}** (expected 0 / 1 only)"
        )

    # Class imbalance
    if fraud_rate < 10:
        issues.append(
            f"Severe class imbalance — positive class is only **{fraud_rate:.1f}%** of the data"
        )

    # Outliers (numeric cols, > 3 std)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    outlier_cols = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        n_outliers = int(((s - s.mean()).abs() > 3 * s.std()).sum())
        if n_outliers > 0:
            outlier_cols.append((col, n_outliers))
    if outlier_cols:
        top3 = sorted(outlier_cols, key=lambda x: x[1], reverse=True)[:3]
        parts = [f"`{c}` ({n})" for c, n in top3]
        issues.append(f"Outliers (>3\u03c3) detected in: {', '.join(parts)}")

    if issues:
        st.subheader("Issues Found")
        for issue in issues:
            st.warning(issue, icon="\u26a0\ufe0f")
    else:
        st.success("No major data quality issues detected.")

    st.divider()

    # ── Missing values breakdown ──────────────────────────────────────
    if not cols_with_missing.empty:
        st.subheader("Missing Values")
        mv_df = pd.DataFrame({
            "Column": cols_with_missing.index,
            "Missing": cols_with_missing.values,
            "% of Rows": (cols_with_missing.values / total_rows * 100).round(1),
        }).reset_index(drop=True)
        st.dataframe(mv_df, width="stretch", hide_index=True)
        st.divider()

    # ── Target distribution ───────────────────────────────────────────
    st.subheader("Target Distribution")
    target_counts = valid_target.value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Value": target_counts.index.astype(int),
        "Count": target_counts.values,
        "Share": (target_counts.values / target_counts.sum() * 100).round(1),
    })
    dist_df["Share"] = dist_df["Share"].astype(str) + "%"
    st.dataframe(dist_df, width="stretch", hide_index=True)

    st.divider()

    # ── Column summary ────────────────────────────────────────────────
    st.subheader("Column Summary")
    summary_rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        miss = int(df[col].isna().sum())
        miss_pct = round(miss / total_rows * 100, 1) if total_rows else 0
        nunique = int(df[col].nunique())
        summary_rows.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-null": f"{non_null:,}",
            "Missing %": f"{miss_pct}%" if miss > 0 else "-",
            "Unique": nunique,
        })
    st.dataframe(
        pd.DataFrame(summary_rows),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Section: EDA Visualizations
# ---------------------------------------------------------------------------

_VIZ_FILES = [
    ("Target Distribution", "target_distribution.png"),
    ("Missing Values", "missing_values.png"),
    ("Feature Distributions", "feature_distributions.png"),
    ("Correlation Heatmap", "correlation_heatmap.png"),
    ("Target Box Plots", "target_boxplots.png"),
]


def render_eda_visualizations(output_dir: str = "outputs") -> None:
    """Display the five EDA PNGs generated by the eda node."""
    found_any = False
    for caption, filename in _VIZ_FILES:
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            st.image(path, caption=caption, width="stretch")
            found_any = True

    if not found_any:
        st.info("No visualizations found. Run the pipeline to generate them.")


# ---------------------------------------------------------------------------
# Section: Metrics Comparison (grouped bar chart + table)
# ---------------------------------------------------------------------------

def render_iteration_metrics_chart(history: list) -> None:
    """Grouped bar chart comparing accuracy / precision / recall / F1 across
    iterations, plus a summary table with the best model highlighted."""
    if not history:
        st.info("No model results yet. Run the pipeline first.")
        return

    metric_names = ["accuracy", "precision", "recall", "f1"]
    iterations = [h["iteration"] for h in history]
    x = np.arange(len(iterations))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(6, len(iterations) * 2.5), 4))
    for i, metric in enumerate(metric_names):
        values = [h.get(metric, 0) or 0 for h in history]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Iter {it}" for it in iterations])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics by Iteration")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    best = max(history, key=lambda h: h.get("f1", 0) or 0)
    rows = []
    for h in history:
        model_name = (h.get("model") or "").split("\n")[0].replace("Model: ", "")
        row = {
            "Iteration": h["iteration"],
            "Model": model_name,
            "Accuracy": h.get("accuracy"),
            "Precision": h.get("precision"),
            "Recall": h.get("recall"),
            "F1": h.get("f1"),
        }
        rows.append(row)

    st.table(rows)
    st.success(
        f"**Best model:** Iteration {best['iteration']} — "
        f"F1 = {best.get('f1', 'N/A')}"
    )


# ---------------------------------------------------------------------------
# Section: Key Highlights
# ---------------------------------------------------------------------------

def render_key_highlights(history: list) -> None:
    """Auto-generated insights derived from the iteration history."""
    if not history:
        st.info("No results yet.")
        return

    best = max(history, key=lambda h: h.get("f1", 0) or 0)
    first = history[0]
    last = history[-1]

    highlights = []

    highlights.append(
        f"Ran **{len(history)} iteration(s)** of the ML pipeline."
    )
    highlights.append(
        f"Best F1 score: **{best.get('f1')}** achieved in iteration "
        f"**{best['iteration']}**."
    )

    if len(history) > 1:
        delta = (last.get("f1", 0) or 0) - (first.get("f1", 0) or 0)
        direction = "improved" if delta > 0 else "decreased" if delta < 0 else "unchanged"
        highlights.append(
            f"F1 {direction} by **{abs(delta):.4f}** from iteration "
            f"{first['iteration']} to {last['iteration']}."
        )

    if best.get("recall", 0) and best.get("precision", 0):
        if best["recall"] < 0.5:
            highlights.append(
                "Recall is below 0.5 — the model may be missing many positive cases."
            )
        if best["precision"] < 0.5:
            highlights.append(
                "Precision is below 0.5 — the model has a high false-positive rate."
            )

    for line in highlights:
        st.markdown(f"- {line}")
