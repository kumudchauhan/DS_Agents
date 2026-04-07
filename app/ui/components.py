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

    # ── Top-5 issues (sorted by severity) ─────────────────────────────
    issues: list[tuple[float, str]] = []  # (sort_key, message)

    # Missing values
    missing_by_col = df.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if not cols_with_missing.empty:
        worst_pct = cols_with_missing.iloc[0] / total_rows * 100
        issues.append((
            worst_pct,
            f"**{len(cols_with_missing)}** of {total_cols} columns have missing values "
            f"(worst: `{cols_with_missing.index[0]}` at {worst_pct:.1f}%)",
        ))

    # Duplicates
    if duplicate_rows > 0:
        dup_pct = duplicate_rows / total_rows * 100
        issues.append((
            dup_pct,
            f"**{duplicate_rows}** duplicate rows detected ({dup_pct:.1f}% of data)",
        ))

    # Target quality
    target_nan = int(df[target_column].isna().sum())
    unexpected = set(valid_target.unique()) - {0.0, 1.0}
    if target_nan > 0:
        issues.append((
            target_nan / total_rows * 100,
            f"Target column `{target_column}` has **{target_nan}** missing values",
        ))
    if unexpected:
        issues.append((
            90,  # high severity — invalid target values
            f"Target column contains unexpected values: "
            f"**{', '.join(str(v) for v in sorted(unexpected))}** (expected 0 / 1 only)",
        ))

    # Class imbalance
    if fraud_rate < 10:
        issues.append((
            80,
            f"Severe class imbalance — positive class is only **{fraud_rate:.1f}%** of the data",
        ))

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
        issues.append((
            top3[0][1],  # sort by worst outlier count
            f"Outliers (>3\u03c3) detected in: {', '.join(parts)}",
        ))

    # Sort high→low and show top 5
    issues.sort(key=lambda x: x[0], reverse=True)
    top_issues = issues[:5]

    if top_issues:
        st.subheader("Top Issues")
        for _, msg in top_issues:
            st.warning(msg, icon="\u26a0\ufe0f")
    else:
        st.success("No major data quality issues detected.")

    st.divider()

    # ── Missing values breakdown (top 5) ──────────────────────────────
    if not cols_with_missing.empty:
        st.subheader("Missing Values (Top 5)")
        top5_missing = cols_with_missing.head(5)
        mv_df = pd.DataFrame({
            "Column": top5_missing.index,
            "Missing": top5_missing.values,
            "% of Rows": (top5_missing.values / total_rows * 100).round(1),
        }).reset_index(drop=True)
        st.dataframe(mv_df, width="stretch", hide_index=True)
        if len(cols_with_missing) > 5:
            st.caption(f"+ {len(cols_with_missing) - 5} more columns with missing values")
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
# Section: Pipeline Details — Cleaning & Feature Engineering
# ---------------------------------------------------------------------------

def render_cleaning_summary(cleaning_report: str) -> None:
    """Render cleaning steps as a structured table."""
    if not cleaning_report:
        return

    imputation_rows = []
    other_steps = []
    for line in cleaning_report.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Filled") and "NaN with" in line:
            # e.g. "Filled amount NaN with median (147.52)"
            parts = line.replace("Filled ", "").split(" NaN with ")
            if len(parts) == 2:
                col = parts[0]
                method_val = parts[1]  # "median (147.52)" or "mode (travel)"
                method = method_val.split("(")[0].strip().capitalize()
                value = method_val.split("(")[-1].rstrip(")")
                imputation_rows.append({
                    "Column": f"`{col}`",
                    "Method": method,
                    "Fill Value": value,
                })
                continue
        if line.startswith("Shape:"):
            other_steps.append(f"**{line}**")
        elif line.startswith("Dropped"):
            other_steps.append(f"**{line}**")
        elif line.startswith("Coerced") or line.startswith("Normalised") or line.startswith("Parsed"):
            other_steps.append(line)

    if other_steps:
        for step in other_steps:
            st.markdown(f"- {step}")

    if imputation_rows:
        st.markdown("**Missing value imputation:**")
        st.dataframe(
            pd.DataFrame(imputation_rows),
            width="stretch",
            hide_index=True,
        )


def render_feature_summary(feature_report: str, iteration: int) -> None:
    """Render feature engineering steps with context on why each was added."""
    if not feature_report:
        return

    _FEATURE_RATIONALE = {
        "account_age_days": "Newer accounts may have higher fraud risk",
        "hour_of_day": "Fraud patterns vary by time of day",
        "day_of_week": "Weekend vs weekday transaction patterns differ",
        "amount_to_avg_ratio": "Unusually high ratio signals anomalous spending",
        "is_high_amount": "Transactions above 95th percentile are higher risk",
        "log_amount": "Log transform reduces outlier impact on model",
        "is_new_account": "Accounts <30 days old are higher fraud risk",
        "is_night_txn": "Late-night transactions correlate with fraud",
        "amount_deviation": "Large deviation from average signals anomaly",
        "high_velocity": "4+ transactions in 24h suggests automated fraud",
        "amount_squared": "Captures non-linear amount effects",
        "amount_x_velocity": "Interaction: high amount + high velocity = risky",
        "is_weekend": "Weekend transactions may have different fraud patterns",
        "txn_per_avg_ratio": "Transaction frequency relative to spending average",
    }

    is_llm_driven = "LLM-recommended" in feature_report

    feature_rows = []
    for line in feature_report.strip().split("\n"):
        line = line.strip()
        if line.startswith("Created:"):
            # New format: "Created: feature_name = description"
            desc_part = line.split("Created:")[-1].strip()
            f = desc_part.split("=")[0].strip()
            rationale = _FEATURE_RATIONALE.get(f, "Derived from domain knowledge")
            source = "LLM-recommended" if is_llm_driven else "Default"
            feature_rows.append({
                "Feature": f"`{f}`",
                "Source": source,
                "Rationale": rationale,
            })
        elif line.startswith("One-hot encoded:"):
            col = line.replace("One-hot encoded: ", "")
            feature_rows.append({
                "Feature": f"`{col}` (one-hot)",
                "Source": "Always applied",
                "Rationale": "Convert categorical to numeric for model input",
            })

    if feature_rows:
        st.dataframe(
            pd.DataFrame(feature_rows),
            width="stretch",
            hide_index=True,
        )

    # Final shape info
    for line in feature_report.strip().split("\n"):
        if "Final feature count" in line or "Shape" in line:
            st.caption(line.strip())


# ---------------------------------------------------------------------------
# Section: Feature Importance
# ---------------------------------------------------------------------------

def render_feature_importance(history: list) -> None:
    """Render feature importance bar charts for each iteration's model."""
    if not history:
        st.info("No model results yet.")
        return

    for h in history:
        model_obj = h.get("model_obj")
        feature_names = h.get("feature_names", [])
        model_name = (h.get("model") or "").split("\n")[0].replace("Model: ", "")
        iteration = h["iteration"]

        if model_obj is None or not feature_names:
            continue

        # Extract importance values
        importances = None
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            importances = np.abs(model_obj.coef_[0])

        if importances is None:
            continue

        # Align lengths (feature names may differ from importance array
        # due to scaling or encoding).
        n = min(len(feature_names), len(importances))
        names = feature_names[:n]
        vals = importances[:n]

        # Sort and take top 15
        idx = np.argsort(vals)[::-1][:15]
        top_names = [names[i] for i in idx]
        top_vals = [vals[i] for i in idx]

        st.markdown(f"**Iteration {iteration} — {model_name}**")

        fig, ax = plt.subplots(figsize=(8, max(3, len(top_names) * 0.35)))
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_vals, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {len(top_names)} Features — {model_name}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Quick interpretation
        if len(top_names) >= 3:
            st.caption(
                f"Top drivers: **{top_names[0]}**, **{top_names[1]}**, "
                f"**{top_names[2]}**"
            )
        st.divider()


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

def render_key_takeaways(history: list) -> None:
    """Diagnostic takeaways derived from the iteration history.

    Surfaces actionable issues like zero-recall, accuracy paradox,
    model regression, and feature-set impact across iterations.
    """
    if not history:
        st.info("No results yet.")
        return

    best = max(history, key=lambda h: h.get("f1", 0) or 0)
    first = history[0]
    last = history[-1]

    # ── Summary metrics ───────────────────────────────────────────────
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Iterations", len(history))
    c2.metric("Best F1", f"{best.get('f1', 0):.4f}",
              help=f"Iteration {best['iteration']}")
    if len(history) > 1:
        delta = (last.get("f1", 0) or 0) - (first.get("f1", 0) or 0)
        c3.metric("F1 Change", f"{delta:+.4f}",
                  delta=f"{delta:+.4f}",
                  delta_color="normal")
    else:
        c3.metric("F1 Change", "N/A", help="Run multiple iterations to compare")

    st.divider()

    # ── Collect diagnostic takeaways ──────────────────────────────────
    warnings: list[str] = []    # problems requiring attention
    insights: list[str] = []    # neutral observations
    positives: list[str] = []   # things that went well

    # --- Per-iteration diagnostics ---
    for h in history:
        it = h["iteration"]
        f1 = h.get("f1", 0) or 0
        acc = h.get("accuracy", 0) or 0
        prec = h.get("precision", 0) or 0
        rec = h.get("recall", 0) or 0
        model_name = (h.get("model") or "").split("\n")[0].replace("Model: ", "")

        # Zero recall — model ignores positive class entirely
        if rec == 0:
            warnings.append(
                f"**Iteration {it}** ({model_name}): Recall is **0** — the "
                f"model classified every sample as negative. This is a "
                f"class-imbalance problem, not a pipeline bug. The LLM critic "
                f"should recommend strategies like SMOTE, class_weight "
                f"adjustments, or switching to a model better suited for "
                f"imbalanced data."
            )
        elif rec < 0.3:
            warnings.append(
                f"**Iteration {it}** ({model_name}): Recall is very low "
                f"(**{rec:.2f}**) — the model misses most positive cases."
            )

        # Zero precision — everything predicted positive is wrong
        if prec == 0 and rec > 0:
            warnings.append(
                f"**Iteration {it}** ({model_name}): Precision is **0** — "
                f"every positive prediction is a false alarm."
            )

        # Accuracy paradox — high accuracy but near-zero F1
        if acc > 0.85 and f1 < 0.15:
            warnings.append(
                f"**Iteration {it}**: Accuracy is **{acc:.1%}** but F1 is "
                f"only **{f1:.4f}**. This is the *accuracy paradox* — high "
                f"accuracy driven by predicting the majority class. F1 is "
                f"the better metric for imbalanced datasets."
            )

    # --- Cross-iteration diagnostics ---
    if len(history) > 1:
        # F1 regression between consecutive iterations
        for i in range(1, len(history)):
            prev, curr = history[i - 1], history[i]
            prev_f1 = prev.get("f1", 0) or 0
            curr_f1 = curr.get("f1", 0) or 0
            if curr_f1 < prev_f1 and prev_f1 > 0:
                drop = prev_f1 - curr_f1
                prev_model = (prev.get("model") or "").split("\n")[0].replace("Model: ", "")
                curr_model = (curr.get("model") or "").split("\n")[0].replace("Model: ", "")
                warnings.append(
                    f"**F1 dropped** from **{prev_f1:.4f}** (iter "
                    f"{prev['iteration']}, {prev_model}) to **{curr_f1:.4f}** "
                    f"(iter {curr['iteration']}, {curr_model}) — a regression "
                    f"of **{drop:.4f}**. The model or feature change may have "
                    f"hurt performance."
                )
            elif curr_f1 > prev_f1:
                gain = curr_f1 - prev_f1
                positives.append(
                    f"F1 improved by **{gain:.4f}** from iteration "
                    f"{prev['iteration']} to {curr['iteration']}."
                )

        # Feature set comparison across iterations
        feat_counts = []
        for h in history:
            fr = h.get("feature_report", "")
            n_created = fr.count("Created:")
            feat_counts.append((h["iteration"], n_created))

        if len(set(c for _, c in feat_counts)) > 1:
            parts = [f"iter {it}: {n} features" for it, n in feat_counts]
            insights.append(
                f"Feature set changed across iterations ({', '.join(parts)}). "
                f"Check the Feature Engineering tab to compare what was added "
                f"or removed."
            )

        # Detect if LLM recommendations were used
        llm_driven = any(
            "LLM-recommended" in (h.get("feature_report") or "")
            for h in history
        )
        default_only = all(
            "default feature set" in (h.get("feature_report") or "").lower()
            for h in history
        )
        if default_only and len(history) > 1:
            warnings.append(
                "All iterations used the **default feature set** — the LLM "
                "critic's feature recommendations were not applied. This may "
                "indicate the LLM was unavailable or returned unparseable "
                "responses. Check the Critic Feedback section below."
            )
        elif llm_driven:
            positives.append(
                "The LLM critic's recommendations drove feature selection "
                "and/or model choice in at least one iteration."
            )

    # --- Best model insight ---
    best_model = (best.get("model") or "").split("\n")[0].replace("Model: ", "")
    if best.get("f1", 0):
        insights.append(
            f"Best performing model: **{best_model}** (iteration "
            f"{best['iteration']}) with F1 = **{best['f1']:.4f}**, "
            f"Precision = **{best.get('precision', 0):.4f}**, "
            f"Recall = **{best.get('recall', 0):.4f}**."
        )

    # ── Render takeaways ──────────────────────────────────────────────
    if warnings:
        st.subheader("Issues Detected")
        for msg in warnings:
            st.warning(msg, icon="\u26a0\ufe0f")

    if positives:
        st.subheader("What Went Well")
        for msg in positives:
            st.success(msg, icon="\u2705")

    if insights:
        st.subheader("Observations")
        for msg in insights:
            st.info(msg, icon="\U0001f4a1")
