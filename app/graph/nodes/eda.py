import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.graph.state import AgentState

OUTPUT_DIR = "outputs"


def _save_fig(fig, name: str) -> str:
    """Save a matplotlib figure to the outputs directory and close it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def _data_quality_report(df: pd.DataFrame, target: str) -> str:
    """Return a plain-text data quality report."""
    total = len(df)
    lines = [
        "--- DATA QUALITY REPORT ---",
        f"Total rows: {total}",
        f"Total columns: {df.shape[1]}",
    ]

    # Missing values summary
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values("pct", ascending=False)
    if len(missing_df) > 0:
        lines.append(f"\nColumns with missing values ({len(missing_df)}):")
        for col, row in missing_df.iterrows():
            lines.append(f"  {col}: {int(row['missing'])} ({row['pct']}%)")
    else:
        lines.append("\nNo missing values found.")

    # Duplicate rows
    dup_count = df.duplicated().sum()
    lines.append(f"\nDuplicate rows: {dup_count}")

    # Numeric column outliers (values beyond 3 std from mean)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    outlier_lines = []
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            outliers = ((df[col] - mean).abs() > 3 * std).sum()
            if outliers > 0:
                outlier_lines.append(f"  {col}: {outliers} outliers (>{3}*std)")
    if outlier_lines:
        lines.append(f"\nPotential outliers (>3 std):")
        lines.extend(outlier_lines)

    # Negative values in columns that are likely non-negative
    for col in ["amount", "avg_amount_7d", "prior_transactions_24h"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                lines.append(f"  {col}: {neg} negative values")

    # Target quality
    if target in df.columns:
        unique_vals = sorted(df[target].dropna().unique())
        lines.append(f"\nTarget column '{target}':")
        lines.append(f"  Unique values: {unique_vals}")
        lines.append(f"  NaN count: {df[target].isnull().sum()}")
        if set(unique_vals) - {0.0, 1.0}:
            lines.append(f"  WARNING: unexpected values {set(unique_vals) - {0.0, 1.0}}")

    # Categorical cardinality
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        lines.append(f"\nCategorical columns ({len(cat_cols)}):")
        for col in cat_cols:
            nuniq = df[col].nunique()
            sample = df[col].dropna().unique()[:5].tolist()
            lines.append(f"  {col}: {nuniq} unique — sample: {sample}")

    return "\n".join(lines)


def _generate_visualizations(df: pd.DataFrame, target: str) -> list[str]:
    """Generate and save EDA visualizations. Returns list of saved paths."""
    saved = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Target distribution bar chart
    if target in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[target].value_counts().sort_index()
        counts.plot.bar(ax=ax, color=["steelblue", "tomato"][:len(counts)])
        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts):
            ax.text(i, v + 5, str(v), ha="center", fontsize=10)
        saved.append(_save_fig(fig, "target_distribution.png"))

    # 2. Missing values heatmap
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].index.tolist()
    if cols_with_missing:
        fig, ax = plt.subplots(figsize=(max(8, len(cols_with_missing)), 4))
        ax.bar(cols_with_missing, missing[cols_with_missing], color="coral")
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        saved.append(_save_fig(fig, "missing_values.png"))

    # 3. Numeric feature distributions (histograms)
    plot_cols = [c for c in numeric_cols if c != target][:8]  # cap at 8
    if plot_cols:
        ncols = min(4, len(plot_cols))
        nrows = (len(plot_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten() if nrows * ncols > 1 else [axes]
        for i, col in enumerate(plot_cols):
            df[col].dropna().hist(ax=axes[i], bins=30, color="steelblue", edgecolor="white")
            axes[i].set_title(col, fontsize=10)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Numeric Feature Distributions", fontsize=12)
        fig.tight_layout()
        saved.append(_save_fig(fig, "feature_distributions.png"))

    # 4. Correlation heatmap
    corr_cols = [c for c in numeric_cols if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(max(8, len(corr_cols)), max(6, len(corr_cols) * 0.7)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_cols, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        saved.append(_save_fig(fig, "correlation_heatmap.png"))

    # 5. Target vs key numeric features (box plots)
    if target in df.columns and len(plot_cols) > 0:
        box_cols = plot_cols[:4]
        fig, axes = plt.subplots(1, len(box_cols), figsize=(4 * len(box_cols), 4))
        if len(box_cols) == 1:
            axes = [axes]
        for i, col in enumerate(box_cols):
            groups = [
                df.loc[df[target] == val, col].dropna()
                for val in sorted(df[target].dropna().unique())
            ]
            labels = [str(int(v)) for v in sorted(df[target].dropna().unique())]
            axes[i].boxplot(groups, labels=labels)
            axes[i].set_title(f"{col} by {target}", fontsize=10)
            axes[i].set_xlabel(target)
        fig.suptitle("Feature Distributions by Target", fontsize=12)
        fig.tight_layout()
        saved.append(_save_fig(fig, "target_boxplots.png"))

    return saved


def eda_node(state: AgentState) -> dict:
    """Load the dataset, produce a data quality report, and generate visualizations."""
    print("\n" + "=" * 60)
    print("  EDA NODE - Iteration", state["iteration"])
    print("=" * 60)

    df = pd.read_csv(state["dataset_path"])

    # Coerce target column to numeric (handles mixed string/float values)
    target = state["target_column"]
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    # --- Basic profile ---
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"\nColumn types:\n{df.dtypes.to_string()}")
    lines.append(f"\nBasic statistics:\n{df.describe().to_string()}")

    if target in df.columns:
        dist = df[target].value_counts()
        lines.append(f"\nTarget distribution ({target}):\n{dist.to_string()}")
        lines.append(f"Fraud percentage: {df[target].mean() * 100:.2f}%")

    # --- Data quality report ---
    quality_report = _data_quality_report(df, target)
    lines.append(f"\n{quality_report}")

    # --- Visualizations ---
    viz_paths = _generate_visualizations(df, target)
    lines.append(f"\nVisualizations saved ({len(viz_paths)}):")
    for p in viz_paths:
        lines.append(f"  {p}")

    report = "\n".join(lines)
    print(report)

    return {"dataframe": df, "eda_report": report}
