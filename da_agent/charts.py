"""EDA visualizations for the DA Agent.

Refactored from app/graph/nodes/eda.py so both the DA Agent and the full
pipeline's EDA node share the same chart generation code.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_fig(fig: plt.Figure, name: str, output_dir: str = "outputs") -> str:
    """Save a matplotlib figure to the outputs directory and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def generate_eda_charts(
    df: pd.DataFrame,
    target: str,
    output_dir: str = "outputs",
) -> list[str]:
    """Generate and save EDA visualizations. Returns list of saved file paths.

    Charts:
    1. Target distribution bar chart
    2. Missing values by column
    3. Numeric feature distributions (histograms, up to 8)
    4. Correlation heatmap
    5. Target vs key features (box plots)
    """
    saved: list[str] = []
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
        saved.append(save_fig(fig, "target_distribution.png", output_dir))

    # 2. Missing values heatmap
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].index.tolist()
    if cols_with_missing:
        fig, ax = plt.subplots(figsize=(max(8, len(cols_with_missing)), 4))
        ax.bar(cols_with_missing, missing[cols_with_missing], color="coral")
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        saved.append(save_fig(fig, "missing_values.png", output_dir))

    # 3. Numeric feature distributions (histograms)
    plot_cols = [c for c in numeric_cols if c != target][:8]
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
        saved.append(save_fig(fig, "feature_distributions.png", output_dir))

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
        saved.append(save_fig(fig, "correlation_heatmap.png", output_dir))

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
        saved.append(save_fig(fig, "target_boxplots.png", output_dir))

    return saved
