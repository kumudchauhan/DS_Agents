"""Convert DA Agent analysis results into a downloadable .ipynb notebook."""

from __future__ import annotations

import json


def _make_cell(cell_type: str, source: str, cell_id: int) -> dict:
    """Build a single notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def generate_notebook(
    dataset_path: str,
    stat_block: str,
    cleaning_report: str,
    data_quality_report: str,
    llm_analysis: str | None = None,
) -> bytes:
    """Generate a Jupyter notebook with DA Agent analysis results.

    Returns notebook content as bytes (ready for st.download_button).
    """
    cells: list[dict] = []
    idx = 0

    # ── 1. Title & Table of Contents ─────────────────────────────
    cells.append(_make_cell("markdown", (
        "# Data Analysis Report\n\n"
        f"Dataset: `{dataset_path}`\n\n"
        "---\n\n"
        "## Table of Contents\n\n"
        "1. **Setup** — Import libraries and configure plotting\n"
        "2. **Load & Preview** — Read the dataset and inspect its shape\n"
        "3. **Summary Statistics** — Descriptive stats and column info\n"
        "4. **Missing Values** — Identify and visualise gaps in the data\n"
        "5. **Correlation Analysis** — Spot multicollinearity and leakage\n"
        "6. **Target Distribution** — Class balance check\n"
        "7. **Feature Distributions** — Histograms for key numeric columns\n"
        "8. **Outlier Detection** — Flag extreme values beyond 3 std\n"
        "9. **Data Quality & Cleaning Summary** — Automated findings\n"
        "10. **Next Steps** — Recommendations for modeling"
    ), idx)); idx += 1

    # ── 2. Imports ────────────────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 1. Setup\n\n"
        "Import the core libraries. Seaborn's `whitegrid` theme keeps plots "
        "clean and readable."
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "# Use a clean grid theme for all plots\n"
        "sns.set_theme(style='whitegrid')\n"
        "%matplotlib inline"
    ), idx)); idx += 1

    # ── 3. Load data ──────────────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 2. Load & Preview\n\n"
        "Load the CSV and take a first look at the data shape and the "
        "first few rows. This tells us how many observations and features "
        "we're working with."
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        f'df = pd.read_csv("{dataset_path}")\n'
        "print(f'Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}')\n"
        "df.head(10)"
    ), idx)); idx += 1

    # ── 4. Summary statistics ─────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 3. Summary Statistics\n\n"
        "**`describe()`** gives the count, mean, std, min, quartiles, and max "
        "for every numeric column. Look for:\n"
        "- Large gaps between the mean and median (50%) — signals skewness\n"
        "- Min values that shouldn't be negative (e.g. amounts)\n"
        "- Max values far from the 75th percentile — possible outliers"
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "df.describe()"
    ), idx)); idx += 1

    cells.append(_make_cell("markdown", (
        "**`info()`** shows the dtype and non-null count per column. "
        "Columns with fewer non-null entries have missing values."
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "df.info()"
    ), idx)); idx += 1

    # ── 5. Missing values ─────────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 4. Missing Values\n\n"
        "Visualise which columns have gaps and how severe they are. "
        "Columns with >30% missing may need to be dropped; smaller gaps "
        "can often be filled with median (numeric) or mode (categorical)."
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "# Count missing values per column, keep only those with gaps\n"
        "missing = df.isnull().sum()\n"
        "missing = missing[missing > 0].sort_values(ascending=False)\n"
        "\n"
        "if len(missing) > 0:\n"
        "    print(f'{len(missing)} column(s) have missing values:\\n')\n"
        "    for col, count in missing.items():\n"
        "        pct = count / len(df) * 100\n"
        "        print(f'  {col}: {count:,} missing ({pct:.1f}%)')\n"
        "    print()\n"
        "    missing.plot.bar(color='coral', title='Missing Values by Column')\n"
        "    plt.ylabel('Count')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No missing values found — the dataset is complete.')"
    ), idx)); idx += 1

    # ── 6. Correlation heatmap ────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 5. Correlation Analysis\n\n"
        "The heatmap shows pairwise Pearson correlations between numeric features.\n\n"
        "**What to look for:**\n"
        "- Values close to **+1 or -1** between two features = multicollinearity "
        "(consider dropping one)\n"
        "- Values close to **+1 or -1** between a feature and the target = "
        "potential **target leakage** (the feature may contain the answer)\n"
        "- Clusters of correlated features = opportunities to combine them"
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "numeric_df = df.select_dtypes(include='number')\n"
        "\n"
        "if numeric_df.shape[1] >= 2:\n"
        "    plt.figure(figsize=(10, 8))\n"
        "    sns.heatmap(\n"
        "        numeric_df.corr(),\n"
        "        annot=True,       # show correlation values in each cell\n"
        "        cmap='RdBu_r',    # red = negative, blue = positive\n"
        "        center=0,\n"
        "        fmt='.2f',\n"
        "        linewidths=0.5,\n"
        "    )\n"
        "    plt.title('Feature Correlation Heatmap')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('Not enough numeric columns for a correlation heatmap.')"
    ), idx)); idx += 1

    # ── 7. Target distribution ────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 6. Target Distribution\n\n"
        "Check the balance between classes. Severe imbalance (e.g. <5% positive) "
        "means accuracy alone is misleading — a model that always predicts the "
        "majority class gets high accuracy but zero recall.\n\n"
        "**If imbalanced**, consider: class weights, SMOTE, or threshold tuning."
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "# Set this to your actual target column name\n"
        "target_column = df.columns[-1]\n"
        "\n"
        "if target_column in df.columns:\n"
        "    counts = df[target_column].value_counts().sort_index()\n"
        "    total = counts.sum()\n"
        "\n"
        "    print(f'Target: {target_column}')\n"
        "    for val, count in counts.items():\n"
        "        print(f'  {val}: {count:,} ({count/total*100:.1f}%)')\n"
        "\n"
        "    counts.plot.bar(\n"
        "        color=['steelblue', 'tomato'][:len(counts)],\n"
        "        title=f'Target Distribution: {target_column}',\n"
        "    )\n"
        "    plt.ylabel('Count')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ), idx)); idx += 1

    # ── 8. Feature distributions ──────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 7. Feature Distributions\n\n"
        "Histograms for the first 8 numeric columns. Look for:\n"
        "- **Skewed distributions** — may benefit from log or sqrt transforms\n"
        "- **Bimodal peaks** — could indicate two distinct sub-populations\n"
        "- **Long tails** — outliers that may dominate model training"
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "# Plot histograms for up to 8 numeric columns\n"
        "numeric_cols = df.select_dtypes(include='number').columns[:8]\n"
        "\n"
        "if len(numeric_cols) > 0:\n"
        "    df[numeric_cols].hist(\n"
        "        bins=30,\n"
        "        figsize=(14, 8),\n"
        "        color='steelblue',\n"
        "        edgecolor='white',\n"
        "    )\n"
        "    plt.suptitle('Numeric Feature Distributions')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ), idx)); idx += 1

    # ── 9. Outlier detection ──────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 8. Outlier Detection\n\n"
        "Flag values that fall more than 3 standard deviations from the mean. "
        "These aren't necessarily errors — they may be rare but legitimate "
        "observations (e.g. very large transactions). Review them before "
        "deciding to clip or remove."
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "# Detect outliers (> 3 std from mean) in each numeric column\n"
        "numeric_cols = df.select_dtypes(include='number').columns\n"
        "outlier_summary = []\n"
        "\n"
        "for col in numeric_cols:\n"
        "    series = df[col].dropna()\n"
        "    if len(series) < 10:\n"
        "        continue\n"
        "    mean, std = series.mean(), series.std()\n"
        "    if std == 0:\n"
        "        continue\n"
        "    n_outliers = int(((series - mean).abs() > 3 * std).sum())\n"
        "    if n_outliers > 0:\n"
        "        outlier_summary.append({\n"
        "            'Column': col,\n"
        "            'Outliers': n_outliers,\n"
        "            '% of Rows': f'{n_outliers / len(df) * 100:.2f}%',\n"
        "            'Mean': f'{mean:.2f}',\n"
        "            'Std': f'{std:.2f}',\n"
        "        })\n"
        "\n"
        "if outlier_summary:\n"
        "    print(f'{len(outlier_summary)} column(s) have outliers beyond 3 std:\\n')\n"
        "    display(pd.DataFrame(outlier_summary))\n"
        "else:\n"
        "    print('No outliers detected beyond 3 standard deviations.')"
    ), idx)); idx += 1

    # ── 10. Data quality report + cleaning summary ────────────────
    report_md = "## 9. Data Quality & Cleaning Summary\n\n"
    report_md += (
        "Below are the automated findings from the DA Agent's analysis pass. "
        "The **Data Quality Report** lists structural issues; the **Cleaning Summary** "
        "shows what was fixed.\n\n"
    )
    if data_quality_report:
        report_md += f"### Data Quality Report\n\n```\n{data_quality_report}\n```\n\n"
    if cleaning_report:
        report_md += f"### Cleaning Summary\n\n```\n{cleaning_report}\n```\n"
    cells.append(_make_cell("markdown", report_md, idx)); idx += 1

    # ── 11. LLM analysis (if available) ───────────────────────────
    if llm_analysis:
        cells.append(_make_cell("markdown", (
            "## AI-Generated Analysis Script\n\n"
            "The following code was generated by the DA Agent's LLM. "
            "Review it before running — it may need adjustments for your "
            "specific dataset and environment."
        ), idx)); idx += 1
        cells.append(_make_cell("code", llm_analysis, idx)); idx += 1

    # ── 12. Next steps ────────────────────────────────────────────
    cells.append(_make_cell("markdown", (
        "## 10. Next Steps\n\n"
        "Based on the analysis above, here are recommended actions:\n\n"
        "1. **Address missing values** — Fill with median/mode or drop columns "
        "with excessive missingness (>50%)\n"
        "2. **Handle outliers** — Clip extreme values or apply log transforms "
        "to reduce their impact\n"
        "3. **Check for leakage** — Remove any feature with near-perfect "
        "correlation to the target\n"
        "4. **Engineer features** — Create interaction terms, time-based features, "
        "or ratio features based on correlation patterns\n"
        "5. **Address class imbalance** — Use class weights, SMOTE, or adjust "
        "the decision threshold\n"
        "6. **Proceed to modeling** — Run the DS Agent End-to-End Pipeline "
        "to train and evaluate models iteratively"
    ), idx)); idx += 1

    # Build notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }

    return json.dumps(notebook, indent=1).encode("utf-8")
