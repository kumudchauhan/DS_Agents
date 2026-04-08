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

    # 1. Title
    cells.append(_make_cell("markdown", f"# Data Analysis Report\n\nDataset: `{dataset_path}`", idx)); idx += 1

    # 2. Imports
    cells.append(_make_cell("code", (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "%matplotlib inline"
    ), idx)); idx += 1

    # 3. Load data
    cells.append(_make_cell("code", (
        f'df = pd.read_csv("{dataset_path}")\n'
        "print(f'Shape: {df.shape}')\n"
        "df.head(10)"
    ), idx)); idx += 1

    # 4. Basic stats
    cells.append(_make_cell("code", (
        "df.describe()"
    ), idx)); idx += 1

    cells.append(_make_cell("code", (
        "df.info()"
    ), idx)); idx += 1

    # 5. Missingness
    cells.append(_make_cell("code", (
        "missing = df.isnull().sum()\n"
        "missing = missing[missing > 0].sort_values(ascending=False)\n"
        "if len(missing) > 0:\n"
        "    missing.plot.bar(color='coral', title='Missing Values by Column')\n"
        "    plt.ylabel('Count')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No missing values found.')"
    ), idx)); idx += 1

    # 6. Correlation heatmap
    cells.append(_make_cell("code", (
        "numeric_df = df.select_dtypes(include='number')\n"
        "if numeric_df.shape[1] >= 2:\n"
        "    plt.figure(figsize=(10, 8))\n"
        "    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')\n"
        "    plt.title('Feature Correlation Heatmap')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ), idx)); idx += 1

    # 7. Target distribution (generic — uses first column as placeholder)
    cells.append(_make_cell("code", (
        "# Update target_column to match your dataset\n"
        "target_column = df.columns[-1]\n"
        "if target_column in df.columns:\n"
        "    df[target_column].value_counts().sort_index().plot.bar(\n"
        "        color=['steelblue', 'tomato'], title=f'Target Distribution: {target_column}'\n"
        "    )\n"
        "    plt.ylabel('Count')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ), idx)); idx += 1

    # 8. Feature distributions
    cells.append(_make_cell("code", (
        "numeric_cols = df.select_dtypes(include='number').columns[:8]\n"
        "if len(numeric_cols) > 0:\n"
        "    df[numeric_cols].hist(bins=30, figsize=(14, 8), color='steelblue', edgecolor='white')\n"
        "    plt.suptitle('Numeric Feature Distributions')\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ), idx)); idx += 1

    # 9. Data quality report + cleaning recommendations
    report_md = "## Data Quality Report\n\n"
    if data_quality_report:
        report_md += f"```\n{data_quality_report}\n```\n\n"
    if cleaning_report:
        report_md += f"## Cleaning Summary\n\n```\n{cleaning_report}\n```\n"
    cells.append(_make_cell("markdown", report_md, idx)); idx += 1

    # 10. LLM analysis (if available)
    if llm_analysis:
        cells.append(_make_cell("markdown", "## AI-Generated Analysis Script"), idx); idx += 1
        cells.append(_make_cell("code", llm_analysis, idx)); idx += 1

    # 11. Next steps
    cells.append(_make_cell("markdown", (
        "## Next Steps\n\n"
        "- Review data quality issues above and apply cleaning steps\n"
        "- Consider feature engineering based on correlation analysis\n"
        "- Proceed to modeling with the DS Agent full pipeline\n"
        "- Address class imbalance if target distribution is skewed"
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
