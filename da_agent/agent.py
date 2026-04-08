"""DA Agent orchestrator — analysis-only pipeline, no modeling."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from da_agent.stats import generate_stat_block
from da_agent.charts import generate_eda_charts
from da_agent.prompts import build_da_prompt
from da_agent.notebook_export import generate_notebook

# Maximum seconds to wait for an LLM response before giving up.
LLM_TIMEOUT_SECONDS = 90


def _run_cleaning(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, str]:
    """Lightweight cleaning pass reusing logic from the DS Agent cleaning node."""
    lines: list[str] = []
    initial_shape = df.shape

    STRING_COLS = ["merchant_category", "transaction_type", "device_type"]
    NUMERIC_FILL_COLS = ["amount", "avg_amount_7d", "is_international", "prior_transactions_24h"]
    TIMESTAMP_COLS = ["timestamp", "user_signup_date"]

    import numpy as np

    # Coerce numeric columns
    for col in NUMERIC_FILL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    lines.append("Coerced numeric columns to proper dtype")

    # Normalise string columns
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "n/a": np.nan, "none": np.nan, "": np.nan})
    lines.append("Normalised string columns to lowercase")

    # Fill numeric NaNs with median
    for col in NUMERIC_FILL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            lines.append(f"Filled {col} NaN with median ({median_val:.2f})")

    # Fill categorical NaNs with mode
    for col in STRING_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            lines.append(f"Filled {col} NaN with mode ({mode_val})")

    # Clean target column
    if target_column in df.columns:
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        before = len(df)
        df = df.dropna(subset=[target_column])
        df = df[df[target_column].isin([0.0, 1.0])]
        dropped = before - len(df)
        if dropped > 0:
            lines.append(f"Dropped {dropped} rows with missing/invalid target")

    # Parse timestamps
    for col in TIMESTAMP_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    lines.append("Parsed timestamp columns")

    lines.append(f"\nShape: {initial_shape} -> {df.shape}")
    return df, "\n".join(lines)


def _build_data_quality_report(df: pd.DataFrame, target: str) -> str:
    """Build a text-based data quality report."""
    import numpy as np

    total = len(df)
    lines = [
        "--- DATA QUALITY REPORT ---",
        f"Total rows: {total}",
        f"Total columns: {df.shape[1]}",
    ]

    # Missing values
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

    # Duplicates
    dup_count = df.duplicated().sum()
    lines.append(f"\nDuplicate rows: {dup_count}")

    # Outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    outlier_lines = []
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            outliers = ((df[col] - mean).abs() > 3 * std).sum()
            if outliers > 0:
                outlier_lines.append(f"  {col}: {outliers} outliers (>3*std)")
    if outlier_lines:
        lines.append("\nPotential outliers (>3 std):")
        lines.extend(outlier_lines)

    # Target quality
    if target in df.columns:
        unique_vals = sorted(df[target].dropna().unique())
        lines.append(f"\nTarget column '{target}':")
        lines.append(f"  Unique values: {unique_vals}")
        lines.append(f"  NaN count: {df[target].isnull().sum()}")

    return "\n".join(lines)


def _noop(*_args: object, **_kwargs: object) -> None:
    pass


def run_da_agent(
    dataset_path: str,
    target_column: str,
    instructions: dict | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Run the DA Agent analysis-only pipeline.

    Parameters
    ----------
    on_progress : callable, optional
        Called with a status string after each step completes, e.g.
        ``on_progress("Loading dataset...")``.  Designed for Streamlit's
        ``st.status.update()``.
    """
    progress = on_progress or _noop
    t0 = time.time()

    def _elapsed() -> str:
        return f"{time.time() - t0:.1f}s"

    # 1. Load CSV
    progress("Loading dataset...")
    df = pd.read_csv(dataset_path)
    progress(f"Dataset loaded — {df.shape[0]:,} rows x {df.shape[1]} columns ({_elapsed()})")

    if target_column in df.columns:
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    # 2. Generate stat block
    progress("Generating data profile...")
    stat_block = generate_stat_block(df, target_column)
    progress(f"Data profile ready ({_elapsed()})")

    # 3. Run cleaning
    progress("Cleaning data...")
    cleaned_df, cleaning_report = _run_cleaning(df.copy(), target_column)
    progress(f"Cleaning complete — {df.shape} -> {cleaned_df.shape} ({_elapsed()})")

    # 4. Generate EDA charts
    progress("Generating visualizations...")
    chart_paths = generate_eda_charts(df, target_column)
    progress(f"Generated {len(chart_paths)} charts ({_elapsed()})")

    # 5. Build data quality report
    progress("Building data quality report...")
    data_quality_report = _build_data_quality_report(df, target_column)
    progress(f"Data quality report ready ({_elapsed()})")

    # 6. LLM analysis — skipped for quick explore (deterministic only)
    llm_analysis: str | None = None
    llm_error: str | None = None

    # Build full EDA report text
    eda_lines = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"\nColumn types:\n{df.dtypes.to_string()}",
        f"\nBasic statistics:\n{df.describe().to_string()}",
    ]
    if target_column in df.columns:
        dist = df[target_column].value_counts()
        eda_lines.append(f"\nTarget distribution ({target_column}):\n{dist.to_string()}")
    eda_lines.append(f"\n{data_quality_report}")
    eda_lines.append(f"\nVisualizations saved ({len(chart_paths)}):")
    for p in chart_paths:
        eda_lines.append(f"  {p}")
    eda_report = "\n".join(eda_lines)

    # 7. Generate notebook
    progress("Generating notebook...")
    notebook_bytes = generate_notebook(
        dataset_path=dataset_path,
        stat_block=stat_block,
        cleaning_report=cleaning_report,
        data_quality_report=data_quality_report,
        llm_analysis=llm_analysis,
    )

    total = _elapsed()
    progress(f"DA Agent complete ({total})")

    return {
        "dataframe": df,
        "cleaned_dataframe": cleaned_df,
        "stat_block": stat_block,
        "cleaning_report": cleaning_report,
        "eda_report": eda_report,
        "chart_paths": chart_paths,
        "data_quality_report": data_quality_report,
        "llm_analysis": llm_analysis,
        "llm_error": llm_error,
        "notebook_bytes": notebook_bytes,
        "elapsed": total,
    }
