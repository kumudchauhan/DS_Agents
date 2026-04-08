"""DA Agent orchestrator — analysis-only pipeline, no modeling."""

from __future__ import annotations

import pandas as pd

from da_agent.stats import generate_stat_block
from da_agent.charts import generate_eda_charts
from da_agent.prompts import build_da_prompt
from da_agent.notebook_export import generate_notebook


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


def run_da_agent(
    dataset_path: str,
    target_column: str,
    instructions: dict | None = None,
    local_mode: bool = True,
    llm_model_name: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Run the DA Agent analysis-only pipeline.

    Steps:
    1. Load CSV -> pd.DataFrame
    2. Generate stat block (stats.py)
    3. Run data cleaning
    4. Generate EDA charts (charts.py)
    5. Build data quality report
    6. Call LLM with DA prompt + stat block -> get analysis script (best-effort)
    7. Generate notebook artifact with analysis cells only

    Returns dict with dataframe, cleaned_dataframe, stat_block, cleaning_report,
    eda_report, chart_paths, data_quality_report, llm_analysis, notebook_bytes.
    """
    print("\n" + "=" * 60)
    print("  DA AGENT — Data Analysis Pipeline")
    print("=" * 60)

    # 1. Load CSV
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Coerce target to numeric early (for stat block)
    if target_column in df.columns:
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    # 2. Generate stat block
    stat_block = generate_stat_block(df, target_column)
    print("Stat block generated.")

    # 3. Run cleaning
    cleaned_df, cleaning_report = _run_cleaning(df.copy(), target_column)
    print(f"Cleaning complete. Shape: {df.shape} -> {cleaned_df.shape}")

    # 4. Generate EDA charts
    chart_paths = generate_eda_charts(df, target_column)
    print(f"Generated {len(chart_paths)} visualizations.")

    # 5. Build data quality report
    data_quality_report = _build_data_quality_report(df, target_column)
    print("Data quality report built.")

    # 6. LLM analysis (best-effort)
    llm_analysis: str | None = None
    llm_error: str | None = None
    try:
        from da_agent.llm_manager import get_da_llm
        llm = get_da_llm(local_mode=local_mode, model_name=llm_model_name, api_key=api_key)
        prompt = build_da_prompt(stat_block, instructions, dataset_path)
        response = llm.invoke(prompt)
        llm_analysis = response.content
        print("LLM analysis generated.")
    except Exception as e:
        llm_error = str(e)
        print(f"LLM unavailable ({e}). Skipping AI analysis — deterministic results still available.")

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
    notebook_bytes = generate_notebook(
        dataset_path=dataset_path,
        stat_block=stat_block,
        cleaning_report=cleaning_report,
        data_quality_report=data_quality_report,
        llm_analysis=llm_analysis,
    )
    print("Notebook generated.")

    print("=" * 60)
    print("  DA AGENT COMPLETE")
    print("=" * 60)

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
    }
