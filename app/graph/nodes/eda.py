import numpy as np
import pandas as pd

from app.graph.state import AgentState
from da_agent.charts import generate_eda_charts, save_fig

OUTPUT_DIR = "outputs"


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

    # --- Visualizations (shared with DA Agent) ---
    viz_paths = generate_eda_charts(df, target, output_dir=OUTPUT_DIR)
    lines.append(f"\nVisualizations saved ({len(viz_paths)}):")
    for p in viz_paths:
        lines.append(f"  {p}")

    report = "\n".join(lines)
    print(report)

    return {"dataframe": df, "eda_report": report}
