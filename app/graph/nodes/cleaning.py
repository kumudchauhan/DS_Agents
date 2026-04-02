import numpy as np
import pandas as pd

from app.graph.state import AgentState

STRING_COLS = ["merchant_category", "transaction_type", "device_type"]
NUMERIC_FILL_COLS = ["amount", "avg_amount_7d", "is_international", "prior_transactions_24h"]
TIMESTAMP_COLS = ["timestamp", "user_signup_date"]


def cleaning_node(state: AgentState) -> dict:
    """Clean the raw dataframe: normalise strings, fill NaNs, parse dates."""
    print("\n" + "=" * 60)
    print("  CLEANING NODE - Iteration", state["iteration"])
    print("=" * 60)

    df = state["dataframe"].copy()
    lines = []
    initial_shape = df.shape

    # --- Coerce numeric columns that may have been read as object ---
    for col in NUMERIC_FILL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    lines.append("Coerced numeric columns to proper dtype")

    # --- Normalise string columns ---
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "n/a": np.nan, "none": np.nan, "": np.nan})
    lines.append("Normalised string columns to lowercase")

    # --- Fill numeric NaNs with median ---
    for col in NUMERIC_FILL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            lines.append(f"Filled {col} NaN with median ({median_val:.2f})")

    # --- Fill categorical NaNs with mode ---
    for col in STRING_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            lines.append(f"Filled {col} NaN with mode ({mode_val})")

    # --- Clean target column: coerce to numeric, keep only valid 0/1 ---
    target = state["target_column"]
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
        before = len(df)
        df = df.dropna(subset=[target])
        # Remove rows with invalid target values (keep only 0 and 1)
        df = df[df[target].isin([0.0, 1.0])]
        dropped = before - len(df)
        if dropped > 0:
            lines.append(f"Dropped {dropped} rows with missing/invalid target")

    # --- Parse timestamps ---
    for col in TIMESTAMP_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    lines.append("Parsed timestamp columns")

    lines.append(f"\nShape: {initial_shape} -> {df.shape}")
    report = "\n".join(lines)
    print(report)

    return {"dataframe": df, "cleaned_dataframe": df.copy(), "cleaning_report": report}
