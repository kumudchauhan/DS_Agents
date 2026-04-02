import numpy as np
import pandas as pd

from app.graph.state import AgentState

DROP_COLS = [
    "transaction_id", "user_id", "timestamp",
    "user_signup_date", "merchant_category",
    "transaction_type", "device_type",
]
CATEGORICAL_COLS = ["merchant_category", "transaction_type", "device_type"]


def feature_engineering_node(state: AgentState) -> dict:
    """Create features from the cleaned dataframe.  Later iterations add more."""
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING NODE - Iteration", state["iteration"])
    print("=" * 60)

    # Always start from the cleaned dataframe so we have raw columns available
    source = state.get("cleaned_dataframe")
    if source is None:
        source = state["dataframe"]
    df = source.copy()
    iteration = state["iteration"]
    lines = []

    # ---- Base features (every iteration) ----

    if "timestamp" in df.columns and "user_signup_date" in df.columns:
        df["account_age_days"] = (df["timestamp"] - df["user_signup_date"]).dt.days
        lines.append("Created: account_age_days")

    if "timestamp" in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        lines.append("Created: hour_of_day, day_of_week")

    if "amount" in df.columns and "avg_amount_7d" in df.columns:
        safe_avg = df["avg_amount_7d"].replace(0, np.nan)
        df["amount_to_avg_ratio"] = (df["amount"] / safe_avg).fillna(1.0)
        lines.append("Created: amount_to_avg_ratio")

    if "amount" in df.columns:
        threshold = df["amount"].quantile(0.95)
        df["is_high_amount"] = (df["amount"] > threshold).astype(int)
        lines.append(f"Created: is_high_amount (threshold={threshold:.2f})")

    # ---- Iteration 1+ features ----

    if iteration >= 1:
        if "amount" in df.columns:
            df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
            lines.append("Created: log_amount")

        if "account_age_days" in df.columns:
            df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
            lines.append("Created: is_new_account")

        if "hour_of_day" in df.columns:
            df["is_night_txn"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)
            lines.append("Created: is_night_txn")

    # ---- Iteration 2+ features ----

    if iteration >= 2:
        if "amount" in df.columns and "avg_amount_7d" in df.columns:
            df["amount_deviation"] = (df["amount"] - df["avg_amount_7d"]).abs()
            lines.append("Created: amount_deviation")

        if "prior_transactions_24h" in df.columns:
            df["high_velocity"] = (df["prior_transactions_24h"] >= 4).astype(int)
            lines.append("Created: high_velocity")

    # ---- Encode categoricals ----

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            lines.append(f"One-hot encoded: {col}")

    # ---- Drop non-feature columns ----

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    lines.append(f"\nFinal feature count: {df.shape[1]}")
    lines.append(f"Shape: {df.shape}")
    report = "\n".join(lines)
    print(report)

    return {"dataframe": df, "feature_report": report}
