import numpy as np
import pandas as pd

from app.graph.state import AgentState

DROP_COLS = [
    "transaction_id", "user_id", "timestamp",
    "user_signup_date", "merchant_category",
    "transaction_type", "device_type",
]
CATEGORICAL_COLS = ["merchant_category", "transaction_type", "device_type"]

# Default features used on iteration 0 (before any LLM recommendations exist).
DEFAULT_FEATURES = [
    "account_age_days", "hour_of_day", "day_of_week",
    "amount_to_avg_ratio", "is_high_amount",
]


# ---------------------------------------------------------------------------
# Feature registry — each entry is a callable(df) -> (df, description | None)
# ---------------------------------------------------------------------------

def _account_age_days(df):
    if "timestamp" in df.columns and "user_signup_date" in df.columns:
        df["account_age_days"] = (df["timestamp"] - df["user_signup_date"]).dt.days
        return df, "account_age_days = (timestamp - signup_date).days"
    return df, None


def _hour_of_day(df):
    if "timestamp" in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
        return df, "hour_of_day = timestamp.hour"
    return df, None


def _day_of_week(df):
    if "timestamp" in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        return df, "day_of_week = timestamp.dayofweek"
    return df, None


def _amount_to_avg_ratio(df):
    if "amount" in df.columns and "avg_amount_7d" in df.columns:
        safe_avg = df["avg_amount_7d"].replace(0, np.nan)
        df["amount_to_avg_ratio"] = (df["amount"] / safe_avg).fillna(1.0)
        return df, "amount_to_avg_ratio = amount / avg_amount_7d"
    return df, None


def _is_high_amount(df):
    if "amount" in df.columns:
        threshold = df["amount"].quantile(0.95)
        df["is_high_amount"] = (df["amount"] > threshold).astype(int)
        return df, f"is_high_amount = (amount > {threshold:.2f})"
    return df, None


def _log_amount(df):
    if "amount" in df.columns:
        df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
        return df, "log_amount = log1p(amount)"
    return df, None


def _is_new_account(df):
    if "account_age_days" not in df.columns:
        if "timestamp" in df.columns and "user_signup_date" in df.columns:
            df["account_age_days"] = (df["timestamp"] - df["user_signup_date"]).dt.days
        else:
            return df, None
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    return df, "is_new_account = (account_age_days < 30)"


def _is_night_txn(df):
    if "hour_of_day" not in df.columns:
        if "timestamp" in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
        else:
            return df, None
    df["is_night_txn"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)
    return df, "is_night_txn = (hour >= 23 or hour <= 5)"


def _amount_deviation(df):
    if "amount" in df.columns and "avg_amount_7d" in df.columns:
        df["amount_deviation"] = (df["amount"] - df["avg_amount_7d"]).abs()
        return df, "amount_deviation = |amount - avg_amount_7d|"
    return df, None


def _high_velocity(df):
    if "prior_transactions_24h" in df.columns:
        df["high_velocity"] = (df["prior_transactions_24h"] >= 4).astype(int)
        return df, "high_velocity = (prior_transactions_24h >= 4)"
    return df, None


def _amount_squared(df):
    if "amount" in df.columns:
        df["amount_squared"] = df["amount"] ** 2
        return df, "amount_squared = amount^2"
    return df, None


def _amount_x_velocity(df):
    if "amount" in df.columns and "prior_transactions_24h" in df.columns:
        df["amount_x_velocity"] = df["amount"] * df["prior_transactions_24h"]
        return df, "amount_x_velocity = amount * prior_transactions_24h"
    return df, None


def _is_weekend(df):
    if "day_of_week" not in df.columns:
        if "timestamp" in df.columns:
            df["day_of_week"] = df["timestamp"].dt.dayofweek
        else:
            return df, None
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df, "is_weekend = (day_of_week >= 5)"


def _txn_per_avg_ratio(df):
    if "prior_transactions_24h" in df.columns and "avg_amount_7d" in df.columns:
        safe_avg = df["avg_amount_7d"].replace(0, np.nan)
        df["txn_per_avg_ratio"] = (df["prior_transactions_24h"] / safe_avg).fillna(0.0)
        return df, "txn_per_avg_ratio = prior_transactions_24h / avg_amount_7d"
    return df, None


FEATURE_REGISTRY = {
    "account_age_days": _account_age_days,
    "hour_of_day": _hour_of_day,
    "day_of_week": _day_of_week,
    "amount_to_avg_ratio": _amount_to_avg_ratio,
    "is_high_amount": _is_high_amount,
    "log_amount": _log_amount,
    "is_new_account": _is_new_account,
    "is_night_txn": _is_night_txn,
    "amount_deviation": _amount_deviation,
    "high_velocity": _high_velocity,
    "amount_squared": _amount_squared,
    "amount_x_velocity": _amount_x_velocity,
    "is_weekend": _is_weekend,
    "txn_per_avg_ratio": _txn_per_avg_ratio,
}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def feature_engineering_node(state: AgentState) -> dict:
    """Create features based on LLM recommendations (or defaults for iter 0)."""
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING NODE - Iteration", state["iteration"])
    print("=" * 60)

    source = state.get("cleaned_dataframe")
    if source is None:
        source = state["dataframe"]
    df = source.copy()
    lines = []

    # Determine which features to build.
    recommendations = state.get("recommendations") or {}
    requested = recommendations.get("features_to_add", None)

    if not requested:
        requested = list(DEFAULT_FEATURES)
        lines.append("Using default feature set (no LLM recommendations yet)")
    else:
        lines.append(f"Using LLM-recommended features ({len(requested)})")

    # Apply instruction overrides (iteration 0 only, before LLM has spoken).
    instructions = state.get("instructions") or {}
    feat_instructions = instructions.get("features", {})
    if feat_instructions:
        must_include = feat_instructions.get("must_include", [])
        avoid = feat_instructions.get("avoid", [])
        # Merge must_include features that aren't already requested
        for f in must_include:
            if f in FEATURE_REGISTRY and f not in requested:
                requested.append(f)
                lines.append(f"Added from instructions: {f}")
        # Remove avoided features
        if avoid:
            before = len(requested)
            requested = [f for f in requested if f not in avoid]
            removed = before - len(requested)
            if removed:
                lines.append(f"Removed {removed} features per instructions (avoid: {', '.join(avoid)})")

    # Apply features from registry.
    for name in requested:
        fn = FEATURE_REGISTRY.get(name)
        if fn is None:
            lines.append(f"Skipped unknown feature: {name}")
            continue
        df, desc = fn(df)
        if desc:
            lines.append(f"Created: {desc}")

    # Encode categoricals (always applied).
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            lines.append(f"One-hot encoded: {col}")

    # Drop non-feature columns.
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    lines.append(f"\nFinal feature count: {df.shape[1]}")
    lines.append(f"Shape: {df.shape}")
    report = "\n".join(lines)
    print(report)

    return {"dataframe": df, "feature_report": report}
