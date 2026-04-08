"""Pre-flight stat block generator.

Pure Python — no LLM needed. Output is a markdown string fed into the DA
Agent prompt so the LLM doesn't guess data shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_stat_block(df: pd.DataFrame, target_column: str) -> str:
    """Generate a markdown stat block for the DA Agent.

    Includes shape, dtypes summary, describe(), missingness report,
    target distribution, column cardinality, and potential target leakage
    flags.
    """
    lines: list[str] = []

    # --- Shape ---
    lines.append(f"## Dataset Shape\n{df.shape[0]} rows x {df.shape[1]} columns\n")

    # --- Dtypes summary (grouped) ---
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime = df.select_dtypes(include="datetime").columns.tolist()
    other = [c for c in df.columns if c not in numeric + categorical + datetime]

    lines.append("## Column Types")
    lines.append(f"- Numeric ({len(numeric)}): {', '.join(numeric[:10])}"
                 + (" ..." if len(numeric) > 10 else ""))
    if categorical:
        lines.append(f"- Categorical ({len(categorical)}): {', '.join(categorical[:10])}"
                     + (" ..." if len(categorical) > 10 else ""))
    if datetime:
        lines.append(f"- Datetime ({len(datetime)}): {', '.join(datetime)}")
    if other:
        lines.append(f"- Other ({len(other)}): {', '.join(other)}")
    lines.append("")

    # --- Describe ---
    lines.append("## Descriptive Statistics")
    lines.append(df.describe().to_string())
    lines.append("")

    # --- Missingness report ---
    missing = df.isnull().sum()
    total = len(df)
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    if len(missing_cols) > 0:
        lines.append(f"## Missing Values ({len(missing_cols)} columns)")
        for col in missing_cols.index:
            count = int(missing_cols[col])
            pct = count / total * 100
            lines.append(f"- {col}: {count} ({pct:.1f}%)")
    else:
        lines.append("## Missing Values\nNone")
    lines.append("")

    # --- Target distribution ---
    if target_column in df.columns:
        lines.append(f"## Target Distribution ({target_column})")
        target_series = pd.to_numeric(df[target_column], errors="coerce")
        vc = target_series.value_counts().sort_index()
        for val, count in vc.items():
            pct = count / total * 100
            lines.append(f"- {val}: {count} ({pct:.1f}%)")
        if len(vc) == 2:
            minority = vc.min()
            lines.append(f"- Class balance ratio: {minority / total * 100:.1f}% minority")
    lines.append("")

    # --- Column cardinality ---
    lines.append("## Column Cardinality")
    for col in df.columns:
        nuniq = df[col].nunique()
        lines.append(f"- {col}: {nuniq} unique")
    lines.append("")

    # --- Target leakage flags ---
    if target_column in df.columns:
        target_numeric = pd.to_numeric(df[target_column], errors="coerce")
        num_df = df[numeric].copy()
        if target_column in num_df.columns:
            leakage_flags = []
            for col in num_df.columns:
                if col == target_column:
                    continue
                try:
                    corr = num_df[col].corr(target_numeric)
                    if abs(corr) > 0.9:
                        leakage_flags.append(f"- {col}: correlation={corr:.3f}")
                except Exception:
                    pass
            if leakage_flags:
                lines.append("## Potential Target Leakage")
                lines.extend(leakage_flags)
                lines.append("")

    return "\n".join(lines)
