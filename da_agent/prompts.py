"""DA Agent system prompt and instruction threading."""

from __future__ import annotations

DA_SYSTEM_PROMPT = """You are a Senior Data Analyst. Your goal is to provide immediate \
statistical sanity in one shot.

Prioritize Speed: Use standard libraries (Pandas, Seaborn).
Statistical Rigor: Always start by identifying data types, missingness, \
and potential target leakage.
Deliverable: Output a clean Python script for data exploration that the user \
can download or run immediately.

You do NOT build models or do feature engineering. Focus exclusively on:
- Data profiling and quality assessment
- Distribution analysis
- Correlation analysis
- Anomaly and outlier detection
- Data cleaning recommendations
- Summary statistics and insights"""


def build_da_prompt(
    stat_block: str,
    instructions: dict | None = None,
    dataset_path: str = "",
) -> str:
    """Build the full DA Agent prompt with stat block + user instructions.

    Sections:
    1. System prompt
    2. Stat block (data profile from stats.py)
    3. User instructions (if provided)
    4. Task directive
    """
    parts: list[str] = [DA_SYSTEM_PROMPT, ""]

    # Stat block
    parts.append("# Data Profile")
    parts.append(stat_block)
    parts.append("")

    # User instructions
    if instructions:
        injected = []

        dataset_info = instructions.get("dataset", {})
        if dataset_info:
            items = ", ".join(f"{k}: {v}" for k, v in dataset_info.items())
            injected.append(f"Dataset context: {items}")

        priorities = instructions.get("priorities", [])
        if priorities:
            injected.append("User priorities: " + "; ".join(priorities))

        viz_prefs = instructions.get("visualization", [])
        if viz_prefs:
            injected.append("Visualization preferences: " + "; ".join(viz_prefs))

        if injected:
            parts.append("# User Instructions")
            parts.extend(injected)
            parts.append("")

    # Task
    parts.append("# Task")
    if dataset_path:
        parts.append(f'Dataset file: "{dataset_path}"')
    parts.append(
        "Generate a complete data exploration Python script for this dataset. "
        "Include imports, data loading, profiling, distribution plots, "
        "correlation analysis, outlier detection, and a summary of findings. "
        "Output ONLY the Python script, no extra commentary."
    )

    return "\n".join(parts)
