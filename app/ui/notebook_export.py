"""Export the data-science pipeline as a standalone Jupyter notebook (.ipynb)."""

import json
import re


# ---------------------------------------------------------------------------
# Feature name -> (dependencies, inline code)
# Each code string may contain \n for multi-line statements.
# ---------------------------------------------------------------------------

_FEATURE_CODE = {
    "account_age_days": (
        [],
        'df["account_age_days"] = (df["timestamp"] - df["user_signup_date"]).dt.days',
    ),
    "hour_of_day": (
        [],
        'df["hour_of_day"] = df["timestamp"].dt.hour',
    ),
    "day_of_week": (
        [],
        'df["day_of_week"] = df["timestamp"].dt.dayofweek',
    ),
    "amount_to_avg_ratio": (
        [],
        'df["amount_to_avg_ratio"] = (df["amount"] / df["avg_amount_7d"].replace(0, np.nan)).fillna(1.0)',
    ),
    "is_high_amount": (
        [],
        '_q95 = df["amount"].quantile(0.95)\n'
        'df["is_high_amount"] = (df["amount"] > _q95).astype(int)',
    ),
    "log_amount": (
        [],
        'df["log_amount"] = np.log1p(df["amount"].clip(lower=0))',
    ),
    "is_new_account": (
        ["account_age_days"],
        'df["is_new_account"] = (df["account_age_days"] < 30).astype(int)',
    ),
    "is_night_txn": (
        ["hour_of_day"],
        'df["is_night_txn"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)',
    ),
    "amount_deviation": (
        [],
        'df["amount_deviation"] = (df["amount"] - df["avg_amount_7d"]).abs()',
    ),
    "high_velocity": (
        [],
        'df["high_velocity"] = (df["prior_transactions_24h"] >= 4).astype(int)',
    ),
    "amount_squared": (
        [],
        'df["amount_squared"] = df["amount"] ** 2',
    ),
    "amount_x_velocity": (
        [],
        'df["amount_x_velocity"] = df["amount"] * df["prior_transactions_24h"]',
    ),
    "is_weekend": (
        ["day_of_week"],
        'df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)',
    ),
    "txn_per_avg_ratio": (
        [],
        'df["txn_per_avg_ratio"] = (df["prior_transactions_24h"] / df["avg_amount_7d"].replace(0, np.nan)).fillna(0.0)',
    ),
}

_CATEGORICAL_COLS = ["merchant_category", "transaction_type", "device_type"]

_DROP_COLS = [
    "transaction_id", "user_id", "timestamp",
    "user_signup_date", "merchant_category",
    "transaction_type", "device_type",
]

_NEEDS_SCALING = {"LogisticRegression", "SVC"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cell(cell_type: str, source: str, cell_id: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        "id": cell_id,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _parse_features_used(feature_report: str) -> list[str]:
    """Extract feature names from a feature_report string."""
    features = []
    for line in feature_report.split("\n"):
        if line.strip().startswith("Created:"):
            part = line.split("Created:")[-1].strip()
            name = part.split("=")[0].strip()
            if name in _FEATURE_CODE:
                features.append(name)
    return features


def _parse_model_info(model_str: str) -> tuple[str, dict]:
    """Extract model name and hyperparameters from model report string."""
    first_line = model_str.split("\n")[0] if model_str else ""
    match = re.match(r"Model:\s*(\w+)\((.*?)\)", first_line)
    if not match:
        return "LogisticRegression", {"C": 1.0}

    name = match.group(1)
    hp_str = match.group(2)
    hp: dict = {}
    for pair in hp_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                hp[k] = float(v) if "." in v else int(v)
            except ValueError:
                hp[k] = v
    return name, hp


def _build_feature_code(features_used: list[str]) -> list[str]:
    """Return code lines for feature engineering in dependency order."""
    added: set[str] = set()
    lines: list[str] = []

    def _add(name: str) -> None:
        if name in added or name not in _FEATURE_CODE:
            return
        deps, code = _FEATURE_CODE[name]
        for dep in deps:
            _add(dep)
        lines.append(f"# {name}")
        lines.append(code)
        lines.append("")
        added.add(name)

    for f in features_used:
        _add(f)

    return lines


def _build_model_code(model_name: str, hp: dict) -> str:
    """Return a single line instantiating the sklearn model."""
    if model_name == "LogisticRegression":
        c = hp.get("C", 1.0)
        return f"model = LogisticRegression(C={c}, max_iter=1000, random_state=42, class_weight='balanced')"

    if model_name == "RandomForestClassifier":
        ne = int(hp.get("n_estimators", 100))
        md = hp.get("max_depth")
        md_str = str(int(md)) if md is not None else "None"
        return (
            f"model = RandomForestClassifier(n_estimators={ne}, max_depth={md_str}, "
            f"random_state=42, class_weight='balanced')"
        )

    if model_name == "GradientBoostingClassifier":
        ne = int(hp.get("n_estimators", 150))
        lr = hp.get("learning_rate", 0.1)
        md = int(hp.get("max_depth", 5))
        return (
            f"model = GradientBoostingClassifier(n_estimators={ne}, learning_rate={lr}, "
            f"max_depth={md}, random_state=42)"
        )

    if model_name == "SVC":
        c = hp.get("C", 1.0)
        kernel = hp.get("kernel", "rbf")
        return (
            f"model = SVC(C={c}, kernel='{kernel}', class_weight='balanced', "
            f"probability=True, random_state=42)"
        )

    # Fallback
    return "model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_notebook(
    history: list[dict],
    target_column: str,
    dataset_filename: str = "dataset.csv",
) -> str:
    """Build a Jupyter notebook JSON string from the pipeline history.

    Returns the notebook as a JSON string ready for download.
    """
    best = max(history, key=lambda h: h.get("f1", 0) or 0)
    features_used = _parse_features_used(best.get("feature_report", ""))
    model_name, hp = _parse_model_info(best.get("model", ""))

    cells: list[dict] = []
    cell_idx = 0

    def add(cell_type: str, source_lines: list[str]) -> None:
        nonlocal cell_idx
        cell_idx += 1
        src = "\n".join(source_lines)
        cells.append(_make_cell(cell_type, src, f"cell-{cell_idx}"))

    # ── 1. Title ─────────────────────────────────────────────────────
    add("markdown", [
        "# Data Science Pipeline",
        "",
        "Exported from the **Autonomous Data Science Agent**.",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Target column | `{target_column}` |",
        f"| Best iteration | {best['iteration']} |",
        f"| F1 score | {best.get('f1', 'N/A')} |",
        f"| Model | {model_name} |",
    ])

    # ── 2. Imports ───────────────────────────────────────────────────
    add("code", [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.svm import SVC",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.metrics import (",
        "    accuracy_score,",
        "    classification_report,",
        "    confusion_matrix,",
        "    f1_score,",
        "    precision_score,",
        "    recall_score,",
        ")",
    ])

    # ── 3. Load Data ─────────────────────────────────────────────────
    add("markdown", ["## 1. Load Data"])
    add("code", [
        f'df = pd.read_csv("{dataset_filename}")',
        f'df["{target_column}"] = pd.to_numeric(df["{target_column}"], errors="coerce")',
        "",
        'print(f"Shape: {df.shape}")',
        "df.head()",
    ])

    # ── 4. EDA ───────────────────────────────────────────────────────
    add("markdown", [
        "## 2. Exploratory Data Analysis",
    ])
    add("code", [
        "df.describe(include='all')",
    ])
    add("code", [
        "# Missing values",
        "missing = df.isnull().sum()",
        "missing = missing[missing > 0].sort_values(ascending=False)",
        'print("Columns with missing values:")',
        "print(missing)",
        "",
        "# Target distribution",
        f'print("\\nTarget distribution:")',
        f'print(df["{target_column}"].value_counts())',
        f'_pos_rate = df["{target_column}"].mean() * 100',
        'print(f"Positive class rate: {_pos_rate:.2f}%")',
    ])
    add("code", [
        "fig, ax = plt.subplots(figsize=(6, 4))",
        f'counts = df["{target_column}"].value_counts().sort_index()',
        "counts.plot.bar(ax=ax, color=['steelblue', 'tomato'][:len(counts)])",
        f'ax.set_title("Target Distribution: {target_column}")',
        f'ax.set_xlabel("{target_column}")',
        'ax.set_ylabel("Count")',
        "plt.tight_layout()",
        "plt.show()",
    ])

    # ── 5. Data Cleaning ─────────────────────────────────────────────
    add("markdown", [
        "## 3. Data Cleaning",
        "",
        "Normalise strings, fill missing values, parse timestamps, and remove invalid target rows.",
    ])
    add("code", [
        'NUMERIC_FILL_COLS = ["amount", "avg_amount_7d", "is_international", "prior_transactions_24h"]',
        'STRING_COLS = ["merchant_category", "transaction_type", "device_type"]',
        'TIMESTAMP_COLS = ["timestamp", "user_signup_date"]',
        "",
        "# Coerce numeric columns",
        "for col in NUMERIC_FILL_COLS:",
        "    if col in df.columns:",
        "        df[col] = pd.to_numeric(df[col], errors='coerce')",
        "",
        "# Normalise string columns",
        "for col in STRING_COLS:",
        "    if col in df.columns:",
        "        df[col] = df[col].astype(str).str.strip().str.lower()",
        "        df[col] = df[col].replace({'nan': np.nan, 'n/a': np.nan, 'none': np.nan, '': np.nan})",
        "",
        "# Fill numeric NaNs with median",
        "for col in NUMERIC_FILL_COLS:",
        "    if col in df.columns and df[col].isnull().any():",
        "        median_val = df[col].median()",
        "        df[col] = df[col].fillna(median_val)",
        '        print(f"Filled {col} NaN with median ({median_val:.2f})")',
        "",
        "# Fill categorical NaNs with mode",
        "for col in STRING_COLS:",
        "    if col in df.columns and df[col].isnull().any():",
        "        mode_val = df[col].mode()[0]",
        "        df[col] = df[col].fillna(mode_val)",
        '        print(f"Filled {col} NaN with mode ({mode_val})")',
        "",
        "# Clean target column — keep only valid 0/1 rows",
        f'df = df.dropna(subset=["{target_column}"])',
        f'df = df[df["{target_column}"].isin([0.0, 1.0])]',
        "",
        "# Parse timestamps",
        "for col in TIMESTAMP_COLS:",
        "    if col in df.columns:",
        "        df[col] = pd.to_datetime(df[col], errors='coerce')",
        "",
        'print(f"Cleaned shape: {df.shape}")',
    ])

    # ── 6. Feature Engineering ───────────────────────────────────────
    feature_lines = _build_feature_code(features_used)

    onehot_lines = [
        "# One-hot encode categorical columns",
        f"CATEGORICAL_COLS = {_CATEGORICAL_COLS}",
        "for col in CATEGORICAL_COLS:",
        "    if col in df.columns:",
        "        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)",
        "        df = pd.concat([df, dummies], axis=1)",
    ]

    drop_lines = [
        "",
        "# Drop non-feature columns",
        f"DROP_COLS = {_DROP_COLS}",
        "df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')",
        "",
        'print(f"Feature matrix shape: {df.shape}")',
        "df.head()",
    ]

    add("markdown", [
        "## 4. Feature Engineering",
        "",
        f"Features used in the best iteration (iteration {best['iteration']}).",
    ])
    add("code", feature_lines + onehot_lines + drop_lines)

    # ── 7. Model Training ────────────────────────────────────────────
    model_line = _build_model_code(model_name, hp)

    train_lines = [
        "# Prepare features and target",
        f'X = df.drop(columns=["{target_column}"])',
        f'y = df["{target_column}"]',
        "X = X.fillna(0)",
        "",
        "# Train/test split",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42, stratify=y,",
        ")",
    ]

    if model_name in _NEEDS_SCALING:
        train_lines += [
            "",
            "# Feature scaling",
            "scaler = StandardScaler()",
            "X_train = scaler.fit_transform(X_train)",
            "X_test = scaler.transform(X_test)",
        ]

    train_lines += [
        "",
        "# Train model",
        model_line,
        "model.fit(X_train, y_train)",
        "",
        'print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")',
        f'print("Model: {model_name}")',
    ]

    add("markdown", [
        "## 5. Model Training",
        "",
        f"Training **{model_name}** with the configuration from the best iteration.",
    ])
    add("code", train_lines)

    # ── 8. Evaluation ────────────────────────────────────────────────
    add("markdown", ["## 6. Evaluation"])
    add("code", [
        "y_pred = model.predict(X_test)",
        "",
        "metrics = {",
        '    "Accuracy": accuracy_score(y_test, y_pred),',
        '    "Precision": precision_score(y_test, y_pred, zero_division=0),',
        '    "Recall": recall_score(y_test, y_pred, zero_division=0),',
        '    "F1": f1_score(y_test, y_pred, zero_division=0),',
        "}",
        "",
        "for name, value in metrics.items():",
        '    print(f"{name}: {value:.4f}")',
        "",
        "print()",
        'print("Classification Report:")',
        "print(classification_report(y_test, y_pred, zero_division=0))",
        "",
        'print("Confusion Matrix:")',
        "print(confusion_matrix(y_test, y_pred))",
    ])

    # ── Build notebook JSON ──────────────────────────────────────────
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

    return json.dumps(notebook, indent=1)
