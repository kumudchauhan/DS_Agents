from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.graph.state import AgentState

# Models that benefit from feature scaling.
NEEDS_SCALING = {"LogisticRegression", "SVC"}

# Default model config for iteration 0 (before any LLM recommendations).
DEFAULT_MODEL_CONFIG = {
    "model_name": "LogisticRegression",
    "hyperparameters": {"C": 1.0},
}


def _build_model(model_name: str, hyperparameters: dict):
    """Instantiate a sklearn model from name + hyperparameters."""
    hp = dict(hyperparameters)

    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=hp.get("C", 1.0),
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
    elif model_name == "RandomForestClassifier":
        max_depth = hp.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
        )
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(
            n_estimators=int(hp.get("n_estimators", 150)),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=int(hp.get("max_depth", 5)),
            random_state=42,
        )
    elif model_name == "SVC":
        return SVC(
            C=hp.get("C", 1.0),
            kernel=hp.get("kernel", "rbf"),
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    else:
        # Fallback
        return RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced",
        )


def modeling_node(state: AgentState) -> dict:
    """Train a classifier based on LLM recommendations (or defaults for iter 0)."""
    print("\n" + "=" * 60)
    print("  MODELING NODE - Iteration", state["iteration"])
    print("=" * 60)

    df = state["dataframe"].copy()
    target = state["target_column"]

    X = df.drop(columns=[target])
    y = df[target]
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Get model config from LLM recommendations.
    recommendations = state.get("recommendations") or {}
    model_config = recommendations.get("model_config", None)

    if not model_config or "model_name" not in model_config:
        model_config = dict(DEFAULT_MODEL_CONFIG)
        print("Using default model config (no LLM recommendations yet)")

    model_name = model_config["model_name"]
    hyperparameters = model_config.get("hyperparameters", {})

    # Apply scaling for models that need it.
    if model_name in NEEDS_SCALING:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = _build_model(model_name, hyperparameters)
    hp_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())
    full_name = f"{model_name}({hp_str})" if hp_str else model_name

    print(f"Training: {full_name}")
    model.fit(X_train, y_train)

    report = f"Model: {full_name}\nTrain size: {len(X_train)}, Test size: {len(X_test)}"
    print(report)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_report": report,
    }
