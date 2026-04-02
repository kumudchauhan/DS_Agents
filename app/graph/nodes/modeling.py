from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.graph.state import AgentState


def modeling_node(state: AgentState) -> dict:
    """Train a classifier. The model choice rotates across iterations."""
    print("\n" + "=" * 60)
    print("  MODELING NODE - Iteration", state["iteration"])
    print("=" * 60)

    df = state["dataframe"].copy()
    target = state["target_column"]
    iteration = state["iteration"]

    X = df.drop(columns=[target])
    y = df[target]

    # Fill any stray NaN so sklearn doesn't complain
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Rotate model by iteration
    if iteration == 0:
        model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced",
        )
        model_name = "LogisticRegression"
        # Logistic regression benefits from scaled features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif iteration == 1:
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced",
        )
        model_name = "RandomForestClassifier"
    elif iteration == 2:
        model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, random_state=42,
        )
        model_name = "GradientBoostingClassifier"
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, class_weight="balanced",
        )
        model_name = "RandomForestClassifier (tuned)"

    print(f"Training: {model_name}")
    model.fit(X_train, y_train)

    report = f"Model: {model_name}\nTrain size: {len(X_train)}, Test size: {len(X_test)}"
    print(report)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_report": report,
    }
