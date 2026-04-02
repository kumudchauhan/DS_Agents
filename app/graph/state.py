from typing import Any, TypedDict


class AgentState(TypedDict):
    dataset_path: str
    dataframe: Any
    cleaned_dataframe: Any  # preserved copy so the loop can restart feature eng
    target_column: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    model: Any
    metrics: dict
    feedback: str
    iteration: int
    max_iterations: int
    eda_report: str
    cleaning_report: str
    feature_report: str
    model_report: str
    evaluation_report: str
    history: list
    should_continue: bool
