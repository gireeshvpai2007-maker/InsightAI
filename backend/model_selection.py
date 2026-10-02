import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)


def detect_task(y):
    """
    Automatically determine whether the target is
    a regression or classification problem.
    """

    # Non-numeric targets are treated as classification
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    unique_values = y.nunique()

    # Binary / low-cardinality numeric targets
    # are treated as classification.
    if unique_values <= 10:
        return "classification"

    return "regression"


def get_models(task):
    """
    Return candidate models for the detected task.
    """

    if task == "regression":

        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            )
        }

    if task == "classification":

        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            )
        }

    raise ValueError(
        f"Unsupported task: {task}"
    )


def compare_models(X, y, task, cv=5):
    """
    Compare candidate models using cross-validation.

    Returns:
        DataFrame containing model performance.
    """

    models = get_models(task)

    results = []

    if task == "regression":

        scoring = "r2"

    else:

        scoring = "f1_weighted"

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "model": name,
            "mean_score": scores.mean(),
            "std_score": scores.std()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "mean_score",
        ascending=False
    ).reset_index(drop=True)

    return results_df


def select_best_model(results):
    """
    Select the model with the highest
    cross-validation score.
    """

    if results.empty:
        raise ValueError(
            "No model results available."
        )

    return results.iloc[0]["model"]