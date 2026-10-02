import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_selection import (
    detect_task,
    compare_models,
    get_models
)

from preprocessing import create_preprocessor

from evaluation import (
    evaluate_regression,
    evaluate_classification
)


def train_dataset(
    df,
    target_column,
    test_size=0.2,
    cv=5,
    random_state=42
):
    """
    Train and evaluate InsightAI on a user-provided dataset.

    Parameters:
        df: Input DataFrame
        target_column: Target column name
        test_size: Fraction of data reserved for testing
        cv: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Dictionary containing the trained pipeline,
        task, selected model, metrics, and model
        comparison results.
    """

    # ========================================================
    # VALIDATE TARGET
    # ========================================================

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' not found."
        )


    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # ========================================================
    # DETECT TASK
    # ========================================================

    task = detect_task(y)


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    if task == "classification":

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results = compare_models(
        X_train,
        y_train,
        task,
        cv=cv
    )


    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = results.iloc[0]["model"]

    models = get_models(task)

    best_model = models[
        best_model_name
    ]


    # ========================================================
    # CREATE PIPELINE
    # ========================================================

    preprocessor = create_preprocessor(
        X_train
    )

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                best_model
            )
        ]
    )


    # ========================================================
    # TRAIN FINAL MODEL
    # ========================================================

    model_pipeline.fit(
        X_train,
        y_train
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    y_pred = model_pipeline.predict(
        X_test
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    if task == "regression":

        metrics = evaluate_regression(
            y_test,
            y_pred
        )

    else:

        y_probability = None

        if hasattr(
            model_pipeline,
            "predict_proba"
        ):

            y_probability = (
                model_pipeline.predict_proba(
                    X_test
                )
            )

        metrics = evaluate_classification(
            y_test,
            y_pred,
            y_probability
        )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "pipeline": model_pipeline,
        "task": task,
        "target_column": target_column,
        "model_name": best_model_name,
        "model_comparison": results,
        "metrics": metrics,
        "test_size": len(X_test),
        "train_size": len(X_train)
    }