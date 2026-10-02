import os
import joblib
import pandas as pd


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_saved_model(model_dir):
    """
    Load the saved InsightAI pipeline and metadata.
    """

    pipeline_path = os.path.join(
        model_dir,
        "model_pipeline.pkl"
    )

    metadata_path = os.path.join(
        model_dir,
        "metadata.pkl"
    )

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            "Saved model pipeline not found."
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Saved model metadata not found."
        )

    pipeline = joblib.load(
        pipeline_path
    )

    metadata = joblib.load(
        metadata_path
    )

    return pipeline, metadata


# ============================================================
# EXPECTED FEATURES
# ============================================================

def get_expected_features(pipeline):
    """
    Return the feature columns expected by
    the trained model.
    """

    if not hasattr(
        pipeline,
        "feature_names_in_"
    ):
        raise ValueError(
            "Unable to determine expected input features."
        )

    return list(
        pipeline.feature_names_in_
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_dataset(
    df,
    model_dir
):
    """
    Generate predictions for a new dataset.

    The input dataset must contain the feature
    columns used during model training.
    """

    pipeline, metadata = load_saved_model(
        model_dir
    )

    expected_features = get_expected_features(
        pipeline
    )

    missing_features = [
        column
        for column in expected_features
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    input_data = df[
        expected_features
    ]

    predictions = pipeline.predict(
        input_data
    )

    result = df.copy()

    result["prediction"] = predictions

    return result, metadata