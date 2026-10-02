import os
import joblib
import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "model_pipeline.pkl"
)


# ============================================================
# LOAD SAVED PIPELINE
# ============================================================

model_pipeline = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_price(
    area,
    bedrooms,
    bathrooms,
    stories,
    parking,
    age
):
    """
    Predict house price using the saved
    InsightAI machine-learning pipeline.
    """

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Stories": [stories],
        "Parking": [parking],
        "Age": [age]
    })

    prediction = model_pipeline.predict(
        input_data
    )

    return float(prediction[0])