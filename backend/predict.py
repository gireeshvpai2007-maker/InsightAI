import os
import joblib
import pandas as pd


MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "linear_regression.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.pkl"
)


def load_model():
    """Load trained model, scaler and feature information."""

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


def predict_price(input_data):
    """
    Predict house price from input features.

    input_data should be a dictionary containing
    the same features used during training.
    """

    model, scaler, features = load_model()

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Make sure features are in the same order
    df = df[features]

    # Scale input
    scaled_data = scaler.transform(df)

    # Generate prediction
    prediction = model.predict(scaled_data)

    return float(prediction[0])


if __name__ == "__main__":

    sample_house = {
        "Area": 2000,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Stories": 2,
        "Parking": 2,
        "Age": 5
    }

    predicted_price = predict_price(sample_house)

    print("\n========== HOUSE PRICE PREDICTION ==========")
    print(f"Predicted Price: ₹{predicted_price:,.2f}")