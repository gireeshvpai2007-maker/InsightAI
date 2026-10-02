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

print("\n========== SAVED PIPELINE ==========")
print("Pipeline loaded successfully.")


# ============================================================
# NEW DATA
# ============================================================

new_data = pd.DataFrame({
    "Area": [1500, 2200],
    "Bedrooms": [3, 4],
    "Bathrooms": [2, 3],
    "Stories": [2, 2],
    "Parking": [1, 2],
    "Age": [5, 3]
})


print("\n========== NEW DATA ==========")
print(new_data)


# ============================================================
# PREDICTION
# ============================================================

predictions = model_pipeline.predict(
    new_data
)


print("\n========== PREDICTIONS ==========")

for i, prediction in enumerate(
    predictions,
    start=1
):

    print(
        f"Prediction {i}: {prediction:.2f}"
    )