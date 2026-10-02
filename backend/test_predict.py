import os
import pandas as pd

from predict import (
    load_saved_model,
    get_expected_features,
    predict_dataset
)


BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "house_price.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# Load test dataset
df = pd.read_csv(
    DATASET_PATH
)

# Remove target column because this is prediction input
prediction_data = df.drop(
    columns=["Price"]
).head(5)


print("\n========== MODEL LOAD ==========")

pipeline, metadata = load_saved_model(
    MODEL_DIR
)

print("Pipeline loaded successfully.")
print("Task:", metadata["task"])
print("Model:", metadata["model_name"])
print("Target:", metadata["target_column"])


print("\n========== EXPECTED FEATURES ==========")

features = get_expected_features(
    pipeline
)

for feature in features:
    print(feature)


print("\n========== INPUT DATA ==========")

print(
    prediction_data
)


print("\n========== PREDICTIONS ==========")

result, _ = predict_dataset(
    prediction_data,
    MODEL_DIR
)

print(result)


print("\n========== ERROR HANDLING ==========")

invalid_data = prediction_data.drop(
    columns=[features[0]]
)

try:
    predict_dataset(
        invalid_data,
        MODEL_DIR
    )
except ValueError as error:
    print("Missing feature detected successfully.")
    print(error)


print("\nPrediction test completed successfully.")