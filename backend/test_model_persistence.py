import os
import joblib

from utils import load_dataset

from training import (
    train_dataset,
    save_training_result
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = load_dataset(
    DATASET_PATH
)


print("\n========== DATASET ==========")
print(
    "Rows:",
    df.shape[0]
)
print(
    "Columns:",
    df.shape[1]
)


# ============================================================
# TRAIN
# ============================================================

result = train_dataset(
    df,
    target_column="Price"
)


print("\n========== TRAINING ==========")

print(
    "Task:",
    result["task"]
)

print(
    "Model:",
    result["model_name"]
)


# ============================================================
# SAVE
# ============================================================

paths = save_training_result(
    result,
    MODEL_DIR
)


print("\n========== MODEL SAVED ==========")

print(
    "Pipeline:",
    paths["pipeline_path"]
)

print(
    "Metadata:",
    paths["metadata_path"]
)


# ============================================================
# VERIFY PIPELINE
# ============================================================

saved_pipeline = joblib.load(
    paths["pipeline_path"]
)

print("\n========== PIPELINE LOAD ==========")

print(
    "Pipeline loaded successfully."
)


# ============================================================
# VERIFY METADATA
# ============================================================

metadata = joblib.load(
    paths["metadata_path"]
)

print("\n========== METADATA ==========")

for key, value in metadata.items():

    print(
        f"{key}: {value}"
    )