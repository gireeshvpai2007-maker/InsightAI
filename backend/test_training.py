import os

from utils import load_dataset
from training import train_dataset


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = load_dataset(
    DATASET_PATH
)


print("\n========== DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# TRAIN DATASET
# ============================================================

result = train_dataset(
    df,
    target_column="Price"
)


# ============================================================
# TASK
# ============================================================

print("\n========== TASK ==========")
print(
    "Task:",
    result["task"]
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n========== MODEL COMPARISON ==========")

print(
    result["model_comparison"].to_string(
        index=False
    )
)


# ============================================================
# SELECTED MODEL
# ============================================================

print("\n========== SELECTED MODEL ==========")

print(
    "Model:",
    result["model_name"]
)


# ============================================================
# DATA SPLIT
# ============================================================

print("\n========== DATA SPLIT ==========")

print(
    "Training rows:",
    result["train_size"]
)

print(
    "Testing rows:",
    result["test_size"]
)


# ============================================================
# EVALUATION
# ============================================================

print("\n========== EVALUATION ==========")

for metric, value in result["metrics"].items():

    if value is None:

        print(
            f"{metric}: N/A"
        )

    else:

        print(
            f"{metric}: {value:.4f}"
        )