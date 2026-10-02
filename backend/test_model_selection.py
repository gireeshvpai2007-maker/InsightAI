import os

from utils import load_dataset
from model_selection import (
    detect_task,
    compare_models,
    select_best_model
)


dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)

df = load_dataset(dataset_path)

# Target column
target_column = "Price"

X = df.drop(columns=[target_column])
y = df[target_column]


print("\n========== TASK DETECTION ==========")

task = detect_task(y)

print("Target:", target_column)
print("Detected task:", task)


print("\n========== MODEL COMPARISON ==========")

results = compare_models(
    X,
    y,
    task,
    cv=5
)

print(results.to_string(index=False))


print("\n========== BEST MODEL ==========")

best_model = select_best_model(results)

print("Selected model:", best_model)