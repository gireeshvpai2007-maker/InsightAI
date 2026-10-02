import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import load_dataset

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


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)

TARGET_COLUMN = "Price"

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = load_dataset(DATASET_PATH)

print("\n========== DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET_COLUMN]
)

y = df[TARGET_COLUMN]


# ============================================================
# DETECT TASK
# ============================================================

task = detect_task(y)

print("\n========== TASK DETECTION ==========")
print("Target:", TARGET_COLUMN)
print("Task:", task)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

if task == "classification":

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


print("\n========== DATA SPLIT ==========")
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n========== MODEL COMPARISON ==========")

results = compare_models(
    X_train,
    y_train,
    task,
    cv=5
)

print(
    results.to_string(index=False)
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results.iloc[0]["model"]

models = get_models(task)

best_model = models[best_model_name]

print("\n========== SELECTED MODEL ==========")
print("Model:", best_model_name)


# ============================================================
# CREATE FINAL PIPELINE
# ============================================================

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


# ============================================================
# TRAIN FINAL PIPELINE
# ============================================================

print("\n========== TRAINING ==========")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# FINAL TEST SET PREDICTION
# ============================================================

y_pred = model_pipeline.predict(
    X_test
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

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


for metric, value in metrics.items():

    if value is None:

        print(
            f"{metric}: N/A"
        )

    else:

        print(
            f"{metric}: {value:.4f}"
        )


# ============================================================
# SAVE COMPLETE PIPELINE
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model_pipeline,
    os.path.join(
        MODEL_DIR,
        "model_pipeline.pkl"
    )
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "target_column": TARGET_COLUMN,
    "task": task,
    "model_name": best_model_name,
    "metrics": metrics
}


joblib.dump(
    metadata,
    os.path.join(
        MODEL_DIR,
        "metadata.pkl"
    )
)


print("\n========== MODEL SAVED ==========")
print(
    "Pipeline:",
    os.path.join(
        MODEL_DIR,
        "model_pipeline.pkl"
    )
)

print("Model:", best_model_name)
print("Task:", task)
print("Target:", TARGET_COLUMN)