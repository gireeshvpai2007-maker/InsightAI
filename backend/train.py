import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import load_dataset
from model_selection import (
    detect_task,
    compare_models,
    get_models
)
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
# NUMERICAL FEATURE SCALING
# ============================================================

numeric_columns = X_train.select_dtypes(
    include="number"
).columns.tolist()

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if numeric_columns:

    X_train_scaled[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_test_scaled[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n========== MODEL COMPARISON ==========")

results = compare_models(
    X_train_scaled,
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
# TRAIN SELECTED MODEL
# ============================================================

best_model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# FINAL TEST SET PREDICTION
# ============================================================

y_pred = best_model.predict(
    X_test_scaled
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

    if hasattr(best_model, "predict_proba"):

        y_probability = best_model.predict_proba(
            X_test_scaled
        )

    metrics = evaluate_classification(
        y_test,
        y_pred,
        y_probability
    )


for metric, value in metrics.items():

    if value is None:

        print(f"{metric}: N/A")

    else:

        print(f"{metric}: {value:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "model.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

joblib.dump(
    numeric_columns,
    os.path.join(
        MODEL_DIR,
        "features.pkl"
    )
)

joblib.dump(
    {
        "target_column": TARGET_COLUMN,
        "task": task,
        "model_name": best_model_name,
        "metrics": metrics
    },
    os.path.join(
        MODEL_DIR,
        "metadata.pkl"
    )
)


print("\n========== MODEL SAVED ==========")
print("Model:", best_model_name)
print("Task:", task)
print("Target:", TARGET_COLUMN)
print("Model directory:", MODEL_DIR)