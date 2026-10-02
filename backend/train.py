import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils import clean_dataset

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)


def train_model(dataset_path, target_column="Price"):
    """
    Train a Linear Regression model on a CSV dataset.
    """

    # =========================
    # 1. LOAD DATASET
    # =========================

    print("\n========== LOADING DATASET ==========")

    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    # =========================
    # 2. DATA CLEANING
    # =========================

    print("\n========== DATA CLEANING ==========")

    print("Missing values before cleaning:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    df = clean_dataset(df)

    print("\nDataset shape after cleaning:", df.shape)

    # =========================
    # 3. VALIDATE TARGET
    # =========================

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # =========================
    # 4. PREPARE FEATURES
    # =========================

    print("\n========== FEATURE PREPARATION ==========")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Keep numeric features for the baseline model
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("No numeric features available for training.")

    print("Features used:")
    print(list(X.columns))

    print("\nTarget:")
    print(target_column)

    # =========================
    # 5. TRAIN / TEST SPLIT
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n========== DATA SPLIT ==========")
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # =========================
    # 6. FEATURE SCALING
    # =========================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # 7. MODEL TRAINING
    # =========================

    print("\n========== MODEL TRAINING ==========")

    model = LinearRegression()

    model.fit(X_train_scaled, y_train)

    print("Linear Regression model trained successfully.")

    # =========================
    # 8. PREDICTION
    # =========================

    y_pred = model.predict(X_test_scaled)

    # =========================
    # 9. MODEL EVALUATION
    # =========================

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n========== MODEL EVALUATION ==========")

    print(f"Mean Squared Error : {mse:.2f}")
    print(f"R² Score           : {r2:.4f}")

    print("\nFirst 5 predictions:")

    for actual, predicted in zip(
        y_test.iloc[:5],
        y_pred[:5]
    ):
        print(
            f"Actual: {actual:.2f} | "
            f"Predicted: {predicted:.2f}"
        )

    # =========================
    # 10. SAVE MODEL
    # =========================

    print("\n========== SAVING MODEL ==========")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "linear_regression.pkl"
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )

    features_path = os.path.join(
        MODEL_DIR,
        "features.pkl"
    )

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X.columns), features_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Features saved to: {features_path}")

    # =========================
    # 11. RETURN RESULTS
    # =========================

    return {
        "model": "Linear Regression",
        "target": target_column,
        "features": list(X.columns),
        "mse": mse,
        "r2": r2,
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }


if __name__ == "__main__":

    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "datasets",
        "house_price.csv"
    )

    results = train_model(dataset_path)

    print("\n========== TRAINING COMPLETE ==========")
    print(results)