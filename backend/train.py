import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset

def train_model(dataset_path):
        df = pd.read_csv(dataset_path)
        print("========== DATA INSPECTION ==========")
        print("========== DATASET ==========")
        print(df.head())

        print("\n========== INFO ==========")
        df.info()

        print("\n========== DESCRIPTION ==========")
        print(df.describe())
        print(df.shape)
        print("========== DATA PREPROCESSING ==========")
        print(df.isnull().sum())
        print("Duplicate rows:", df.duplicated().sum())
        
        print("========== MODEL TRAINING ==========")
        # Separate features and target

        X = df.drop("Price", axis=1)
        y = df["Price"]

        print("Features:")
        print(X.head())

        print("\nTarget:")
        print(y.head())
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
        # Scale the features
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Training Features:", X_train.shape)
        print("Testing Features :", X_test.shape)

        print("Training Target  :", y_train.shape)
        print("Testing Target   :", y_test.shape)

        model = LinearRegression()
        model.fit(X_train, y_train)
        # Predict on test data
        y_pred = model.predict(X_test)
        
        print("========== MODEL EVALUATION ==========")
        print("First 5 Predictions:")
        print(y_pred[:5])

        print("\nFirst 5 Actual Values:")
        print(y_test.iloc[:5].values)
        for actual, predicted in zip(y_test.iloc[:5], y_pred[:5]):
            print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {
            "model": "Linear Regression",
            "mse": mse,
            "r2": r2
        }

        print("\n========== MODEL METRICS ==========")
        print(f"Model : {metrics['model']}")
        print(f"MSE   : {metrics['mse']:.2f}")
        print(f"R²    : {metrics['r2']:.4f}") 
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        
        print("========== SAVING MODEL ==========")
        joblib.dump(model, "../models/linear_regression.pkl")
        joblib.dump(scaler, "../models/scaler.pkl")

        print("Model and scaler saved successfully!")
if __name__ == "__main__":
    train_model("../datasets/house_price.csv")
