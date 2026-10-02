import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from utils import load_dataset
from evaluation import evaluate_regression


dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)

df = load_dataset(dataset_path)

target_column = "Price"

X = df.drop(columns=[target_column])
y = df[target_column]


# Create an untouched test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# Predict on test set
y_pred = model.predict(X_test)


# Evaluate
metrics = evaluate_regression(
    y_test,
    y_pred
)


print("\n========== TEST SET EVALUATION ==========")

for metric, value in metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )