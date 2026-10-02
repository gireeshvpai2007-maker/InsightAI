import pandas as pd

from preprocessing import create_preprocessor


# Training data
train_df = pd.DataFrame({
    "Age": [21, 25, 30, 28, 35],
    "Salary": [30000, 45000, 60000, 50000, 70000],
    "City": [
        "Manipal",
        "Udupi",
        "Mangalore",
        "Udupi",
        "Manipal"
    ],
    "Plan": [
        "Basic",
        "Premium",
        "Premium",
        "Basic",
        "Premium"
    ]
})


# Test data contains:
# 1. Missing numeric value
# 2. Missing categorical value
# 3. Completely unseen city
test_df = pd.DataFrame({
    "Age": [24, None, 32],
    "Salary": [40000, 55000, None],
    "City": [
        "Manipal",
        "Bangalore",
        None
    ],
    "Plan": [
        "Basic",
        None,
        "Premium"
    ]
})


print("\n========== TRAINING DATA ==========")
print(train_df)


print("\n========== TEST DATA ==========")
print(test_df)


# Create preprocessor using training data only
preprocessor = create_preprocessor(
    train_df
)


# Fit only on training data
train_transformed = preprocessor.fit_transform(
    train_df
)


# Transform test data
test_transformed = preprocessor.transform(
    test_df
)


print("\n========== PREPROCESSING ==========")

print(
    "Training shape:",
    train_transformed.shape
)

print(
    "Test shape:",
    test_transformed.shape
)


print("\n========== TRANSFORMED TEST DATA ==========")

print(test_transformed)