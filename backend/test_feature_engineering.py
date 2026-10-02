import os

from utils import load_dataset
from feature_engineering import create_ratio_features


dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)


df = load_dataset(dataset_path)

print("\n========== ORIGINAL DATASET ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

transformed_df, report = create_ratio_features(
    df,
    target_column="Price",
    max_features=10
)


print("\n========== FEATURE ENGINEERING REPORT ==========")

print(report.to_string(index=False))


print("\n========== TRANSFORMED DATASET ==========")

print("Rows:", transformed_df.shape[0])
print("Columns:", transformed_df.shape[1])


print("\n========== NEW FEATURES ==========")

new_columns = [
    column
    for column in transformed_df.columns
    if column not in df.columns
]

print(new_columns)