import os
import numpy as np

from utils import load_dataset
from utils import transform_numeric_features


dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)


df = load_dataset(dataset_path)


# Create an artificially skewed feature for testing
df["Skewed_Test_Feature"] = np.exp(
    np.linspace(0, 5, len(df))
)


print("\n========== TRANSFORMATION ANALYSIS ==========")

transformed_df, report = transform_numeric_features(df)

print(report.to_string(index=False))