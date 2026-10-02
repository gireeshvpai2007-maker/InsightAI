import os

from utils import load_dataset
from utils import get_column_profile
from utils import detect_outliers


dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)


df = load_dataset(dataset_path)

print("\n========== COLUMN PROFILE ==========")

profile = get_column_profile(df)

print(profile.to_string(index=False))


print("\n========== OUTLIER ANALYSIS ==========")

outliers = detect_outliers(df)

print(outliers.to_string(index=False))