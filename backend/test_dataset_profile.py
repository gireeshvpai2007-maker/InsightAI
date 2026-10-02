from utils import (
    load_dataset,
    generate_dataset_profile
)


# ============================================================
# LOAD DATASET
# ============================================================

import os

from utils import (
    load_dataset,
    generate_dataset_profile
)


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "house_price.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = load_dataset(
    DATASET_PATH
)


# ============================================================
# GENERATE PROFILE
# ============================================================

profile = generate_dataset_profile(
    df
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n========== DATASET SUMMARY ==========")

for key, value in profile["summary"].items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# COLUMN PROFILE
# ============================================================

print("\n========== COLUMN PROFILE ==========")

print(
    profile["column_profile"].to_string(
        index=False
    )
)


# ============================================================
# OUTLIER SUMMARY
# ============================================================

print("\n========== OUTLIER SUMMARY ==========")

if profile["outliers"].empty:

    print(
        "No numerical outlier information available."
    )

else:

    print(
        profile["outliers"].to_string(
            index=False
        )
    )

# ============================================================
# GENERATE PROFILE
# ============================================================

profile = generate_dataset_profile(
    df
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n========== DATASET SUMMARY ==========")

for key, value in profile["summary"].items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# COLUMN PROFILE
# ============================================================

print("\n========== COLUMN PROFILE ==========")

print(
    profile["column_profile"].to_string(
        index=False
    )
)


# ============================================================
# OUTLIER SUMMARY
# ============================================================

print("\n========== OUTLIER SUMMARY ==========")

if profile["outliers"].empty:

    print("No numerical outlier information available.")

else:

    print(
        profile["outliers"].to_string(
            index=False
        )
    )