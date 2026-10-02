import os
import pandas as pd

from insights import generate_insights


BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "house_price.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# GENERATE INSIGHTS
# ============================================================

insights = generate_insights(
    df,
    target_column="Price"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========== INSIGHTAI INSIGHTS ==========")

for item in insights:

    print(
        f"[{item['category']}] "
        f"{item['insight']}"
    )


print("\nInsight generation completed successfully.")