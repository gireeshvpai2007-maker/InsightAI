import pandas as pd


def generate_insights(df, target_column=None):
    """
    Generate automated factual insights from a dataset.

    Returns a list of insight dictionaries.
    """

    insights = []

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    rows, columns = df.shape

    insights.append({
        "category": "Dataset",
        "insight": (
            f"The dataset contains {rows:,} rows "
            f"and {columns} columns."
        )
    })

    # ========================================================
    # MISSING VALUES
    # ========================================================

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:

        insights.append({
            "category": "Data Quality",
            "insight": "No missing values were detected."
        })

    else:

        for column, count in missing.items():

            percentage = (count / rows) * 100

            insights.append({
                "category": "Data Quality",
                "insight": (
                    f"'{column}' contains {count:,} missing "
                    f"values ({percentage:.2f}% of rows)."
                )
            })

    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:

        insights.append({
            "category": "Data Quality",
            "insight": "No duplicate rows were detected."
        })

    else:

        percentage = (
            duplicate_count / rows
        ) * 100

        insights.append({
            "category": "Data Quality",
            "insight": (
                f"{duplicate_count:,} duplicate rows "
                f"were detected ({percentage:.2f}% of rows)."
            )
        })

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty or series.nunique() <= 1:
            continue

        skewness = series.skew()

        if abs(skewness) >= 1:

            direction = (
                "right-skewed"
                if skewness > 0
                else "left-skewed"
            )

            insights.append({
                "category": "Distribution",
                "insight": (
                    f"'{column}' is strongly {direction} "
                    f"(skewness: {skewness:.2f})."
                )
            })

    # ========================================================
    # OUTLIERS
    # ========================================================

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty or series.nunique() <= 1:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            (
                (series < lower_bound)
                | (series > upper_bound)
            ).sum()
        )

        if outlier_count > 0:

            percentage = (
                outlier_count / len(series)
            ) * 100

            insights.append({
                "category": "Outliers",
                "insight": (
                    f"'{column}' contains "
                    f"{outlier_count:,} potential outliers "
                    f"({percentage:.2f}% of available values)."
                )
            })

    # ========================================================
    # TARGET ANALYSIS
    # ========================================================

    if target_column is not None:

        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        target = df[target_column]

        insights.append({
            "category": "Target",
            "insight": (
                f"'{target_column}' is the selected "
                f"target column."
            )
        })

        if pd.api.types.is_numeric_dtype(target):

            correlations = df[numeric_columns].corr(
                numeric_only=True
            )[target_column].drop(
                labels=[target_column],
                errors="ignore"
            )

            correlations = correlations.dropna()

            if not correlations.empty:

                strongest_feature = correlations.abs().idxmax()
                strongest_correlation = correlations[
                    strongest_feature
                ]

                insights.append({
                    "category": "Relationship",
                    "insight": (
                        f"'{strongest_feature}' has the strongest "
                        f"linear correlation with '{target_column}' "
                        f"among the numeric features "
                        f"(correlation: "
                        f"{strongest_correlation:.2f})."
                    )
                })

    return insights