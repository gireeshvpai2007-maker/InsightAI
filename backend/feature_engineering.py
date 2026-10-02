import pandas as pd


def create_ratio_features(
    df,
    target_column=None,
    max_features=10
):
    """
    Create a limited number of ratio features from numeric columns.

    Parameters:
        df: Input DataFrame
        target_column: Optional target column to exclude
        max_features: Maximum number of ratio features to create

    Returns:
        transformed_df: DataFrame with generated features
        report: DataFrame describing generated features
    """

    df = df.copy()

    report = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Exclude target column
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)

    # Generate each pair only once
    feature_count = 0

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            if feature_count >= max_features:
                break

            numerator = numeric_columns[i]
            denominator = numeric_columns[j]

            denominator_values = df[denominator]

            # Skip columns containing too many zero values
            zero_ratio = (
                denominator_values.eq(0).mean()
            )

            if zero_ratio > 0.05:
                continue

            feature_name = (
                f"{numerator}_per_{denominator}"
            )

            df[feature_name] = (
                df[numerator] /
                denominator_values.replace(0, pd.NA)
            )

            report.append({
                "feature": feature_name,
                "type": "ratio",
                "source_columns": (
                    f"{numerator}, {denominator}"
                )
            })

            feature_count += 1

        if feature_count >= max_features:
            break

    return df, pd.DataFrame(report)
def evaluate_features(
    df,
    target_column,
    original_columns=None
):
    """
    Evaluate numerical features based on absolute
    Pearson correlation with the target.

    Parameters:
        df: DataFrame containing features and target
        target_column: Target column
        original_columns: Original columns before
                          feature engineering

    Returns:
        DataFrame containing feature correlations.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_columns.remove(target_column)

    # Only evaluate newly generated features
    if original_columns is not None:

        numeric_columns = [
            column
            for column in numeric_columns
            if column not in original_columns
        ]

    evaluation = []

    for column in numeric_columns:

        correlation = df[column].corr(
            df[target_column]
        )

        if pd.notna(correlation):

            evaluation.append({
                "feature": column,
                "correlation": float(correlation),
                "absolute_correlation": float(
                    abs(correlation)
                )
            })

    result = pd.DataFrame(evaluation)

    if not result.empty:

        result = result.sort_values(
            "absolute_correlation",
            ascending=False
        ).reset_index(drop=True)

    return result
def select_features(
    evaluation,
    min_correlation=0.25
):
    """
    Select generated features based on absolute
    correlation with the target.

    Parameters:
        evaluation: Feature evaluation DataFrame
        min_correlation: Minimum absolute correlation required

    Returns:
        DataFrame containing selected features.
    """

    if evaluation.empty:
        return evaluation.copy()

    selected = evaluation[
        evaluation["absolute_correlation"]
        >= min_correlation
    ].copy()

    selected = selected.reset_index(drop=True)

    return selected