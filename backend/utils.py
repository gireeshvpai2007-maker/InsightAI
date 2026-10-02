import pandas as pd

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


def load_dataset(dataset_path):
    """Load a CSV dataset."""
    return pd.read_csv(dataset_path)


def get_dataset_summary(df):
    """Return basic dataset information."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def get_numeric_columns(df):
    """Return numeric columns."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df):
    """Return categorical columns."""
    return df.select_dtypes(exclude="number").columns.tolist()


def clean_dataset(df):
    """Basic dataset cleaning."""
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill numeric missing values with median
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill categorical missing values with mode
    categorical_columns = df.select_dtypes(exclude="number").columns

    for column in categorical_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])

    return df


def get_column_profile(df):
    """
    Generate statistical profile for every column.
    """

    profile = []

    for column in df.columns:

        series = df[column]

        information = {
            "column": column,
            "dtype": str(series.dtype),
            "missing": int(series.isnull().sum()),
            "unique": int(series.nunique()),
        }

        # Numerical statistics
        if pd.api.types.is_numeric_dtype(series):

            information.update({
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            })

        else:

            information.update({
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "skewness": None,
                "kurtosis": None,
            })

        profile.append(information)

    return pd.DataFrame(profile)


def detect_outliers(df):
    """
    Detect outliers in numerical columns using the IQR method.
    """

    outlier_summary = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        outlier_summary.append({
            "column": column,
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(len(outliers)),
        })

    return pd.DataFrame(outlier_summary)


def transform_numeric_features(df, skew_threshold=1.0):
    """
    Automatically transform highly skewed numerical features.

    Returns:
        transformed_df: DataFrame with transformed features
        transformation_report: DataFrame describing transformations
    """

    df = df.copy()

    report = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        series = df[column]

        # Skip columns with constant values
        if series.nunique() <= 1:

            report.append({
                "column": column,
                "original_skewness": 0.0,
                "transformation": "None",
                "final_skewness": 0.0
            })

            continue

        original_skewness = series.skew()

        transformation = "None"

        # Transform only highly skewed features
        if abs(original_skewness) > skew_threshold:

            # Positive-only data → Box-Cox
            if (series > 0).all():

                transformed, _ = boxcox(series)

                new_skewness = pd.Series(
                    transformed
                ).skew()

                # Use Box-Cox only if it improves skewness
                if abs(new_skewness) < abs(original_skewness):

                    df[column] = transformed
                    transformation = "Box-Cox"

                else:

                    new_skewness = original_skewness

            # Data containing zero/negative values → Yeo-Johnson
            else:

                transformer = PowerTransformer(
                    method="yeo-johnson",
                    standardize=False
                )

                transformed = transformer.fit_transform(
                    series.to_numpy().reshape(-1, 1)
                ).flatten()

                new_skewness = pd.Series(
                    transformed
                ).skew()

                if abs(new_skewness) < abs(original_skewness):

                    df[column] = transformed
                    transformation = "Yeo-Johnson"

                else:

                    new_skewness = original_skewness

        else:

            new_skewness = original_skewness

        report.append({
            "column": column,
            "original_skewness": float(original_skewness),
            "transformation": transformation,
            "final_skewness": float(new_skewness)
        })

    return df, pd.DataFrame(report)