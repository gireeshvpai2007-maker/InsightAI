import pandas as pd


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