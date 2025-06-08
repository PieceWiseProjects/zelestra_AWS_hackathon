import pandas as pd


def load_train_data(file_path: str) -> pd.DataFrame:
    """
    Load training data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing training data.

    Returns:
        pd.DataFrame: DataFrame containing the training data.
    """
    return pd.read_csv(file_path, index_col=0)


def load_test_data(file_path: str) -> pd.DataFrame:
    """
    Load test data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing test data.

    Returns:
        pd.DataFrame: DataFrame containing the test data.
    """
    return pd.read_csv(file_path, index_col=0)


def change_dtype(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """
    Change the data type of a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column (str): The name of the column to change.
        dtype (str): The new data type for the column.

    Returns:
        pd.DataFrame: DataFrame with the updated column data type.
    """
    df[column] = df[column].astype(dtype)
    return df


if __name__ == "__main__":
    # Example usage
    train_data = load_train_data("data/raw/train.csv")
    test_data = load_test_data("data/raw/test.csv")

    print("Train Data:")
    print(train_data.head())

    print("\nTest Data:")
    print(test_data.head())
