import os
import re
import typing as ty

import pandas as pd


def read_csv_files(directory: str) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """
    Read all csv files in a directory and return a dictionary of dataframes and a list of the csv files

    Args:
        directory (str): The directory containing the csv files.

    Returns:
        tuple: A tuple containing a list of csv file names (without extension) and a dictionary of dataframes.

    Examples:
        >>> filenames, dfs = read_csv_files("data/")
    """
    csv_files = [f[:-4] for f in os.listdir(directory) if f.endswith(".csv")]
    f_csv_files = [
        f for f in os.listdir(directory) if f.endswith(".csv")
    ]  # full file names

    dataframes: dict[str, pd.DataFrame] = {}
    for file in f_csv_files:
        file_path = os.path.join(directory, file)
        dataframes[file[:-4]] = pd.read_csv(file_path, encoding="latin-1")

    return csv_files, dataframes


def extract_features(df: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """Extracts features from the dataset

    Args:
        df (pd.DataFrame): The dataframe to extract features from.

    Returns:
        tuple: A tuple containing the target variable, feature variables, and other variables.
    """
    Y = [column for column in df.columns if column.startswith("T_")]
    X = [column for column in df.columns if re.match("S\\d+_.*", column)]
    Z = [column for column in df.columns if column not in Y + X]
    return Y[0], X, Z


def extract_row_data(
    row: pd.Series, features: ty.Iterable[str], prefix="Original_"
) -> dict[ty.Any, ty.Any]:
    """
    Extracts feature values from a row of a DataFrame.

    Args:
        row (pd.Series): A row from a DataFrame.
        features (Iterable[str]): A list of feature names to extract.
        prefix (str): The prefix to use for the feature names in the row.
            Defaults to "Original_".
    Returns:
        dict: A dictionary with feature names as keys and their corresponding values from the row.
    """
    return {feat: row[f"{prefix}{feat}"] for feat in features}


# def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Combine training and testing DataFrames into a single DataFrame.

#     Parameters:
#     -----------
#     train_df : pd.DataFrame
#         Training dataset
#     test_df : pd.DataFrame
#         Testing dataset

#     Returns:
#     --------
#     pd.DataFrame
#         Combined DataFrame with an additional column indicating the source (train/test)
#     """

#     train_df["In_Test_Set"] = False
#     test_df["In_Test_Set"] = True

#     combined_df = pd.concat([train_df, test_df], ignore_index=True)

#     return combined_df
