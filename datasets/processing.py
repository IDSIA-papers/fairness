import typing as ty
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


def detect_column_types(
    data: pd.DataFrame, threshold: int = 10
) -> ty.Tuple[list[str], list[str]]:
    """
    Detect the types of columns in a dataframe, based on the number of unique values

    Args:
        data (pd.DataFrame): The dataframe to analyze
        threshold (int): The threshold for the number of unique values to consider a column as continuous

    Returns:
        tuple: A tuple containing two lists: continuous columns and categorical columns
    """
    continuous_columns = []
    categorical_columns = []

    for column in data.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            # Further check if it has more unique values than a threshold (e.g., 10)
            if data[column].nunique() > threshold:
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)
        else:
            categorical_columns.append(column)

    return continuous_columns, categorical_columns


_encode_type: ty.TypeAlias = ty.Literal[
    "onehot", "onehot-dense", "ordinal"
]  # type of encoding
_strategy_type: ty.TypeAlias = ty.Literal[
    "uniform", "quantile", "kmeans"
]  # type of strategy


def discretize_continuous_columns(
    df: pd.DataFrame,
    continuous_columns: list[str],
    n_bins: int = 5,
    encode: _encode_type = "ordinal",
    strategy: _strategy_type = "uniform",
) -> pd.DataFrame:
    """
    Discretize continuous columns in a dataframe using the KBinsDiscretizer from scikit-learn

    Args:
        df (pd.DataFrame): The dataframe to discretize
        continuous_columns (list): List of continuous columns to discretize
        n_bins (int): The number of bins to discretize the continuous columns
        encode (Literal): The encoding method to use. Can be 'onehot', 'onehot-dense', or 'ordinal'
        strategy (Literal): The strategy to use for discretization. Can be 'uniform', 'quantile', or 'kmeans'

    Returns:
        pd.DataFrame: The dataframe with discretized continuous columns
    """

    for column in continuous_columns:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        df[column] = discretizer.fit_transform(df[[column]]).astype(int)

        bin_edges = discretizer.bin_edges_[0]
        range_labels = [
            f"{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}"
            for i in range(len(bin_edges) - 1)
        ]

        df[column] = df[column].map(lambda x: range_labels[x])

    return df


def make_columns_categorical(
    df: pd.DataFrame,
    threshold: int = 10,
    n_bins: int = 5,
    encode: _encode_type = "ordinal",
    strategy: _strategy_type = "uniform",
) -> pd.DataFrame:
    """
    Make continuous columns categorical

    Args:
        df (pandas.DataFrame): The dataframe to discretize
        threshold (int): Threshold of the number of unique values necessary to consider a column as continuous
        n_bins (int): The number of bins to discretize the continuous columns
        encode (Literal): The encoding method to use. Can be 'onehot', 'onehot-dense', or 'ordinal'
        strategy (Literal): The strategy to use for discretization. Can be 'uniform', 'quantile', or 'kmeans'

    Returns:
        pandas.DataFrame: The dataframe with continuous columns discretized
    """

    continous_columns, categorical_columns = detect_column_types(
        df, threshold=threshold
    )
    df_disc = discretize_continuous_columns(
        df, continous_columns, n_bins=n_bins, encode=encode, strategy=strategy
    )

    for column in df_disc.columns:
        if df_disc[column].dtype == "float64":
            df_disc[column] = df_disc[column].astype("int64")
    return df_disc


def split_dataset(
    df: pd.DataFrame, target_column: str, test_size: float = 0.5, random_state: int = 42
) -> ty.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into training and testing sets with stratification on the target variable. Ensures that
    all unique values for each column are present in the training set.

    Args:
        df (pd.DataFrame): The dataframe to split.
        target_column (str): The name of the target column to stratify on.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.5.
        random_state (int, optional): Controls shuffling before the split. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and testing datasets.
    """
    # Make sure target column exists in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe")

    # Step 1: Ensure representation of all unique values in the training set
    # Create a set to store indices of rows that must be in the training set
    required_indices = set()

    # For each column, find one example of each unique value
    for column in df.columns:
        for unique_val in df[column].unique():
            # Find indices of rows that have this value
            indices = df[df[column] == unique_val].index
            if len(indices) > 0:
                # Add the first occurrence to required indices
                required_indices.add(indices[0])

    required_df = df.loc[list(required_indices)].copy()
    remaining_df = df.drop(index=list(required_indices)).copy()

    # Step 2: Calculate how many more samples we need for a 50/50 split
    total_train_size = int(len(df) * (1 - test_size))
    additional_train_size = total_train_size - len(required_df)

    # Step 3: Split the remaining data with stratification
    if len(remaining_df) > 0 and additional_train_size > 0:
        # Stratify the remaining portion
        remaining_train, test_df = train_test_split(
            remaining_df,
            test_size=len(remaining_df) - additional_train_size,
            random_state=random_state,
            stratify=remaining_df[target_column]
            if len(remaining_df[target_column].unique()) > 1
            else None,
        )

        # Combine the required rows with the additional stratified training rows
        train_df = pd.concat([required_df, remaining_train])
    else:
        # If we've already selected too many required samples or have no remaining data
        if len(required_df) <= total_train_size:
            train_df = required_df
            test_df = df.drop(index=list(required_indices))
        else:
            # If we have more required samples than our target train size,
            # we need to move some to the test set
            train_indices = list(required_indices)[:total_train_size]
            train_df = df.loc[train_indices].copy()
            test_df = df.drop(index=train_indices).copy()

    # Reset indices for both dataframes
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print split statistics
    logger.info(
        f"Dataset split into train ({len(train_df)} rows) and test ({len(test_df)} rows)"
    )
    logger.info(
        f"Target distribution in train: {train_df[target_column].value_counts(normalize=True).to_dict()}"
    )
    logger.info(
        f"Target distribution in test: {test_df[target_column].value_counts(normalize=True).to_dict()}"
    )

    return train_df, test_df


def compute_time_ratios(
    individual_fairness_bn: pd.DataFrame,
    individual_fairness_mrf: pd.DataFrame,
    remove_outliers: bool = True,
    z_score_threshold: float = 3.0,
    save_path: str | Path = None,
) -> pd.DataFrame:
    """
    Compute the ratio of processing times between two individual fairness dataframes.
    Args:
        individual_fairness_bn (pd.DataFrame): DataFrame containing processing times for the BN model.
        individual_fairness_mrf (pd.DataFrame): DataFrame containing processing times for the MRF model.
    Returns:
        pd.Series: A series containing the ratio of processing times for each row.
    """
    individual_fairness_bn = individual_fairness_bn.dropna(subset=["ID_row"])
    individual_fairness_mrf = individual_fairness_mrf.dropna()

    joined_df = individual_fairness_bn.merge(
        individual_fairness_mrf,
        left_on="ID_row",
        right_on=individual_fairness_mrf.index,
        how="right",
    )
    joined_df = (
        joined_df[["Time_row", "Row_Processing_Time"]].drop_duplicates().dropna()
    )
    joined_df["Ratio"] = joined_df["Row_Processing_Time"] / joined_df["Time_row"]

    if remove_outliers:
        # Remove outliers based on z-score
        z_scores = (joined_df["Ratio"] - joined_df["Ratio"].mean()) / joined_df[
            "Ratio"
        ].std()
        joined_df = joined_df[z_scores.abs() <= z_score_threshold]

    if save_path:
        save_path = Path(save_path)
        joined_df.to_csv(save_path / "time_ratios.csv", index=False)
    return joined_df
