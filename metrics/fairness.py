import heapq
import os
import time
import typing as ty
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum as gum
import tqdm
from loguru import logger
from scipy.special import kl_div
from sklearn.metrics.pairwise import manhattan_distances

from bayesian.inference import (
    build_inference_engine,
    select_optimal_sensitive_feature_perturbation,
)
from datasets.data import extract_features, extract_row_data
from metrics.evaluate import compute_brier_scores
from mrf.inference.VE import (
    AssignmentOperation,
    ProjectionOperation,
    VariableElimination,
)
from mrf.network.builder import BNtoRatioMRF
from mrf.network.utils import names_dict

MetricsEntry = dict[str, float | str]


def save_metrics_dict(
    dataset_name: str,
    save_dict: dict,
    kind: str,
    top_kls: list[dict],
    top_kls_against: list[dict],
    top_manhattans: list[dict],
    top_manhattans_against: list[dict],
) -> dict:
    """
    Save the metrics to the dictionary for the dataset.

    """
    save_dict_copy = save_dict.copy()

    if dataset_name not in save_dict_copy:
        save_dict_copy[dataset_name] = {}

    if kind not in save_dict_copy[dataset_name]:
        save_dict_copy[dataset_name][kind] = {}

    save_dict_copy[dataset_name][kind]["top_kls"] = top_kls
    save_dict_copy[dataset_name][kind]["top_kls_against"] = top_kls_against
    save_dict_copy[dataset_name][kind]["top_manhattans"] = top_manhattans
    save_dict_copy[dataset_name][kind]["top_manhattans_against"] = (
        top_manhattans_against
    )

    return save_dict_copy


def compute_group_fairness_metrics(
    base_posterior: np.typing.NDArray[np.float64] | float,
    all_posteriors_comb: dict[str, dict[str, np.typing.NDArray[np.float64]]],
    target: str,
    tops: int = 30,
    save_path: ty.Optional[str | Path] = None,
    verbose: bool = False,
) -> tuple[
    list[MetricsEntry],
    list[MetricsEntry],
    list[MetricsEntry],
    list[MetricsEntry],
]:
    """
    Compute and log the top KL divergences and Manhattan distances for group fairness analysis.

    Parameters:
        base_posterior: The marginal posterior distribution of the target variable.
        all_posteriors_comb: Dictionary of posteriors for all combinations of sensible feature states.
        target: The target variable.
        tops: Number of top divergences to display (default: 30).

    Returns:
        A tuple containing four lists:
        top_kls: Top KL divergences between posterior distributions of the original and modified states
        top_kls_against: Top KL divergences between posterior distributions of different sensible feature states configurations
        top_manhattans: Top Manhattan distances between posterior distributions of the original and modified states
        top_manhattans_against: Top Manhattan distances between posterior distributions of different sensible feature states configurations
    """
    top_kls = []
    top_kls_against = []
    top_manhattans = []
    top_manhattans_against = []
    counter = 0

    for att, step in all_posteriors_comb.items():
        for state, posterior in step.items():
            KL = kl_div(base_posterior, posterior).sum()
            MAN = manhattan_distances(np.array([base_posterior]), np.array([posterior]))
            kl_entry = (
                KL,
                counter,
                {
                    "KL": KL,
                    "att1": att,
                    "state1": state,
                },
            )
            manhattan_entry = (
                MAN[0][0],
                counter,
                {
                    "MAN": MAN[0][0],
                    "att1": att,
                    "state1": state,
                },
            )
            counter += 1

            if len(top_kls) < tops:
                heapq.heappush(top_kls, kl_entry)
            elif kl_entry[0] > top_kls[tops - 1][0]:
                heapq.heappushpop(top_kls, kl_entry)

            if len(top_manhattans) < tops:
                heapq.heappush(top_manhattans, manhattan_entry)
            elif manhattan_entry[0] > top_manhattans[tops - 1][0]:
                heapq.heappushpop(top_manhattans, manhattan_entry)

            for att_other, step_other in all_posteriors_comb.items():
                for state_other, posterior_other in step_other.items():
                    # If the sensible feature is the same but the state is different
                    if att_other == att and state_other != state:
                        KL2 = kl_div(posterior, posterior_other).sum()
                        MAN2 = manhattan_distances(
                            np.array([posterior]), np.array([posterior_other])
                        )
                        if KL2 < 0.05:
                            continue

                        kl_entry2 = (
                            KL2,
                            counter,
                            {
                                "KL": KL2,
                                "att1": att,
                                "state1": state,
                                "att2": att_other,
                                "state2": state_other,
                            },
                        )
                        manhattan_entry2 = (
                            MAN2[0][0],
                            counter,
                            {
                                "MAN": MAN2[0][0],
                                "att1": att,
                                "state1": state,
                                "att2": att_other,
                                "state2": state_other,
                            },
                        )
                        counter += 1

                        if len(top_kls_against) < tops:
                            heapq.heappush(top_kls_against, kl_entry2)
                        elif kl_entry2[0] > top_kls_against[tops - 1][0]:
                            heapq.heappushpop(top_kls_against, kl_entry2)

                        if len(top_manhattans_against) < tops:
                            heapq.heappush(top_manhattans_against, manhattan_entry2)
                        elif manhattan_entry2[0] > top_manhattans_against[tops - 1][0]:
                            heapq.heappushpop(top_manhattans_against, manhattan_entry2)

    sorted_results = sorted(top_kls, key=lambda x: -x[0])
    top_kls = [item[2] for item in sorted_results]

    sorted_results_against = sorted(top_kls_against, key=lambda x: -x[0])
    top_kls_against = [item[2] for item in sorted_results_against]

    sorted_results = sorted(top_manhattans, key=lambda x: -x[0])
    top_manhattans = [item[2] for item in sorted_results]

    sorted_results_against = sorted(top_manhattans_against, key=lambda x: -x[0])
    top_manhattans_against = [item[2] for item in sorted_results_against]

    if verbose:
        logger.info(f"\nTop {tops} highest KL divergences from original:")
        for i, kl_data in enumerate(top_kls, 1):
            logger.info(
                f"{i}. MAN: {kl_data['MAN']:.6f} between P({target}) and P({target}|{kl_data['att1']}={kl_data['state1']})"
            )

        logger.info(f"\nTop {tops} highest KL divergences within:")
        for i, kl_data in enumerate(top_kls_against, 1):
            logger.info(
                f"{i}. MAN: {kl_data['MAN']:.6f} between P({target}|{kl_data['att1']}={kl_data['state1']}) and P({target}|{kl_data['att2']}={kl_data['state2']})"
            )

        logger.info(f"\nTop {tops} highest Manhattans distances from original:")
        for i, man_data in enumerate(top_manhattans, 1):
            logger.info(
                f"{i}. MAN: {man_data['MAN']:.6f} between P({target}) and P({target}|{man_data['att1']}={man_data['state1']})"
            )

        logger.info(f"\nTop {tops} highest Manhattans distances within:")
        for i, man_data in enumerate(top_manhattans_against, 1):
            logger.info(
                f"{i}. MAN: {man_data['MAN']:.6f} between P({target}|{man_data['att1']}={man_data['state1']}) and P({target}|{man_data['att2']}={man_data['state2']})"
            )

    if save_path:
        logger.info(f"Saving results to {save_path}")
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save top KL divergences
        pd.DataFrame(top_kls).to_csv(
            save_path / "group_fairness_top_kls.csv", index=False, header=False
        )
        pd.DataFrame(top_kls_against).to_csv(
            save_path / "group_fairness_top_kls_against.csv", index=False, header=False
        )

        # Save top Manhattan distances
        pd.DataFrame(top_manhattans).to_csv(
            save_path / "group_fairness_top_manhattans.csv", index=False, header=False
        )
        pd.DataFrame(top_manhattans_against).to_csv(
            save_path / "group_fairness_top_manhattans_against.csv",
            index=False,
            header=False,
        )

    return top_kls, top_kls_against, top_manhattans, top_manhattans_against


def compute_individual_fairness(
    name: str,
    df: ty.Optional[pd.DataFrame] = None,
    markov_blanket: ty.Optional[gum.BayesNet] = None,
    target: ty.Optional[str] = None,
    sensitive_cols: ty.Optional[ty.Iterable[str]] = None,
    sensitive_columns_val: ty.Optional[dict[str, ty.Any]] = None,
    learning_method: ty.Optional[str] = None,
    save_path: ty.Optional[str] = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Compute individual fairness metrics for a dataset by comparing posteriors when combinations of sensitive attributes are changed.

    If a saved result file exists, loads and returns it immediately, requiring only file_name, learning_method, and save_path.
    If not, computes the metrics and saves/returns them as before.

    Parameters:
        name: Name of the dataset
        df: DataFrame containing the dataset
        markov_blanket: Bayesian network
        target: Target variable
        sensitive_cols: List of sensitive columns
        sensitive_columns_val: Dictionary mapping sensitive columns to their possible values
        learning_method: Learning method used to learn the Bayesian network
        save_path: Path to save results. If None, results are not saved.
        drop_duplicates: Whether to drop duplicate rows in the DataFrame

    Returns:
        DataFrame with individual fairness metrics for each row and sensitive attribute combination
    """
    # Try to load existing results first
    if save_path and learning_method:
        result_file = f"{save_path}/{name}_{learning_method}_individual_fairness.csv"
        if os.path.exists(result_file):
            logger.info(f"Loading existing results from {result_file}")
            read_df = pd.read_csv(result_file)

            def parse_posterior(val: str) -> np.typing.NDArray[np.float64]:
                return np.fromstring(val.strip("[]"), sep=" ")

            read_df["Posterior_Original"] = read_df["Posterior_Original"].apply(
                parse_posterior
            )
            read_df["Posterior_Modified"] = read_df["Posterior_Modified"].apply(
                parse_posterior
            )

            return read_df
        else:
            logger.info("No saved file found, computing individual fairness metrics.")

    logger.info(f"Computing individual fairness for {name}...")
    results = []

    if not (
        df is not None
        and markov_blanket is not None
        and target is not None
        and sensitive_cols is not None
        and sensitive_columns_val is not None
        and learning_method is not None
    ):
        raise ValueError(
            "If no saved file is found, all parameters must be provided to compute individual fairness."
        )

    if not sensitive_cols:
        logger.info(f"No sensitive columns in {name}, skipping")
        return pd.DataFrame()

    ie = build_inference_engine(markov_blanket)

    df_new = df.copy()
    df_new = df_new[
        list(markov_blanket.names())
        + [
            "Brier_Score",
            "Predicted",
        ]
    ]
    if drop_duplicates:
        df_new.drop_duplicates(
            subset=list(markov_blanket.names()), inplace=True, ignore_index=False
        )  # type: ignore

        logger.info(
            f"Duplicate rows dropped from {len(df)} to {len(df_new)}, # dropped: {len(df) - len(df_new)}"
        )

    # Process each row in the dataset
    for row_id, row_data in tqdm.tqdm(
        df_new.iterrows(), desc="Processing rows", total=len(df_new)
    ):
        # Extract precomputed values from the dataset
        brier_score = row_data.get("Brier_Score", None)
        predicted_value = row_data.get("Predicted", None)
        true_value = str(row_data[target])
        match_flag = (
            predicted_value == true_value
            if type(predicted_value) == type(true_value)
            else -1
        )

        row_start_time = time.time()

        original_values = {}
        for column in df_new.columns:
            if column != target and column not in [
                "Brier_Score",
                "Predicted",
            ]:
                original_values[column] = str(row_data[column])

        # Get posterior probability by setting as evidence the entire row
        ie.setEvidence(original_values)
        ie.makeInference()
        posterior_original = ie.posterior(
            target
        ).toarray()  # P(T|X) X=values of that row, sensible + non sensible
        ie.eraseAllEvidence()

        # Store all posteriors for this row to compute robustness
        all_posteriors = [posterior_original]

        sensitive_feat_to_states = {
            feat: list(df[feat].unique()) for feat in sensitive_cols
        }

        man_dists = []
        kl_dists = []
        for states_pair in product(*sensitive_feat_to_states.values()):
            # Create a new row with the alternative values
            modified_row_data = original_values.copy()

            # Track which attributes were modified in this combination
            modified_feature = []
            original_states = []
            modified_states = []

            for i, attr in enumerate(sensitive_feat_to_states.keys()):
                original_value = str(row_data[attr])
                new_value = str(states_pair[i])

                if new_value != original_value:
                    modified_row_data[attr] = new_value
                    modified_feature.append(attr)
                    original_states.append(original_value)
                    modified_states.append(new_value)

            if modified_feature == []:
                continue

            # Calculate posterior with modified values
            ie.setEvidence(modified_row_data)
            ie.makeInference()
            posterior_modified = ie.posterior(target).toarray()
            all_posteriors.append(posterior_modified)

            # Calculate KL divergence between original and modified posteriors
            manhattan_distance = (
                manhattan_distances(
                    np.array([posterior_original]), np.array([posterior_modified])
                ).item()
                / 2
            )  # Divide by 2 to normalize the distance
            man_dists.append(manhattan_distance)
            kl_div_value = kl_div(posterior_original, posterior_modified).sum()
            kl_dists.append(kl_div_value)
            ie.eraseAllEvidence()

            # Create result entry
            result = {
                "Dataset": name,
                "Target": target,
                "target_value": str(row_data[target]),
                "ID_row": row_id,
                "Modified_Attributes": " and ".join(modified_feature),
                "Original_States": " and ".join(original_states),
                "Modified_States": " and ".join(modified_states),
                "Posterior_Original": posterior_original,
                "Posterior_Modified": posterior_modified,
                "kl_div": kl_div_value,
                "Manhattan_Distance": manhattan_distance,
                "Predicted_Value": predicted_value,
                "Prediction_Correct": match_flag,
            }

            # Add Brier score if available
            if brier_score is not None:
                result["Brier_Score"] = brier_score

            # Add all original column values to the result
            for col, val in original_values.items():
                result[f"Original_{col}"] = val

            # Add modified values to the result
            for i, attr in enumerate(sensitive_feat_to_states.keys()):
                result[f"Modified_{attr}"] = states_pair[i]

            results.append(result)

        # Robustness as the maximum manhattan distance observed
        row_end_time = time.time() - row_start_time
        max_manhattan = max(man_dists) if man_dists else 0
        max_kl = max(kl_dists) if kl_dists else 0

        # Update all the results for this row with robustness metrics
        for result in results:
            if result["ID_row"] == row_id:
                result["Man_Robustness"] = max_manhattan
                result["KL_Robustness"] = max_kl
                result["Row_Processing_Time"] = row_end_time

    logger.success(f"Done with {name}. Processed {len(results)} combinations.")

    results_df = pd.DataFrame(results)

    # # Add robustness metrics to the DataFrame
    # if row_robustness:
    #     robustness_df = pd.DataFrame.from_dict(
    #         row_robustness, orient="index"
    #     ).reset_index()
    #     robustness_df.rename(columns={"index": "ID_row"}, inplace=True)

    #     # Merge robustness data with the results DataFrame
    #     results_df = pd.merge(results_df, robustness_df, on="ID_row", how="left")

    if save_path:
        logger.info(f"Saving results to {save_path}")
        results_df.to_csv(
            f"{save_path}/{name}_{learning_method}_individual_fairness.csv",
            index=False,
        )

    return results_df


def analyze_individual_fairness_metrics(
    individual_fairness: pd.DataFrame,
    target: str,
    tops: int = 30,
    verbose: bool = False,
) -> tuple[
    list[MetricsEntry],
    list[MetricsEntry],
    list[MetricsEntry],
    list[MetricsEntry],
]:
    """
    Compute and log the top KL divergences and Manhattan distances for individual fairness analysis.

    Parameters:
        individual_fairness: DataFrame containing individual fairness evaluation results
        target: The target variable.
        tops: Number of top divergences to display (default: 30).

    Returns:
        A tuple containing four lists:
        top_kls: Top KL divergences between posterior distributions of the original and modified states
        top_kls_against: Top KL divergences between posterior distributions of different sensible feature states configurations
        top_manhattans: Top Manhattan distances between posterior distributions of the original and modified states
        top_manhattans_against: Top Manhattan distances between posterior distributions of different sensible feature states configurations
    """
    previous_kl = 0
    previous_manhattan = 0
    top_kls_individual = []
    top_kls_individual_against = []
    top_manhattans_individual = []
    top_manhattans_individual_against = []

    for idx, row in enumerate(individual_fairness.iterrows()):
        row_data = row[1]

        if row_data["kl_div"] < 0.05:  # threshold
            continue

        kl_entry = (
            row_data["kl_div"],
            idx,
            {
                "KL": row_data["kl_div"],
                "att1": row_data["Modified_Attributes"],
                "state1": row_data["Original_States"],
                "att2": row_data["Modified_Attributes"],
                "state2": row_data["Modified_States"],
                "other_att": [
                    key.split("_", 1)[1]
                    for key in row_data.keys()
                    if key.startswith("Original_")
                ],
                "other_states": [
                    value
                    for key, value in row_data.items()
                    if key.startswith("Original_")
                ],
            },
        )
        manhattan_entry = (
            row_data["Manhattan_Distance"],
            idx,
            {
                "MAN": row_data["Manhattan_Distance"],
                "att1": row_data["Modified_Attributes"],
                "state1": row_data["Original_States"],
                "att2": row_data["Modified_Attributes"],
                "state2": row_data["Modified_States"],
                "other_att": [
                    key.split("_", 1)[1]
                    for key in row_data.keys()
                    if key.startswith("Original_")
                ],
                "other_states": [
                    value
                    for key, value in row_data.items()
                    if key.startswith("Original_")
                ],
            },
        )

        if len(top_kls_individual) < tops:
            heapq.heappush(top_kls_individual, kl_entry)
        elif kl_entry[0] > top_kls_individual[0][0]:
            heapq.heappushpop(top_kls_individual, kl_entry)

        if len(top_manhattans_individual) < tops:
            heapq.heappush(top_manhattans_individual, manhattan_entry)
        elif manhattan_entry[0] > top_manhattans_individual[0][0]:
            heapq.heappushpop(top_manhattans_individual, manhattan_entry)

        if False:  # to avoid code being too slow
            same_rows = individual_fairness.loc[
                individual_fairness["ID_row"] == row_data["ID_row"]
            ]
            for idx2, sub_row in enumerate(same_rows.iterrows()):
                sub_row_data = sub_row[1]

                # When loading there are from csv there are problems with the format of the posteriors
                if isinstance(row_data["Posterior_Modified"], str):
                    posterior1 = row_data["Posterior_Modified"][1:-1].split()
                    posterior2 = sub_row_data["Posterior_Modified"][1:-1].split()
                    posteriors_mod = (
                        [float(i) for i in posterior1],
                        [float(i) for i in posterior2],
                    )
                else:
                    posteriors_mod = (
                        row_data["Posterior_Modified"],
                        sub_row_data["Posterior_Modified"],
                    )

                new_KL = kl_div(posteriors_mod[0], posteriors_mod[1]).sum().round(6)
                new_MAN = manhattan_distances(
                    [posteriors_mod[0]], [posteriors_mod[1]]
                ).round(6)[0][0]
                # print(f"KL row P({target}|{row_data['Modified_Attributes']}={row_data['Modified_States']}) || P({target}|{sub_row_data['Modified_Attributes']}={sub_row_data['Modified_States']}) = {new_KL}")

                if new_KL < 0.05:  # threshold
                    continue

                kl_entry = (
                    new_KL,
                    idx + idx2 / 10,
                    {
                        "KL": new_KL,
                        "att1": row_data["Modified_Attributes"],
                        "state1": row_data["Modified_States"],
                        "att2": sub_row_data["Modified_Attributes"],
                        "state2": sub_row_data["Modified_States"],
                        "other_att": [
                            key.split("_")[1]
                            for key in row_data.keys()
                            if key.startswith("Original_")
                        ],
                        "other_states": [
                            value
                            for key, value in row_data.items()
                            if key.startswith("Original_")
                        ],
                    },
                )

                manhattan_entry = (
                    new_MAN,
                    idx + idx2 / 10,
                    {
                        "MAN": new_MAN,
                        "att1": row_data["Modified_Attributes"],
                        "state1": row_data["Modified_States"],
                        "att2": sub_row_data["Modified_Attributes"],
                        "state2": sub_row_data["Modified_States"],
                        "other_att": [
                            key.split("_")[1]
                            for key in row_data.keys()
                            if key.startswith("Original_")
                        ],
                        "other_states": [
                            value
                            for key, value in row_data.items()
                            if key.startswith("Original_")
                        ],
                    },
                )

                if kl_entry[0] == previous_kl:
                    continue

                if manhattan_entry[0] == previous_manhattan:
                    continue

                if len(top_kls_individual_against) < tops:
                    heapq.heappush(top_kls_individual_against, kl_entry)
                elif kl_entry[0] > top_kls_individual_against[tops - 1][0]:
                    heapq.heappushpop(top_kls_individual_against, kl_entry)

                if len(top_manhattans_individual_against) < tops:
                    heapq.heappush(top_manhattans_individual_against, manhattan_entry)
                elif (
                    manhattan_entry[0] > top_manhattans_individual_against[tops - 1][0]
                ):
                    heapq.heappushpop(
                        top_manhattans_individual_against, manhattan_entry
                    )

                previous_kl = kl_entry[0]
                previous_manhattan = manhattan_entry[0]

    # Sort results and extract data
    sorted_results_indiv = sorted(top_kls_individual, key=lambda x: -x[0])
    top_kls_individual = [item[2] for item in sorted_results_indiv]

    sorted_results_indiv_against = sorted(
        top_kls_individual_against, key=lambda x: -x[0]
    )
    top_kls_individual_against = [item[2] for item in sorted_results_indiv_against]

    sorted_results_indiv = sorted(top_manhattans_individual, key=lambda x: -x[0])
    top_manhattans_individual = [item[2] for item in sorted_results_indiv]

    sorted_results_indiv_against = sorted(
        top_manhattans_individual_against, key=lambda x: -x[0]
    )
    top_manhattans_individual_against = [
        item[2] for item in sorted_results_indiv_against
    ]

    # Display the top KL divergences
    if verbose:
        logger.info(f"\nTop {tops} highest KL divergences from original:")
        for i, kl_data in enumerate(top_kls_individual, 1):
            logger.info(
                f"Original states public features: {', '.join([f'{x}={y}' for x, y in zip(kl_data.get('other_att', []), kl_data.get('other_states', []))])}"
                f"\n{i}. KL: {kl_data['KL']:.6f} between P({target}|) and P({target}|{kl_data['att1']}={kl_data['state1']})"
            )

        logger.info(f"\nTop {tops} highest KL divergences within:")
        for i, kl_data in enumerate(top_kls_individual_against, 1):
            logger.info(
                f"Original states public features: {', '.join([f'{x}={y}' for x, y in zip(kl_data.get('other_att', []), kl_data.get('other_states', []))])}"
                f"\n{i}. KL: {kl_data['KL']:.6f} between P({target}|{kl_data['att1']}={kl_data['state1']}) and P({target}|{kl_data['att2']}={kl_data['state2']})"
            )

        logger.info(f"\nTop {tops} highest Manhattans distances from original:")
        for i, man_data in enumerate(top_manhattans_individual, 1):
            logger.info(
                f"Original states public features: {', '.join([f'{x}={y}' for x, y in zip(man_data.get('other_att', []), man_data.get('other_states', []))])}"
                f"\n{i}. KL: {man_data['KL']:.6f} between P({target}|) and P({target}|{man_data['att1']}={man_data['state1']})"
            )

        logger.info(f"\nTop {tops} highest Manhattans distances within:")
        for i, man_data in enumerate(top_manhattans_individual_against, 1):
            logger.info(
                f"Original states public features: {', '.join([f'{x}={y}' for x, y in zip(man_data.get('other_att', []), man_data.get('other_states', []))])}"
                f"\n{i}. KL: {man_data['KL']:.6f} between P({target}|{man_data['att1']}={man_data['state1']}) and P({target}|{man_data['att2']}={man_data['state2']})"
            )

    return (
        top_kls_individual,
        top_kls_individual_against,
        top_manhattans_individual,
        top_manhattans_individual_against,
    )


def compute_individual_fairness_MRF(
    markov_blanket: gum.BayesNet,
    test_df: pd.DataFrame,
    individual_fairness_df: pd.DataFrame,
    save_path: ty.Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compute individual fairness metrics using a Markov Random Field (MRF) based on the provided Bayesian network.

    Parameters:
        markov_blanket: The Markov blanket as a Bayesian network.
        test_df: DataFrame containing the test dataset.
        individual_fairness_df: DataFrame containing individual fairness evaluation results.
        save_path: Path to save the MRF inference results. If None, results are not saved.
    Returns:
        A dataframe with MRF inference results for each row in the test dataset.
    """
    if save_path:
        save_path = Path(save_path)
        filename = save_path / "mrf_inference_results.csv"
        if filename.exists():
            logger.info(f"Checking for existing MRF results at {save_path}")
            logger.info(f"Loading existing MRF results from {save_path}")
            read_df = pd.read_csv(
                save_path / "mrf_inference_results.csv", index_col=0, sep=";"
            )

            def parse_posterior(val: str) -> np.typing.NDArray[np.float64]:
                return np.fromstring(val.strip("[]"), sep=" ")

            read_df["Posterior_Original"] = read_df["Posterior_Original"].apply(
                parse_posterior
            )
            read_df["Posterior_Modified"] = read_df["Posterior_Modified"].apply(
                parse_posterior
            )

            read_df["Posterior_MAX"] = read_df["Posterior_MAX"].apply(parse_posterior)
            read_df["Posterior_MIN"] = read_df["Posterior_MIN"].apply(parse_posterior)
            read_df["Posterior_Star"] = read_df["Posterior_Star"].apply(parse_posterior)

            logger.success(
                f"Loaded {len(read_df)} MRF inference records from {save_path}"
            )
            return read_df

    target, sensible_features, public_features = extract_features(test_df)  # type: ignore

    try:
        mrf = BNtoRatioMRF(markov_blanket, target)
    except Exception as e:
        logger.error(
            f"Error converting Bayesian network to MRF: {e}. Ensure the target variable is in the Markov Blanket."
        )
        raise

    features_mrf: set[str] = set([mrf.variable(node).name() for node in mrf.nodes()])
    public_features: set[str] = set(public_features) & set(features_mrf)
    sensible_features: set[str] = set(sensible_features) & set(features_mrf)
    logger.info(
        f"Features in MRF: {features_mrf}, Public features: {public_features}, "
        f"Sensible features: {sensible_features}"
    )

    ie_markov_blanket = build_inference_engine(markov_blanket)

    # We will use the row IDs to reference the dataset rows
    row_ids = individual_fairness_df["ID_row"].unique()
    mrf_inference_records = []
    for id in tqdm.tqdm(row_ids, total=len(row_ids), desc="MRF Inference"):
        row_start_time = time.time()
        mrf_inference_record = {"ID_row": id}
        individual_fairness_row = individual_fairness_df.query(
            f"ID_row == {id}"
        ).sort_values(by="Manhattan_Distance", ascending=False)

        original_public_data = extract_row_data(
            individual_fairness_row.iloc[0], public_features
        )

        # The star assignment is the one that maximizes the manhattan distance,
        # since the row is sorted by it
        bn_star_assignment = extract_row_data(
            individual_fairness_row.iloc[0], sensible_features, prefix="Modified_"
        )

        mrf_inference_record["Target"] = individual_fairness_row["target_value"].iloc[0]
        mrf_inference_record["BN_Star_Assignment"] = bn_star_assignment
        mrf_inference_record["Posterior_Original"] = individual_fairness_row[
            "Posterior_Original"
        ].iloc[0]
        mrf_inference_record["Posterior_Modified"] = individual_fairness_row[
            "Posterior_Modified"
        ].iloc[0]

        inference_starting_time = time.time()
        ie_mrf = VariableElimination(mrf)

        # overline x
        mrf_max_assignment, _ = ie_mrf.MAP(
            # targets=sensible_features,
            evidence=original_public_data,
            verbose=False,
        )

        inference_max_ckpt = time.time() - inference_starting_time

        mrf_min_assignment, _ = ie_mrf.MAP(
            evidence=original_public_data,
            projection_operation=ProjectionOperation.MIN,
            assignment_operation=AssignmentOperation.ARGMIN,
            verbose=False,
        )

        inference_min_ckpt = time.time() - inference_max_ckpt - inference_starting_time

        ie_markov_blanket.eraseAllEvidence()
        max_evidence_data = original_public_data.copy()
        max_evidence_data.update(mrf_max_assignment)
        ie_markov_blanket.setEvidence(max_evidence_data)
        ie_markov_blanket.makeInference()
        p_max = ie_markov_blanket.posterior(target).toarray()

        ie_markov_blanket.eraseAllEvidence()
        min_evidence_data = original_public_data.copy()
        min_evidence_data.update(mrf_min_assignment)
        ie_markov_blanket.setEvidence(min_evidence_data)
        ie_markov_blanket.makeInference()
        p_min = ie_markov_blanket.posterior(target).toarray()

        # Compute the star assignment(s)
        mrf_star_assignment, posterior_star = (
            select_optimal_sensitive_feature_perturbation(
                baseline_posterior=individual_fairness_row["Posterior_Original"].iloc[
                    0
                ],
                assignment_lower=mrf_max_assignment,
                posterior_lower=p_max,
                assignment_upper=mrf_min_assignment,
                posterior_upper=p_min,
            )
        )

        # Compute the manhattan distance between the posterior of the star assignment and
        # the baseline posterior
        manhattan_distance = (
            manhattan_distances(
                np.array([individual_fairness_row["Posterior_Original"].iloc[0]]),
                np.array([posterior_star]),
            ).item()
            / 2
        )
        kl_divergence = kl_div(
            individual_fairness_row["Posterior_Original"].iloc[0], posterior_star
        ).sum()

        # Records the data
        mrf_inference_record["MRF_assignment_MAX"] = names_dict(mrf, mrf_max_assignment)
        mrf_inference_record["Posterior_MAX"] = p_max
        mrf_inference_record["Time_MAX"] = inference_max_ckpt
        mrf_inference_record["MRF_assignment_MIN"] = names_dict(mrf, mrf_min_assignment)
        mrf_inference_record["Posterior_MIN"] = p_min
        mrf_inference_record["Time_MIN"] = inference_min_ckpt
        mrf_inference_record["MRF_Star_Assignment"] = names_dict(
            mrf, mrf_star_assignment
        )
        mrf_inference_record["Posterior_Star"] = posterior_star
        mrf_inference_record["Manhattan_Distance"] = manhattan_distance
        mrf_inference_record["KL_Divergence"] = kl_divergence

        mrf_inference_record["Match_assignments"] = (
            mrf_inference_record["BN_Star_Assignment"]
            == mrf_inference_record["MRF_Star_Assignment"]
        )

        mrf_inference_record["Time_row"] = time.time() - row_start_time
        mrf_inference_records.append(mrf_inference_record)

    mrf_inference_df = pd.DataFrame(mrf_inference_records)

    mrf_inference_df["Brier_Score"] = compute_brier_scores(
        markov_blanket,
        target,
        mrf_inference_df["Posterior_Star"],
        mrf_inference_df["Target"],
    )

    mrf_inference_df["Prediction_Correct"] = (
        mrf_inference_df["Posterior_Star"].apply(
            lambda x: markov_blanket.variable(target).labels()[(np.argmax(x))]
        )
        == mrf_inference_df["Target"]
    )
    if save_path:
        logger.info(f"Saving MRF inference results to {save_path}")
        save_path = Path(save_path)
        mrf_inference_df.to_csv(
            save_path / "mrf_inference_results.csv",
            index=False,
            sep=";",
        )
    logger.success(
        f"MRF inference completed in {mrf_inference_df['Time_row'].sum():.2f} seconds"
    )

    return mrf_inference_df
