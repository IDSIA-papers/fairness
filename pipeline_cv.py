import argparse
import typing as ty
from pathlib import Path

import pandas as pd
from loguru import logger

from bayesian.inference import (
    build_inference_engine,
    # compute_posteriors_given_sensible_features_combinations,
)
from bayesian.learn import display_and_save_gum_bn, learn_bayesian_network
from bayesian.modifiers import (
    add_public_to_target_arcs,
    add_sensible_to_target_arcs,
    simplification_1,
)
from data.preprocess_data_pers import read_csv_files
from datasets.data import extract_features
from datasets.processing import (
    compute_time_ratios,
    make_columns_categorical,
    stratified_kfold_split,
)
from metrics.evaluate import evaluate_bn_performance, evaluate_bn_performance_aggregate
from metrics.fairness import (
    # compute_group_fairness_metrics,
    compute_individual_fairness_cv,
    compute_individual_fairness_MRF_cv,
    # save_metrics_dict,
)
from visualization.metrics import (
    plot_boxplot_timeratios,
    plot_brier_vs_robustness,
    plot_stacked_histograms_brier_accuracy,
)
from visualization.utils import linear_n_bins_chooser

# Suppress pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'


def load_and_process_datasets(data_path: Path) -> dict[str, ty.Any]:
    """
    Load datasets from the specified directory.

    Args:
        data_path (Path): Path to the directory containing CSV files.

    Returns:
        dict: A dictionary where keys are dataset names and values are DataFrames.
    """
    _, dfs = read_csv_files(data_path)

    # Preprocess datasets - make continuous columns categorical
    for key in dfs.keys():
        dfs[key] = make_columns_categorical(
            dfs[key], threshold=10, n_bins=5, encode="ordinal", strategy="uniform"
        )
    return dfs


def main(
    learning_method: ty.Literal["tabu", "greedy", "miic", "k2"] = "tabu",
    force: ty.Optional[bool] = None,
    data_path: str | Path = "./data",
    drop_duplicates: bool = False,
    n_splits: int = 10,
    random_state: int = 42,
):
    # Configuration
    data_path = Path("./data")
    data_path.mkdir(parents=True, exist_ok=True)

    data_preprocessed_path = data_path / "preprocessed_data"

    save_path_output = data_path / (
        "output_forced" if force == 1 else "output_non_forced"
    )

    save_path_output.mkdir(parents=True, exist_ok=True)

    logger.info("Parameters:")
    logger.info(f"Learning method: {learning_method}")
    logger.info(f"Data path: {data_path}, preprocessed path: {data_preprocessed_path}")
    logger.info(f"Save path for output: {save_path_output}")
    logger.info(f"Drop duplicates: {drop_duplicates}")
    logger.info(f"Number of splits for cross-validation: {n_splits}")
    logger.info(f"Random state: {random_state}")

    # Load datasets
    logger.info("Loading and preprocessing datasets...")
    dfs = load_and_process_datasets(data_path=data_preprocessed_path)
    logger.success("Datasets loaded and preprocessed successfully.")

    logger.info(f"Datasets to process: {list(dfs.keys())}")

    # top_metrics_dict = {}
    for name, dataset in dfs.items():
        logger.info(f"Processing dataset: {name}")
        logger.info(f"Dataset shape: {dataset.shape}")
        save_path_output_dataset = save_path_output / name
        save_path_output_dataset.mkdir(parents=True, exist_ok=True)

        # Remove all the files in the output directory that are not .csv
        # for file in save_path_output_dataset.glob("*"):
        #     if file.suffix != ".csv":
        #         file.unlink()

        # Extract features
        target, sensible_features, public_features = extract_features(dataset)

        kfold_results = {}
        for fold, (ktrain_df, kval_df) in enumerate(
            stratified_kfold_split(
                dataset,
                target_column=target,
                n_splits=n_splits,
                random_state=random_state,
                shuffle=True,
                keep_original_indexes=True,
            )
        ):
            logger.info(f"Processing fold {fold + 1}/{n_splits} for dataset {name}...")

            if force:
                # We are forcing
                learning_params = add_sensible_to_target_arcs(
                    learning_params={},
                    sensible_features=sensible_features,
                    target=target,
                )

                learning_params = add_public_to_target_arcs(
                    learning_params={},
                    public_features=public_features,
                    target=target,
                )
            else:
                learning_params = {}
            logger.info(f"Learning parameters: {learning_params}")

            # Learn Bayesian network on the training dataset
            bn = learn_bayesian_network(
                dataset=ktrain_df,
                df_name=name,
                target=target,
                show=False,
                learning_method=learning_method,
                learning_params=learning_params,
                # save_path=save_path_output_dataset.as_posix(),
            )

            display_and_save_gum_bn(
                bn,
                df_name=name,
                learning_method=learning_method,
                save_path=save_path_output_dataset.as_posix(),
                prefix=f"fold{fold + 1}",
            )

            # Simplify network by removing independent nodes
            # This will be used to compute the performance metrics
            bn = simplification_1(
                bn,
                name,
                target,
                # save_path=save_path_output_dataset.as_posix(),
                name_prefix=f"{name}_{learning_method}_simple1",
            )

            display_and_save_gum_bn(
                bn,
                df_name=name,
                learning_method=learning_method,
                save_path=save_path_output_dataset.as_posix(),
                prefix=f"fold{fold + 1}_simple1",
            )

            base_ie = build_inference_engine(bn)
            base_posterior = base_ie.posterior(target).toarray()

            # Evaluate performance
            performance_results, val_df_metrics = evaluate_bn_performance(
                bn=bn,
                ie=base_ie,
                test_df=kval_df,
                target=target,
                save_path=save_path_output_dataset,
                verbose=True,
                drop_duplicates=drop_duplicates,
                matrix_plot_prefix=f"{name}_fold{fold + 1}_{learning_method}",
                curve_plot_prefix=f"{name}_fold{fold + 1}_{learning_method}",
            )

            # Add fold number to the dataframe
            val_df_metrics.loc[:, "Fold"] = fold + 1

            kfold_results[fold] = {
                "bn": bn,
                "learning_params": learning_params,
                "performance_results": performance_results,
                "val_df_metrics": val_df_metrics,
                "accuracy_per_instance": performance_results["accuracy"],
                "brier_scores_per_instance": performance_results["brier_scores"],
                "brier_scores_abs_per_instance": performance_results[
                    "brier_scores_abs"
                ],
            }

        performance_aggregate_results = evaluate_bn_performance_aggregate(
            kfold_results,
            target,
            save_path_output_dataset,
            matrix_plot_prefix=f"{name}_aggregate_{learning_method}",
            curve_plot_prefix=f"{name}_aggregate_{learning_method}",
        )

        individual_fairness_all_folds_df = compute_individual_fairness_cv(
            kfold_results=kfold_results,
            name=name,
            target=target,
            learning_method=learning_method,
            performance_results=performance_aggregate_results,
            save_path_output_dataset=save_path_output_dataset,
            drop_duplicates=drop_duplicates,
        )

        simplified_df = individual_fairness_all_folds_df[
            individual_fairness_all_folds_df["Man_Robustness_Individual"]
            == individual_fairness_all_folds_df["Man_Robustness_Max"]
        ].drop_duplicates("ID_row")[
            [
                "Man_Robustness_Individual",
                "Brier_Score",
                "Prediction_Correct",
            ]
        ]

        simplified_df.to_csv(
            save_path_output_dataset / f"{name}_individual_fairness_simplified.csv",
        )

        individual_fairness_MRF_all_folds_df = compute_individual_fairness_MRF_cv(
            kfold_results=kfold_results,
            individual_fairness_all_folds_df=individual_fairness_all_folds_df,
            name=name,
            target=target,
            save_path_output_dataset=save_path_output_dataset,
        )

        _ = plot_stacked_histograms_brier_accuracy(
            name=name,
            data=individual_fairness_all_folds_df,
            robustness_column_key="Man_Robustness_Max",
            n_bins=linear_n_bins_chooser(
                len(individual_fairness_all_folds_df), 8, 20, samples_per_bin=1000
            ),
            save_path=save_path_output_dataset,
        )

        _ = plot_stacked_histograms_brier_accuracy(
            name=name,
            data=individual_fairness_all_folds_df,
            robustness_column_key="KL_Robustness_Max",
            n_bins=linear_n_bins_chooser(
                len(individual_fairness_all_folds_df), 8, 20, samples_per_bin=1000
            ),
            save_path=save_path_output_dataset,
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness_all_folds_df,
            filename_prefix=f"{name}_individual_fairness_man",
            robustness_column_key="Man_Robustness_Max",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=linear_n_bins_chooser(
                len(individual_fairness_all_folds_df), samples_per_bin=1000
            ),
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
            metric_name="Brier",
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness_all_folds_df,
            filename_prefix=f"{name}_individual_fairness_man",
            robustness_column_key="Man_Robustness_Max",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=linear_n_bins_chooser(
                len(individual_fairness_all_folds_df), samples_per_bin=1000
            ),
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
            metric_name="Accuracy",
        )

        timeratios = compute_time_ratios(
            individual_fairness_bn=individual_fairness_all_folds_df,
            individual_fairness_mrf=individual_fairness_MRF_all_folds_df,
            save_path=save_path_output_dataset,
        )

        plot_boxplot_timeratios(
            name=name,
            learning_method=learning_method,
            timeratios=timeratios,
            save_path=save_path_output_dataset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the fairness analysis pipeline using Bayesian networks."
    )
    parser.add_argument(
        "--learning_method",
        type=str,
        choices=["tabu", "greedy", "miic", "k2"],
        default="tabu",
        help="Learning method for Bayesian network structure.",
    )

    parser.add_argument("--force", action="store_true")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to the directory containing the datasets.",
    )

    # parser.add_argument(
    #     "--save_path",
    #     type=str,
    #     default="./data/output_forced",
    #     help="Path to save the output results.",
    # )

    parser.add_argument(
        "--drop_duplicates",
        action="store_true",
        help="Whether to drop duplicate rows in the datasets.",
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Number of splits for cross-validation.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )
    args = parser.parse_args()

    main(
        learning_method=args.learning_method,
        force=args.force,
        data_path=args.data_path,
        # save_path=args.save_path,
        drop_duplicates=args.drop_duplicates,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
