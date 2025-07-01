import argparse
import json
import typing as ty
from pathlib import Path

from loguru import logger

from bayesian.inference import (
    build_inference_engine,
    compute_posteriors_given_sensible_features_combinations,
)
from bayesian.learn import learn_bayesian_network
from bayesian.modifiers import (
    add_sensible_to_target_arcs,
    extract_markov_blanket,
    filter_relevant_features,
    simplification_1,
)

# Import custom modules
from datasets.data import extract_features, read_csv_files
from datasets.processing import (
    compute_time_ratios,
    make_columns_categorical,
    split_dataset,
)
from metrics.evaluate import (
    evaluate_bn_performance,
    # visualize_and_export_metrics,
)
from metrics.fairness import (
    analyze_individual_fairness_metrics,
    compute_group_fairness_metrics,
    compute_individual_fairness,
    compute_individual_fairness_MRF,
    save_metrics_dict,
)
from visualization.bayesnet import visualize_bn
from visualization.metrics import plot_boxplot_timeratios, plot_brier_vs_robustness


def main(
    learning_method: ty.Literal["tabu", "greedy", "miic", "k2"] = "tabu",
    learning_parameters=None,
    data_path: str | Path = "./data",
    save_path: str | Path = "./data/output_non_forced",
    drop_duplicates: bool = False,
):
    """
    Main function to run the fairness analysis pipeline using Bayesian networks.
    """
    # Configuration
    learning_parameters = {} if learning_parameters is None else learning_parameters
    data_path = Path("./data")
    data_path.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        save_path_output = data_path / "output_non_forced"
    else:
        save_path_output = Path(save_path)

    save_path_output.mkdir(parents=True, exist_ok=True)

    # save_path_individual_results = data_path / "individual_results"
    # save_path_individual_results.mkdir(parents=True, exist_ok=True)

    logger.info("Parameters:")
    logger.info(f"Learning method: {learning_method}")
    logger.info(f"Learning parameters: {learning_parameters}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Save path for output: {save_path_output}")
    logger.info(f"Drop duplicates: {drop_duplicates}")

    # Load datasets
    logger.info("Loading and preprocessing datasets...")
    _, dfs = read_csv_files((data_path / "preprocessed_data").as_posix())

    # Preprocess datasets - make continuous columns categorical
    for key in dfs.keys():
        dfs[key] = make_columns_categorical(
            dfs[key], threshold=10, n_bins=5, encode="ordinal", strategy="uniform"
        )
    logger.success("Datasets loaded and preprocessed successfully.")

    # Dictionary to store metrics for all datasets
    top_metrics_to_dataset = {}

    # Process each dataset
    for name, dataset in dfs.items():
        logger.info("-" * 70)
        logger.info(f"Fairness analysis on dataset: {name}")

        save_path_output_dataset = save_path_output / name
        save_path_output_dataset.mkdir(parents=True, exist_ok=True)

        # Remove all the files in the output directory that are not .csv
        for file in save_path_output_dataset.glob("*"):
            if file.suffix != ".csv":
                file.unlink()

        # Extract features
        target, sensible_features, public_features = extract_features(dataset)

        # Split dataset into train and test sets
        train_df, test_df = split_dataset(
            df=dataset, target_column=target, test_size=0.5, random_state=42
        )

        # Add arcs from sensible features to target
        learning_params = add_sensible_to_target_arcs(
            learning_params=learning_parameters,
            sensible_features=sensible_features,
            target=target,
        )

        logger.debug(
            f"Learning parameters after adding sensible features: {learning_params}"
        )

        # Learn Bayesian network on the training dataset
        bn = learn_bayesian_network(
            dataset=train_df,
            df_name=name,
            target=target,
            show=False,
            learning_method=learning_method,
            learning_params=learning_params,
            save_path=save_path_output_dataset.as_posix(),
        )

        # Simplify network by removing independent nodes
        # This will be used to compute the performance metrics
        bn = simplification_1(bn, name, target)

        # Check which sensible features remain relevant
        sensible_features_copy = sensible_features.copy()
        for sensible_feature in sensible_features_copy:
            if sensible_feature not in bn.names():
                logger.info(
                    f"Sensible feature {sensible_feature} is not relevant for the target"
                )
                sensible_features.remove(sensible_feature)

        # Save the simplified network
        visualize_bn(
            bn,
            name,
            learning_method=learning_method,
            save_path=save_path_output_dataset.as_posix(),
            simple="simple1",
        )

        # Create inference engine
        base_ie = build_inference_engine(bn)

        # Get base posterior of the target variable (marginal distribution)
        base_posterior = base_ie.posterior(target).toarray()

        # Evaluate performance
        performance_results, test_df_metrics = evaluate_bn_performance(
            bn=bn,
            ie=base_ie,
            test_df=test_df,
            target_column=target,
            save_path=save_path_output_dataset,
            verbose=False,
            drop_duplicates=drop_duplicates,
        )

        # Get all possible states of the sensible features
        sensible_states = {
            feat: list(train_df[feat].unique()) for feat in sensible_features
        }

        # Get all possible posteriors given the sensible features
        all_posteriors_comb = compute_posteriors_given_sensible_features_combinations(
            base_ie, sensible_states, target
        )

        # Add target domain and true values to performance results
        performance_results["target_domain"] = list(bn.variable(target).labels())
        performance_results["true_values"] = test_df[target].values.tolist()

        # Analyze group fairness
        logger.info("Analyzing group fairness...")
        group_fairness_results = compute_group_fairness_metrics(
            base_posterior,
            all_posteriors_comb,
            target,
            tops=30,
            save_path=save_path_output_dataset,
        )

        # Save group fairness metrics
        top_kls, top_kls_against, top_manhattans, top_manhattans_against = (
            group_fairness_results
        )

        top_metrics_to_dataset = save_metrics_dict(
            name,
            top_metrics_to_dataset,
            "group_fairness",
            top_kls,
            top_kls_against,
            top_manhattans,
            top_manhattans_against,
        )

        # Get Markov blanket
        markov_blanket = extract_markov_blanket(
            bn, target, save_path=save_path_output_dataset
        )

        # Check which sensible features are relevant in Markov blanket
        new_sensible_features = filter_relevant_features(
            markov_blanket, sensible_features
        )

        # Save the Markov blanket
        visualize_bn(
            markov_blanket,
            name,
            learning_method=learning_method,
            save_path=save_path_output_dataset.as_posix(),
            simple="blanket",
        )

        # Skip if no sensible features remain
        if not new_sensible_features:
            print(f"No sensible features in {name} after independency check")
            continue

        # combine train and test datasets to distinguish between them
        # dataset = combine_train_test(train_df, test_df_metrics)

        # Analyze individual fairness
        print("Analyzing individual fairness...")
        individual_fairness = compute_individual_fairness(
            name,
            test_df_metrics,
            markov_blanket,
            target,
            new_sensible_features,
            sensible_states,
            learning_method=learning_method,
            save_path=save_path_output_dataset.as_posix(),
            drop_duplicates=drop_duplicates,
        )

        # Analyze individual fairness metrics
        individual_fairness_results = analyze_individual_fairness_metrics(
            individual_fairness, target, tops=30
        )

        # Save individual fairness metrics
        top_kls_ind, top_kls_ind_against, top_man_ind, top_man_ind_against = (
            individual_fairness_results
        )
        top_metrics_to_dataset = save_metrics_dict(
            name,
            top_metrics_to_dataset,
            "individual_fairness",
            top_kls_ind,
            top_kls_ind_against,
            top_man_ind,
            top_man_ind_against,
        )

        top_metrics_to_dataset[name]["generic"] = {
            "accuracy": performance_results["accuracy"],
            "brier_score": performance_results["brier_scores"],
        }

        # Compute individual fairness metrics for Markov blanket
        individual_fairness_mrf = compute_individual_fairness_MRF(
            markov_blanket=markov_blanket,
            test_df=test_df_metrics,
            individual_fairness_df=individual_fairness,
            save_path=save_path_output_dataset.as_posix(),
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness,
            filename_prefix=f"{name}_individual_fairness",
            robustness_column_key="Man_Robustness",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=8,
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness,
            filename_prefix=f"{name}_individual_fairness_no_top",
            robustness_column_key="Man_Robustness",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=8,
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
            show_top_subplot=False,
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness,
            filename_prefix=f"{name}_individual_fairness_KL",
            robustness_column_key="KL_Robustness",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=8,
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
            reference_diagonal_lines=False,
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness_mrf,
            filename_prefix=f"{name}_individual_fairness_mrf",
            robustness_column_key="Manhattan_Distance",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=8,
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
        )

        plot_brier_vs_robustness(
            fairness_analysis_data=individual_fairness_mrf,
            filename_prefix=f"{name}_individual_fairness_mrf_KL",
            robustness_column_key="KL_Divergence",
            robustness_bins_strategy="quantile",
            n_bins_brier=5,
            n_bins_robustness=8,
            drop_duplicates=drop_duplicates,
            save_path=save_path_output_dataset,
            reference_diagonal_lines=False,
        )

        timeratios = compute_time_ratios(
            individual_fairness_bn=individual_fairness,
            individual_fairness_mrf=individual_fairness_mrf,
            save_path=save_path_output_dataset,
        )

        plot_boxplot_timeratios(
            name=name,
            learning_method=learning_method,
            timeratios=timeratios,
            save_path=save_path_output_dataset,
        )

    # if top_metrics_to_dataset:
    # print("\nVisualizing and exporting fairness metrics...")
    # visualize_and_export_metrics(
    #     top_metrics=top_metrics_to_dataset, output_dir=save_path_output.as_posix()
    # )


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

    parser.add_argument(
        "--learning_parameters",
        type=str,
        default="{}",
        help="JSON string of learning parameters for the Bayesian network.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to the directory containing the datasets.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/output_non_forced",
        help="Path to save the output results.",
    )

    parser.add_argument(
        "--drop_duplicates",
        action="store_false",
        help="Whether to drop duplicate rows in the datasets.",
    )

    args = parser.parse_args()

    # Convert learning_parameters from string to dictionary
    try:
        learning_parameters = json.loads(args.learning_parameters)
    except json.JSONDecodeError:
        logger.error("Invalid JSON string for learning parameters.")
        learning_parameters = {}

    main(
        learning_method=args.learning_method,
        learning_parameters=learning_parameters,
        data_path=args.data_path,
        save_path=args.save_path,
        drop_duplicates=args.drop_duplicates,
    )
