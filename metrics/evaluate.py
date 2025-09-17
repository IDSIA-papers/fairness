import os
import typing as ty
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pyagrum as gum
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from visualization.metrics import plot_confusion_matrix_with_counts, plot_roc_pr_curves


def compute_brier_scores_abs(
    bn: gum.BayesNet,
    target_column: str,
    posteriors: pd.Series,
    y_true: pd.Series,
):
    """
    Computes the Brier scores (ABSOLUTE) for a Bayesian network's predictions.

    Args:
        bn (gum.BayesNet): The Bayesian network used for inference.
        target_column (str): The name of the target column in the dataset.
        posteriors (pd.Series): Series containing posterior probabilities for each instance.
        y_true (pd.Series): Series containing true labels for each instance.
    Returns:
        pd.Series: A series containing the Brier scores for each instance.
    """
    # Convert the indexes labels to indices
    y_true_idxs = y_true.apply(
        lambda label: bn.variable(target_column).index(str(label))
        if isinstance(label, str)
        else label
    )
    selected_posteriors = pd.Series(
        [
            posterior[y_true_idx]
            for y_true_idx, posterior in zip(y_true_idxs, posteriors)
        ],
        index=posteriors.index,
    )

    brier_scores = 1 - selected_posteriors
    return brier_scores


def compute_brier_scores(
    bn: gum.BayesNet,
    target_column: str,
    posteriors: pd.Series,
    y_true: pd.Series,
):
    """
    Computes the Brier scores for a Bayesian network's predictions.

    Args:
        bn (gum.BayesNet): The Bayesian network used for inference.
        target_column (str): The name of the target column in the dataset.
        posteriors (pd.Series): Series containing posterior probabilities for each instance.
        y_true (pd.Series): Series containing true labels for each instance.

    Returns:
        pd.Series: A series containing the Brier scores for each instance.
    """
    target_var = bn.variable(target_column)
    K = target_var.domainSize()

    # map labels ("no"/"yes") -> class indices (0..K-1)
    y_idx = y_true.apply(lambda lbl: target_var.index(str(lbl))).astype(int)

    scores = []
    for idx, posterior in zip(y_idx, posteriors):
        # one-hot outcome
        o = np.zeros(K, dtype=float)
        o[idx] = 1.0

        # ensure probability vector aligned to BN domain order
        if isinstance(posterior, dict):
            if all(isinstance(k, (int, np.integer)) for k in posterior.keys()):
                p = np.array([posterior[i] for i in range(K)], dtype=float)
            else:
                # try mapping by string labels (e.g., {"no":0.3,"yes":0.7})
                labels = [target_var.label(i) for i in range(K)]
                p = np.array([posterior[str(lbl)] for lbl in labels], dtype=float)
        else:
            p = np.asarray(posterior, dtype=float)
            if p.shape[0] != K:
                raise ValueError(
                    f"Posterior length {p.shape[0]} != domain size {K} for {target_column}."
                )

        score = float(np.sum((p - o) ** 2) / K)
        scores.append(score)

    return pd.Series(scores, index=y_true.index, name="brier_score")


def compute_predictions(
    df: pd.DataFrame, bn: gum.BayesNet, ie: gum.LazyPropagation, target: str
) -> tuple[pd.Series, pd.Series]:
    """
    Computes predictions and posterior probabilities for a given DataFrame using a Bayesian network.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        bn (gum.BayesNet): The Bayesian network.
        ie (gum.LazyPropagation): The inference engine for the Bayesian network.
        target (str): The target variable to predict.
    Returns:
        tuple[pd.Series, pd.Series]: A tuple containing:
            - y_pred: Series of predicted classes for each instance.
            - y_probs: Series of posterior probabilities for the target variable.
    """
    # Remove the target column from evidence
    evidence_cols = list(set(bn.names()) - set([target]))  # type: ignore

    y_pred = []
    y_probs = []
    target_domain = bn.variable(target).labels()

    for row_id, row_data in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing test data",
    ):
        ie.eraseAllEvidence()
        evidence = row_data[evidence_cols].astype(str).to_dict()
        # evidence = {col: str(row_data[col]) for col in evidence_cols}

        try:
            ie.setEvidence(evidence)
            ie.makeInference()
            posterior = ie.posterior(target).toarray()
        except gum.InvalidArgument as e:
            logger.warning(
                f"\nWarning: Could not set evidence for a test instance: {e}"
            )
            # Use prior distribution as fallback
            ie.eraseAllEvidence()
            ie.makeInference()
            posterior = ie.posterior(target).toarray()

        most_probable_idx = int(np.argmax(posterior))
        most_probable_class = target_domain[most_probable_idx]

        y_probs.append(posterior)
        y_pred.append(most_probable_class)

    y_probs = pd.Series(y_probs, index=df.index)
    y_pred = pd.Series(y_pred, index=df.index)
    return y_pred, y_probs


def evaluate_bn_performance(
    bn: gum.BayesNet,
    ie: gum.LazyPropagation,
    test_df: pd.DataFrame,
    target: str,
    save_path: ty.Optional[str | Path] = None,
    curve_plot_prefix: str = "test",
    matrix_plot_prefix: str = "test",
    drop_duplicates: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, ty.Any], pd.DataFrame]:
    """
    Evaluates the performance of a Bayesian network as a classifier on a test dataset using an existing PyAgrum inference engine.

    Args:
        bn (pyAgrum.BayesNet): The trained Bayesian network to evaluate.
        ie (pyAgrum.LazyPropagation): The existing inference engine for the Bayesian network.
        test_df (pandas.DataFrame): The dataset to evaluate the network on.
        target_column (str): The name of the target column to predict.
        save_path (Path, optional): Path to save the confusion matrix plot. Defaults to None.
        drop_duplicates (bool, optional): Whether to drop duplicate rows from the given dataset. Defaults to True.
        verbose (bool, optional): Whether to print detailed logs. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - A dictionary with performance metrics including posterior probabilities, accuracy, Brier score, confusion matrix, classification report, predictions, target domain, and true values.
            - A DataFrame with the test dataset including predictions, true values, posterior probabilities, and Brier scores.
    """

    logger.info(f"Evaluating Bayesian network performance for target: {target}")

    if drop_duplicates:
        original_len = len(test_df)
        test_df_wk = test_df.drop_duplicates(ignore_index=False)
        dedup_len = len(test_df_wk)
        if dedup_len < original_len:
            logger.debug(
                f"Dropped {original_len - dedup_len} duplicate rows from test set ({original_len} → {dedup_len})"
            )
    else:
        test_df_wk = test_df.copy()
        logger.debug("Duplicates not dropped from the test set")

    # Get target modality once
    target_modality = bn.variable(target).labels()

    # Initialize arrays for true values and predictions
    y_true = test_df_wk[target].astype(str)
    y_pred, y_probs = compute_predictions(
        df=test_df_wk,
        bn=bn,
        ie=ie,
        target=target,
    )

    brier_scores = compute_brier_scores(
        bn=bn,
        target_column=target,
        posteriors=y_probs,  # the actual posteriors computed on the bn
        y_true=y_true,  # the gt label
    )

    brier_scores_abs = compute_brier_scores_abs(
        bn=bn,
        target_column=target,
        posteriors=y_probs,  # the actual posteriors computed on the bn
        y_true=y_true,  # the gt label
    )

    brier_score_mean = np.mean(brier_scores) if len(brier_scores) > 0 else None

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(
        y_true,
        y_pred,
    )
    class_report = classification_report(y_true, y_pred, output_dict=True)

    roc_aucs, pr_aucs, fig1 = plot_roc_pr_curves(
        y_true=y_true,
        y_probs=y_probs,
        class_labels=target_modality,
        title_prefix=curve_plot_prefix,
        save_path=save_path,
    )

    # Display results
    if verbose:
        logger.info(f"Accuracy: {accuracy:.4f}")
        if brier_score_mean is not None:
            logger.info(f"Brier Score mean: {brier_score_mean:.4f} (lower is better)")
        logger.info(dedent(f"Confusion Matrix: \n{conf_matrix}"))
        logger.info(
            dedent(f"Classification Report: \n{classification_report(y_true, y_pred)}")
        )

    fig2 = plot_confusion_matrix_with_counts(
        conf_matrix=conf_matrix,
        class_labels=target_modality,
        accuracy=float(accuracy),
        figsize=(6, 4),
        save_path=save_path,
        prefix=matrix_plot_prefix,
    )

    pd.options.mode.chained_assignment = None
    test_df_wk["Target_Modality"] = [list(target_modality)] * len(test_df_wk)
    # test_df_wk["GT_Class"] = y_true
    test_df_wk["Predicted_Class"] = y_pred
    test_df_wk["Posterior_Probabilities"] = y_probs
    test_df_wk["Brier_Score"] = brier_scores
    test_df_wk["Brier_Score_Abs"] = brier_scores_abs
    pd.options.mode.chained_assignment = "warn"

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        results_file = save_path / f"{matrix_plot_prefix}_results.csv"
        test_df_wk.to_csv(results_file, index=False)
        logger.success(f"Saved dataset with results to {results_file}")

    plt.close("all")
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "brier_score_mean": brier_score_mean,
        "brier_scores": brier_scores,
        "brier_scores_abs": brier_scores_abs,
        "confusion_matrix": conf_matrix,
        "class_report": class_report,
        "predictions": y_pred,
        "posterior_probabilities": y_probs,
        "target_domain": target_modality,
        "true_values": y_true,
        "roc_auc": roc_aucs,
        "pr_auc": pr_aucs,
    }, test_df_wk


def evaluate_bn_performance_aggregate(
    kfold_results: dict[int, dict],
    target: str,
    save_path_output_dataset: ty.Optional[str | Path] = None,
    matrix_plot_prefix: str = "Aggregate",
    curve_plot_prefix: str = "Aggregate",
) -> dict:
    """
    Evaluate the aggregate performance of the Bayesian network across all folds.

    Args:
        kfold_results (dict[int, dict]): Dictionary containing results for each fold.
        target (str): The target variable to evaluate.
        save_path_output_dataset (str|Path, optional): Path to save the plots. Defaults
    """
    target_modality = kfold_results[0]["bn"].variable(target).labels()
    true_parts: list[pd.Series] = []
    probs_parts: list[pd.Series] = []
    pred_parts: list[pd.Series] = []

    for data in kfold_results.values():
        df = data["val_df_metrics"]
        true_parts.append(
            pd.Series(df[target].astype(str).values, index=df.index, name="y_true")
        )
        probs_parts.append(
            pd.Series(
                df["Posterior_Probabilities"].values,
                index=df.index,
                name="y_prob",
            )
        )
        pred_parts.append(
            pd.Series(
                df["Predicted_Class"].astype(str).values,
                index=df.index,
                name="y_pred",
            )
        )

    y_true_s = pd.concat(true_parts)
    y_prob_s = pd.concat(probs_parts)
    y_pred_s = pd.concat(pred_parts)

    roc_aucs, pr_aucs, fig = plot_roc_pr_curves(
        y_true=y_true_s,
        y_probs=y_prob_s,
        class_labels=target_modality,
        title_prefix=curve_plot_prefix,
        figsize=(7, 4),
        save_path=save_path_output_dataset,
    )

    conf_matrix = confusion_matrix(
        y_true_s,
        y_pred_s,
    )
    accuracy = accuracy_score(y_true_s, y_pred_s)
    _ = plot_confusion_matrix_with_counts(
        conf_matrix=conf_matrix,
        class_labels=target_modality,
        accuracy=float(accuracy),
        figsize=(5, 3),
        save_path=save_path_output_dataset,
        prefix=matrix_plot_prefix,
    )
    # # Average classification report
    cls_report = classification_report(y_true_s, y_pred_s, output_dict=True)
    print(
        dedent(
            f"Classification Report average cross validation: \n{classification_report(y_true_s, y_pred_s)}"
        )
    )

    return {
        "classification_report": cls_report,
        "roc_aucs": roc_aucs,
        "pr_aucs": pr_aucs,
        "confusion_matrix": conf_matrix,
        "y_true": y_true_s,
        "y_probs": y_prob_s,
        "y_pred": y_pred_s,
        "figure_roc_pr": fig,
        "figure_confusion_matrix": fig,
    }


def visualize_and_export_metrics(top_metrics: dict, output_dir: str | Path) -> dict:
    """
    Process fairness metrics and export them to exactly two CSV files:
    1. One CSV file per dataset with all metrics for that dataset
    2. One CSV file with accuracy and Brier scores from all datasets

    Parameters:
    top_metrics: (dict) Dictionary containing fairness metrics for different datasets
    output_dir: (str) Directory to save CSV files, default='metrics_output'

    Returns:
    dict: Dictionary containing DataFrames of all metrics
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to store dataframes
    metrics_dfs = {}

    # Single dataframe for performance summary across all datasets
    performance_summary = []

    # Process each dataset
    for dataset_name, metrics_types in top_metrics.items():
        logger.info(f"{'-' * 80}")
        logger.info(f"DATASET: {dataset_name}")
        logger.info(f"{'-' * 80}")

        # Create a list to store all metrics for this dataset
        dataset_metrics = []

        # Process generic metrics
        if "generic" in metrics_types:
            logger.info("\nOVERALL PERFORMANCE METRICS:")
            accuracy = metrics_types["generic"].get("accuracy", float("nan"))
            brier_score = metrics_types["generic"].get("brier_score", float("nan"))

            logger.info(f"  Accuracy:    {accuracy:.4f}")
            logger.info(f"  Brier Score: {brier_score:.4f} (lower is better)")

            # Add to performance summary
            performance_summary.append(
                {
                    "dataset": dataset_name,
                    "fairness_type": "generic",
                    "accuracy": accuracy,
                    "brier_score": brier_score,
                }
            )

            # Add to dataset metrics
            dataset_metrics.append(
                {"metric_type": "generic", "metric_name": "accuracy", "value": accuracy}
            )
            dataset_metrics.append(
                {
                    "metric_type": "generic",
                    "metric_name": "brier_score",
                    "value": brier_score,
                }
            )

        # Process fairness metrics
        for fairness_type in metrics_types:
            if fairness_type == "generic":
                continue

            metrics = metrics_types[fairness_type]
            logger.info(f"\n{fairness_type.upper()} FAIRNESS METRICS:")

            # Add accuracy and Brier score if available
            if "accuracy" in metrics:
                performance_summary.append(
                    {
                        "dataset": dataset_name,
                        "fairness_type": fairness_type,
                        "accuracy": metrics["accuracy"],
                        "brier_score": metrics.get("brier_score", float("nan")),
                    }
                )

                dataset_metrics.append(
                    {
                        "metric_type": fairness_type,
                        "metric_name": "accuracy",
                        "value": metrics["accuracy"],
                    }
                )
                dataset_metrics.append(
                    {
                        "metric_type": fairness_type,
                        "metric_name": "brier_score",
                        "value": metrics.get("brier_score", float("nan")),
                    }
                )

            # Process other metrics like KL divergence and Manhattan distance
            for metric_type in [
                "top_kls",
                "top_kls_against",
                "top_manhattans",
                "top_manhattans_against",
            ]:
                if metric_type in metrics and metrics[metric_type]:
                    value_col = "KL" if "kls" in metric_type else "MAN"

                    # Calculate and display max value
                    max_value = max([item[value_col] for item in metrics[metric_type]])
                    logger.info(f"  Max {value_col}: {max_value:.4f}")

                    dataset_metrics.append(
                        {
                            "metric_type": fairness_type,
                            "metric_name": f"max_{metric_type}",
                            "value": max_value,
                        }
                    )

                    # Print top 3 values
                    top_values = sorted(
                        metrics[metric_type], key=lambda x: -x[value_col]
                    )[:3]
                    logger.info(f"  Top 3 {metric_type} values:")
                    for i, item in enumerate(top_values):
                        # Print description
                        desc = f"    {item[value_col]:.4f}"
                        if "att1" in item:
                            desc += f" - {item['att1']}"
                            if "state1" in item:
                                desc += f"={item['state1']}"
                        logger.info(desc)

                        # Add to dataset metrics
                        entry = {
                            "metric_type": fairness_type,
                            "metric_name": f"{metric_type}_rank{i + 1}",
                            "value": item[value_col],
                        }

                        if "att1" in item:
                            entry["attribute"] = item["att1"]
                            if "state1" in item:
                                entry["state"] = item["state1"]

                        dataset_metrics.append(entry)

        dataset_df = pd.DataFrame(dataset_metrics)
        dataset_file = (
            Path(output_dir) / f"{dataset_name}" / f"{dataset_name}_all_metrics.csv"
        )
        dataset_df.to_csv(dataset_file, index=False)
        logger.success(f"All metrics for {dataset_name} saved to {dataset_file}")

        metrics_dfs[dataset_name] = dataset_df

    performance_df = pd.DataFrame(performance_summary)

    # Check if the output dir contains the dataset nam
    performance_file = Path(output_dir) / "all_datasets_performance.csv"
    performance_df.to_csv(performance_file, index=False)
    logger.success(f"Performance summary for all datasets saved to {performance_file}")

    metrics_dfs["performance_summary"] = performance_df

    return metrics_dfs
