import itertools
import typing as ty
from operator import le
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import bbox_artist
from matplotlib.transforms import Bbox
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_curve

from visualization.utils import plot_reference_lines


def plot_boxplot_timeratios(
    name: str,
    learning_method: str,
    timeratios: pd.DataFrame,
    save_path: ty.Optional[str | Path] = None,
):
    """
    Plots a boxplot of time ratios for a given learning method and saves it to the specified path.

    Args:
        name (str): Name of the dataset.
        learning_method (str): Name of the learning method.
        timeratios (pd.DataFrame): DataFrame containing time ratios with a column "Ratio".
        save_path (str | Path, optional): If provided, saves the plot to this directory.
    """
    if len(timeratios) == 0:
        logger.warning("No timeratios data provided or data is empty. Skipping plot.")
        return

    plt.figure(figsize=(10, 1))
    plt.boxplot(
        timeratios["Ratio"],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="gray", color="black"),
        medianprops=dict(color="black"),
        flierprops=dict(
            markerfacecolor="black",
            markeredgecolor="none",
            markersize=3,
            alpha=0.3,
        ),
    )

    plt.yticks([])
    plt.grid(axis="x")
    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving boxplot to {save_path}")
        plt.savefig(
            save_path / f"{name}_{learning_method}_time_ratio.png",
            bbox_inches="tight",
            format="pdf",
            dpi=600,
        )


def plot_roc_pr_curves(
    y_true: pd.Series,
    y_probs: pd.Series,
    class_labels: list,
    title_prefix: str = "",
    figsize: tuple[int, int] = (7, 4),
    save_path: ty.Optional[str | Path] = None,
) -> tuple[dict, dict, Figure]:
    """
    Plots ROC and Precision-Recall curves for binary or multiclass classification.
    Returns the matplotlib Figure object for further use.

    Args:
        y_true (pd.Series): True labels (as strings or ints).
        y_probs (pd.Series): Series of posterior probability arrays.
        class_labels (list): List of class labels.
        title_prefix (str): Prefix for plot titles.
        save_path (str|Path, optional): If provided, saves the plots to this directory.

    Returns:
        tuple: (roc_aucs, pr_aucs, fig) where roc_aucs and pr_aucs are dicts with per-class AUCs,
               and fig is the matplotlib Figure object.
    """

    y_true_bin = y_true.apply(
        lambda x: class_labels.index(x) if isinstance(x, str) else x
    )

    roc_aucs = {}
    pr_aucs = {}
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC and PR curves for each class (one-vs-rest)
    for i, label in enumerate(class_labels):
        y_score = y_probs.apply(lambda x: x[i])
        fpr, tpr, _ = roc_curve(y_true_bin == i, y_score)
        precision, recall, _ = precision_recall_curve(y_true_bin == i, y_score)
        roc_auc_val = auc(fpr, tpr)
        pr_auc_val = auc(recall, precision)
        roc_aucs[label] = roc_auc_val
        pr_aucs[label] = pr_auc_val

        axes[0].plot(fpr, tpr, label=f"{label} (ROC-AUC={roc_auc_val:.2f})")
        axes[1].plot(recall, precision, label=f"{label} (PR-AUC={pr_auc_val:.2f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].set_title(f"{title_prefix.capitalize()} Set ROC Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Recall (R)")
    axes[1].set_ylabel("Precision (P)")
    axes[1].set_title(f"{title_prefix.capitalize()} Set PR Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            Path(save_path) / f"{title_prefix}_roc_pr_curves_multiclass.pdf",
            format="pdf",
            dpi=600,
        )

    return roc_aucs, pr_aucs, fig


def plot_confusion_matrix_with_counts(
    conf_matrix: np.typing.NDArray,
    class_labels: list,
    accuracy: float,
    save_path: ty.Optional[str | Path] = None,
    figsize: tuple = (5, 3),
    cmap: str = "Blues",
    prefix: str = "confusion_matrix",
) -> Figure:
    """
    Plots a confusion matrix as a heatmap with count annotations, axis labels, and saves the plot if requested.
    Returns the matplotlib Figure object for further use.

    Args:
        conf_matrix (np.ndarray): The confusion matrix to plot.
        class_labels (list): List of class labels for axes.
        accuracy (float): Accuracy value to display in the plot title.
        save_path (str | Path, optional): Directory to save the plot as PDF. If None, does not save. Default: None.
        figsize (tuple, optional): Figure size. Default: (10, 8).
        cmap (str, optional): Colormap for the heatmap. Default: "Blues".

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        conf_matrix,
        interpolation="nearest",
        cmap=cmap,
    )
    ax.set_title(f"Confusion matrix - Accuracy: {accuracy:.4f}")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.grid(True, alpha=0.3)

    # Annotate each cell with the count
    thresh = conf_matrix.max() / 2
    for i, j in itertools.product(
        range(conf_matrix.shape[0]), range(conf_matrix.shape[1])
    ):
        ax.text(
            j,
            i,
            format(conf_matrix[i, j], "d"),
            ha="center",
            va="center",
            color="white" if conf_matrix[i, j] > thresh else "black",
        )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            (save_path / f"{prefix}_confusion_matrix").with_suffix(".pdf"),
            format="pdf",
            dpi=600,
        )

    return fig


def plot_boxplot_robustness_per_brier_score_bins(
    ax: Axes,
    data: pd.DataFrame,
    n_bins: int = 10,
    robustness_column_key: str = "Man_Robustness",
    score_column: str = "Brier_Score",
    correct_color="gray",
    incorrect_color="gray",
    correct_edge="black",
    incorrect_edge="black",
    correct_median="black",
    incorrect_median="black",
    alpha=0.7,
):
    """
    Plots boxplots of robustness for each Brier score bin, with correct predictions on the LEFT
    and incorrect predictions on the RIGHT within each bin.

    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        data (pd.DataFrame): DataFrame containing individual fairness evaluation results.
        n_bins (int): Number of bins to create for Brier scores.
        robustness_column_key (str): Column name for robustness values.
        score_column (str): Column name for Brier scores.
        correct_color (str): Color for correct predictions boxplots.
        incorrect_color (str): Color for incorrect predictions boxplots.
        correct_edge (str): Edge color for correct predictions boxplots.
        incorrect_edge (str): Edge color for incorrect predictions boxplots.
        correct_median (str): Color for median line of correct predictions boxplots.
        incorrect_median (str): Color for median line of incorrect predictions boxplots.
        alpha (float): Transparency level for boxplots.
        save_path (str | Path, optional): If provided, saves the plot to this directory.
    """
    data = data.copy()

    bin_min = data[score_column].min()
    bin_max = data[score_column].max()

    # Compute bin edges
    total_bins = 2 * n_bins
    bin_edges = np.linspace(bin_min, bin_max, total_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Bin the data
    data["bin"] = pd.cut(data[score_column], bins=bin_edges, include_lowest=True)

    # Prepare categories
    correct = data[data["Prediction_Correct"].astype(bool)]
    incorrect = data[~data["Prediction_Correct"].astype(bool)]

    for i, interval in enumerate(data["bin"].cat.categories):
        center = bin_centers[i]
        width = bin_widths[i]
        box_width = width * 0.7

        bin_correct = correct[correct["bin"] == interval]
        bin_incorrect = incorrect[incorrect["bin"] == interval]

        pos_correct = center
        pos_incorrect = center

        if not bin_correct.empty:
            ax.boxplot(
                bin_correct[robustness_column_key],
                positions=[pos_correct],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=correct_color, color=correct_edge, alpha=alpha),
                medianprops=dict(color=correct_median, linewidth=1.5),
                whiskerprops=dict(color=correct_edge),
                capprops=dict(color=correct_edge),
                flierprops=dict(
                    markerfacecolor=correct_median,
                    markeredgecolor=correct_edge,
                    markersize=3,
                    alpha=0.15,
                ),
            )

        if not bin_incorrect.empty:
            ax.boxplot(
                bin_incorrect[robustness_column_key],
                positions=[pos_incorrect],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(
                    facecolor=incorrect_color, color=incorrect_edge, alpha=alpha
                ),
                medianprops=dict(color=incorrect_median, linewidth=1.5),
                whiskerprops=dict(color=incorrect_edge),
                capprops=dict(color=incorrect_edge),
                flierprops=dict(
                    markerfacecolor=incorrect_median,
                    markeredgecolor=incorrect_edge,
                    markersize=3,
                    alpha=0.15,
                ),
            )

    ax.set_xticks(bin_centers)
    ax.set_xticklabels([""] * len(bin_centers), fontsize=8)
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.set_ylabel("FRL")
    ax.set_xlim(bin_min - 0.05, bin_max + 0.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="both")
    return ax


def plot_histogram_by_eqbins(
    ax: Axes,
    data: pd.DataFrame,
    robustness_column_key: str,
    n_bins: int = 10,
    bins_strategy: str = "quantile",  # "quantile" or "equal_width"
    metric_name: ty.Literal["Accuracy", "Brier"] = "Accuracy",
):
    """
    Plots a horizontal bar histogram of accuracy/brier-score by equal-width bins of robustness.

    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        data (pd.DataFrame): DataFrame containing individual fairness evaluation results.
        robustness_column_key (str): Column name for robustness values.
        n_bins (int): Number of bins to create for robustness.
        bins_strategy (str): Strategy for binning ("quantile" or "equal_width").
        filename_prefix (str): Prefix for the saved plot filename.
    """
    instances_robustness = data[robustness_column_key]
    if bins_strategy == "equal_width":
        quantiles = pd.cut(
            instances_robustness,
            bins=pd.Series(np.linspace(0, 1, n_bins + 1)),
            include_lowest=True,
        )
    elif bins_strategy == "quantile":
        quantiles = pd.qcut(
            instances_robustness,
            q=n_bins,
            duplicates="drop",
        )
    else:
        raise ValueError(
            f"Invalid mode '{bins_strategy}'. Use 'quantile' or 'equal_width'."
        )

    if len(quantiles.cat.categories) < 2:
        logger.warning(
            f"Not enough unique robustness values to create {n_bins} quantile bins. Skipping plot."
        )
        return

    if metric_name == "Brier":
        metric_values = data.groupby(quantiles, observed=False)["Brier_Score"].mean()
        counts = data.groupby(quantiles, observed=False).size()
    elif metric_name == "Accuracy":
        metric_values = data.groupby(quantiles, observed=False)[
            "Prediction_Correct"
        ].mean()
        counts = data.groupby(quantiles, observed=False).size()

    bin_edges = [interval.left for interval in metric_values.index.categories] + [
        metric_values.index.categories[-1].right
    ]
    bin_midpoints = [interval.mid for interval in metric_values.index.categories]

    ax.barh(
        bin_midpoints,
        metric_values.values,
        height=np.diff(bin_edges),
        color="grey",
        edgecolor="black",
        alpha=0.7,
    )

    for i, (acc, _, y_pos) in enumerate(
        zip(metric_values.values, counts.values, bin_midpoints)
    ):
        bar_height = np.diff(bin_edges)[i]
        if metric_name == "Accuracy":
            x_pos = acc - 0.1 if acc < 0.8 else acc - 0.1
        elif metric_name == "Brier":
            x_pos = acc - 0.1 if acc > 0.3 else acc + 0.1
        if bar_height > 0.03:
            ax.text(
                x_pos,
                y_pos,
                f"{acc:.2f}",
                va="center",
                ha="center",
                fontsize=7,
            )

        ax.set_xlabel(metric_name)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, linestyle=":")

    ax.tick_params(axis="y", labelleft=False)
    return ax


def plot_scatter_brier_vs_frl(
    data: pd.DataFrame,
    ax_main: Axes,
    robustness_column_key: str,
) -> Axes:
    """Main scatter plot of Brier Score vs FRL (or robustness).

    Args:
        data (pd.DataFrame): DataFrame containing individual fairness evaluation results.
        ax_main (Axes): Matplotlib Axes object to plot on.
        robustness_column_key (str): Column name for robustness values.
    """

    ax_main.scatter(
        data["Brier_Score"],
        data[robustness_column_key],
        color="black",
        # alpha=0.05,
        s=3,
    )

    ax_main.set_xticks(np.linspace(0, 1, 11))
    ax_main.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 11)], fontsize=10)
    ax_main.set_xlim(-0.05, 1.05)
    ax_main.set_ylim(0, 1.05)
    ax_main.set_xlabel("Brier Score", fontsize=12)
    ax_main.set_ylabel("FRL", fontsize=12)
    ax_main.grid(True, alpha=0.3, axis="both")

    return ax_main


def plot_brier_vs_robustness(
    fairness_analysis_data: pd.DataFrame,
    robustness_column_key: str,
    filename_prefix: str = "",
    save_path: ty.Optional[str | Path] = None,
    figsize=(10, 8),
    drop_duplicates: bool = False,
    n_bins_brier: int = 10,
    robustness_bins_strategy: ty.Literal["quantile", "equal_width"] = "quantile",
    n_bins_robustness: int = 10,
    metric_name: ty.Literal["Accuracy", "Brier"] = "Accuracy",
):
    """
    Brier vs PRL plot function.

    Parameters:
        fairness_analysis_data: DataFrame containing individual fairness evaluation results.
        robustness_column_key: Column name for robustness values.
        filename_prefix: Prefix for the saved plot filename.
        save_path: Optional path to save the plot.
        figsize: Figure size (default=(12, 10)).
        drop_duplicates: Whether to drop duplicate instances based on "ID_row" (default=False).
        n_bins_brier: Number of bins for Brier score (default=10).
        robustness_bins_strategy: Strategy for robustness bins ("quantile" or "equal_width").
        n_bins_robustness: Number of bins for robustness (default=10).
    """

    if fairness_analysis_data is None or len(fairness_analysis_data) == 0:
        logger.warning(
            "No fairness analysis data provided or data is empty. Skipping plot."
        )
        return

    filtered_analysis_data = fairness_analysis_data[
        fairness_analysis_data["Man_Robustness_Individual"]
        == fairness_analysis_data["Man_Robustness_Max"]
    ]

    if (
        "Brier_Score" not in filtered_analysis_data.columns
        or robustness_column_key not in filtered_analysis_data.columns
    ):
        missing_cols = []
        if "Brier_Score" not in filtered_analysis_data.columns:
            missing_cols.append("Brier_Score")
        if robustness_column_key not in filtered_analysis_data.columns:
            missing_cols.append(robustness_column_key)
        logger.error(
            f"Error: Required columns {', '.join(missing_cols)} not found in the dataframe"
        )
        return

    fig = plt.figure(figsize=figsize, facecolor="white")

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.02,
        hspace=0.10,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_hist_top = fig.add_subplot(gs[0, 0])
    ax_hist_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    _ = plot_boxplot_robustness_per_brier_score_bins(
        ax=ax_hist_top,
        data=filtered_analysis_data,
        robustness_column_key=robustness_column_key,
        n_bins=n_bins_brier,
    )

    _ = plot_histogram_by_eqbins(
        ax=ax_hist_right,
        data=filtered_analysis_data,
        bins_strategy=robustness_bins_strategy,
        n_bins=n_bins_robustness,
        robustness_column_key=robustness_column_key,
        metric_name=metric_name,
    )

    if drop_duplicates:
        filtered_analysis_data = fairness_analysis_data.reset_index().drop_duplicates(
            subset=["ID_row"]
        )

    _ = plot_scatter_brier_vs_frl(
        data=filtered_analysis_data,
        ax_main=ax_main,
        robustness_column_key=robustness_column_key,
    )

    if save_path is not None:
        save_path = Path(save_path)

        filename = (
            "brier_vs_robustness"
            if not filename_prefix
            else f"{filename_prefix}_{metric_name}_brier_vs_robustness"
        )

        plt.savefig(
            (save_path / filename).with_suffix(".pdf"),
            format="pdf",
            dpi=1200,
        )
        logger.success(f"Plot saved to {save_path / filename}")

    return fig


def plot_stacked_histograms_brier_accuracy(
    name: str,
    data: pd.DataFrame,
    robustness_column_key: str,
    n_bins: int = 10,
    save_path: ty.Optional[Path] = None,
):
    # data = data[data[robustness_column_key] == data["Man_Robustness_Individual"]]
    if len(data) == 0:
        logger.warning("No data provided or data is empty.")
        return

    quantiles = pd.qcut(
        data[robustness_column_key],
        q=n_bins,
        duplicates="drop",
    )

    if len(quantiles.cat.categories) < 2:
        logger.warning(
            f"Not enough unique robustness values to create {n_bins} quantile bins. Skipping plot."
        )
        return

    brier_scores = data.groupby(quantiles)["Brier_Score"].mean()
    brier_scores.index = [f"Q{i + 1}" for i in range(len(brier_scores))]
    accuracy_scores = data.groupby(quantiles).apply(
        lambda x: accuracy_score(x["target_value"].values, x["Predicted_Value"].values)
    )
    accuracy_scores.index = [f"Q{i + 1}" for i in range(len(accuracy_scores))]

    bin_edges = [interval.left for interval in quantiles.cat.categories] + [
        quantiles.cat.categories[-1].right
    ]
    bin_midpoints = [interval.mid for interval in quantiles.cat.categories]

    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ax[0].bar(
        bin_midpoints,
        brier_scores,
        width=np.diff(bin_edges),
        align="center",
        color="grey",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )
    ax[0].set_ylabel("Brier Loss")
    ax[0].set_ylim(0, 1)
    ax[0].grid(axis="y")
    ax[0].set_yticks(np.linspace(0, 1, 11))
    ax[0].set_xlim(0, 1)

    ax[1].bar(
        bin_midpoints,
        accuracy_scores,
        width=np.diff(bin_edges),
        align="center",
        color="grey",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1)  # Accuracy ranges from 0
    ax[1].grid(axis="y")
    ax[1].set_yticks(np.linspace(0, 1, 11))
    ax[1].set_xlabel("FRL")
    ax[0].set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path / f"{name}_brier_accuracy_vs_robustness.pdf",
            format="pdf",
            dpi=600,
        )

    return fig
