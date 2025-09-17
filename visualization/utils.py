from matplotlib.axes import Axes


def plot_reference_lines(ax: Axes, color: str = "gray") -> Axes:
    """
    Adds reference lines to the main plot: a diagonal from the lower left to the upper right.

    Args:
        ax (Axes): The axes to plot on.
        color (str): The color of the reference lines. Default is "gray".
    Returns:
        Axes: The axes with the reference lines added.
    """

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=color,
        linewidth=1,
    )

    ax.plot(
        [1, 0],
        [0, 1],
        "--",
        color=color,
        linewidth=1,
    )

    return ax


def linear_n_bins_chooser(
    n_samples: int, min_bins: int = 8, max_bins: int = 20, samples_per_bin: int = 500
) -> int:
    """
    Chooses the number of bins linearly based on the number of samples.

    Args:
        n_samples (int): The number of samples in the dataset.
        min_bins (int): Minimum number of bins. Default is 8.
        max_bins (int): Maximum number of bins. Default is 20.
        samples_per_bin (int): Number of samples per bin. Default is 5000.

    Returns:
        int: The chosen number of bins."""
    return min(max(n_samples // samples_per_bin, min_bins), max_bins)
