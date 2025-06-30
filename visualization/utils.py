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
