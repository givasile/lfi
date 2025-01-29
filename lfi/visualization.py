import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typing

def plot_pairwise_posterior(
        samples: np.ndarray,  # (N, Dy)
        subset_dims: typing.Optional[list[int]] = None,
        limits: typing.Optional[typing.List[float]] = None,
        title: typing.Optional[str] = None,
        savefig: typing.Optional[str] = None,
        samples_gt: typing.Optional[np.ndarray] = None,
):
    """
    Plots pairwise relationships and marginal distributions of posterior samples.

    Args:
        samples (np.ndarray): Array of shape (N, Dy) containing posterior samples.
        subset_dims (list[int] or None): Indices of dimensions to include in the plot.
                                         If None, all dimensions are included.
        limits (list[tuple] or None): Limits for each dimension as [(xmin, xmax), ...].
                                       If None, limits are inferred automatically.
        title (str or None): Title of the plot. If None, no title is set.
        savefig (str or None): Path to save the figure. If None, the figure is not saved.

    Returns:
        sns.PairGrid: The Seaborn PairGrid object for further customization.
    """
    # Use all dimensions if subset_dims is None
    if subset_dims is None:
        subset_dims = list(range(samples.shape[1]))

    # Convert samples to DataFrame for easier handling with Seaborn
    samples_df = pd.DataFrame(
        samples,
        columns=[f"x_{i + 1}" for i in range(samples.shape[1])]
    )
    samples_df["Type"] = "Inferred"

    if samples_gt is not None:
        samples_gt_df = pd.DataFrame(
            samples_gt,
            columns=[f"x_{i + 1}" for i in range(samples_gt.shape[1])]
        )
        samples_gt_df["Type"] = "Ground truth"
        samples_df = pd.concat([samples_df, samples_gt_df], ignore_index=True)

    # Select columns corresponding to subset_dims
    selected_vars = [f"x_{i + 1}" for i in subset_dims]

    plt.figure()
    # Create the pairplot
    g = sns.pairplot(
        data=samples_df,
        hue="Type",
        markers=["o", "D"],
        vars=selected_vars,
        kind="scatter",  # Pairplot style
        diag_kind="kde",  # KDE for diagonal plots
        plot_kws={'alpha': 0.5},  # Additional keyword arguments for scatter plots
    )

    # Add the title if provided
    if title:
        g.fig.suptitle(title)

    # # Set limits if provided
    # TODO: fix it
    if limits is not None:
        for i in range(len(selected_vars)):
            for j in range(len(selected_vars)):
                if i != j:
                    g.axes[i, j].set_xlim(limits)
                    g.axes[i, j].set_ylim(limits)
                else:
                    g.axes[i, j].set_xlim(limits)


    # Save the figure if a file path is specified
    if savefig is not None:
        g.savefig(savefig)

    plt.show(block=False)
