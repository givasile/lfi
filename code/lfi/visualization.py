import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import typing

def plot_pairwise_posterior(
        samples: np.ndarray, # (N, Dy)
        subset_dims: typing.Optional[list[int]]=None,
        limits: typing.Optional[typing.List[float]]=None,
        title: typing.Optional[str]=None,
        savefig: typing.Optional[str]=None,
        samples_gt: typing.Optional[np.ndarray]=None,
        max_dims_to_plot: int = 10,
        show_legend: bool = True,
):
    """
    Plots pairwise relationships and marginal distributions of posterior samples.
    
    Args:
        samples (np.ndarray): Array of shape (N, D) containing posterior samples.
        subset_dims (list[int] or None): Indices of dimensions to include in the plot.
                                         if None, all dimensions are included.
        limits (list[int] or None): Limits of each dimension as [(xmin, xmax), ...].
                                    If None, limits are inferred automatically.
        title (str or None): Title of the plot. If None, no title is set.
        savefig (str or None): Path to save the plot. If None, the figure is not saved.
        max_dims_to_plot(int): Maximum number of dimensions to plot (default:10)
    Returns:
        sns.PairGrid: The Seaborn PairGrid object for further customization.
    """

    # Use all dimensions if subset_dim is None
    if subset_dims is None:
        subset_dims = list(range(samples.shape[1]))

    # Limit the number of dimensions to plot if it exceeds max_dims_to_plot

    if len(subset_dims) > max_dims_to_plot:
        print(f"Warning: Only plotting first {max_dims_to_plot} dimensions, (requested {len(subset_dims)}). ")
        subset_dims = subset_dims[:max_dims_to_plot]

    # Convert samples to DataFrame for easier handling with Seaborn
    samples_df = pd.DataFrame(samples, 
                              columns=[f"x_{i+1}" for i in range(samples.shape[1])]
                              )
    samples_df["Type"] = "Inferred"

    if samples_gt is not None:
        samples_gt_df = pd.DataFrame(
            samples_gt,
            columns=[f"x_{i+1}" for i in range(samples_gt.shape[1])]
        )
        samples_gt_df["Type"] = "Ground truth"
        samples_df = pd.concat([samples_df, samples_gt_df], ignore_index=True)

    # Select columns corresponding to subset_dims
    selected_vars = [f"x_{i+1}" for i in subset_dims]


    # Create the pairplot
    g = sns.pairplot(
        data = samples_df,
        hue="Type",
        vars = selected_vars,
        kind='scatter',
        diag_kind='kde',
        palette={"Inferred": "royalblue", "Ground truth": "crimson"},
        plot_kws={"alpha":0.7, "s": 30},
    )

    # --- KEY STEP: Adjust alpha per group ---
    for ax in g.axes.flat:
        if ax is not None:
            for collection in ax.collections:
                if collection.get_label() == "Inferred":
                    collection.set_alpha(0.7)  # More opaque
                elif collection.get_label() == "Ground truth":
                    collection.set_alpha(0.2)  # More transparent
                    collection.set_sizes([10])  # Optional: Smaller points
        
    # Show or hide legend
    if not show_legend:
        g.legend.remove()

    # Add the title if provided
    if title:
        g.fig.suptitle(title)

    # Set the limits if provided
    if limits is not None:
        for i in range(len(selected_vars)):
            for j in range(len(selected_vars)):
                if i != j: 
                    g.axes[i,j].set_xlim(limits)
                    g.axes[i,j].set_ylim(limits)
                else:
                    g.axes[i,j].set_xlim(limits)

    # Save the figure if a file path is specified
    if savefig is not None:
        g.savefig(savefig)

    return g
