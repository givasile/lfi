import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_table(results_root, metric, method, agg="mean"):
    """
    Build a DataFrame (budgets x dimensions) with c2st scores for a given method.

    Parameters
    ----------
    results_root : str
        Path to the results root (e.g. "results/mog_benchmark/two_modes")
    metric : str
        Metric name (e.g. "c2st", "runtime")
    method : str
        Method name (e.g. "npec", "bayes_flow", "flow_matching", "r2omc")
    agg : str
        Aggregation mode: "mean" or "best"

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by budget, columns are dimensions (int).
    """
    data = {}

    for d_dir in os.listdir(results_root):
        if not d_dir.startswith("D_"):
            continue
        dim = int(d_dir.split("_")[1])
        dim_path = os.path.join(results_root, d_dir)

        for mb in os.listdir(dim_path):
            if not mb.startswith(method):
                continue
            # extract budget
            try:
                budget = int(mb.split("_")[-1])
            except (IndexError, ValueError):
                continue
            mb_path = os.path.join(dim_path, mb)

            # collect all runs
            scores = []
            for run in os.listdir(mb_path):
                if metric == "c2st":
                    run_path = os.path.join(mb_path, run, "c2st.csv")
                elif metric == "runtime":
                    run_path = os.path.join(mb_path, run, "runtime.csv")
                else:
                    raise ValueError("metric must be 'c2st' or 'runtime'")
                if os.path.exists(run_path):
                    try:
                        val = pd.read_csv(run_path, header=None).values.squeeze()
                        # allow either single value or a column
                        if val.ndim > 0:
                            val = float(val[0])
                        scores.append(val)
                    except Exception:
                        continue
            if not scores:
                continue
            if agg == "mean":
                score = np.mean(scores)
            elif agg == "best":
                score = np.min(scores)
            else:
                raise ValueError("agg must be 'mean' or 'best'")

            data.setdefault(budget, {})[dim] = score

    # build DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "budget"
    df = df.sort_index().sort_index(axis=1)
    return df


def plot_c2st_vs_budget(df, method_name):
    """
    Plot c2st score vs budget with separate curves for each dimension.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from load_c2st_table (index=budget, columns=dimensions).
    method_name : str
        Method name (for title/labeling).
    """
    plt.figure(figsize=(7, 5))

    for dim in df.columns:
        plt.plot(df.index, df[dim], marker="o", label=f"D={dim}")

    plt.xlabel("Budget")
    plt.ylabel("C2ST Score")
    plt.title(f"{method_name} – C2ST vs Budget")
    plt.legend(title="Dimensions")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)


def plot_c2st_vs_dim(df, method_name):
    """
    Plot c2st score vs dimensionality with separate curves for each budget.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from load_c2st_table (index=budget, columns=dimensions).
    method_name : str
        Name of the method (for title/labeling).
    """
    plt.figure(figsize=(7, 5))

    for budget in df.index:
        plt.plot(df.columns, df.loc[budget], marker="o", label=f"Budget {budget}")

    plt.xlabel("Dimensionality (D)")
    plt.ylabel("C2ST Score")
    plt.title(f"{method_name} – C2ST vs Dimensionality")
    plt.legend(title="Budgets")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)


def plot_c2st_heatmap(df, method_name, savepath=None):
    """
    Plot a heatmap of c2st scores (rows=budget, cols=dimension).
    """
    plt.figure(figsize=(8, 6))
    desired_index = [1_000, 5_000, 10_000, 50_000, 100_000]
    df = df.reindex(desired_index)
    sns.heatmap(df, annot=True, annot_kws={"size": 12}, fmt=".2f", cmap="viridis", vmin=0.5, vmax=1)
    plt.xlabel("Dimensions (D)", fontsize=20)
    plt.ylabel("Budget", fontsize=20)
    plt.title(f"{method_name} – C2ST Heatmap", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().invert_yaxis()  # flip so smaller budgets are at the bottom
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"{method_name}_c2st_heatmap.png"))
        plt.savefig(os.path.join(savepath, f"{method_name}_c2st_heatmap.pdf"))
    plt.show(block=False)


def plot_c2st_success_frontier(loadpath, methods, threshold=0.75, agg="mean", savepath=None):
    plt.figure(figsize=(8, 6))

    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    colors = ["#1f77b4",  # blue
              "#ff7f0e",  # orange
              "#2ca02c",  # green
              "#d62728"]  # red
    for ii, method in enumerate(methods):
        df = load_table(loadpath, "c2st", method, agg=agg)
        frontier = {}
        for dim in df.columns:
            # find budgets that satisfy condition
            ok_budgets = df.index[df[dim] <= threshold]
            if len(ok_budgets) > 0:
                frontier[dim] = ok_budgets.min()
            else:
                frontier[dim] = np.nan  # never reached

        dims = sorted(frontier.keys())
        budgets = [frontier[d] for d in dims]
        budgets = [b if not np.isnan(b) else 110_000 for b in budgets]  # for plotting

        plt.plot(
            dims,
            budgets,
            marker=markers[ii % len(markers)],
            linestyle=['-', '--', '-.', '-'][ii % 4],
            label=method_names.get(method, method),
            markersize=8,
            linewidth=2,
            markeredgecolor='black',
            color=colors[ii % len(colors)]
        )

    plt.xlabel("Dimensions (D)", fontsize=20)
    plt.ylabel("Budget", fontsize=20)
    # plt.title(f"C2ST Success Frontier", fontsize=20)
    plt.xlim(left=1, right=21)
    plt.ylim(bottom=0, top=120_000)
    plt.xticks([2, 5, 10, 15, 20], fontsize=16)
    plt.yticks(
        [1_000, 5_000, 10_000, 50_000, 100_000],
        labels=["1k", "5k", "10k", "50k", "100k"],
        fontsize=16
    )
    # plt.yscale("log")
    plt.hlines(y=100_000, xmin=0, xmax=22, colors='gray', linewidth=4, linestyles='dotted')
    # write "over 100K" over the line and
    plt.text(6, 102_000, 'Over 100k', color='gray', fontsize=20, verticalalignment='bottom', horizontalalignment='right')
    # if loadpath finishes with "single_mode" add legend
    if loadpath.endswith("single_mode"):
        plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"c2st_frontier.png"))
        plt.savefig(os.path.join(savepath, f"c2st_frontier.pdf"))
    plt.show(block=False)


def plot_runtime_success_frontier(loadpath, methods, threshold=0.75, agg="mean", savepath=None):
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

    colors = ["#1f77b4",  # blue
              "#ff7f0e",  # orange
              "#2ca02c",  # green
              "#d62728"]  # red

    for ii, method in enumerate(methods):
        df_c2st = load_table(loadpath, "c2st", method, agg=agg)
        df_runtime = load_table(loadpath, "runtime", method, agg=agg)

        # estimated over 100k budget runtime
        df_runtime_100k = df_runtime.loc[100_000].max() if 100_000 in df_runtime.index else 0
        df_runtime_50k = df_runtime.loc[50_000].max() if 50_000 in df_runtime.index else 0
        df_runtime_10k = df_runtime.loc[10_000].max() if 10_000 in df_runtime.index else 0
        df_runtime_5k = df_runtime.loc[5_000].max() if 5_000 in df_runtime.index else 0
        df_runtime_1k = df_runtime.loc[1_000].max() if 1_000 in df_runtime.index else 0
        df_max_runtime = max(
            df_runtime_100k,
            df_runtime_50k * 2,
            df_runtime_10k * 10,
            df_runtime_5k * 20,
            df_runtime_1k * 100
        )
        frontier_runtime = {}
        for dim in df_c2st.columns:
            ok_budgets = df_c2st.index[df_c2st[dim] <= threshold]
            if len(ok_budgets) > 0:
                # smallest budget that hits the threshold
                chosen_budget = ok_budgets.min()

                # compute avg runtime across runs
                runtimes = []
                budget_path = os.path.join(loadpath, f"D_{dim}", f"{method}_{chosen_budget}")
                if os.path.exists(budget_path):
                    for run in os.listdir(budget_path):
                        run_path = os.path.join(budget_path, run, "runtime.csv")
                        if os.path.exists(run_path):
                            try:
                                val = pd.read_csv(run_path, header=None).values.squeeze()
                                if val.ndim > 0:
                                    val = float(val[0])
                                runtimes.append(val)
                            except Exception:
                                continue
                frontier_runtime[dim] = np.mean(runtimes) if runtimes else np.nan
            else:
                frontier_runtime[dim] = np.nan  # never reached

        dims = sorted(frontier_runtime.keys())
        runtimes = [frontier_runtime[d] for d in dims]

        runtimes = [r if not np.isnan(r) else df_max_runtime for r in runtimes]  # for plotting

        plt.plot(
            dims,
            np.array(runtimes) / 60.,
            marker=markers[ii % len(markers)],
            label=method_names.get(method, method),
            linestyle=['-', '--', '-.', '-'][ii % 4],
            markersize=8,
            linewidth=2,
            markeredgecolor='black',
            color=colors[ii % len(colors)]
            )

    plt.xlabel("Dimensions (D)", fontsize=20)
    plt.ylabel("Runtime (min)", fontsize=20)
    # plt.title(f"Runtime Success Frontier", fontsize=20)
    plt.xlim(left=1, right=21)
    plt.xticks([2, 5, 10, 15, 20], fontsize=16)
    # use <xx>k for yticks
    plt.yticks(fontsize=16)
    if loadpath.endswith("single_mode"):
        plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f"runtime_frontier.png"))
        plt.savefig(os.path.join(savepath, f"runtime_frontier.pdf"))
    plt.show(block=False)

# main part
experimets = [
    "single_mode",
    "single_mode_distractors",
    "two_modes",
    "two_modes_distractors"
]

method_names = {
    "npec": "NPE",
    "bayes_flow": "BayesFlow",
    "flow_matching": "Flow Matching",
    "r2omc": "R2OMC"
}

for experiment in experimets:
    print(f"Experiment: {experiment}")
    loadpath = os.path.join("./../../results/mog_benchmark", experiment)
    df = load_table(loadpath, metric="c2st", method="r2omc", agg="mean")
    savepath = os.path.join("./../../paper/figures", "mog_benchmark", experiment)

    print("Mean scores:\n", df)
    print("Best scores:\n", df)

    # plot_c2st_vs_dim(df, "NPE-C")
    # plot_c2st_vs_budget(df, "NPE-C")
    for method in ["npec", "bayes_flow", "flow_matching", "r2omc"]:
        df = load_table(loadpath, metric="c2st", method=method, agg="mean")
        # plot_c2st_vs_dim(df, method)
        # plot_c2st_vs_budget(df, method)
        plot_c2st_heatmap(df, method_names.get(method, method), savepath=savepath)

    plot_c2st_success_frontier(
        loadpath,
        methods=["npec", "bayes_flow", "flow_matching", "r2omc"],
        threshold=0.75,
        agg="mean",
        savepath=savepath
    )

    plot_runtime_success_frontier(
        loadpath,
        methods=["npec", "bayes_flow", "flow_matching", "r2omc"],
        threshold=0.75,
        agg="mean",
        savepath=savepath
    )

