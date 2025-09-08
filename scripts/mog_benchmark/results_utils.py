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


def plot_c2st_heatmap(df, method_name):
    """
    Plot a heatmap of c2st scores (rows=budget, cols=dimension).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "C2ST Score"})
    plt.xlabel("Dimensionality (D)")
    plt.ylabel("Budget")
    plt.title(f"{method_name} – C2ST Heatmap")
    plt.gca().invert_yaxis()  # flip so smaller budgets are at the bottom
    plt.tight_layout()
    plt.show(block=False)


def plot_c2st_success_frontier(savepath, methods, threshold=0.75, agg="mean", savefig=True):
    """
    Plot frontier curves: minimum budget per dimension needed to reach a target c2st.

    Parameters
    ----------
    savepath : str
        Path to results root (e.g. "results/mog_benchmark/two_modes")
    methods : list of str
        List of method names to compare.
    threshold : float
        Success threshold for c2st (lower is better).
    agg : str
        Aggregation mode: "mean" or "best".
    """
    plt.figure(figsize=(8, 6))

    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    for ii, method in enumerate(methods):
        df = load_table(savepath, "c2st", method, agg=agg)
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
        plt.plot(dims, budgets, marker=markers[ii % len(markers)], label=method)

    plt.xlabel("Dimensionality (D)")
    plt.ylabel("Minimum Budget to reach target")
    plt.title(f"Success Frontier (threshold={threshold}, agg={agg})")
    plt.xlim(left=0, right=22)
    plt.ylim(bottom=0, top=110_000)
    plt.xticks([2, 5, 10, 15, 20])
    plt.yticks([1_000, 5_000, 10_000, 50_000, 100_000], labels=["1k", "5k", "10k", "50k", "100k"])
    # plt.yscale("log")
    plt.legend(title="Methods")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath, f"c2st_frontier_thr_{threshold}_{agg}.png"))
        plt.savefig(os.path.join(savepath, f"c2st_frontier_thr_{threshold}_{agg}.pdf"))
    plt.show(block=False)


def plot_runtime_success_frontier(savepath, methods, threshold=0.75, agg="mean", savefig=True):
    """
    Plot runtime frontier: minimum runtime per dimension to reach a target c2st.

    Parameters
    ----------
    results_root : str
        Path to results root (e.g. "results/mog_benchmark/two_modes")
    methods : list of str
        List of method names to compare.
    threshold : float
        Success threshold for c2st (lower is better).
    agg : str
        Aggregation mode: "mean" or "best".
    """
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

    for ii, method in enumerate(methods):
        df_c2st = load_table(savepath, "c2st", method, agg=agg)

        frontier_runtime = {}
        for dim in df_c2st.columns:
            ok_budgets = df_c2st.index[df_c2st[dim] <= threshold]
            if len(ok_budgets) > 0:
                # smallest budget that hits the threshold
                chosen_budget = ok_budgets.min()

                # compute avg runtime across runs
                runtimes = []
                budget_path = os.path.join(savepath, f"D_{dim}", f"{method}_{chosen_budget}")
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
        plt.plot(dims, runtimes, marker=markers[ii % len(markers)], label=method)

    plt.xlabel("Dimensionality (D)")
    plt.ylabel("Runtime (s)")
    plt.title(f"Runtime Frontier (threshold={threshold}, agg={agg})")
    plt.xlim(left=0, right=22)
    plt.xticks([2, 5, 10, 15, 20])
    plt.legend(title="Methods")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if savefig:
        plt.savefig(os.path.join(savepath, f"runtime_frontier_thr_{threshold}_{agg}.png"))
        plt.savefig(os.path.join(savepath, f"runtime_frontier_thr_{threshold}_{agg}.pdf"))
    plt.show(block=False)

# main part
if __name__ == "__main__":
    experiment = "single_mode_distractors"
    savepath = os.path.join("./../../results/mog_benchmark", experiment)

    # df_mean = load_table(results_root, metric="c2st", method="npec", agg="mean")
    # df_best = load_table(results_root, metric="c2st", method="npec", agg="best")
    #
    # print("Mean scores:\n", df_mean)
    # print("Best scores:\n", df_best)
    #
    # plot_c2st_vs_dim(df_mean, "NPE-C")
    # plot_c2st_vs_budget(df_mean, "NPE-C")
    # plot_c2st_heatmap(df_mean, "NPE-C")

    plot_c2st_success_frontier(
        savepath,
        methods=["npec", "bayes_flow", "flow_matching", "r2omc"],
        threshold=0.8,
        agg="mean",
        savefig=True
    )

    # plot_c2st_success_frontier(
    #     savepath,
    #     methods=["npec", "bayes_flow", "flow_matching", "r2omc"],
    #     threshold=0.8,
    #     agg="best",
    #     savefig=True
    # )

    plot_runtime_success_frontier(
        savepath,
        methods=["npec", "bayes_flow", "flow_matching", "r2omc"],
        threshold=0.8,
        agg="mean",
        savefig=True
    )

    # plot_runtime_success_frontier(
    #     savepath,
    #     methods=["npec", "bayes_flow", "flow_matching", "r2omc"],
    #     threshold=0.8,
    #     agg="best",
    #     savefig=True
    # )
