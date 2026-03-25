"""
Regenerate pairwise posterior figures for SLCP with improved legends:
- SLCP obs_0..obs_3: no legend (redundant panels)
- SLCP selected: legend kept
"""
import sys, os, shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pairwise_posterior(samples, limits, samples_gt, show_legend, savefig):
    dim = samples.shape[1]
    cols = [f"x_{i+1}" for i in range(dim)]
    df = pd.DataFrame(samples, columns=cols)
    df["Type"] = "Inferred"
    df_gt = pd.DataFrame(samples_gt, columns=cols)
    df_gt["Type"] = "Ground truth"
    df_all = pd.concat([df, df_gt], ignore_index=True)

    g = sns.pairplot(
        data=df_all, hue="Type", vars=cols,
        kind="scatter", diag_kind="kde",
        palette={"Inferred": "royalblue", "Ground truth": "crimson"},
        plot_kws={"alpha": 0.7, "s": 30},
    )
    for ax in g.axes.flat:
        if ax is not None:
            for collection in ax.collections:
                if collection.get_label() == "Inferred":
                    collection.set_alpha(0.7)
                elif collection.get_label() == "Ground truth":
                    collection.set_alpha(0.2)
                    collection.set_sizes([10])
    if limits is not None:
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    g.axes[i, j].set_xlim(limits)
                    g.axes[i, j].set_ylim(limits)
                else:
                    g.axes[i, j].set_xlim(limits)
    if not show_legend:
        if g.legend is not None:
            g.legend.remove()
        # seaborn sometimes stores the legend on the figure instead
        if g.fig.legends:
            for leg in g.fig.legends:
                leg.remove()
    g.savefig(savefig, bbox_inches="tight")
    plt.close()


# ---- SLCP ground truth from sbibm ----
import sbibm
exp_num = 2
budget = 30_000
nof_samples_per_obs = 1000

results_path = f"../../results/sbibm/slcp/budget_{budget}/exp_{exp_num}"
camera_ready_path = "../../../camera-ready/figures/sbibm/slcp"
os.makedirs(camera_ready_path, exist_ok=True)

samples_total   = np.loadtxt(os.path.join(results_path, "samples_total.csv"),   delimiter=",")
samples_selected = np.loadtxt(os.path.join(results_path, "samples_selected.csv"), delimiter=",")

task = sbibm.get_task("slcp")
samples_gt = task.get_reference_posterior_samples(exp_num).numpy()[:200]

# Per-observation panels: no legend
for jj in range(4):
    samples_cur = samples_total[jj * nof_samples_per_obs:(jj + 1) * nof_samples_per_obs]
    fname = f"pairwise_posterior_obs_{jj}.png"
    out = os.path.join(results_path, fname)
    plot_pairwise_posterior(samples_cur, [-3., 3.], samples_gt, show_legend=False, savefig=out)
    shutil.copyfile(out, os.path.join(camera_ready_path, fname))
    print(f"Saved {fname}")

# Selected panel: keep legend
out = os.path.join(results_path, "pairwise_posterior_selected.png")
plot_pairwise_posterior(samples_selected, [-3., 3.], samples_gt, show_legend=False, savefig=out)
shutil.copyfile(out, os.path.join(camera_ready_path, "pairwise_posterior_selected.png"))
print("Saved pairwise_posterior_selected.png")

print("SLCP done.")
