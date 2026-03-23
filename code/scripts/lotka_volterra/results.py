import numpy as np
import matplotlib.pyplot as plt


def compute_sbc_uniformity(ranks, nsamples):
    """
    Compute SBC uniformity statistic: how close the rank histogram is to uniform.

    Args:
        ranks: ndarray (n_rep, dim)
        nsamples: int, number of posterior samples per inference

    Returns:
        dict with:
          - 'expected_uniform_freq' = 1 / (nsamples + 1)
          - 'histograms': list of histograms per parameter (freq counts)
    """
    n_rep, dim = ranks.shape
    histograms = []
    for d in range(dim):
        counts, _ = np.histogram(ranks[:, d], bins=np.arange(nsamples+2)-0.5)
        freq = counts / n_rep
        histograms.append(freq)
    expected_uniform = 1.0 / (nsamples + 1)
    return {
        'expected_uniform_freq': expected_uniform,
        'histograms': histograms,
    }



def compute_coverage(theta_true_list, th_post_list, alphas=(0.5, 0.9)):
    """
    Compute marginal coverage of credible intervals.

    Args:
        theta_true_list: list length n_rep, arrays (1, dim)
        th_post_list: list length n_rep, arrays (nsamples, dim)
        alphas: tuple of credible interval levels

    Returns:
        dict: alpha -> coverage ndarray (dim,)
    """
    n_rep = len(theta_true_list)
    dim = theta_true_list[0].shape[-1]
    coverage = {alpha: np.zeros(dim) for alpha in alphas}

    for i in range(n_rep):
        gt = theta_true_list[i].reshape(-1)
        samples = th_post_list[i]
        for alpha in alphas:
            lower = np.quantile(samples, (1 - alpha) / 2, axis=0)
            upper = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
            inside = (gt >= lower) & (gt <= upper)
            coverage[alpha] += inside.astype(float)

    for alpha in alphas:
        coverage[alpha] /= n_rep
    return coverage


def compute_param_recovery(theta_true_list, th_post_list, point_estimate='median'):
    """
    Compute RMSE and MAE between true parameters and posterior point estimates.

    Args:
        theta_true_list: list length n_rep, arrays (1, dim)
        th_post_list: list length n_rep, arrays (nsamples, dim)
        point_estimate: 'mean' or 'median'

    Returns:
        dict with 'RMSE_per_dim', 'MAE_per_dim', 'RMSE_mean', 'MAE_mean'
    """
    n_rep = len(theta_true_list)
    dim = theta_true_list[0].shape[-1]
    errors = np.zeros((n_rep, dim))

    for i in range(n_rep):
        gt = theta_true_list[i].reshape(-1)
        samples = th_post_list[i]
        if point_estimate == 'mean':
            est = samples.mean(axis=0)
        elif point_estimate == 'median':
            est = np.median(samples, axis=0)
        else:
            raise ValueError("point_estimate must be 'mean' or 'median'")
        errors[i] = est - gt

    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    return {
        'MAE_per_dim': mae,
        'RMSE_per_dim': rmse,
        'MAE_mean': mae.mean(),
        'RMSE_mean': rmse.mean(),
    }


def plot_sbc_rank_histograms(ranks, nsamples, param_names=None):
    """
    Plot SBC rank histograms for each parameter.

    Args:
        ranks: ndarray of shape (n_rep, dim), ranks of true params among posterior samples
        nsamples: int, number of posterior samples per inference
        param_names: list of strings, optional names for each parameter (length = dim)
    """
    n_rep, dim = ranks.shape
    bins = np.arange(nsamples + 2) - 0.5  # bins for histogram (ranks go 0..nsamples)
    expected_freq = 1.0 / (nsamples + 1)

    if param_names is None:
        param_names = [f"Param {i+1}" for i in range(dim)]

    fig, axes = plt.subplots(dim, 1, figsize=(8, 2.5 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    for d in range(dim):
        counts, _ = np.histogram(ranks[:, d], bins=bins)
        freqs = counts / n_rep

        ax = axes[d]
        ax.bar(range(nsamples + 1), freqs, color='skyblue', edgecolor='k', alpha=0.7)
        ax.axhline(expected_freq, color='red', linestyle='--', label='Expected uniform freq')
        ax.set_title(f"SBC Rank Histogram — {param_names[d]}")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, max(max(freqs) * 1.1, expected_freq * 2))
        ax.legend()

    axes[-1].set_xlabel("Rank of true parameter among posterior samples")
    plt.tight_layout()
    plt.show()


# main part

# load results
path = "../../results/lotka_volterra/"
data = np.load(path + "lotka_volterra_sbc_results.npz", allow_pickle=True)
theta_true_list = data['theta_true_list']   # (150, 1, 4)
th_post_list    = data['th_post_list']       # (150, 1000, 4)
d_post_list     = data['d_post_list']        # (150, 1000)
n_rep           = len(theta_true_list)
dim             = theta_true_list[0].shape[-1]

# posterior samples = top-K optimisation endpoints by smallest distance (per rep)
K = 100
samples = [th_post_list[i][np.argsort(d_post_list[i])[:K]] for i in range(n_rep)]
nsamples = K

param_names = ["alpha", "beta", "gamma", "delta"]
param_names_long = [
    "alpha (prey birth rate)",
    "beta  (predation rate)",
    "gamma (predator death rate)",
    "delta (reproduction efficiency)",
]

# ranks
ranks = np.zeros((n_rep, dim), dtype=int)
for i in range(n_rep):
    gt = theta_true_list[i].reshape(-1)
    for d in range(dim):
        ranks[i, d] = np.sum(samples[i][:, d] < gt[d])

# chi2 with K+1 bins (df=K)
from scipy.stats import chisquare
bins = np.arange(K + 2) - 0.5
chi2_stats, chi2_pvals = [], []
for d in range(dim):
    counts, _ = np.histogram(ranks[:, d], bins=bins)
    stat, p = chisquare(counts)
    chi2_stats.append(stat)
    chi2_pvals.append(p)

# coverage
coverage_results = compute_coverage(theta_true_list, samples, alphas=(0.5, 0.9, 0.95))

# print table
header = f"{'Parameter':<32} {'SBC chi2':>9} {'p-value':>9} {'Cov 50%':>9} {'Cov 90%':>9} {'Cov 95%':>9}"
print(header)
print("-" * len(header))
for d in range(dim):
    print(
        f"{param_names_long[d]:<32} "
        f"{chi2_stats[d]:>9.2f} "
        f"{chi2_pvals[d]:>9.3f} "
        f"{coverage_results[0.5][d]:>9.3f} "
        f"{coverage_results[0.9][d]:>9.3f} "
        f"{coverage_results[0.95][d]:>9.3f}"
    )

# coverage calibration curve
alphas_fine = np.linspace(0.05, 0.95, 19)
coverage_fine = compute_coverage(theta_true_list, samples, alphas=tuple(alphas_fine))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # colorblind-friendly
labels = [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"]

fig, ax = plt.subplots(figsize=(4, 4))
for d in range(dim):
    empirical = [coverage_fine[a][d] for a in alphas_fine]
    ax.plot(alphas_fine, empirical, marker="o", markersize=3,
            color=colors[d], label=labels[d])
ax.plot([0, 1], [0, 1], "k-", lw=1.5, zorder=10, label="Ideal")

ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.set_aspect("equal")
plt.tight_layout()

fig_path = "../../results/lotka_volterra/coverage_calibration.pdf"
plt.savefig(fig_path, bbox_inches="tight")
plt.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nCoverage calibration curve saved to {fig_path}")
plt.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()

# SBC rank histograms — separate figure per parameter, two bin resolutions
for nbins_plot in [10, 5]:
    expected_freq = 1.0 / nbins_plot
    for d in range(dim):
        fig, ax = plt.subplots(figsize=(3, 2.5))
        counts, bin_edges = np.histogram(ranks[:, d], bins=nbins_plot, range=(-0.5, K + 0.5))
        freqs = counts / n_rep
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, freqs, width=bar_width * 0.9,
               color=colors[d], alpha=0.75, edgecolor="white")
        ax.axhline(expected_freq, color="black", linestyle="--", lw=1.2, label="Uniform")
        ax.set_title(labels[d], fontsize=11)
        ax.set_xlabel("Rank", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_xlim(-0.5, K + 0.5)
        ax.set_ylim(0, expected_freq * 3)
        ax.legend(fontsize=8)
        plt.tight_layout()

        tag = param_names[d]
        hist_path = f"../../results/lotka_volterra/sbc_rank_hist_{tag}_bins{nbins_plot}.pdf"
        plt.savefig(hist_path, bbox_inches="tight")
        plt.savefig(hist_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"SBC rank histograms (bins={nbins_plot}) saved to results/lotka_volterra/")

# pairwise posterior plot — pick the rep whose theta_true has the best rank
# (median rank across dims, closest to K/2 = perfectly centred in the posterior)
median_ranks = np.median(np.abs(ranks - K / 2), axis=1)
rep = int(np.argmin(median_ranks))

th_post_rep = samples[rep]          # (K, 4)
theta_true   = theta_true_list[rep, 0, :]  # (4,)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for row in range(4):
    for col in range(4):
        ax = axes[row, col]
        if row == col:
            # diagonal: marginal histogram
            ax.hist(th_post_rep[:, col], bins=12, color=colors[col],
                    alpha=0.7, edgecolor="white", density=True)
            ax.axvline(theta_true[col], color="black", lw=1.5, linestyle="--")
            ax.set_yticks([])
        elif row > col:
            # lower triangle: scatter
            ax.scatter(th_post_rep[:, col], th_post_rep[:, row],
                       s=6, alpha=0.4, color="#555555", rasterized=True)
            ax.scatter(theta_true[col], theta_true[row],
                       marker="*", s=120, color="black", zorder=5)
        else:
            ax.axis("off")

        # axis labels on outer edges only
        if row == 3:
            ax.set_xlabel(labels[col], fontsize=8)
        if col == 0 and row > 0:
            ax.set_ylabel(labels[row], fontsize=8)

        # tick numbers only on bottom row and leftmost column
        show_x = (row == 3)
        show_y = (col == 0 and row > 0)
        ax.tick_params(labelsize=6,
                       labelbottom=show_x, labelleft=show_y)

pairwise_path = "../../results/lotka_volterra/posterior_pairwise.pdf"
plt.savefig(pairwise_path, bbox_inches="tight")
plt.savefig(pairwise_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Pairwise posterior saved to {pairwise_path}")
