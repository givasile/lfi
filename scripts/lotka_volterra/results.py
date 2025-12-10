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
path = "./../results/lotka_volterra/"
data = np.load(path + "lotka_volterra_sbc_results.npz", allow_pickle=True)
theta_true_list = data['theta_true_list']
th_post_list = data['th_post_list']
d_post_list = data['d_post_list']
n_rep = len(theta_true_list)
nsamples = th_post_list[0].shape[0]
dim = theta_true_list[0].shape[-1]


# # plot_sbc_rank_histograms(ranks, nsamples, param_names=["alpha", "beta", "gamma", "delta"])

# sbc_results = compute_sbc_uniformity(ranks, nsamples)
# coverage_results = compute_coverage(theta_true_list, th_post_list)
# param_recovery_results = compute_param_recovery(theta_true_list, th_post_list)

# print("SBC expected uniform freq:", sbc_results['expected_uniform_freq'])
# print("Coverage at 50%:", coverage_results[0.5])
# print("Coverage at 90%:", coverage_results[0.9])
# print("Parameter recovery RMSE:", param_recovery_results['RMSE_per_dim'])
# print("Parameter recovery MAE:", param_recovery_results['MAE_per_dim'])

import numpy as np

# d_post: shape (N,)
# th_post: shape (N, D)
# K: number of top entries to select
K = 100

idx = np.argpartition(d_post, -K)[-K:]      # indices of the top-K distances (unordered)
topk_ordered = idx[np.argsort(-d_post[idx])]  # sort them descending by distance

th_topk = th_post[topk_ordered]            # shape (K, D)
d_topk = d_post[topk_ordered]              # shape (K,)  # optional
