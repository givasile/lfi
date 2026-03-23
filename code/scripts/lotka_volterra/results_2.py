import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def compute_sbc_uniformity(ranks, nsamples):
    """
    Compute SBC uniformity with chi-squared test.

    Args:
        ranks: ndarray (n_rep, dim)
        nsamples: int, number of posterior samples per inference

    Returns:
        dict with histograms, chi-squared statistics, and p-values
    """
    n_rep, dim = ranks.shape
    histograms = []
    chi2_stats = []
    p_values = []

    expected_uniform = 1.0 / (nsamples + 1)
    expected_count = n_rep * expected_uniform

    for d in range(dim):
        counts, _ = np.histogram(ranks[:, d], bins=np.arange(nsamples + 2) - 0.5)
        freq = counts / n_rep
        histograms.append(freq)

        # Chi-squared test for uniformity
        chi2 = np.sum((counts - expected_count)**2 / expected_count)
        p_val = 1 - stats.chi2.cdf(chi2, df=nsamples)
        chi2_stats.append(chi2)
        p_values.append(p_val)

    print("SBC Uniformity Test:")
    for d in range(dim):
        print(f"  Param {d+1}: chi2={chi2_stats[d]:.3f}, p={p_values[d]:.3f}")

    return {
        'expected_uniform_freq': expected_uniform,
        'histograms': histograms,
        'chi2_stats': chi2_stats,
        'p_values': p_values,
    }


def compute_coverage(theta_true_list, th_post_list, alphas=(0.5, 0.9)):
    """
    Compute marginal coverage with confidence intervals.

    Args:
        theta_true_list: list of arrays (1, dim) or (dim,)
        th_post_list: list of arrays (nsamples, dim)
        alphas: tuple of credible interval levels

    Returns:
        dict with coverage results and confidence intervals
    """
    n_rep = len(theta_true_list)
    dim = theta_true_list[0].reshape(-1).shape[0]

    coverage = {alpha: np.zeros(dim) for alpha in alphas}

    for i in range(n_rep):
        gt = theta_true_list[i].reshape(-1)
        samples = th_post_list[i]

        for alpha in alphas:
            lower = np.quantile(samples, (1 - alpha) / 2, axis=0)
            upper = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
            inside = (gt >= lower) & (gt <= upper)
            coverage[alpha] += inside.astype(float)

    # Compute coverage and confidence intervals
    results = {}
    for alpha in alphas:
        cov = coverage[alpha] / n_rep
        # Binomial confidence interval (Wilson score)
        se = np.sqrt(cov * (1 - cov) / n_rep)
        ci_lower = cov - 1.96 * se
        ci_upper = cov + 1.96 * se

        results[alpha] = {
            'coverage': cov,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'expected': alpha,
        }

    print("\nCoverage Results:")
    for alpha in alphas:
        cov = results[alpha]['coverage']
        print(f"  Alpha={alpha}: {cov} (expected={alpha})")
        for d in range(dim):
            ci_l = results[alpha]['ci_lower'][d]
            ci_u = results[alpha]['ci_upper'][d]
            in_ci = ci_l <= alpha <= ci_u
            status = "✓" if in_ci else "✗"
            print(f"    Param {d+1}: {cov[d]:.3f} [{ci_l:.3f}, {ci_u:.3f}] {status}")

    return results


def compute_param_recovery(theta_true_list, th_post_list, point_estimate='mean'):
    """
    Compute RMSE, MAE, and bias for parameter recovery.
    """
    n_rep = len(theta_true_list)
    dim = theta_true_list[0].reshape(-1).shape[0]
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
    bias = np.mean(errors, axis=0)

    print("\nParameter Recovery Results:")
    print(f"  MAE per dim: {mae}")
    print(f"  RMSE per dim: {rmse}")
    print(f"  Bias per dim: {bias}")

    return {
        'MAE_per_dim': mae,
        'RMSE_per_dim': rmse,
        'Bias_per_dim': bias,
        'MAE_mean': mae.mean(),
        'RMSE_mean': rmse.mean(),
        'Bias_mean': bias.mean(),
    }


def compute_posterior_width(th_post_list, quantiles=(0.05, 0.95)):
    """
    Compute posterior credible interval widths to check contraction.
    """
    n_rep = len(th_post_list)
    dim = th_post_list[0].shape[1]
    widths = np.zeros((n_rep, dim))

    for i in range(n_rep):
        samples = th_post_list[i]
        lower = np.quantile(samples, quantiles[0], axis=0)
        upper = np.quantile(samples, quantiles[1], axis=0)
        widths[i] = upper - lower

    mean_width = widths.mean(axis=0)
    std_width = widths.std(axis=0)

    print("\nPosterior Width (90% CI):")
    print(f"  Mean width per dim: {mean_width}")
    print(f"  Std width per dim: {std_width}")

    return {
        'mean_width': mean_width,
        'std_width': std_width,
        'all_widths': widths,
    }


def plot_sbc_rank_histograms(ranks, nsamples):
    """Plot SBC rank histograms with expected uniform distribution."""
    dim = ranks.shape[1]
    n_rep = ranks.shape[0]

    fig, axes = plt.subplots(dim, 1, figsize=(8, 2.5 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    expected_freq = 1.0 / (nsamples + 1)

    for d in range(dim):
        ax = axes[d]
        ax.hist(ranks[:, d], bins=nsamples + 1,
                range=(-0.5, nsamples + 0.5),
                density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.axhline(expected_freq, color='red', linestyle='--',
                   label=f'Expected uniform ({expected_freq:.4f})')
        ax.set_title(f"SBC Rank Histogram — Parameter {d+1}")
        ax.set_ylabel("Density")
        ax.legend()

    axes[-1].set_xlabel("Rank of true parameter among posterior samples")
    plt.tight_layout()
    plt.savefig("sbc_histograms.png", dpi=150, bbox_inches='tight')
    plt.show(block=False)


def plot_coverage_diagnostics(coverage_results, dim):
    """Plot coverage vs expected coverage."""
    alphas = sorted(coverage_results.keys())

    fig, axes = plt.subplots(1, dim, figsize=(5 * dim, 4))
    if dim == 1:
        axes = [axes]

    for d in range(dim):
        ax = axes[d]
        empirical = [coverage_results[alpha]['coverage'][d] for alpha in alphas]
        ci_lower = [coverage_results[alpha]['ci_lower'][d] for alpha in alphas]
        ci_upper = [coverage_results[alpha]['ci_upper'][d] for alpha in alphas]

        ax.plot(alphas, empirical, 'o-', label='Empirical', markersize=8)
        ax.fill_between(alphas, ci_lower, ci_upper, alpha=0.3, label='95% CI')
        ax.plot(alphas, alphas, 'r--', label='Ideal', linewidth=2)
        ax.set_xlabel('Nominal Coverage')
        ax.set_ylabel('Empirical Coverage')
        ax.set_title(f'Parameter {d+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("coverage_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.show(block=False)


# ============================================================================
# Main evaluation script
# ============================================================================

# Load results
path = "./../../results/lotka_volterra/"
data = np.load(path + "lotka_volterra_sbc_results.npz", allow_pickle=True)

theta_true_list = list(data['theta_true_list'])
th_post_list = list(data['th_post_list'])
d_post_list = list(data['d_post_list'])

n_rep = len(theta_true_list)
nsamples = th_post_list[0].shape[0]
dim = theta_true_list[0].reshape(-1).shape[0]

print(f"Loaded {n_rep} replications, {nsamples} samples per posterior, {dim} dimensions")

# WARNING: Distance-based filtering biases posterior samples
# Only use if distances represent a valid posterior approximation metric
# For unbiased SBC, use all samples or a random subset

# Sort samples by distance (if using distance-based posterior approximation)
for i in range(n_rep):
    sorted_indices = np.argsort(d_post_list[i])
    th_post_list[i] = th_post_list[i][sorted_indices]
    d_post_list[i] = d_post_list[i][sorted_indices]

# Select K samples for evaluation
K = 100
th_post_selected = [samples[:K] for samples in th_post_list]

# Compute ranks for SBC
ranks = np.zeros((n_rep, dim), dtype=int)
for i in range(n_rep):
    theta_true = theta_true_list[i].reshape(-1)
    th_post = th_post_selected[i]

    for d in range(dim):
        rank = np.sum(th_post[:, d] < theta_true[d])
        ranks[i, d] = rank

# Run all diagnostics
print("=" * 70)
print("SBI EVALUATION RESULTS")
print("=" * 70)

sbc_results = compute_sbc_uniformity(ranks, K)
coverage_results = compute_coverage(theta_true_list, th_post_selected,
                                   alphas=(0.1, 0.5, 0.9, 0.95))
param_recovery = compute_param_recovery(theta_true_list, th_post_selected,
                                       point_estimate='mean')
width_results = compute_posterior_width(th_post_selected)

# Generate diagnostic plots
plot_sbc_rank_histograms(ranks, K)
plot_coverage_diagnostics(coverage_results, dim)

print("\n" + "=" * 70)
print("SUMMARY FOR PAPER")
print("=" * 70)
print(f"SBC p-values: {sbc_results['p_values']}")
print(f"Coverage (90%): {coverage_results[0.9]['coverage']}")
print(f"RMSE: {param_recovery['RMSE_per_dim']}")
print(f"Mean posterior width (90% CI): {width_results['mean_width']}")
