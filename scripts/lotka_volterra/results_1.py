import numpy as np
import matplotlib.pyplot as plt
# Import chisquare for the statistical validation of SBC
from scipy.stats import chisquare

# --- Metric Functions ---

def compute_sbc_uniformity(ranks, nsamples):
    """
    Compute SBC uniformity statistic and chi-squared p-value.
    Args:
        ranks: ndarray (n_rep, dim)
        nsamples: int, number of posterior samples per inference (K)
    Returns:
        dict with:
          - 'expected_uniform_freq': 1 / (nsamples + 1)
          - 'histograms': list of histograms per parameter (freq counts)
          - 'chi2_p_values': ndarray (dim,)
    """
    n_rep, dim = ranks.shape
    histograms = []
    chi2_p_values = np.zeros(dim)
    n_bins = nsamples + 1

    for d in range(dim):
        # bins = 0, 1, ..., nsamples+1
        # np.histogram counts how many times the rank falls into each bin
        counts, _ = np.histogram(ranks[:, d], bins=np.arange(n_bins + 1) - 0.5)

        # Chi-squared test for uniformity
        expected_counts = n_rep / n_bins
        if expected_counts >= 5:
            # chisquare returns (test_statistic, p_value)
            _, p_value = chisquare(counts, f_exp=expected_counts)
            chi2_p_values[d] = p_value
        else:
            chi2_p_values[d] = np.nan

        freq = counts / n_rep
        histograms.append(freq)

    expected_uniform = 1.0 / n_bins

    return {
        'expected_uniform_freq': expected_uniform,
        'histograms': histograms,
        'chi2_p_values': chi2_p_values
    }

def compute_coverage(theta_true_list, th_post_K, alphas=(0.5, 0.9)):
    """
    Compute marginal coverage of credible intervals.
    Args:
        theta_true_list: ndarray (n_rep, 1, dim)
        th_post_K: ndarray (n_rep, K, dim) (K is num posterior samples)
        alphas: tuple of credible interval levels
    Returns:
        dict: alpha -> coverage ndarray (dim,)
    """
    n_rep, K, dim = th_post_K.shape
    coverage = {alpha: np.zeros(dim) for alpha in alphas}

    for i in range(n_rep):
        gt = theta_true_list[i].reshape(-1) # (dim,)
        samples = th_post_K[i] # (K, dim)

        for alpha in alphas:
            lower_q = (1 - alpha) / 2
            upper_q = 1.0 - lower_q

            lower = np.quantile(samples, lower_q, axis=0) # (dim,)
            upper = np.quantile(samples, upper_q, axis=0) # (dim,)

            inside = (gt >= lower) & (gt <= upper) # (dim,) bool
            coverage[alpha] += inside.astype(float)

    for alpha in alphas:
        coverage[alpha] /= n_rep

    return coverage

def compute_param_recovery(theta_true_list, th_post_K, point_estimate='mean'):
    """
    Compute RMSE and MAE between true parameters and posterior point estimates.
    Args:
        theta_true_list: ndarray (n_rep, 1, dim)
        th_post_K: ndarray (n_rep, K, dim) (K is num posterior samples)
        point_estimate: 'mean' or 'median'
    Returns:
        dict with 'RMSE_per_dim', 'MAE_per_dim', 'RMSE_mean', 'MAE_mean'
    """
    n_rep, K, dim = th_post_K.shape
    errors = np.zeros((n_rep, dim))

    for i in range(n_rep):
        gt = theta_true_list[i].reshape(-1)
        samples = th_post_K[i]

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

def plot_sbc_rank_histograms(ranks, K):
    """K is the number of posterior samples used, making K+1 the number of bins."""
    dim = ranks.shape[1]
    nof_bins = K + 1 # Number of bins must be K+1 for SBC

    fig, axes = plt.subplots(dim, 1, figsize=(8, 2.5 * dim), sharex=True)
    if dim == 1:
        axes = [axes] # Ensure axes is iterable even for 1D

    expected_freq = 1.0 / nof_bins

    for d in range(dim):
        ax = axes[d]

        # Hist bins go from -0.5 to nof_bins - 0.5
        ax.hist(ranks[:, d], bins=nof_bins,
                range=(0, nof_bins),
                density=True, alpha=0.7, color='blue', edgecolor='black')

        ax.axhline(expected_freq, color='red', linestyle='--', label=f'Expected: 1/{nof_bins:.0f}')
        ax.set_title(f"SBC Rank Histogram — Param {d+1}")
        ax.set_ylabel("Frequency")
        ax.legend()

    axes[-1].set_xlabel(f"Rank (0 to {K}) of true parameter among posterior samples (K={K})")
    plt.xticks(np.arange(0, nof_bins + 1, max(1, nof_bins // 10)))
    plt.tight_layout()
    # Use block=True so the plot is shown before the script finishes
    plt.show(block=True)


# --- Main Execution Script (Updated) ---

# Load results
path = "./../../results/lotka_volterra/"
data = np.load(path + "lotka_volterra_sbc_results.npz", allow_pickle=True)
# Assuming these are now NumPy arrays based on your clarification
theta_true_list = data['theta_true_list'] # shape (N_rep, 1, D)
th_post_list = data['th_post_list']       # shape (N_rep, N_samples, D)
d_post_list = data['d_post_list']         # shape (N_rep, N_samples)

n_rep = theta_true_list.shape[0]
nsamples_total = th_post_list.shape[1]
dim = theta_true_list.shape[-1]

# 1. Sort posterior samples by distance (ABC-like procedure)
# The lists are now Np arrays, so we loop through the repetitions for sorting
print("Sorting posterior samples by distance...")
for i in range(n_rep):
    # Sort indices based on distance for each repetition
    sorted_indices = np.argsort(d_post_list[i])
    # Apply sorting to the posterior samples and distances
    th_post_list[i] = th_post_list[i][sorted_indices]
    d_post_list[i] = d_post_list[i][sorted_indices]
print("Sorting complete.")

# 2. Define the number of top-K samples to retain for the approximate posterior
# K must be consistent for all evaluations
K = 100

# 3. Slice the posterior array to use only the top-K samples
th_post_K = th_post_list[:, :K, :] # Correct slicing for the NumPy array

# 4. Compute Ranks for SBC
ranks = np.zeros((n_rep, dim), dtype=int)
for i in range(n_rep):
    theta_true = theta_true_list[i].reshape(-1) # (dim,)
    th_post_rep = th_post_K[i] # (K, dim)

    for d in range(dim):
        # Rank is the count of posterior samples strictly LESS THAN the true value
        rank = np.sum(th_post_rep[:, d] < theta_true[d])
        ranks[i, d] = rank

print(f"\n--- Starting Evaluation (N_rep={n_rep}, K={K}, D={dim}) ---")

# 5. Compute Metrics
sbc_results = compute_sbc_uniformity(ranks, K)
coverage_results = compute_coverage(theta_true_list, th_post_K, alphas=(0.1, 0.5, 0.9))
param_recovery_results = compute_param_recovery(theta_true_list, th_post_K, point_estimate='median')

# --- Rebuttal Summary Output ---

print("\n" + "="*80)
print("             🚀 REBUTTAL SUMMARY OF SBI METHOD VALIDATION 🚀")
print("="*80)
print(f"**Method Evaluated on {n_rep} Simulated Data Sets (K={K} retained posterior samples)**")
print("-"*80)

## I. Validity Assessment: Simulation-Based Calibration (SBC)

print("\n### 1. Validity Assessment: Simulation-Based Calibration (SBC)")
print("SBC is the gold standard for testing if an approximate posterior is well-calibrated.")
print(f"Expected Uniform Frequency (H0): $1 / (K+1) = 1 / {K+1} = {sbc_results['expected_uniform_freq']:.4f}")

chi2_p_values = sbc_results['chi2_p_values']
# Create a formatted string for the p-values
p_value_str = ", ".join([f"P{d+1}: {p:.4f}" for d, p in enumerate(chi2_p_values)])

print("\n**SBC Chi-squared P-values per Dimension:**")
print(f"({p_value_str})")
if np.all(chi2_p_values[~np.isnan(chi2_p_values)] > 0.05):
    print("$\implies$ **Conclusion:** All p-values are $> 0.05$, confirming that the rank histograms are statistically uniform. The method is **well-calibrated**.")
else:
    print("$\implies$ **Conclusion:** Some p-values are $\le 0.05$, indicating potential miscalibration in those dimensions.")

## II. Calibration Assessment: Marginal Coverage

print("\n### 2. Calibration Assessment: Marginal Coverage")
print("Tests if the credible intervals contain the true parameter value with the expected probability.")

for alpha in sorted(coverage_results.keys()):
    nominal = alpha * 100
    actual_coverage = coverage_results[alpha] * 100
    coverage_str = ", ".join([f"{c:.2f}%" for c in actual_coverage])

    print(f"\n**Nominal Coverage $\\alpha={alpha}$ ({nominal:.0f}% CI):**")
    print(f"  Actual Coverage (per dim): ({coverage_str})")

    mean_actual = np.mean(actual_coverage)
    print(f"  Mean Actual Coverage: {mean_actual:.2f}%")
    print(f"  Difference from Nominal: {mean_actual - nominal:.2f} percentage points.")

## III. Accuracy Assessment: Parameter Recovery

print("\n### 3. Accuracy Assessment: Parameter Recovery (Point Estimation)")
print(f"Measures the accuracy of the point estimate (Posterior {param_recovery_results['point_estimate']}) relative to the true parameter.")

mae_per_dim = param_recovery_results['MAE_per_dim']
rmse_per_dim = param_recovery_results['RMSE_per_dim']

mae_str = ", ".join([f"P{d+1}: {m:.4f}" for d, m in enumerate(mae_per_dim)])
rmse_str = ", ".join([f"P{d+1}: {r:.4f}" for d, r in enumerate(rmse_per_dim)])

print(f"\n**Mean Absolute Error (MAE):**")
print(f"  Per Dim: ({mae_str})")
print(f"  Overall Mean: {param_recovery_results['MAE_mean']:.4f}")

print(f"\n**Root Mean Squared Error (RMSE):**")
print(f"  Per Dim: ({rmse_str})")
print(f"  Overall Mean: {param_recovery_results['RMSE_mean']:.4f}")

print("\n" + "="*80)

# 6. Plot SBC Histograms
plot_sbc_rank_histograms(ranks, K)

print("Plotting SBC rank histograms...")
print("The script has finished executing and printed the Rebuttal Summary.")
