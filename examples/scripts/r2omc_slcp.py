"""
R2OMC on the SLCP (Simple Likelihood, Complex Posterior) problem.

Simulator
---------
    theta ~ Uniform(-3, 3)^5
    mu      = theta[:2]
    s1, s2  = theta[2]^2, theta[3]^2
    rho     = tanh(theta[4])
    Sigma   = [[s1^2, rho*s1*s2], [rho*s1*s2, s2^2]]
    y | theta ~ N(mu, Sigma)          (dim_y = 2)

Because the covariance is determined by theta[2:5], two draws from the same
theta can look very different.  A single observation therefore leaves a wide
posterior; combining multiple observations from the same theta constrains it
much more tightly.
"""
import numpy as np
import jax
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import SLCP
from lfi.inference.r2omc import R2OMC, R2OMCMultiObs

# ── Setup ─────────────────────────────────────────────────────────────────────

KEY         = jax.random.PRNGKey(21)
SEED        = 42
DIM         = 5
DIM_Y       = 2
N_OBS       = 3        # number of independent observations from theta_true
BUDGET      = 5_000
NOF_SAMPLES = 500

prior = UniformPrior(dim=DIM, low=-3, high=3)
sim   = SLCP(dim=DIM, dim_y=DIM_Y)

# ── Sample theta_true and N_OBS independent observations ──────────────────────

np.random.seed(SEED)
theta_true = prior.sample_numpy(1)                             # (1, 5)
obs_list   = [sim.sample_numpy(theta_true) for _ in range(N_OBS)]  # list of (1, 2)
obs_multi  = np.concatenate(obs_list, axis=0)                  # (N_OBS, 2)

print(f"true theta   : {theta_true[0].round(3)}")
for i, obs in enumerate(obs_list):
    print(f"observation {i+1}: {obs[0].round(3)}")

# ── Single-observation R2OMC (first observation only) ─────────────────────────

print("\n── Single observation ──")
method_single  = R2OMC(prior, sim, obs_list[0])
samples_single = method_single.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY, "pcg_to_keep": 0.5},
    sample_kwargs={},
)

# ── Multi-observation R2OMC ────────────────────────────────────────────────────

print(f"\n── Multi-observation ({N_OBS} obs) ──")
method_multi  = R2OMCMultiObs(prior, sim, obs_multi)
samples_multi = method_multi.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY, "pcg_to_keep": 0.5},
    sample_kwargs={},
)

# ── eps diagnostics ───────────────────────────────────────────────────────────

print("\n── eps values per observation ──")
for i, r in enumerate(method_multi.r2omc_list):
    print(f"  obs {i+1}: eps_1={r.eps_1:.4f}  eps_2={r.eps_2:.4f}")
print(f"  temperature used: {method_multi.temperature:.4e}")

# ── Weight diagnostics ────────────────────────────────────────────────────────

w      = method_multi.w_norm          # (N_total,)  normalised weights
w_in   = method_multi.weight_inside   # (N_total,)  #seeds inside × all obs
n_total    = len(w)
n_positive = (w > 0).sum()
n_accepted = len(method_multi.th_accepted)

print(f"\n── Weight diagnostics ──")
print(f"  Total candidates  : {n_total}")
print(f"  Accepted (w > 0)  : {n_positive}  ({100*n_positive/n_total:.1f}%)")
print(f"  Effective N (1/Σw²): {1/np.sum(w**2):.1f}")
print(f"  weight_inside  min={w_in.min():.0f}  max={w_in.max():.0f}  "
      f"mean={w_in[w_in>0].mean():.1f}  (zeros: {(w_in==0).sum()})")
print(f"  w_norm         min={w[w>0].min():.2e}  max={w.max():.2e}")

# histogram of positive weights
fig_w, axes_w = plt.subplots(1, 2, figsize=(11, 4))

axes_w[0].hist(w_in[w_in > 0], bins=40, color="steelblue", edgecolor="white")
axes_w[0].set_xlabel("weight_inside  (# agreeing seeds, product over obs)")
axes_w[0].set_ylabel("count")
axes_w[0].set_title(f"weight_inside > 0  ({n_positive}/{n_total} candidates)")

axes_w[1].hist(w[w > 0], bins=40, color="steelblue", edgecolor="white")
axes_w[1].set_xlabel("w_norm  (normalised weight)")
axes_w[1].set_ylabel("count")
axes_w[1].set_title("Normalised weights (positive only)")

fig_w.suptitle(f"R2OMCMultiObs weight distribution — SLCP  ({N_OBS} obs)", fontsize=11)
fig_w.tight_layout()
plt.show(block=False)

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, title, samples in zip(
    axes,
    [f"R2OMC — 1 observation", f"R2OMCMultiObs — {N_OBS} observations"],
    [samples_single, samples_multi],
):
    # plot the first two dimensions (mu = theta[:2])
    ax.scatter(samples[:, 0], samples[:, 1],
               s=6, alpha=0.4, color="tab:blue", label="posterior samples")
    ax.scatter(theta_true[0, 0], theta_true[0, 1],
               s=200, marker="*", color="red", zorder=5, label="true θ")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("θ₁  (mu_1)")
    ax.set_ylabel("θ₂  (mu_2)")
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle("SLCP — posterior over θ₁, θ₂  (marginal)", fontsize=11)
fig.tight_layout()
plt.show(block=False)
