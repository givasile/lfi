"""
R2OMC on the TwoMoons problem with two observations from the same true theta.

The idea: a single theta_true generates two independent observations.
R2OMCMultiObs fits R2OMC independently for each observation, then combines
the samples by keeping only those that are consistent with *all* observations,
yielding a tighter posterior than any single observation alone.
"""
import numpy as np
import jax
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import TwoMoons
from lfi.inference.r2omc import R2OMC, R2OMCMultiObs

# ── Setup ─────────────────────────────────────────────────────────────────────

KEY         = jax.random.PRNGKey(21)
SEED        = 42
DIM         = 2
BUDGET      = 2_000
NOF_SAMPLES = 500

prior = UniformPrior(dim=DIM, low=-3, high=3)
sim   = TwoMoons(dim=DIM, dim_y=DIM)

# sample one true theta and generate two independent observations from it
np.random.seed(SEED)
theta_true = prior.sample_numpy(1)           # (1, 2)
obs1       = sim.sample_numpy(theta_true)    # (1, 2)
obs2       = sim.sample_numpy(theta_true)    # (1, 2)
obs_multi  = np.concatenate([obs1, obs2], axis=0)  # (2, 2)

print(f"true theta  : {theta_true[0]}")
print(f"observation 1: {obs1[0]}")
print(f"observation 2: {obs2[0]}")

# ── Single-observation baselines ───────────────────────────────────────────────

print("\n── Single observation 1 ──")
method1  = R2OMC(prior, sim, obs1)
samples1 = method1.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={},
)

print("\n── Single observation 2 ──")
method2  = R2OMC(prior, sim, obs2)
samples2 = method2.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={},
)

# ── Multi-observation R2OMC ────────────────────────────────────────────────────

print("\n── Multi-observation (obs1 + obs2) ──")
method_multi  = R2OMCMultiObs(prior, sim, obs_multi)
samples_multi = method_multi.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={},
)

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["R2OMC — obs 1 only", "R2OMC — obs 2 only", "R2OMCMultiObs — obs 1 + 2"]
samples_list = [samples1, samples2, samples_multi]

for ax, title, samples in zip(axes, titles, samples_list):
    ax.scatter(samples[:, 0], samples[:, 1],
               s=6, alpha=0.4, color="tab:blue", label="posterior samples")
    ax.scatter(theta_true[0, 0], theta_true[0, 1],
               s=200, marker="*", color="red", zorder=5, label="true θ")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("θ₁")
    ax.set_ylabel("θ₂")
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.tight_layout()
plt.show(block=False)
