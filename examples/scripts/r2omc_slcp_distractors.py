"""
R2OMC on the SLCP-Distractors problem.

Simulator
---------
    theta ~ Uniform(-3, 3)^5
    mu      = theta[:2]
    s1, s2  = theta[2]^2, theta[3]^2
    rho     = tanh(theta[4])
    Sigma   = [[s1^2, rho*s1*s2], [rho*s1*s2, s2^2]]

    y_informative | theta ~ N(mu, Sigma)           (dim_y = 2)
    y_distractors          ~ noise (independent of theta)
    y = [y_informative, y_distractors]             (dim_y = 2 + DIM_DISTRACTORS)

R2OMC Step 1 automatically detects the 2 informative output dims and ignores
the distractors, so inference accuracy should match plain SLCP.
"""
import numpy as np

from lfi.priors import UniformPrior
from lfi.simulators import SLCPDistractors
from lfi.inference.r2omc import R2OMC, R2OMCMultiObs

# ── Setup ─────────────────────────────────────────────────────────────────────

SEED            = 42
DIM             = 5
DIM_DISTRACTORS = 8                    # noise output dims
DIM_Y           = 2 + DIM_DISTRACTORS  # total output dim = 10
N_OBS           = 3
BUDGET          = 5_000

prior = UniformPrior(dim=DIM, low=-3, high=3)
sim   = SLCPDistractors(dim=DIM, dim_y=DIM_Y, dim_distractors=DIM_DISTRACTORS)

# ── Sample theta_true and N_OBS independent observations ──────────────────────

np.random.seed(SEED)
theta_true = prior.sample_numpy(1)                                  # (1, 5)
obs_list   = [sim.sample_numpy(theta_true) for _ in range(N_OBS)]  # list of (1, 10)
obs_multi  = np.concatenate(obs_list, axis=0)                       # (N_OBS, 10)

print(f"true theta   : {theta_true[0].round(3)}")
for i, obs in enumerate(obs_list):
    print(f"observation {i+1}: {obs[0].round(3)}")

# ── Single-observation R2OMC ──────────────────────────────────────────────────

print("\n── Single observation ──")
method_single  = R2OMC(prior, sim, obs_list[0])
samples_single = method_single.fit_and_sample(
    budget=BUDGET,
    nof_samples=1_000,
    fit_kwargs={"pcg_to_keep": 0.5},
    sample_kwargs={},
)
print(f"  informative dims detected: {method_single.informative_dims.tolist()}")

# ── Multi-observation R2OMC ────────────────────────────────────────────────────

print(f"\n── Multi-observation ({N_OBS} obs) ──")
method_multi = R2OMCMultiObs(prior, sim, obs_multi)
method_multi.fit(budget=BUDGET, fit_kwargs={"pcg_to_keep": 0.5})

SAMPLE_KWARGS = {"quantile": 0.05, "nof_samples_per_obs": 5_000}

print("\n── Sampling 50 (very tight) ──")
samples_50  = method_multi.sample(nof_samples=50,  sample_kwargs=SAMPLE_KWARGS)

print("\n── Sampling 300 (high accuracy) ──")
samples_300 = method_multi.sample(nof_samples=300, sample_kwargs=SAMPLE_KWARGS)

print("\n── Sampling 750 (broader coverage) ──")
samples_750 = method_multi.sample(nof_samples=750, sample_kwargs=SAMPLE_KWARGS)

# ── Posterior plots — all 5 dims ──────────────────────────────────────────────

method_single.plot_posterior_samples(
    th_true=theta_true[0],
    title=f"SLCP+{DIM_DISTRACTORS} distractors — R2OMC — 1 observation",
    limits=[-3, 3],
)

method_multi.plot_posterior_samples(
    samples=samples_50,
    th_true=theta_true[0],
    title=f"SLCP+{DIM_DISTRACTORS} distractors — R2OMCMultiObs — {N_OBS} obs — top 50",
    limits=[-3, 3],
)

method_multi.plot_posterior_samples(
    samples=samples_300,
    th_true=theta_true[0],
    title=f"SLCP+{DIM_DISTRACTORS} distractors — R2OMCMultiObs — {N_OBS} obs — top 300",
    limits=[-3, 3],
)

method_multi.plot_posterior_samples(
    samples=samples_750,
    th_true=theta_true[0],
    title=f"SLCP+{DIM_DISTRACTORS} distractors — R2OMCMultiObs — {N_OBS} obs — top 750",
    limits=[-3, 3],
)

import matplotlib.pyplot as plt
plt.show(block=True)
