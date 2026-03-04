"""
Methods comparison on the Bimodal Gaussian simulator.

Problem
-------
    theta ~ Uniform(-3, 3)^2
    y | theta ~ 0.5 * N(theta + shift, sigma^2 I) + 0.5 * N(theta - shift, sigma^2 I)
    y_0 = (0, 0)   ->   true posterior: N((-1.5,-1.5), sigma^2 I) + N((+1.5,+1.5), sigma^2 I)

Runs all available methods and plots posterior samples against the ground truth.
"""

import time
import numpy as np
import jax
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import BimodalGaussian
from lfi.inference.native import ABCRejection, SMCInference, ABCRejectionJAX, SMCInferenceJAX
from lfi.inference.r2omc import R2OMC
from lfi.inference.sbi import NPEASingleRound, NPECSingleRound, FMPESingleRound
from lfi.inference.elfi import RejectionSampling, SMCRejection
from lfi.evaluation import c2st

# ── configuration ──────────────────────────────────────────────────────────────

BUDGET      = 10_000
NOF_SAMPLES = 500
DIM         = 2
SIGMA       = 0.1
SHIFT       = 1.5
KEY         = jax.random.PRNGKey(0)

# ── problem setup ──────────────────────────────────────────────────────────────

prior = UniformPrior(dim=DIM, low=-3, high=3)
sim   = BimodalGaussian(dim=DIM, dim_y=DIM, sigma_noise=SIGMA, shift=SHIFT)
OBS   = np.zeros((1, DIM), dtype=np.float32)

# ground truth: equal mixture of N(-shift, sigma^2 I) and N(+shift, sigma^2 I)
rng  = np.random.default_rng(42)
half = NOF_SAMPLES // 2
gt_a = rng.normal(-SHIFT, SIGMA, size=(half, DIM)).astype(np.float32)
gt_b = rng.normal(+SHIFT, SIGMA, size=(NOF_SAMPLES - half, DIM)).astype(np.float32)
gt_samples = np.concatenate([gt_a, gt_b], axis=0)
rng.shuffle(gt_samples)

# ── method registry ────────────────────────────────────────────────────────────

METHODS = [
    # (display name, class, fit_kwargs, sample_kwargs)
    # ("ABCRejection", ABCRejection, {"quantile": 0.1}, {}),
    ("SMCInference", SMCInference, {"nof_rounds": 4, "nof_samples": NOF_SAMPLES}, {}),
    # ("ABCRejectionJAX", ABCRejectionJAX, {"key": KEY, "quantile": 0.1}, {}),
    ("SMCInferenceJAX", SMCInferenceJAX, {"key": KEY, "nof_rounds": 4, "nof_samples": NOF_SAMPLES}, {}),
    # ("R2OMC", R2OMC, {}, {}),
    # ("NPEASingleRound", NPEASingleRound, {}, {}),
    # ("NPECSingleRound", NPECSingleRound, {}, {}),
    # ("FMPESingleRound", FMPESingleRound, {}, {}),
    # ("RejectionSampling (elfi)", RejectionSampling, {}, {}),
    ("SMCRejection (elfi)", SMCRejection, {}, {"nof_iterations": 4}),
]

# ── run loop ───────────────────────────────────────────────────────────────────

results = {}

for name, cls, fit_kwargs, sample_kwargs in METHODS:
    print(f"\n{'─'*50}\n  {name}\n{'─'*50}")
    method = cls(prior, sim, OBS)

    t0 = time.perf_counter()
    method.fit(budget=BUDGET, fit_kwargs=fit_kwargs, verbose=1)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = method.sample(nof_samples=NOF_SAMPLES, sample_kwargs=sample_kwargs, verbose=1)
    t_sample = time.perf_counter() - t0

    results[name] = {
        "method":   method,
        "samples":  samples,
        "t_fit":    t_fit,
        "t_sample": t_sample,
        "c2st":     c2st(gt_samples, samples),
    }

# ── results table ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}\n  Results\n{'='*60}")
print(f"  {'Method':<28}  {'t_fit':>7}  {'t_samp':>7}  {'C2ST':>6}")
print(f"  {'─'*55}")
for name, r in results.items():
    print(f"  {name:<28}  {r['t_fit']:>6.1f}s  {r['t_sample']:>6.1f}s  {r['c2st']:>6.3f}")
print(f"\n  C2ST: 0.5 = perfect match, 1.0 = completely different")

# ── scatter plot ───────────────────────────────────────────────────────────────

ncols = 3
nrows = -(-len(results) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, (name, r) in zip(axes, results.items()):
    s = r["samples"]
    ax.scatter(gt_samples[:, 0], gt_samples[:, 1], s=6, alpha=0.3, color="tab:gray", label="ground truth")
    ax.scatter(s[:, 0],          s[:, 1],          s=6, alpha=0.4, color="tab:blue", label=name)
    ax.set_title(f"{name}\nC2ST={r['c2st']:.3f}", fontsize=9)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")

for ax in axes[len(results):]:
    ax.set_visible(False)

fig.suptitle(
    f"BimodalGaussian — posterior samples vs ground truth  (budget={BUDGET}, samples={NOF_SAMPLES})",
    fontsize=11,
)
fig.tight_layout()
plt.show(block=False)
