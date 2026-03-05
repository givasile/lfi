"""
Simulator
---------
    y | theta ~ 0.5 * N(theta, sigma^2) + 0.5 * N(theta + shift, sigma^2)

Two observations (known ground truth)
--------------------------------------
    obs_1 = 0      ->  single-obs posterior modes: theta=0  and theta=-shift
    obs_2 = shift  ->  single-obs posterior modes: theta=shift and theta=0

Expected result
---------------
    Single obs 1  :  two modes at  0  and  -shift
    Single obs 2  :  two modes at  0  and  +shift
    Multi-obs     :  only theta=0 survives (the only theta consistent with both)
"""
import numpy as np
import jax
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import ShiftedBimodalGaussian
from lfi.inference.r2omc import R2OMC, R2OMCMultiObs

# ── Setup ─────────────────────────────────────────────────────────────────────

KEY    = jax.random.PRNGKey(0)
DIM    = 2
SIGMA  = 0.05
SHIFT  = 1.5
BUDGET = 2_000
NOF_SAMPLES = 500

prior = UniformPrior(dim=DIM, low=-3, high=3)
sim   = ShiftedBimodalGaussian(dim=DIM, dim_y=DIM, sigma_noise=SIGMA, shift=SHIFT)

# fixed observations — the ground truth is theta_true = (0, 0)
obs1      = np.zeros((1, DIM),  dtype=np.float32)                    # y=(0, 0)
obs2      = np.full((1, DIM),   SHIFT, dtype=np.float32)             # y=(shift, shift)
obs_multi = np.concatenate([obs1, obs2], axis=0)                     # (2, 2)

print(f"obs_1 = {obs1[0]}  (posterior modes: theta=(0,0) and theta=({-SHIFT},{-SHIFT}))")
print(f"obs_2 = {obs2[0]}  (posterior modes: theta=({SHIFT},{SHIFT}) and theta=(0,0))")
print(f"expected after cross-filter: theta ≈ (0,0) only\n")

# ── Single-obs runs ───────────────────────────────────────────────────────────

print("── Single observation 1 ──")
m1 = R2OMC(prior, sim, obs1)
s1 = m1.fit_and_sample(BUDGET, NOF_SAMPLES, fit_kwargs={"key": KEY}, sample_kwargs={})

print("\n── Single observation 2 ──")
m2 = R2OMC(prior, sim, obs2)
s2 = m2.fit_and_sample(BUDGET, NOF_SAMPLES, fit_kwargs={"key": KEY}, sample_kwargs={})

# ── Multi-obs ─────────────────────────────────────────────────────────────────

print("\n── Multi-observation: step (i) + step (ii) ──")
m_multi  = R2OMCMultiObs(prior, sim, obs_multi)
s_multi  = m_multi.fit_and_sample(
    BUDGET, NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={"quantile": 0.5},
)

print("\n── Multi-observation: step (ii) only  (quantile=None) ──")
m_multi2 = R2OMCMultiObs(prior, sim, obs_multi)
s_multi2 = m_multi2.fit_and_sample(
    BUDGET, NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={"quantile": None},
)

# ── Plot ──────────────────────────────────────────────────────────────────────

# expected posterior modes in 2D
modes = {
    "true θ=(0,0)":            ( 0.0,   0.0,  "red",    200, "*"),
    f"θ=({-SHIFT},{-SHIFT})":  (-SHIFT,-SHIFT, "orange",  80, "^"),
    f"θ=({SHIFT},{SHIFT})":    ( SHIFT, SHIFT, "green",   80, "s"),
}

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, samples, title in zip(
    axes,
    [s1, s2, s_multi, s_multi2],
    [f"Single obs  (y=(0,0))",
     f"Single obs  (y=({SHIFT},{SHIFT}))",
     "Multi-obs  (i)+(ii)",
     "Multi-obs  (ii) only"],
):
    ax.scatter(samples[:, 0], samples[:, 1],
               s=6, alpha=0.4, color="steelblue", label="posterior samples")
    for label, (x, y, color, size, marker) in modes.items():
        ax.scatter(x, y, s=size, marker=marker, color=color, zorder=5, label=label)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("θ₁")
    ax.set_ylabel("θ₂")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle("ShiftedBimodalGaussian — R2OMCMultiObs sanity check", fontsize=11)
fig.tight_layout()
plt.show(block=False)
