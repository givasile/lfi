"""
R2OMC on the Lotka-Volterra predator-prey model.

Simulator
---------
    theta = [alpha, beta, gamma, delta]  (all positive)
    dx/dt =  alpha*x - beta*x*y          (prey)
    dy/dt = -gamma*y + delta*x*y         (predator)
    x(0) = 30,  y(0) = 1

    Output: 10 subsampled time points × 2 species → dim_y = 20
            each entry is a log-normal observation around the ODE trajectory.

Prior
-----
    log(theta) ~ N(mean, 0.5²)
    mean = [-0.125, -3.0, -0.125, -3.0]

    This keeps beta, delta small (≈0.05) so the LV equilibrium
    x*=γ/δ≈17  is close to the initial condition x0=30, avoiding
    violent oscillations that would crash populations to zero.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lfi.priors import LogNormal
from lfi.simulators import LotkaVolterra
from lfi.inference.r2omc import R2OMC

# ── Setup ─────────────────────────────────────────────────────────────────────

SEED        = 42
DIM         = 4
DIM_Y       = 20
BUDGET      = 1_000
NOF_SAMPLES = 100

prior = LogNormal(dim=DIM, mean=[-0.125, -3.0, -0.125, -3.0], std=0.5)
sim   = LotkaVolterra(dim=DIM, dim_y=DIM_Y)

# ── Sample theta_true and observation ─────────────────────────────────────────
# Use the JAX simulator for the observation — same as used internally by R2OMC,
# so the observation is consistent with gradient computations and never NaN.

np.random.seed(SEED)
theta_true = prior.sample_numpy(1)                                    # (1, 4)
obs_jax    = sim.sample_jax(jnp.array(theta_true[0]), seed=SEED)     # (20,)
obs        = np.array(obs_jax)[None, :]                               # (1, 20)

print(f"true theta : {theta_true[0].round(3)}")
print(f"observation: {obs[0].round(3)}")

# ── R2OMC ─────────────────────────────────────────────────────────────────────

print("\n── R2OMC ──")
method  = R2OMC(prior, sim, obs)
method.fit(
    budget=BUDGET,
    fit_kwargs={
        "find_informative_dims": False,
        "epochs": 8,
        "nof_gd_steps": 10,
        "alpha": 0.001,
        "pcg_to_keep": 0.05,
        "box_algorithm": "blind",
        "dx": 0.1,
        "nof_ls_steps": 10,
        "step_size": 0.02,
    }
)
samples = method.sample(
    nof_samples=NOF_SAMPLES,
    sample_kwargs={"return_th_star": True},
)

# ── Posterior plot ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

PRIOR_MEAN = np.array([-0.125, -3.0, -0.125, -3.0])
PRIOR_STD  = 0.5
lim_lo = np.exp(PRIOR_MEAN - 2 * PRIOR_STD)
lim_hi = np.exp(PRIOR_MEAN + 2 * PRIOR_STD)
param_names = ["alpha", "beta", "gamma", "delta"]

fig, axes = plt.subplots(DIM, DIM, figsize=(10, 10))
for i in range(DIM):
    for j in range(DIM):
        ax = axes[i, j]
        if i == j:
            ax.hist(samples[:, i], bins=15, color="royalblue", edgecolor="white")
            ax.axvline(theta_true[0, i], color="red", ls="--", lw=1.5)
            ax.set_xlim(lim_lo[i], lim_hi[i])
        else:
            ax.scatter(samples[:, j], samples[:, i],
                       s=15, alpha=0.6, color="royalblue")
            ax.scatter(theta_true[0, j], theta_true[0, i],
                       marker="*", color="red", s=200, zorder=5)
            ax.set_xlim(lim_lo[j], lim_hi[j])
            ax.set_ylim(lim_lo[i], lim_hi[i])
        if i == DIM - 1:
            ax.set_xlabel(param_names[j])
        if j == 0:
            ax.set_ylabel(param_names[i])

fig.suptitle("Lotka-Volterra — R2OMC — th_star samples", fontsize=12)
fig.tight_layout()
plt.show(block=True)
