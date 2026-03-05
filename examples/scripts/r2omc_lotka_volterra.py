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
    theta_i ~ LogNormal(mean=0, std=0.5)   i.e.  log(theta) ~ N(0, 0.5²)
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
BUDGET      = 500
NOF_SAMPLES = 100

prior = LogNormal(dim=DIM, mean=0, std=0.5)
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
samples = method.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"pcg_to_keep": 0.5, "epochs": 10, "nof_gd_steps": 10},
    sample_kwargs={},
)
print(f"  informative dims detected: {method.informative_dims.tolist()}")

# ── Posterior plot ─────────────────────────────────────────────────────────────

method.plot_posterior_samples(
    th_true=theta_true[0],
    title="Lotka-Volterra — R2OMC — single observation",
)

import matplotlib.pyplot as plt
plt.show(block=True)
