"""
Minimal diagnostic script for LotkaVolterra R2OMC sampling.
Budget=10 so it runs in seconds; reveals exactly where things break.
"""
import time
import warnings
import numpy as np
import jax
import jax.numpy as jnp

from lfi.priors import LogNormal
from lfi.simulators import LotkaVolterra
from lfi.inference.r2omc import R2OMC

warnings.filterwarnings("error")   # turn warnings into errors so nothing is silent

prior = LogNormal(dim=4, mean=[-0.125, -3.0, -0.125, -3.0], std=0.5)
sim   = LotkaVolterra(dim=4, dim_y=20)

np.random.seed(42)
theta_true = prior.sample_numpy(1)
obs = np.array(sim.sample_jax(jnp.array(theta_true[0]), seed=42))[None, :]
print(f"theta_true : {theta_true[0].round(4)}")
print(f"obs        : {obs[0].round(3)}")

# ── Fit with tiny budget ───────────────────────────────────────────────────────
method = R2OMC(prior, sim, obs)
method.fit(
    budget=10,
    fit_kwargs={
        "find_informative_dims": False,
        "epochs": 2,
        "nof_gd_steps": 5,
        "alpha": 0.001,
        "pcg_to_keep": 0.5,
        "box_algorithm": "blind",
        "dx": 0.1,
        "nof_ls_steps": 10,
        "step_size": 0.02,
    },
    verbose=2,
)
print(f"\nth_star range: {method.th_star.min():.4f} – {method.th_star.max():.4f}")
print(f"eps_1 = {method.eps_1:.4f}")

# ── Inspect box samples directly ──────────────────────────────────────────────
print("\n── Manual _get_samples_per_region (seed 0) ──")
key = jax.random.PRNGKey(99)
t0 = time.time()
samples_raw, log_w = method._get_samples_per_region(key, nof_samples=5,
                                                     i_seed=0, i_th0=0,
                                                     eps_3=method.eps_1 * 10)
print(f"  done in {time.time()-t0:.2f}s")
print(f"  samples range: {np.array(samples_raw).min():.4f} – {np.array(samples_raw).max():.4f}")
print(f"  log_weights  : {np.array(log_w).round(2)}")
print(f"  any negative theta? {(np.array(samples_raw) < 0).any()}")

# ── Full sample() ──────────────────────────────────────────────────────────────
print("\n── method.sample() ──")
t0 = time.time()
samples = method.sample(
    nof_samples=5,
    sample_kwargs={
        "eps_3":              method.eps_1 * 10,
        "samples_per_region": 5,
        "replace":            True,
    },
)
print(f"  done in {time.time()-t0:.2f}s")
print(f"  samples shape : {samples.shape}")
print(f"  samples:\n{samples.round(4)}")
