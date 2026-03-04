import numpy as np
import jax
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import TwoMoons
from lfi.inference.r2omc import R2OMC

# ── Setup ─────────────────────────────────────────────────────────────────────

KEY         = jax.random.PRNGKey(21)
SEED        = 42
DIM         = 2
BUDGET      = 2_000
NOF_SAMPLES = 500

prior = UniformPrior(dim=DIM, low=-3, high=3)
sim   = TwoMoons(dim=DIM, dim_y=DIM)

# sample one true theta from the prior and simulate the corresponding observation
np.random.seed(SEED)
theta_true = prior.sample_numpy(1)           # (1, 2)
obs        = sim.sample_numpy(theta_true)    # (1, 2)

print(f"true theta : {theta_true[0]}")
print(f"observation: {obs[0]}")

# ── Fit & sample ──────────────────────────────────────────────────────────────
method  = R2OMC(prior, sim, obs)
samples = method.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={},
)

# ── Plot ──────────────────────────────────────────────────────────────────────
method.plot_posterior_samples(
    samples=samples,
    th_true=theta_true[0],
    limits=[-4, 4],
    title="R2OMC — TwoMoons",
    show=False,
)
plt.show(block=False)
