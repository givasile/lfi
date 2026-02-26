"""
Manual test script for lfi inference methods.
Tests one method from each inference script:
  - native.py  -> ABCRejection
  - r2omc.py   -> R2OMC
  - sbi.py     -> NPECSingleRound  (requires lfi[torch-cpu/gpu])
  - elfi.py    -> RejectionSampling (requires lfi[elfi])
"""
import numpy as np
import lfi

# ── shared setup ──────────────────────────────────────────────────────────────

prior      = lfi.priors.UniformPrior(dim=2, low=-3, high=3)
simulator  = lfi.simulators.GaussianNoise(dim=2, dim_y=2, sigma_noise=0.1)
obs        = np.array([[1.5, 1.5]])

BUDGET     = 1_000
NOF_SAMPLES = 100


# ── native.py : ABCRejection ──────────────────────────────────────────────────

print("\n=== native.ABCRejection ===")
method = lfi.inference.native.ABCRejection(
    prior=prior, simulator=simulator, observation=obs
)
samples = method.fit_and_sample(budget=BUDGET, nof_samples=NOF_SAMPLES, fit_kwargs={"quantile": 0.1})
print(f"samples shape: {samples.shape}")
method.plot_posterior_samples(limits=(-3, 3), show=True)


# ── r2omc.py : R2OMC ─────────────────────────────────────────────────────────

print("\n=== r2omc.R2OMC ===")
method = lfi.inference.r2omc.R2OMC(
    prior=prior, simulator=simulator, observation=obs
)
samples = method.fit_and_sample(budget=BUDGET, nof_samples=NOF_SAMPLES)
print(f"samples shape: {samples.shape}")
method.plot_posterior_samples(limits=(-3, 3), show=True)


# ── sbi.py : NPECSingleRound ──────────────────────────────────────────────────

print("\n=== sbi.NPECSingleRound ===")
method = lfi.inference.sbi.NPECSingleRound(
    prior=prior, simulator=simulator, observation=obs
)
samples = method.fit_and_sample(budget=BUDGET, nof_samples=NOF_SAMPLES)
print(f"samples shape: {samples.shape}")
method.plot_posterior_samples(limits=(-3, 3), show=True)


# ── elfi.py : RejectionSampling ───────────────────────────────────────────────

print("\n=== elfi.RejectionSampling ===")
method = lfi.inference.elfi.RejectionSampling(
    prior=prior, simulator=simulator, observation=obs
)
samples = method.fit_and_sample(budget=BUDGET, nof_samples=NOF_SAMPLES)
print(f"samples shape: {samples.shape}")
method.plot_posterior_samples(limits=(-3, 3), show=True)
