"""
Tests for lfi.inference.native — JAX variants.

Covers ABCRejectionJAX and SMCInferenceJAX on a simple GaussianNoise problem
where the true posterior is known: theta ~ Uniform(-3, 3), y = theta + noise,
so the posterior given obs=0 should concentrate near 0.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lfi.priors import UniformPrior
from lfi.simulators import GaussianNoise
from lfi.inference.native import ABCRejectionJAX, SMCInferenceJAX

KEY = jax.random.PRNGKey(42)

DIM = 2
DIM_Y = 2
PRIOR = UniformPrior(dim=DIM, low=-3, high=3)
SIMULATOR = GaussianNoise(dim=DIM, dim_y=DIM_Y, sigma_noise=0.1)
OBS = np.zeros((1, DIM_Y), dtype=np.float32)  # observation at theta=0


# ── ABCRejectionJAX ───────────────────────────────────────────────────────────

class TestABCRejectionJAX:

    def test_fit_and_sample_shape(self):
        method = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        method.fit(budget=500, fit_kwargs={"key": KEY, "quantile": 0.1})
        samples = method.sample(nof_samples=40)
        assert samples.shape == (40, DIM)

    def test_posterior_is_set_after_fit(self):
        method = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        assert method.posterior is None
        method.fit(budget=500, fit_kwargs={"key": KEY})
        assert method.posterior is not None

    def test_sample_raises_before_fit(self):
        method = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        with pytest.raises(ValueError, match="not computed"):
            method.sample()

    def test_eps_mode(self):
        method = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        method.fit(budget=500, fit_kwargs={"key": KEY, "eps": 0.5, "quantile": None})
        samples = method.sample(nof_samples=10)
        assert samples.ndim == 2 and samples.shape[1] == DIM

    def test_samples_close_to_observation(self):
        """Accepted samples should be concentrated near the true theta (0, 0)."""
        method = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        method.fit(budget=2000, fit_kwargs={"key": KEY, "quantile": 0.05})
        samples = method.sample(nof_samples=50)
        assert np.abs(np.mean(samples, axis=0)).max() < 0.5

    def test_reproducibility(self):
        """Same key must produce identical posterior samples."""
        m1 = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        m2 = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        m1.fit(budget=500, fit_kwargs={"key": KEY})
        m2.fit(budget=500, fit_kwargs={"key": KEY})
        assert jnp.array_equal(m1.posterior, m2.posterior)

    def test_fit_and_sample_convenience(self):
        method = ABCRejectionJAX(PRIOR, SIMULATOR, OBS)
        samples = method.fit_and_sample(
            budget=500, nof_samples=20,
            fit_kwargs={"key": KEY, "quantile": 0.1},
        )
        assert samples.shape == (20, DIM)


# ── SMCInferenceJAX ───────────────────────────────────────────────────────────

class TestSMCInferenceJAX:

    def test_fit_returns_all_particles(self):
        method = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        all_particles = method.fit(
            budget=300,
            fit_kwargs={"key": KEY, "tolerance_sequence": [1.0, 0.5]},
        )
        assert isinstance(all_particles, list)
        assert len(all_particles) == 2

    def test_posterior_is_set_after_fit(self):
        method = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        assert method.posterior is None
        method.fit(budget=300, fit_kwargs={"key": KEY, "tolerance_sequence": [1.0, 0.5]})
        assert method.posterior is not None

    def test_sample_shape(self):
        method = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        method.fit(budget=300, fit_kwargs={"key": KEY, "tolerance_sequence": [1.0, 0.5]})
        samples = method.sample(nof_samples=20)
        assert samples.shape == (20, DIM)

    def test_sample_raises_before_fit(self):
        method = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        with pytest.raises(ValueError, match="not completed"):
            method.sample()

    def test_later_rounds_have_smaller_tolerance(self):
        """Particles from the last round should be closer to obs than the first."""
        method = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        method.fit(
            budget=500,
            fit_kwargs={"key": KEY, "tolerance_sequence": [2.0, 1.0, 0.5]},
        )
        first_round = method.all_particles[0]
        last_round = method.all_particles[-1]
        dist_first = np.linalg.norm(first_round - OBS, axis=1).mean()
        dist_last = np.linalg.norm(last_round - OBS, axis=1).mean()
        assert dist_last < dist_first

    def test_samples_close_to_observation(self):
        """Final posterior samples should concentrate near the true theta (0, 0)."""
        method = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        method.fit(
            budget=500,
            fit_kwargs={"key": KEY, "tolerance_sequence": [2.0, 1.0, 0.5, 0.2]},
        )
        samples = method.sample(nof_samples=50)
        assert np.abs(np.mean(samples, axis=0)).max() < 0.5

    def test_reproducibility(self):
        """Same key must produce identical all_particles across two runs."""
        m1 = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        m2 = SMCInferenceJAX(PRIOR, SIMULATOR, OBS)
        m1.fit(budget=300, fit_kwargs={"key": KEY, "tolerance_sequence": [1.0, 0.5]})
        m2.fit(budget=300, fit_kwargs={"key": KEY, "tolerance_sequence": [1.0, 0.5]})
        for p1, p2 in zip(m1.all_particles, m2.all_particles):
            assert np.array_equal(p1, p2)
