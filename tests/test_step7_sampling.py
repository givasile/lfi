"""Tests for R2OMC Step 7: weighted_sampling + importance_resampling."""
import warnings
import pytest
import numpy as np
import jax

from lfi.priors import UniformPrior
from lfi.simulators import GaussianNoise
from lfi.inference.r2omc import R2OMC

KEY = jax.random.PRNGKey(0)
DIM = 2
NOF_SEEDS = 20
NOF_TH0 = 2
PCG_TO_KEEP = 0.5
SAMPLES_PER_REGION = 5


# ── helpers / fixtures ────────────────────────────────────────────────────────

def make_method():
    prior = UniformPrior(dim=DIM, low=-3, high=3)
    sim   = GaussianNoise(dim=DIM, dim_y=DIM, sigma_noise=0.1)
    obs   = np.zeros((1, DIM), dtype=np.float32)
    return R2OMC(prior=prior, simulator=sim, observation=obs)


@pytest.fixture
def boxes_method():
    """R2OMC after all steps up to and including get_boxes."""
    m = make_method()
    m.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
    m.optimize(nof_gd_steps=50, alpha=0.1)
    m.filter_solutions_inside_prior()
    m.filter_solutions(pcg_to_keep=PCG_TO_KEEP)
    m.get_directions(method="blind")
    m.get_boxes_blind(dx=0.5)
    return m


@pytest.fixture
def sampled_method(boxes_method):
    """R2OMC after weighted_sampling."""
    boxes_method.weighted_sampling(
        sampling_seed=42,
        samples_per_region=SAMPLES_PER_REGION,
        eps_3=1.0,
    )
    return boxes_method


# ── weighted_sampling ─────────────────────────────────────────────────────────

class TestWeightedSampling:

    def test_samples_shape(self, sampled_method):
        """samples must have shape (S, T, N_per_region, D)."""
        m = sampled_method
        S = m.nof_seeds_accept
        assert m.samples.shape == (S, NOF_TH0, SAMPLES_PER_REGION, DIM)

    def test_weights_shape(self, sampled_method):
        """samples_weights must have same shape as samples minus last dim."""
        m = sampled_method
        S = m.nof_seeds_accept
        assert m.samples_weights.shape == (S, NOF_TH0, SAMPLES_PER_REGION)

    def test_flat_shapes(self, sampled_method):
        """Flat arrays must have total length S*T*N_per_region."""
        m = sampled_method
        total = m.nof_seeds_accept * NOF_TH0 * SAMPLES_PER_REGION
        assert m.samples_flat.shape           == (total, DIM)
        assert m.samples_weights_flat.shape   == (total,)

    def test_weights_sum_to_one(self, sampled_method):
        """Normalised weights must sum to 1."""
        assert np.isclose(sampled_method.samples_weights_flat.sum(), 1.0, atol=1e-6)

    def test_weights_are_nonneg(self, sampled_method):
        """All weights must be non-negative."""
        assert np.all(sampled_method.samples_weights_flat >= 0)

    def test_reproducibility(self, boxes_method):
        """Same sampling_seed must produce identical results."""
        boxes_method.weighted_sampling(sampling_seed=7, samples_per_region=SAMPLES_PER_REGION)
        s1 = boxes_method.samples_flat.copy()
        w1 = boxes_method.samples_weights_flat.copy()

        boxes_method.weighted_sampling(sampling_seed=7, samples_per_region=SAMPLES_PER_REGION)
        assert np.array_equal(s1, boxes_method.samples_flat)
        assert np.array_equal(w1, boxes_method.samples_weights_flat)

    def test_different_seeds_differ(self, boxes_method):
        """Different sampling seeds must produce different samples."""
        boxes_method.weighted_sampling(sampling_seed=1, samples_per_region=SAMPLES_PER_REGION)
        s1 = boxes_method.samples_flat.copy()
        boxes_method.weighted_sampling(sampling_seed=2, samples_per_region=SAMPLES_PER_REGION)
        assert not np.array_equal(s1, boxes_method.samples_flat)

    def test_numerical_stability_all_outside_eps3(self, boxes_method):
        """When ALL samples are outside eps_3, weights must not be NaN.

        All log-weights become -inf → max = -inf → naive max-shift gives nan.
        The guard falls back to uniform weights and issues a warning.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            boxes_method.weighted_sampling(sampling_seed=0, samples_per_region=SAMPLES_PER_REGION, eps_3=1e-10)
        assert not np.any(np.isnan(boxes_method.samples_weights_flat))
        assert np.isclose(boxes_method.samples_weights_flat.sum(), 1.0, atol=1e-6)
        # a warning must have been issued
        assert any("eps_3" in str(w.message).lower() or "zero weight" in str(w.message).lower() for w in caught)


# ── importance_resampling ─────────────────────────────────────────────────────

class TestImportanceResampling:

    def test_output_shape(self, sampled_method):
        """importance_resampling must return exactly nof_samples rows."""
        m = sampled_method
        result = m.importance_resampling(m.samples_flat, m.samples_weights_flat, nof_samples=10)
        assert result.shape == (10, DIM)

    def test_samples_stored(self, sampled_method):
        """samples_final must be set on the object."""
        m = sampled_method
        result = m.importance_resampling(m.samples_flat, m.samples_weights_flat, nof_samples=10)
        assert m.samples_final is not None
        assert np.array_equal(m.samples_final, result)

    def test_samples_come_from_pool(self, sampled_method):
        """Every returned sample must appear in the original pool."""
        m = sampled_method
        result = m.importance_resampling(m.samples_flat, m.samples_weights_flat, nof_samples=10)
        for row in result:
            assert np.any(np.all(np.isclose(m.samples_flat, row), axis=1))

    def test_no_replace_returns_unique(self, sampled_method):
        """With replace=False, returned samples must all be unique rows."""
        m = sampled_method
        nof_samples = min(10, int(np.sum(m.samples_weights_flat > 0)))
        result = m.importance_resampling(
            m.samples_flat, m.samples_weights_flat, nof_samples=nof_samples, replace=False
        )
        # check no duplicate rows
        unique_rows = np.unique(result, axis=0)
        assert unique_rows.shape[0] == result.shape[0]

    def test_fallback_warns_when_few_positive_weights(self, sampled_method):
        """A warning must be issued when positive-weight samples < nof_samples."""
        m = sampled_method
        # force almost all weights to zero
        weights = np.zeros_like(m.samples_weights_flat)
        weights[0] = 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.importance_resampling(m.samples_flat, weights, nof_samples=5)
        assert len(caught) == 1
        assert "positive weight" in str(caught[0].message).lower()

    def test_fallback_returns_correct_shape(self, sampled_method):
        """Fallback (top-k) path must still return nof_samples rows."""
        m = sampled_method
        weights = np.zeros_like(m.samples_weights_flat)
        weights[0] = 1.0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = m.importance_resampling(m.samples_flat, weights, nof_samples=5)
        assert result.shape == (5, DIM)
