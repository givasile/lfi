"""
Tests for R2OMC Step 2: sample_objective_functions + optimize.

One test is expected to FAIL before the code changes are applied:
  TestSampleObjectiveFunctions::test_default_nof_th0_is_1
  (current default is 10, desired is 1)
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lfi.priors import UniformPrior
from lfi.simulators import GaussianNoise
from lfi.inference.r2omc import R2OMC

KEY = jax.random.PRNGKey(0)
DIM = 2
NOF_SEEDS = 10
NOF_TH0 = 2


# ── helpers / fixtures ────────────────────────────────────────────────────────

def make_method():
    prior = UniformPrior(dim=DIM, low=-3, high=3)
    sim   = GaussianNoise(dim=DIM, dim_y=DIM, sigma_noise=0.1)
    obs   = np.zeros((1, DIM), dtype=np.float32)
    return R2OMC(prior=prior, simulator=sim, observation=obs)


@pytest.fixture
def method():
    return make_method()


@pytest.fixture
def sampled_method():
    """R2OMC with sample_objective_functions already called."""
    m = make_method()
    m.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
    return m


# ── sample_objective_functions ────────────────────────────────────────────────

class TestSampleObjectiveFunctions:

    def test_output_shapes(self, method):
        """All five output arrays must have the expected shapes."""
        method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert method.seeds_init.shape    == (NOF_SEEDS,)
        assert method.th0_init.shape      == (NOF_SEEDS, NOF_TH0, DIM)
        assert method.d0_init.shape       == (NOF_SEEDS, NOF_TH0)
        assert method.th_star_init.shape  == (NOF_SEEDS, NOF_TH0, DIM)
        assert method.d_star_init.shape   == (NOF_SEEDS, NOF_TH0)

    def test_key_consumed(self, method):
        """Returned key must differ from the input key."""
        new_key = method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert not jnp.array_equal(new_key, KEY)

    def test_reproducibility(self, method):
        """Same key must produce identical arrays across two fresh calls."""
        m1, m2 = make_method(), make_method()
        m1.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        m2.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert np.array_equal(m1.seeds_init,   m2.seeds_init)
        assert np.array_equal(m1.th0_init,     m2.th0_init)
        assert np.array_equal(m1.d0_init,      m2.d0_init)

    def test_seeds_in_valid_range(self, method):
        """Seeds must be non-negative integers below 2**31."""
        method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert np.all(method.seeds_init >= 0)
        assert np.all(method.seeds_init < 2**31)

    def test_th0_inside_prior(self, method):
        """All initial thetas must lie within the prior support [-3, 3]."""
        method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert np.all(method.th0_init >= -3)
        assert np.all(method.th0_init <=  3)

    def test_d0_is_nonneg(self, method):
        """Initial distances must be non-negative."""
        method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert np.all(method.d0_init >= 0)

    def test_working_copies_equal_init(self, method):
        """th_star_init and d_star_init must be copies of th0_init and d0_init."""
        method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        assert np.array_equal(method.th_star_init, method.th0_init)
        assert np.array_equal(method.d_star_init,  method.d0_init)

    def test_default_nof_th0_is_1(self, method):
        """Default nof_th0 must be 1 (matching fit() default).
        FAILS before the fix (current default is 10)."""
        method.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS)
        assert method.th0_init.shape == (NOF_SEEDS, 1, DIM)
        assert method.nof_th0 == 1


# ── optimize ──────────────────────────────────────────────────────────────────

class TestOptimize:

    def test_shapes_unchanged(self, sampled_method):
        """optimize must not change the shape of th_star_init or d_star_init."""
        th_shape = sampled_method.th_star_init.shape
        d_shape  = sampled_method.d_star_init.shape
        sampled_method.optimize(nof_gd_steps=10, alpha=0.1)
        assert sampled_method.th_star_init.shape == th_shape
        assert sampled_method.d_star_init.shape  == d_shape

    def test_distances_nonneg(self, sampled_method):
        """Optimised distances must be non-negative."""
        sampled_method.optimize(nof_gd_steps=10, alpha=0.1)
        assert np.all(sampled_method.d_star_init >= 0)

    def test_optimizer_moves_params(self, sampled_method):
        """th_star_init must differ from th0_init after optimization."""
        th0_copy = sampled_method.th0_init.copy()
        sampled_method.optimize(nof_gd_steps=50, alpha=0.1)
        assert not np.array_equal(sampled_method.th_star_init, th0_copy)

    def test_distances_decrease(self, sampled_method):
        """Mean optimised distance must be lower than mean initial distance."""
        d0_mean = sampled_method.d0_init.mean()
        sampled_method.optimize(nof_gd_steps=50, alpha=0.1)
        assert sampled_method.d_star_init.mean() < d0_mean

    def test_zero_steps_leaves_state_unchanged(self, sampled_method):
        """With nof_gd_steps=0, parameters and distances must be unchanged."""
        th0_copy = sampled_method.th0_init.copy()
        d0_copy  = sampled_method.d0_init.copy()
        sampled_method.optimize(nof_gd_steps=0, alpha=0.1)
        assert np.allclose(sampled_method.th_star_init, th0_copy)
        assert np.allclose(sampled_method.d_star_init,  d0_copy)

    def test_more_steps_lower_distance(self):
        """More gradient steps must yield a lower mean distance."""
        m_few  = make_method()
        m_many = make_method()
        m_few.sample_objective_functions(KEY,  nof_seeds_total=NOF_SEEDS, nof_th0=1)
        m_many.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=1)
        m_few.optimize(nof_gd_steps=5,   alpha=0.1)
        m_many.optimize(nof_gd_steps=100, alpha=0.1)
        assert m_many.d_star_init.mean() < m_few.d_star_init.mean()

    def test_convergence_on_gaussian(self):
        """On GaussianNoise, sufficient optimization steps must drive d_star near 0."""
        m = make_method()
        m.sample_objective_functions(KEY, nof_seeds_total=20, nof_th0=1)
        m.optimize(nof_gd_steps=200, alpha=0.1)
        assert m.d_star_init.mean() < 1e-3
