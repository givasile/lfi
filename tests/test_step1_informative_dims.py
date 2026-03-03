"""
Tests for R2OMC Step 1: find_informative_dims.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lfi.priors import UniformPrior
from lfi.simulators import (
    GaussianNoise,
    GaussianNoiseDistractors,
    BimodalGaussianDistractors,
    SLCPDistractors,
)
from lfi.simulators.base import BaseSimulator
from lfi.inference.r2omc import R2OMC

KEY = jax.random.PRNGKey(0)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_r2omc(prior, simulator, obs):
    return R2OMC(prior=prior, simulator=simulator, observation=obs)


class ConstantSimulator(BaseSimulator):
    """Simulator whose output is always zero, independent of theta.
    Jacobian is identically zero — used to test the degenerate guard."""

    def __init__(self, dim, dim_y):
        super().__init__("constant", dim, dim_y)

    def sample_jax(self, theta, seed):
        return jnp.zeros(self.dim_y)

    def sample_numpy(self, theta):
        return np.zeros((theta.shape[0], self.dim_y))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(params=[
    "gaussian_noise_distractor",
    "bimodal_gaussian_distractor",
    "slcp_distractor",
])
def distractor_case(request):
    """Fresh (simulator, prior, obs, expected_inf_dims) for each distractor type."""
    if request.param == "gaussian_noise_distractor":
        return (
            GaussianNoiseDistractors(dim=2, dim_y=4, dim_distractors=2),
            UniformPrior(dim=2, low=-3, high=3),
            np.zeros((1, 4)),
            [True, True, False, False],
        )
    if request.param == "bimodal_gaussian_distractor":
        return (
            BimodalGaussianDistractors(dim=2, dim_y=4, dim_distractors=2),
            UniformPrior(dim=2, low=-3, high=3),
            np.zeros((1, 4)),
            [True, True, False, False],
        )
    if request.param == "slcp_distractor":
        return (
            SLCPDistractors(dim=5, dim_y=12, dim_distractors=10),
            UniformPrior(dim=5, low=-3, high=3),
            np.zeros((1, 12)),
            [True, True] + [False] * 10,
        )


# ── 1. Output contract ────────────────────────────────────────────────────────

class TestInformativeDimsOutput:
    """find_informative_dims satisfies its basic output contract."""

    def test_shape(self, distractor_case):
        simulator, prior, obs, _ = distractor_case
        method = make_r2omc(prior, simulator, obs)
        method.find_informative_dims(KEY)
        assert method.informative_dims.shape == (simulator.dim_y,)

    def test_dtype_is_bool(self, distractor_case):
        simulator, prior, obs, _ = distractor_case
        method = make_r2omc(prior, simulator, obs)
        method.find_informative_dims(KEY)
        assert method.informative_dims.dtype == jnp.bool_

    def test_mask_propagated_to_both_objects(self, distractor_case):
        """The same mask must be written to both the R2OMC object and the simulator."""
        simulator, prior, obs, _ = distractor_case
        method = make_r2omc(prior, simulator, obs)
        assert simulator.informative_dims is None  # simulator starts unset
        method.find_informative_dims(KEY)
        assert jnp.array_equal(method.informative_dims, simulator.informative_dims)

    def test_reproducibility(self, distractor_case):
        """Same key must produce identical results across two fresh instances."""
        simulator, prior, obs, _ = distractor_case
        m1 = make_r2omc(prior, simulator, obs)
        m2 = make_r2omc(prior, simulator, obs)
        m1.find_informative_dims(KEY)
        m2.find_informative_dims(KEY)
        assert jnp.array_equal(m1.informative_dims, m2.informative_dims)


# ── 2. Correctness ────────────────────────────────────────────────────────────

class TestInformativeDimsCorrectness:
    """find_informative_dims correctly identifies informative vs distractor dims."""

    def test_detects_correct_dims(self, distractor_case):
        simulator, prior, obs, expected = distractor_case
        method = make_r2omc(prior, simulator, obs)
        method.find_informative_dims(KEY)
        result = [bool(v) for v in method.informative_dims]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_all_dims_informative_when_no_distractors(self):
        """GaussianNoise with no distractors → every output dim is informative."""
        sim   = GaussianNoise(dim=2, dim_y=2, sigma_noise=0.1)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 2))
        method = make_r2omc(prior, sim, obs)
        method.find_informative_dims(KEY)
        assert all(bool(v) for v in method.informative_dims)


# ── 3. Threshold behaviour ────────────────────────────────────────────────────

class TestInformativeDimsThreshold:

    def test_zero_threshold_still_drops_zero_jacobian_dims(self):
        """threshold=0 means aggregated > 0: dims with exact zero Jacobian stay False."""
        sim   = GaussianNoiseDistractors(dim=2, dim_y=4, dim_distractors=2)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 4))
        method = make_r2omc(prior, sim, obs)
        method.find_informative_dims(KEY, inf_dims_threshold=0.0)
        result = [bool(v) for v in method.informative_dims]
        assert result == [True, True, False, False]

    def test_higher_threshold_never_keeps_more_dims(self):
        """Increasing the threshold can only drop dims, never add them."""
        sim   = GaussianNoiseDistractors(dim=2, dim_y=4, dim_distractors=2)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 4))
        m_low  = make_r2omc(prior, sim, obs)
        m_high = make_r2omc(prior, sim, obs)
        m_low.find_informative_dims(KEY, inf_dims_threshold=0.01)
        m_high.find_informative_dims(KEY, inf_dims_threshold=0.5)
        assert int(jnp.sum(m_high.informative_dims)) <= int(jnp.sum(m_low.informative_dims))

    def test_threshold_one_returns_all_false(self):
        """threshold=1.0 means aggregated > max(aggregated), which is always False.
        Documents the edge case: all dims dropped when threshold equals 1."""
        sim   = GaussianNoiseDistractors(dim=2, dim_y=4, dim_distractors=2)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 4))
        method = make_r2omc(prior, sim, obs)
        method.find_informative_dims(KEY, inf_dims_threshold=1.0)
        assert not any(bool(v) for v in method.informative_dims)


# ── 4. Degenerate simulator ───────────────────────────────────────────────────

class TestInformativeDimsDegenerate:

    def test_constant_simulator_keeps_all_dims(self):
        """When max Jacobian < 1e-6 the guard returns all-True (keep everything)."""
        sim   = ConstantSimulator(dim=2, dim_y=3)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 3))
        method = make_r2omc(prior, sim, obs)
        method.find_informative_dims(KEY)
        assert all(bool(v) for v in method.informative_dims)


# ── 5. Boundary inputs ────────────────────────────────────────────────────────

class TestInformativeDimsBoundary:

    def test_min_nof_th_and_seeds(self):
        """nof_th=1 and nof_seeds=1 must run without error and return a valid mask."""
        sim   = GaussianNoiseDistractors(dim=2, dim_y=4, dim_distractors=2)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 4))
        method = make_r2omc(prior, sim, obs)
        method.find_informative_dims(KEY, inf_dims_nof_th=1, inf_dims_nof_seeds=1)
        assert method.informative_dims.shape == (4,)
        assert method.informative_dims.dtype == jnp.bool_

    def test_qualitative_stability_across_keys(self):
        """The informative/distractor split must be stable across different random keys."""
        sim   = GaussianNoiseDistractors(dim=2, dim_y=4, dim_distractors=2)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 4))
        results = []
        for seed in range(5):
            method = make_r2omc(prior, sim, obs)
            method.find_informative_dims(jax.random.PRNGKey(seed))
            results.append([bool(v) for v in method.informative_dims])
        assert all(r == results[0] for r in results), \
            f"Unstable results across keys: {results}"


# ── 6. Aggregation modes ──────────────────────────────────────────────────────

class TestInformativeDimsAggregation:

    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_mean_and_max_detect_correct_dims(self, aggregation, distractor_case):
        """mean and max must correctly separate informative from distractor dims."""
        simulator, prior, obs, expected = distractor_case
        method = make_r2omc(prior, simulator, obs)
        method.find_informative_dims(KEY, inf_dims_aggregation=aggregation)
        result = [bool(v) for v in method.informative_dims]
        assert result == expected, \
            f"aggregation='{aggregation}': expected {expected}, got {result}"

    def test_median_never_promotes_distractor_dims(self, distractor_case):
        """median may miss informative dims on sparse Jacobians (false negatives are
        acceptable), but must never mark a zero-Jacobian distractor dim as informative."""
        simulator, prior, obs, expected = distractor_case
        method = make_r2omc(prior, simulator, obs)
        method.find_informative_dims(KEY, inf_dims_aggregation="median")
        result = [bool(v) for v in method.informative_dims]
        for i, is_informative in enumerate(expected):
            if not is_informative:
                assert not result[i], \
                    f"median incorrectly marked distractor dim {i} as informative"

    def test_invalid_aggregation_raises(self):
        """An unrecognised aggregation string must raise ValueError."""
        sim   = GaussianNoise(dim=2, dim_y=2, sigma_noise=0.1)
        prior = UniformPrior(dim=2, low=-3, high=3)
        obs   = np.zeros((1, 2))
        method = make_r2omc(prior, sim, obs)
        with pytest.raises(ValueError, match="inf_dims_aggregation"):
            method.find_informative_dims(KEY, inf_dims_aggregation="sum")
