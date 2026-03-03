"""Tests for R2OMC Steps 3 & 4: filter_solutions_inside_prior + filter_solutions."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from lfi.priors import UniformPrior
from lfi.simulators import GaussianNoise
from lfi.inference.r2omc import R2OMC

KEY = jax.random.PRNGKey(0)
DIM = 2
NOF_SEEDS = 20
NOF_TH0 = 2


# ── helpers / fixtures ────────────────────────────────────────────────────────

def make_method(low=-3.0, high=3.0):
    prior = UniformPrior(dim=DIM, low=low, high=high)
    sim   = GaussianNoise(dim=DIM, dim_y=DIM, sigma_noise=0.1)
    obs   = np.zeros((1, DIM), dtype=np.float32)
    return R2OMC(prior=prior, simulator=sim, observation=obs)


@pytest.fixture
def optimized_method():
    """R2OMC after sample_objective_functions + optimize."""
    m = make_method()
    m.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
    m.optimize(nof_gd_steps=50, alpha=0.1)
    return m


@pytest.fixture
def inside_prior_method(optimized_method):
    """R2OMC after filter_solutions_inside_prior."""
    optimized_method.filter_solutions_inside_prior()
    return optimized_method


# ── filter_solutions_inside_prior ─────────────────────────────────────────────

class TestFilterSolutionsInsidePrior:

    def test_count_matches_array_length(self, inside_prior_method):
        """nof_seeds_inside_prior must equal len(seeds_inside_prior)."""
        m = inside_prior_method
        assert int(m.nof_seeds_inside_prior) == len(m.seeds_inside_prior)

    def test_output_shapes(self, inside_prior_method):
        """Output arrays must have consistent shapes."""
        m = inside_prior_method
        n = int(m.nof_seeds_inside_prior)
        assert m.seeds_inside_prior.shape     == (n,)
        assert m.th_star_inside_prior.shape   == (n, NOF_TH0, DIM)
        assert m.d_star_inside_prior.shape    == (n, NOF_TH0)

    def test_all_kept_seeds_have_at_least_one_inside(self, inside_prior_method):
        """Every kept seed must have at least one theta inside the prior support."""
        m = inside_prior_method
        n = int(m.nof_seeds_inside_prior)
        th_flat = m.th_star_inside_prior.reshape(-1, DIM)
        inside  = m.prior.has_mass(th_flat).reshape(n, NOF_TH0)
        assert np.all(inside.sum(axis=1) > 0)

    def test_seeds_are_subset_of_init(self, inside_prior_method):
        """seeds_inside_prior must be a subset of seeds_init."""
        m = inside_prior_method
        assert set(m.seeds_inside_prior.tolist()) <= set(m.seeds_init.tolist())

    def test_excludes_seeds_with_all_thetas_outside(self, optimized_method):
        """Seeds whose every th_star is outside the prior must be filtered out."""
        m = optimized_method
        # force first 5 seeds entirely outside prior [-3, 3]
        m.th_star_init[:5] = 999.0
        m.filter_solutions_inside_prior()
        assert int(m.nof_seeds_inside_prior) <= NOF_SEEDS - 5

    def test_wide_prior_keeps_all_seeds(self):
        """With a very wide prior all seeds should survive the filter."""
        m = make_method(low=-1e6, high=1e6)
        m.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        m.optimize(nof_gd_steps=10, alpha=0.1)
        m.filter_solutions_inside_prior()
        assert int(m.nof_seeds_inside_prior) == NOF_SEEDS


# ── filter_solutions ──────────────────────────────────────────────────────────

class TestFilterSolutions:

    def test_output_shapes(self, inside_prior_method):
        """Output arrays must have shapes consistent with nof_seeds_accept."""
        m = inside_prior_method
        m.filter_solutions(pcg_to_keep=0.5)
        n = m.nof_seeds_accept
        assert m.seeds.shape    == (n,)
        assert m.th_star.shape  == (n, NOF_TH0, DIM)
        assert m.d_star.shape   == (n, NOF_TH0)

    def test_nof_seeds_accept_with_pcg(self, inside_prior_method):
        """nof_seeds_accept must equal ceil(nof_seeds_total * pcg_to_keep)."""
        m = inside_prior_method
        pcg = 0.5
        expected = int(np.ceil(NOF_SEEDS * pcg))
        m.filter_solutions(pcg_to_keep=pcg)
        assert m.nof_seeds_accept == expected

    def test_accepted_have_lowest_distances(self, inside_prior_method):
        """Accepted seeds must have lower best-distances than rejected seeds."""
        m = inside_prior_method
        m.filter_solutions(pcg_to_keep=0.5)
        all_best      = np.sort(m.d_star_inside_prior.min(axis=1))
        accepted_best = m.d_star.min(axis=1)
        rejected_best = all_best[m.nof_seeds_accept:]
        if len(rejected_best) > 0:
            assert accepted_best.max() <= rejected_best.min() + 1e-6

    def test_eps_1_is_set(self, inside_prior_method):
        """eps_1 must be stored and non-negative after filtering."""
        m = inside_prior_method
        m.filter_solutions(pcg_to_keep=0.5)
        assert m.eps_1 is not None
        assert m.eps_1 >= 0

    def test_eps_1_equals_worst_accepted_distance(self, inside_prior_method):
        """eps_1 must equal the best distance of the worst accepted seed."""
        m = inside_prior_method
        m.filter_solutions(pcg_to_keep=0.5)
        assert np.isclose(m.eps_1, m.d_star.min(axis=1).max(), atol=1e-6)

    def test_eps_1_mode(self, inside_prior_method):
        """With eps_1 provided, all accepted best-distances must be <= eps_1."""
        m = inside_prior_method
        eps_1 = float(np.median(m.d_star_inside_prior.min(axis=1)))
        m.filter_solutions(pcg_to_keep=None, eps_1=eps_1)
        accepted_best = m.d_star.min(axis=1)
        assert np.all(accepted_best <= eps_1 + 1e-6)

    def test_seeds_are_subset_of_inside_prior(self, inside_prior_method):
        """Accepted seeds must be a subset of seeds_inside_prior."""
        m = inside_prior_method
        m.filter_solutions(pcg_to_keep=0.5)
        assert set(m.seeds.tolist()) <= set(m.seeds_inside_prior.tolist())

    def test_too_high_pcg_raises_valueerror(self, optimized_method):
        """pcg_to_keep requiring more seeds than inside prior must raise ValueError."""
        m = optimized_method
        # force most seeds outside the prior so nof_seeds_inside_prior < nof_seeds_total
        m.th_star_init[2:] = 999.0
        m.filter_solutions_inside_prior()
        # nof_seeds_inside_prior <= 2, but pcg_to_keep=1.0 wants NOF_SEEDS
        with pytest.raises(ValueError, match="pcg_to_keep"):
            m.filter_solutions(pcg_to_keep=1.0)

    def test_valueerror_message_hints_at_fix(self, optimized_method):
        """ValueError message must mention both pcg_to_keep and nof_seeds_total."""
        m = optimized_method
        m.th_star_init[2:] = 999.0
        m.filter_solutions_inside_prior()
        with pytest.raises(ValueError) as exc_info:
            m.filter_solutions(pcg_to_keep=1.0)
        msg = str(exc_info.value)
        assert "pcg_to_keep" in msg
        assert "nof_seeds_total" in msg
