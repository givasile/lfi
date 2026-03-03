"""Tests for R2OMC Step 6: _get_distances, _process_limit, get_boxes, get_boxes_blind."""
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
NOF_LS_STEPS = 5
STEP_SIZE = 0.1


# ── helpers / fixtures ────────────────────────────────────────────────────────

def make_method():
    prior = UniformPrior(dim=DIM, low=-3, high=3)
    sim   = GaussianNoise(dim=DIM, dim_y=DIM, sigma_noise=0.1)
    obs   = np.zeros((1, DIM), dtype=np.float32)
    return R2OMC(prior=prior, simulator=sim, observation=obs)


@pytest.fixture
def directions_method():
    """R2OMC after all steps up to and including get_directions."""
    m = make_method()
    m.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
    m.optimize(nof_gd_steps=50, alpha=0.1)
    m.filter_solutions_inside_prior()
    m.filter_solutions(pcg_to_keep=PCG_TO_KEEP)
    m.get_directions(method="blind")  # blind for speed; shapes independent of method
    return m


@pytest.fixture
def boxes_method(directions_method):
    """R2OMC after get_boxes."""
    directions_method.get_boxes(eps_2=1.0, nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
    return directions_method


# ── _get_distances ────────────────────────────────────────────────────────────

class TestGetDistances:

    def test_output_shape(self, directions_method):
        """Output must have shape (S, T, D, 2*nof_ls_steps+1)."""
        m = directions_method
        dist = m._get_distances(nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
        S = m.nof_seeds_accept
        assert dist.shape == (S, NOF_TH0, DIM, 2 * NOF_LS_STEPS + 1)

    def test_distances_are_nonneg(self, directions_method):
        """All distances must be non-negative."""
        dist = directions_method._get_distances(nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
        assert np.all(dist >= 0)

    def test_center_has_lowest_distance(self, directions_method):
        """Distance at center (index nof_ls_steps) must be <= distances at other steps."""
        m = directions_method
        dist = m._get_distances(nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
        center_idx = NOF_LS_STEPS
        center = dist[..., center_idx]          # (S, T, D)
        assert np.all(center <= dist.max(axis=-1) + 1e-6)

    def test_more_steps_same_center(self, directions_method):
        """Center distance must be the same regardless of nof_ls_steps."""
        m = directions_method
        d1 = m._get_distances(nof_ls_steps=2, step_size=STEP_SIZE)
        d2 = m._get_distances(nof_ls_steps=5, step_size=STEP_SIZE)
        assert np.allclose(d1[..., 2], d2[..., 5], atol=1e-5)


# ── get_boxes ─────────────────────────────────────────────────────────────────

class TestGetBoxes:

    def test_limits_shape(self, boxes_method):
        """limits must have shape (S, T, D, 2)."""
        m = boxes_method
        assert m.limits.shape == (m.nof_seeds_accept, NOF_TH0, DIM, 2)

    def test_log_volumes_shape(self, boxes_method):
        """log_volumes must have shape (S, T)."""
        m = boxes_method
        assert m.log_volumes.shape == (m.nof_seeds_accept, NOF_TH0)

    def test_lower_limit_is_nonpositive(self, boxes_method):
        """Lower box limits must be <= 0 (boxes are centered at theta*)."""
        assert np.all(boxes_method.limits[..., 0] <= 0)

    def test_upper_limit_is_nonneg(self, boxes_method):
        """Upper box limits must be >= 0."""
        assert np.all(boxes_method.limits[..., 1] >= 0)

    def test_limits_are_positive_width(self, boxes_method):
        """Upper limit must be strictly greater than lower limit."""
        m = boxes_method
        assert np.all(m.limits[..., 1] > m.limits[..., 0])

    def test_log_volumes_are_finite(self, boxes_method):
        """log_volumes must be finite (no zero-width boxes)."""
        assert np.all(np.isfinite(boxes_method.log_volumes))

    def test_larger_eps2_gives_larger_boxes(self, directions_method):
        """A larger eps_2 must produce boxes at least as large."""
        m1, m2 = directions_method, make_method()
        # reuse same state for m2
        m2.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
        m2.optimize(nof_gd_steps=50, alpha=0.1)
        m2.filter_solutions_inside_prior()
        m2.filter_solutions(pcg_to_keep=PCG_TO_KEEP)
        m2.get_directions(method="blind")

        m1.get_boxes(eps_2=0.5, nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
        m2.get_boxes(eps_2=2.0, nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
        widths1 = m1.limits[..., 1] - m1.limits[..., 0]
        widths2 = m2.limits[..., 1] - m2.limits[..., 0]
        assert np.all(widths2 >= widths1 - 1e-8)

    def test_stored_eps2(self, directions_method):
        """eps_2 must be stored on the object."""
        directions_method.get_boxes(eps_2=1.0, nof_ls_steps=NOF_LS_STEPS, step_size=STEP_SIZE)
        assert directions_method.eps_2 == 1.0


# ── get_boxes_blind ───────────────────────────────────────────────────────────

class TestGetBoxesBlind:

    def test_limits_shape(self, directions_method):
        """limits must have shape (S, T, D, 2)."""
        m = directions_method
        m.get_boxes_blind(dx=0.2)
        assert m.limits.shape == (m.nof_seeds_accept, NOF_TH0, DIM, 2)

    def test_limits_are_symmetric(self, directions_method):
        """Blind boxes must be symmetric: lower = -dx, upper = +dx."""
        m = directions_method
        dx = 0.2
        m.get_boxes_blind(dx=dx)
        assert np.allclose(m.limits[..., 0], -dx)
        assert np.allclose(m.limits[..., 1],  dx)

    def test_log_volumes_correct(self, directions_method):
        """log_volumes must equal D * log(2*dx) for each (seed, th0)."""
        m = directions_method
        dx = 0.3
        m.get_boxes_blind(dx=dx)
        expected = DIM * np.log(2 * dx)
        assert np.allclose(m.log_volumes, expected, atol=1e-6)
