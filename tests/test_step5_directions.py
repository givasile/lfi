"""Tests for R2OMC Step 5: get_directions."""
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
PCG_TO_KEEP = 0.5


# ── helpers / fixtures ────────────────────────────────────────────────────────

def make_method():
    prior = UniformPrior(dim=DIM, low=-3, high=3)
    sim   = GaussianNoise(dim=DIM, dim_y=DIM, sigma_noise=0.1)
    obs   = np.zeros((1, DIM), dtype=np.float32)
    return R2OMC(prior=prior, simulator=sim, observation=obs)


@pytest.fixture
def filtered_method():
    """R2OMC after sample_objective_functions + optimize + both filter steps."""
    m = make_method()
    m.sample_objective_functions(KEY, nof_seeds_total=NOF_SEEDS, nof_th0=NOF_TH0)
    m.optimize(nof_gd_steps=50, alpha=0.1)
    m.filter_solutions_inside_prior()
    m.filter_solutions(pcg_to_keep=PCG_TO_KEEP)
    return m


# ── get_directions ────────────────────────────────────────────────────────────

class TestGetDirectionsOutputShapes:

    @pytest.mark.parametrize("method", ["hessian", "jacobian", "blind"])
    def test_output_shapes(self, filtered_method, method):
        """hessians, eig_val, eig_vec must have expected shapes for all methods."""
        m = filtered_method
        m.get_directions(method=method)
        S, T, D = m.nof_seeds_accept, NOF_TH0, DIM
        assert m.hessians.shape == (S, T, D, D)
        assert m.eig_val.shape  == (S, T, D)
        assert m.eig_vec.shape  == (S, T, D, D)


class TestGetDirectionsBlind:

    def test_eig_vec_is_identity(self, filtered_method):
        """Blind mode must set eigenvectors to the identity matrix."""
        m = filtered_method
        m.get_directions(method="blind")
        for i in range(m.nof_seeds_accept):
            for j in range(NOF_TH0):
                assert np.allclose(m.eig_vec[i, j], np.eye(DIM))

    def test_eig_val_are_ones(self, filtered_method):
        """Blind mode must set all eigenvalues to 1."""
        m = filtered_method
        m.get_directions(method="blind")
        assert np.allclose(m.eig_val, 1.0)


class TestGetDirectionsHessian:

    def test_eigenvalues_are_real(self, filtered_method):
        """eigh must return real eigenvalues (no imaginary part)."""
        m = filtered_method
        m.get_directions(method="hessian")
        assert m.eig_val.dtype in (np.float32, np.float64)
        assert np.all(np.isfinite(m.eig_val))

    def test_eigenvectors_are_orthonormal(self, filtered_method):
        """Eigenvector matrices must be orthonormal: V @ V.T ≈ I."""
        m = filtered_method
        m.get_directions(method="hessian")
        for i in range(m.nof_seeds_accept):
            for j in range(NOF_TH0):
                V = m.eig_vec[i, j]
                assert np.allclose(V @ V.T, np.eye(DIM), atol=1e-5)

    def test_hessian_reconstructed_from_eigen(self, filtered_method):
        """H ≈ V @ diag(λ) @ V.T must hold for each (seed, th0)."""
        m = filtered_method
        m.get_directions(method="hessian")
        for i in range(m.nof_seeds_accept):
            for j in range(NOF_TH0):
                H = m.hessians[i, j]
                V = m.eig_vec[i, j]
                lam = m.eig_val[i, j]
                H_reconstructed = V @ np.diag(lam) @ V.T
                assert np.allclose(H, H_reconstructed, atol=1e-4)


class TestGetDirectionsJacobian:

    def test_eigenvalues_are_real(self, filtered_method):
        """eigh on the outer-product approximation must return real eigenvalues."""
        m = filtered_method
        m.get_directions(method="jacobian")
        assert m.eig_val.dtype in (np.float32, np.float64)
        assert np.all(np.isfinite(m.eig_val))

    def test_eigenvectors_are_orthonormal(self, filtered_method):
        """Eigenvector matrices must be orthonormal: V @ V.T ≈ I."""
        m = filtered_method
        m.get_directions(method="jacobian")
        for i in range(m.nof_seeds_accept):
            for j in range(NOF_TH0):
                V = m.eig_vec[i, j]
                assert np.allclose(V @ V.T, np.eye(DIM), atol=1e-5)

    def test_hessian_is_psd(self, filtered_method):
        """Outer-product approximation must be positive semi-definite."""
        m = filtered_method
        m.get_directions(method="jacobian")
        assert np.all(m.eig_val >= -1e-6)


class TestGetDirectionsInvalidMethod:

    def test_invalid_method_raises_valueerror(self, filtered_method):
        """An unknown method string must raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            filtered_method.get_directions(method="unknown")
