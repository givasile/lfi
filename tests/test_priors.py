import pytest
import numpy as np
import jax
import jax.numpy as jnp

import lfi
from lfi.priors import UniformPrior, Normal, LogNormal

KEY = jax.random.PRNGKey(42)
N_STAT = 10_000   # large N for statistical assertions
N_SMALL = 50      # small N for shape / dtype checks

# ── prior factory ─────────────────────────────────────────────────────────────

PRIORS = {
    "uniform":   lambda dim: UniformPrior(dim, -2.0, 2.0),
    "normal":    lambda dim: Normal(dim, 0.0, 1.0),
    "lognormal": lambda dim: LogNormal(dim, 0.0, 0.5),
}


@pytest.fixture(params=list(PRIORS.keys()))
def prior3(request):
    """Each prior instantiated with dim=3."""
    return PRIORS[request.param](3)


# ── 1. Shape ──────────────────────────────────────────────────────────────────

class TestShape:

    @pytest.mark.parametrize("prior_name", list(PRIORS.keys()))
    @pytest.mark.parametrize("dim", [1, 3, 10])
    @pytest.mark.parametrize("N", [1, 10, 100])
    def test_sample_numpy(self, prior_name, dim, N):
        s = PRIORS[prior_name](dim).sample_numpy(N)
        assert s.shape == (N, dim)

    @pytest.mark.parametrize("prior_name", list(PRIORS.keys()))
    @pytest.mark.parametrize("dim", [1, 3, 10])
    @pytest.mark.parametrize("N", [1, 10, 100])
    def test_sample_jax(self, prior_name, dim, N):
        s = PRIORS[prior_name](dim).sample_jax(KEY, N)
        assert s.shape == (N, dim)

    @pytest.mark.parametrize("prior_name", list(PRIORS.keys()))
    @pytest.mark.parametrize("dim", [1, 3, 10])
    @pytest.mark.parametrize("N", [1, 10, 100])
    def test_sample_pytorch(self, prior_name, dim, N):
        torch = pytest.importorskip("torch")
        s = PRIORS[prior_name](dim).sample_pytorch(N)
        assert s.shape == torch.Size([N, dim])


# ── 2. Dtype ──────────────────────────────────────────────────────────────────

class TestDtype:

    def test_numpy_returns_ndarray_float32(self, prior3):
        s = prior3.sample_numpy(N_SMALL)
        assert isinstance(s, np.ndarray)
        assert s.dtype == np.float32

    def test_jax_returns_jax_array_float32(self, prior3):
        s = prior3.sample_jax(KEY, N_SMALL)
        assert isinstance(s, jax.Array)
        assert s.dtype == jnp.float32

    def test_pytorch_returns_tensor_float32(self, prior3):
        torch = pytest.importorskip("torch")
        s = prior3.sample_pytorch(N_SMALL)
        assert isinstance(s, torch.Tensor)
        assert s.dtype == torch.float32


# ── 3. Empirical mean / std / min / max ───────────────────────────────────────

class TestEmpirical:

    def test_uniform_samples_within_bounds(self):
        p = UniformPrior(5, -3.0, 3.0)
        s = p.sample_numpy(N_STAT)
        assert np.all(s >= -3.0) and np.all(s <= 3.0)

    def test_uniform_empirical_mean(self):
        p = UniformPrior(3, -2.0, 2.0)
        s = p.sample_numpy(N_STAT)
        np.testing.assert_allclose(s.mean(axis=0), np.zeros(3), atol=0.05)

    def test_normal_empirical_mean(self):
        p = Normal(3, [1.0, -1.0, 0.0], [0.5, 1.0, 2.0])
        s = p.sample_numpy(N_STAT)
        np.testing.assert_allclose(s.mean(axis=0), [1.0, -1.0, 0.0], atol=0.05)

    def test_normal_empirical_std(self):
        p = Normal(3, [1.0, -1.0, 0.0], [0.5, 1.0, 2.0])
        s = p.sample_numpy(N_STAT)
        np.testing.assert_allclose(s.std(axis=0), [0.5, 1.0, 2.0], atol=0.05)

    def test_lognormal_all_positive(self):
        p = LogNormal(3, 0.0, 0.5)
        s = p.sample_numpy(N_STAT)
        assert np.all(s > 0)

    def test_lognormal_empirical_mean(self):
        loc, scale = 0.0, 0.5
        p = LogNormal(3, loc, scale)
        s = p.sample_numpy(N_STAT)
        expected = np.exp(loc + scale ** 2 / 2)
        np.testing.assert_allclose(s.mean(axis=0), np.full(3, expected), rtol=0.05)

    def test_lognormal_empirical_std(self):
        loc, scale = 0.0, 0.5
        p = LogNormal(3, loc, scale)
        s = p.sample_numpy(N_STAT)
        expected = np.sqrt((np.exp(scale ** 2) - 1) * np.exp(2 * loc + scale ** 2))
        np.testing.assert_allclose(s.std(axis=0), np.full(3, expected), rtol=0.05)


# ── 4. logpdf / pdf consistency ───────────────────────────────────────────────

class TestLogpdfPdfConsistency:

    @pytest.mark.parametrize("prior_name", list(PRIORS.keys()))
    def test_pdf_equals_exp_logpdf(self, prior_name):
        p = PRIORS[prior_name](3)
        x = p.sample_numpy(50)   # valid points drawn from the prior
        np.testing.assert_allclose(p.pdf(x), np.exp(p.logpdf(x)), rtol=1e-5)


# ── 5. logpdf stability at high D ─────────────────────────────────────────────

class TestHighDimStability:

    @pytest.mark.parametrize("D", [int(1e3), int(1e4), int(1e5), int(1e6)])
    def test_uniform_logpdf_finite(self, D):
        p = UniformPrior(D, -1.0, 1.0)
        x = np.zeros((1, D))
        assert np.isfinite(p.logpdf(x)).all()

    @pytest.mark.parametrize("D", [int(1e3), int(1e4), int(1e5), int(1e6)])
    def test_normal_logpdf_finite(self, D):
        p = Normal(D, 0.0, 1.0)
        x = np.zeros((1, D))
        assert np.isfinite(p.logpdf(x)).all()

    @pytest.mark.parametrize("D", [int(1e3), int(1e4), int(1e5), int(1e6)])
    def test_lognormal_logpdf_finite(self, D):
        p = LogNormal(D, 0.0, 0.5)
        x = np.ones((1, D))
        assert np.isfinite(p.logpdf(x)).all()


# ── 6. pdf / logpdf at known points ───────────────────────────────────────────

class TestKnownValues:

    def test_uniform_pdf_inside(self):
        # UniformPrior(dim=1, low=-2, high=2): pdf = 1/(2-(-2)) = 0.25
        p = UniformPrior(1, -2.0, 2.0)
        np.testing.assert_allclose(p.pdf(np.array([[0.0]])), [0.25], rtol=1e-6)

    def test_uniform_pdf_outside(self):
        p = UniformPrior(1, -2.0, 2.0)
        assert p.pdf(np.array([[5.0]]))[0] == 0.0

    def test_uniform_logpdf_inside(self):
        p = UniformPrior(1, -2.0, 2.0)
        np.testing.assert_allclose(p.logpdf(np.array([[0.0]])), [np.log(0.25)], rtol=1e-6)

    def test_uniform_logpdf_outside(self):
        p = UniformPrior(1, -2.0, 2.0)
        assert p.logpdf(np.array([[5.0]]))[0] == -np.inf

    def test_normal_pdf_at_mode(self):
        # Standard normal dim=1: pdf(0) = 1/sqrt(2*pi)
        p = Normal(1, 0.0, 1.0)
        np.testing.assert_allclose(
            p.pdf(np.array([[0.0]])),
            [1.0 / np.sqrt(2 * np.pi)],
            rtol=1e-6,
        )

    def test_normal_logpdf_at_mode(self):
        p = Normal(1, 0.0, 1.0)
        np.testing.assert_allclose(
            p.logpdf(np.array([[0.0]])),
            [-0.5 * np.log(2 * np.pi)],
            rtol=1e-6,
        )

    def test_lognormal_logpdf_at_one(self):
        # LogNormal(dim=1, loc=0, scale=1) at x=1:
        # logpdf = -log(x) - 0.5*log(2*pi) - (log(x)-loc)^2/(2*scale^2) - log(scale)
        #        = 0 - 0.5*log(2*pi) - 0 - 0 = -0.5*log(2*pi)
        p = LogNormal(1, 0.0, 1.0)
        np.testing.assert_allclose(
            p.logpdf(np.array([[1.0]])),
            [-0.5 * np.log(2 * np.pi)],
            rtol=1e-6,
        )


# ── 7. has_mass ───────────────────────────────────────────────────────────────

class TestHasMass:

    def test_uniform_inside(self):
        p = UniformPrior(2, -1.0, 1.0)
        assert bool(p.has_mass(np.array([[0.0, 0.5]])))

    def test_uniform_outside(self):
        p = UniformPrior(2, -1.0, 1.0)
        assert not bool(p.has_mass(np.array([[2.0, 0.0]])))

    def test_uniform_on_boundary(self):
        # boundary points are included (>= low and <= high)
        p = UniformPrior(1, -1.0, 1.0)
        assert bool(p.has_mass(np.array([[-1.0]])))
        assert bool(p.has_mass(np.array([[1.0]])))

    def test_normal_has_mass_everywhere(self):
        p = Normal(2, 0.0, 1.0)
        assert bool(p.has_mass(np.array([[0.0, 0.0]])))
        assert bool(p.has_mass(np.array([[100.0, -100.0]])))  # far from mean but finite

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_lognormal_no_mass_at_zero(self):
        p = LogNormal(1, 0.0, 1.0)
        assert not bool(p.has_mass(np.array([[0.0]])))

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_lognormal_no_mass_at_negative(self):
        p = LogNormal(1, 0.0, 1.0)
        assert not bool(p.has_mass(np.array([[-1.0]])))

    def test_lognormal_has_mass_at_positive(self):
        p = LogNormal(1, 0.0, 1.0)
        assert bool(p.has_mass(np.array([[1.0]])))
