import numpy as np
import jax.numpy as jnp
import jax
import scipy.stats as ss


class BasePrior:
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def __repr__(self):
        return self.name

    def sample(self, key, shape):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def has_mass(self, x):
        return self.logpdf(x) > -jnp.inf


class Uniform(BasePrior):
    def __init__(self, low, high, dim):
        self.low = np.ones(dim) * low if isinstance(low, (int, float)) else low
        self.high = np.ones(dim) * high if isinstance(high, (int, float)) else high
        self.volume = np.prod(self.high - self.low)
        self.log_volume = np.sum(np.log(self.high - self.low))
        super().__init__(f'Uniform({low}, {high})', dim)

    def sample(self, key, shape):
        return jax.random.uniform(key, (*shape, self.dim), minval=self.low, maxval=self.high)

    def logpdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, -self.log_volume, -jnp.inf)

    def pdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, 1/self.volume, 0)


class Gaussian(BasePrior):
    def __init__(self, mean, std, dim):
        self.m_normal = ss.multivariate_normal(mean=mean, cov=np.eye(dim) * std**2)
        self.mean = mean
        self.std = std
        super().__init__(f'Gaussian({mean},{std})', dim)

    def sample(self, key, shape):
        return jax.random.multivariate_normal(key, self.mean, np.eye(self.dim) * self.std**2, shape)

    def logpdf(self, x):
        return self.m_normal.logpdf(x)

    def pdf(self, x):
        return self.m_normal.pdf(x)
