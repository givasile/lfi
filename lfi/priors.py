import numpy as np
import torch
import sbi
import sbi.utils
import jax
from jax import random
import jax.numpy as jnp
import typing
import elfi


class BasePrior:
    def __init__(self, name: str, dim:int, **kwargs):
        self.name = name
        self.dim = dim

    def sample_numpy(self, N):
        raise NotImplementedError
    
    def sample_jax(self, N, keys):
        raise NotImplementedError
    
    def sample_pytorch(self, N):
        raise NotImplementedError
    
    def return_sbi_object(self):
        raise NotImplementedError
    
    def return_elfi_objects(self):
        raise NotImplementedError
    
    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def has_mass(self, x):
        return self.logpdf(x) > -jnp.inf


class UniformPrior(BasePrior):
    def __init__(self, low, high, dim):
        self.low = low
        self.high = high
        self.dim = dim

        self.volume = np.prod(self.high - self.low)
        self.log_volume = np.sum(np.log(self.high - self.low))
        super().__init__("uniform", dim)

    def sample_numpy(self, N):
        return np.random.uniform(self.low, self.high, size = (N, self.dim)).astype(np.float32)
    
    def sample_jax(self, key, shape):
        return jax.random.uniform(key, (*shape, self.dim), minval=self.low, maxval=self.high)

    def sample_pytorch(self, N):
        return torch.rand(N, self.dim)*(self.high-self.low) + self.low
    
    def return_sbi_object(self):
        return sbi.utils.BoxUniform(low=self.low*torch.ones(self.dim), high=self.high*torch.ones(self.dim))
    
    def return_elfi_objects(self):
        return [elfi.Prior("uniform", self.low, self.high - self.low) for _ in range(self.dim)]

    def logpdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, -self.log_volume, -jnp.inf)

    def pdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, 1/self.volume, 0)



class NormalPrior(BasePrior):
    def __init__(self, mean, std, dim):
        self.mean = mean
        self.std = std
        self.dim = dim
        super().__init__("normal", dim)

    def sample_numpy(self, N):
        return np.random.normal(self.mean, self.std, size = (N, self.dim)).astype(np.float32)
    
    def sample_jax(self, N, keys):
        def sample_one(key):
            return random.normal(key, shape=(self.dim,), dtype=jnp.float32)*self.std + self.mean
        return jax.vmap(sample_one)(keys)
    
    def sample_pytorch(self, N):
        return torch.normal(self.mean*torch.ones(N, self.dim), self.std*torch.ones(N, self.dim))
    
    def return_elfi_objects(self):
        return [elfi.Prior("normal", self.mean, self.std) for _ in range(self.dim)]
