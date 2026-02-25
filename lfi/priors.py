from __future__ import annotations
import typing
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

try:
    import torch
    import sbi
    import sbi.utils
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import elfi
    _HAS_ELFI = True
except ImportError:
    _HAS_ELFI = False

_TORCH_MSG = "torch/sbi not installed. Install with: pip install 'lfi[torch-cpu]' or 'lfi[torch-gpu]'"
_ELFI_MSG = "elfi not installed. Install with: pip install 'lfi[elfi]'"


class BasePrior:
    def __init__(self, name: str, dim:int, **kwargs):
        self.name = name
        self.dim = dim

    def sample_numpy(self, N):
        raise NotImplementedError
    
    def sample_jax(self, key, N):
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

        self.log_volume = np.log(high - low) * dim
        super().__init__("uniform", dim)

    def sample_numpy(self, N):
        return np.random.uniform(self.low, self.high, size = (N, self.dim)).astype(np.float32)
    
    def sample_jax(self, key, N):
        return jax.random.uniform(key, (N, self.dim), minval=self.low, maxval=self.high)

    def sample_pytorch(self, N):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return torch.rand(N, self.dim)*(self.high-self.low) + self.low

    def return_sbi_object(self):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return sbi.utils.BoxUniform(low=self.low*torch.ones(self.dim), high=self.high*torch.ones(self.dim))
    
    def return_elfi_objects(self):
        if not _HAS_ELFI:
            raise ImportError(_ELFI_MSG)
        return [elfi.Prior("uniform", self.low, self.high - self.low) for _ in range(self.dim)]

    def logpdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, -self.log_volume, -jnp.inf)

    def pdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, 1/self.volume, 0)


class Normal(BasePrior):
    def __init__(self, dim, mean, std):
        self.mean = mean
        self.std = std
        super().__init__("normal", dim)

    def sample_numpy(self, N):
        return np.random.normal(loc=self.mean, scale=self.std, size=(N, self.dim))

    def sample_jax(self, key, N):
        keys = jax.random.split(key, self.dim)
        z = jnp.stack([
            jax.random.normal(keys[i], shape=(N,)) * self.std[i] + self.mean[i]
            for i in range(self.dim)
        ], axis=-1)
        return z

    def sample_pytorch(self, N):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return torch.normal(mean=torch.tensor(self.mean), std=torch.tensor(self.std)).repeat(N, 1)

    def logpdf(self, x):
        # x: [..., 4]
        logpdf = -0.5 * np.sum(((x - self.mean) / self.std) ** 2, axis=-1) - np.sum(np.log(self.std)) - 0.5 * self.dim * np.log(2 * np.pi)
        return logpdf

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class LogNormal(BasePrior):
    def __init__(self, dim, mean, std):
        self.loc = mean
        self.scale = std
        super().__init__("lognormal", dim)

    def sample_numpy(self, N):
        z = np.random.normal(loc=self.loc, scale=self.scale, size=(N, self.dim))
        return np.exp(z)

    def sample_jax(self, key, N):
        keys = jax.random.split(key, self.dim)
        z = jnp.stack([
            jax.random.normal(keys[i], shape=(N,)) * self.scale[i] + self.loc[i]
            for i in range(self.dim)
        ], axis=-1)
        return jnp.exp(z)

    def sample_pytorch(self, N):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        dist = torch.distributions.LogNormal(
            loc=torch.tensor(self.loc), scale=torch.tensor(self.scale)
        )
        return dist.sample((N,))

    def logpdf(self, x):
        # x: [..., 4]
        # PDF of LogNormal: log(f(x)) = -log(x) + log Normal PDF of log(x)
        logx = np.log(x)
        norm_logpdf = -0.5 * (((logx - self.loc) / self.scale) ** 2) - np.log(self.scale) - 0.5 * np.log(2 * np.pi)
        logpdf = np.sum(norm_logpdf - np.log(x), axis=-1)
        return logpdf

    def pdf(self, x):
        return np.exp(self.logpdf(x))





# class NormalPrior(BasePrior):
#     def __init__(self, mean, std, dim):
#         self.mean = mean
#         self.std = std
#         self.dim = dim
#         super().__init__("normal", dim)

#     def sample_numpy(self, N):
#         return np.random.normal(self.mean, self.std, size = (N, self.dim)).astype(np.float32)
    
#     def sample_jax(self, key, N):
#         def sample_one(key):
#             return random.normal(key, shape=(self.dim,), dtype=jnp.float32)*self.std + self.mean
#         return jax.vmap(sample_one)(keys)
    
#     def sample_pytorch(self, N):
#         return torch.normal(self.mean*torch.ones(N, self.dim), self.std*torch.ones(N, self.dim))
    
#     def return_elfi_objects(self):
#         return [elfi.Prior("normal", self.mean, self.std) for _ in range(self.dim)]


