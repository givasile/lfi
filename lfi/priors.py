from __future__ import annotations
from typing import Union

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
    def __init__(self, name: str, dim: int, **kwargs):
        self.name = name
        self.dim = dim

    def sample_numpy(self, N: int):
        raise NotImplementedError

    def sample_jax(self, key: jax.random.PRNGKey, N: int):
        raise NotImplementedError

    def sample_pytorch(self, N: int):
        raise NotImplementedError

    def return_sbi_object(self):
        raise NotImplementedError

    def return_elfi_objects(self):
        raise NotImplementedError

    def logpdf(self, x: Union[np.ndarray, jnp.ndarray]):
        raise NotImplementedError

    def pdf(self, x: Union[np.ndarray, jnp.ndarray]):
        raise NotImplementedError

    def has_mass(self, x):
        return self.logpdf(x) > -np.inf


class UniformPrior(BasePrior):
    def __init__(self, dim, low, high):
        self.low = low
        self.high = high
        self.log_volume = np.log(high - low) * dim
        super().__init__("uniform", dim)

    def sample_numpy(self, N):
        return np.random.uniform(self.low, self.high, size=(N, self.dim)).astype(np.float32)

    def sample_jax(self, key, N):
        return jax.random.uniform(key, (N, self.dim), minval=self.low, maxval=self.high)

    def sample_pytorch(self, N):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return torch.rand(N, self.dim) * (self.high - self.low) + self.low

    def return_sbi_object(self):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return sbi.utils.BoxUniform(
            low=self.low * torch.ones(self.dim),
            high=self.high * torch.ones(self.dim)
        )

    def return_elfi_objects(self):
        if not _HAS_ELFI:
            raise ImportError(_ELFI_MSG)
        return [elfi.Prior("uniform", self.low, self.high - self.low) for _ in range(self.dim)]

    def logpdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, -self.log_volume, -np.inf)

    def pdf(self, x):
        marginal_inside = np.logical_and(x >= self.low, x <= self.high)
        inside = np.all(marginal_inside, axis=-1)
        return np.where(inside, np.exp(-self.log_volume), 0.0)


class Normal(BasePrior):
    def __init__(self, dim, mean, std):
        self.mean = np.broadcast_to(np.atleast_1d(np.array(mean, dtype=float)), (dim,)).copy()
        self.std = np.broadcast_to(np.atleast_1d(np.array(std, dtype=float)), (dim,)).copy()
        super().__init__("normal", dim)

    def sample_numpy(self, N):
        return np.random.normal(loc=self.mean, scale=self.std, size=(N, self.dim)).astype(np.float32)

    def sample_jax(self, key, N):
        return jax.random.normal(key, shape=(N, self.dim)) * self.std + self.mean

    def sample_pytorch(self, N):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        dist = torch.distributions.Normal(
            loc=torch.tensor(self.mean, dtype=torch.float32),
            scale=torch.tensor(self.std, dtype=torch.float32)
        )
        return dist.sample((N,))

    def return_sbi_object(self):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return sbi.utils.MultipleIndependent(
            [
                torch.distributions.Normal(
                    loc=torch.tensor([self.mean[i]], dtype=torch.float32),
                    scale=torch.tensor([self.std[i]], dtype=torch.float32)
                )
                for i in range(self.dim)
            ]
        )
    
    def return_elfi_objects(self):
        if not _HAS_ELFI:
            raise ImportError(_ELFI_MSG)
        return [elfi.Prior("norm", self.mean[i], self.std[i]) for i in range(self.dim)]

    def logpdf(self, x):
        logpdf = (
            -0.5 * np.sum(((x - self.mean) / self.std) ** 2, axis=-1)
            - np.sum(np.log(self.std))
            - 0.5 * self.dim * np.log(2 * np.pi)
        )
        return logpdf

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class LogNormal(BasePrior):
    def __init__(self, dim, mean, std):
        self.loc = np.broadcast_to(np.atleast_1d(np.array(mean, dtype=float)), (dim,)).copy()
        self.scale = np.broadcast_to(np.atleast_1d(np.array(std, dtype=float)), (dim,)).copy()
        super().__init__("lognormal", dim)

    def sample_numpy(self, N):
        z = np.random.normal(loc=self.loc, scale=self.scale, size=(N, self.dim))
        return np.exp(z).astype(np.float32)

    def sample_jax(self, key, N):
        z = jax.random.normal(key, shape=(N, self.dim)) * self.scale + self.loc
        return jnp.exp(z)

    def sample_pytorch(self, N):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        dist = torch.distributions.LogNormal(
            loc=torch.tensor(self.loc, dtype=torch.float32),
            scale=torch.tensor(self.scale, dtype=torch.float32)
        )
        return dist.sample((N,))

    def return_sbi_object(self):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return sbi.utils.MultipleIndependent(
            [
                torch.distributions.LogNormal(
                    loc=torch.tensor([self.loc[i]], dtype=torch.float32),
                    scale=torch.tensor([self.scale[i]], dtype=torch.float32)
                )
                for i in range(self.dim)
            ]
        )


    def return_elfi_objects(self):
        if not _HAS_ELFI:
            raise ImportError(_ELFI_MSG)
        # scipy.stats.lognorm positional args: lognorm(s, loc, scale)
        # where s = std of underlying normal, loc = 0 (shift), scale = exp(mean of underlying normal)
        return [
            elfi.Prior("lognorm", self.scale[i], 0, np.exp(self.loc[i]))
            for i in range(self.dim)
        ]

    def logpdf(self, x):
        logx = np.log(x)
        norm_logpdf = (
            -0.5 * ((logx - self.loc) / self.scale) ** 2
            - np.log(self.scale)
            - 0.5 * np.log(2 * np.pi)
        )
        return np.sum(norm_logpdf - np.log(x), axis=-1)

    def pdf(self, x):
        return np.exp(self.logpdf(x))
