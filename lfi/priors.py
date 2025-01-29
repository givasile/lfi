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
    def __init__(self, name: str, dim: int, **kwargs):
        self.name = name
        self.dim = dim

    def sample_numpy(self, nof_samples: int):
        raise NotImplementedError

    def sample_pytorch(self, nof_samples: int):
        raise NotImplementedError

    def sample_jax(self, nof_samples: int, keys: typing.List):
        raise NotImplementedError

    def return_sbi_object(self):
        raise NotImplementedError

    def return_elfi_objects(self):
        raise NotImplementedError


class UniformPrior(BasePrior):
    def __init__(self, dim, low, high):
        self.dim = dim
        self.low = low
        self.high = high
        super().__init__("uniform", dim)

    def sample_numpy(self, nof_samples: int) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=(nof_samples, self.dim)).astype(np.float32)

    def sample_pytorch(self, nof_samples: int) -> torch.Tensor:
        return torch.rand(nof_samples, self.dim) * (self.high - self.low) + self.low

    def sample_jax(self, nof_samples: int, keys: typing.List) -> jnp.ndarray:
        def sample_one(key):
            return random.uniform(key, shape=(self.dim,), minval=self.low, maxval=self.high, dtype=jnp.float32)
        return jax.vmap(sample_one)(keys)

    def return_sbi_object(self) -> sbi.utils.BoxUniform:
        return sbi.utils.BoxUniform(low=self.low*torch.ones(self.dim), high=self.high*torch.ones(self.dim))

    def return_elfi_objects(self):
        return [elfi.Prior("uniform", self.low, self.high) for _ in range(self.dim)]

