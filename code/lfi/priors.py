import typing
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import sbi
import sbi.utils
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from sklearn.neighbors import KernelDensity


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


class ImageDatasetPrior(BasePrior):
    def __init__(self,
                 dataset_name: str = "mnist",
                 split: str = "train",
                 nof_samples: int = 100,
                 bandwidth: float = 0.5
                 ):
        """
        Loads and prepares the dataset.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.bandwidth = bandwidth

        self.kde = None

        ds = tfds.load(dataset_name, split=split, as_supervised=True, batch_size=-1)
        data = tfds.as_numpy(ds)
        images, _ = data  # Ignore labels

        images = np.array(images)[:nof_samples]

        # Normalize to [0,1] and flatten
        self.images = jnp.array(images).astype(jnp.float32) / 255.0
        self.images = self.images.reshape(self.images.shape[0], -1)  # (N_total, D)
        super().__init__(f"image_dataset_{dataset_name}_{split}", self.images.shape[1])

    def sample_jax(self, key: jax.Array, N: int) -> jax.Array:
        total = self.images.shape[0]
        key, subkey = jax.random.split(key)
        replace = False if N <= total else True
        indices = jax.random.choice(subkey, total, shape=(N,), replace=replace)
        return self.images[indices]

    def sample_numpy(self, N: int) -> np.ndarray:
        total = self.images.shape[0]
        replace = False if N <= total else True
        indices = np.random.choice(total, size=N, replace=replace)
        return np.array(self.images[indices])

    def sample_pytorch(self, N):
        im = self.sample_numpy(N)
        return torch.tensor(im, dtype=torch.float32)

    def return_sbi_object(self):
        logpdf = self.logpdf
        sample_pytorch = self.sample_pytorch
        dim = self.dim

        class SBIImageDatasetPrior:
            def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
                theta = theta.numpy()
                logprob = logpdf(theta)
                return torch.tensor(logprob, dtype=torch.float32)

            def sample(self, sample_shape: typing.Optional[torch.Size]):
                if sample_shape is None:
                    N = 1
                    y = sample_pytorch(N)
                    return y.squeeze()
                else:
                    N = int(np.prod(sample_shape))
                    y = sample_pytorch(N)
                    new_shape = list(sample_shape) + [dim]
                    return y.reshape(new_shape)
        return SBIImageDatasetPrior()

    def logpdf(self, theta) -> float:
        if self.kde is None:
            self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            self.kde.fit(np.array(self.images))  # scikit-learn expects numpy
        return self.kde.score_samples(np.array(theta))

    def visualize_i(self, i: int):
        samples = self.images[i]
        plt.figure(figsize=(5, 5))
        plt.imshow(samples.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

    def visualize_samples(self, sample: jax.Array):
        plt.figure(figsize=(5, 5))
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

