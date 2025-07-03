import matplotlib.pyplot as plt
import numpy as np
import torch
import sbi
import sbi.utils
import jax
from jax import random
import jax.numpy as jnp
import typing
import elfi
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
                 logpdf_method: str = "kde",
                 bandwidth: float = 0.1
                 ):
        """
        Loads and prepares the dataset.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.logpdf_method = logpdf_method
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
        """
        Samples N images from the dataset.

        Args:
            N: Number of samples.
            key: JAX PRNG key.

        Returns:
            Array of shape (N, D) with clean image samples.
        """
        total = self.images.shape[0]
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(key, total, shape=(N,), replace=False)
        return self.images[indices]

    def sample_numpy(self, N: int) -> np.ndarray:
        total = self.images.shape[0]
        indices = np.random.choice(total, size=N, replace=False)
        return np.array(self.images[indices])

    def logpdf(self, theta: jax.Array) -> float:
        """
        Compute log-density at a single point θ of shape (D,).
        Returns: float
        """
        if self.logpdf_method == "kde":
            if self.kde is None:
                self._fit_kde()
            return self.kde.score_samples(np.array(theta))
        elif self.logpdf_method == "gaussian":
            diff = theta - self.mean
            mahalanobis = diff @ self.precision @ diff
            norm_const = -0.5 * (self.D * jnp.log(2 * jnp.pi) + self.log_det_cov)
            return norm_const - 0.5 * mahalanobis
        elif self.logpdf_method == "uniform":
            if jnp.all((theta >= self.lower) & (theta <= self.upper)):
                return -self.D * jnp.log(1.0)  # uniform over [0,1]^D
            else:
                return -jnp.inf

    def _fit_kde(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(np.array(self.images))  # scikit-learn expects numpy

    def _fit_gaussian(self):
        data_np = np.array(self.images)
        self.mean = jnp.array(data_np.mean(axis=0))
        self.cov = jnp.array(np.cov(data_np.T))
        self.precision = jnp.linalg.inv(self.cov)
        self.log_det_cov = jnp.linalg.slogdet(self.cov)[1]

    def _set_uniform_bounds(self):
        self.lower = jnp.zeros(self.D)
        self.upper = jnp.ones(self.D)

    def visualize_i(self, i: int):
        """
        Visualizes N samples from the dataset.

        Args:
            key: JAX PRNG key.
            N: Number of samples to visualize.
        """
        samples = self.images[i]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(samples.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

    def visualize_samples(self, sample: jax.Array):
        """
        Visualizes a grid of samples.

        Args:
            samples: Array of shape (D, )
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

