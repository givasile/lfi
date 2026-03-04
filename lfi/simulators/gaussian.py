from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import scipy.stats as ss

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_TORCH_MSG = "torch not installed. Install with: pip install 'lfi[torch-cpu]' or 'lfi[torch-gpu]'"

from .base import BaseSimulator


class GaussianNoise(BaseSimulator):
    def __init__(self, dim, dim_y, sigma_noise, shift=0):
        self.sigma_noise = sigma_noise
        self.shift = shift
        super().__init__("gaussian_noise", dim, dim_y)

    def sample_numpy(self, theta):
        return np.random.normal(theta + self.shift, self.sigma_noise)

    def sample_jax(self, theta, seed):
        # seed -> prng key for each sample
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        theta = jnp.asarray(theta)
        shift = jnp.asarray(self.shift)
        y = theta + shift + jax.random.normal(subkey, shape=theta.shape)*self.sigma_noise
        return y

    def sample_pytorch(self, theta):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        return theta + self.shift + torch.randn_like(theta)*self.sigma_noise

    def return_elfi_callable(self):
        def elfi_simulator(*th_params, batch_size=1, random_state=None):
            theta = np.stack(th_params, axis=1)  # (batch_size, dim)
            samples = ss.norm.rvs(
                loc=theta + self.shift,
                scale=self.sigma_noise,
                size=(batch_size, self.dim_y),
                random_state=random_state
            )
            return samples
        return elfi_simulator


class GaussianNoiseDistractors(BaseSimulator):
    def __init__(self, dim, dim_y, dim_distractors, sigma_noise=0.1, shift=0):
        self.sigma_noise = sigma_noise
        self.shift = shift
        self.dim_distractors = dim_distractors
        super().__init__("gaussian_noise_distractor", dim, dim_y)

    def sample_numpy(self, theta):
        yy = np.random.normal(theta + self.shift, self.sigma_noise)

        # add dim_distractor samples from a uniform distribution
        yy_distractors = np.random.uniform(
            low=-3,
            high=3,
            size=(theta.shape[0], self.dim_distractors)
        )
        # concatenate informative and distractor parts
        y = np.concatenate([yy, yy_distractors], axis=-1)
        return y

    def sample_jax(self, theta, seed):
        # seed -> prng key for each sample
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        yy = jax.random.multivariate_normal(subkey, theta + self.shift, jnp.eye(self.dim) * self.sigma_noise**2)

        # add dim_distractor samples from a uniform distribution
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        yy_distractors = jax.random.uniform(
            subkey,
            shape=(self.dim_distractors,),
            minval=-3,
            maxval=3
        )
        y = jnp.concatenate([yy, yy_distractors], axis=-1)
        return y

    def sample_pytorch(self, theta):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        yy = theta + self.shift + torch.randn_like(theta) * self.sigma_noise

        # add dim_distractor samples from a uniform distribution
        yy_distractors = torch.rand((theta.shape[0], self.dim_distractors)) * 6 - 3 # Uniform distribution in [-3, 3]

        # concatenate informative and distractor parts
        y = torch.cat([yy, yy_distractors], dim=-1)
        return y


class BimodalGaussian(BaseSimulator):
    def __init__(self, dim, dim_y, sigma_noise=0.1, shift=3):
        self.sigma_noise = sigma_noise
        self.shift = shift
        super().__init__("bimodal_gaussian", dim, dim_y)

    def sample_numpy(self, theta):
        mode = np.random.choice([0, 1], size=theta.shape[0])
        mean = theta + self.shift
        mean[mode == 1] = theta[mode == 1] - self.shift
        return np.random.normal(mean, self.sigma_noise)

    def sample_jax(self, theta, seed):
        # seed -> prng key for each sample
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        mode = jax.random.randint(subkey, shape=(1,), minval=0, maxval=2)
        mean = jnp.where(mode == 0, theta - self.shift, theta + self.shift)
        key, subkey = jax.random.split(key)
        yy = jax.random.multivariate_normal(subkey, mean, jnp.eye(self.dim_y) * self.sigma_noise ** 2)
        return yy

    def sample_pytorch(self, theta):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        mode = torch.randint(0, 2, (theta.shape[0],))
        mean = theta + self.shift
        mean[mode == 1] = theta[mode == 1] - self.shift
        return mean + torch.randn_like(theta) * self.sigma_noise

    def return_elfi_callable(self):
        def elfi_simulator(*th_params, batch_size=1, random_state=None):
            theta = np.stack(th_params, axis=1)
            mode = np.random.randint(0, 2, size=(batch_size,))
            mean = theta + self.shift
            mean[mode == 1] = theta[mode == 1] - self.shift
            samples_standard_normal = ss.norm.rvs(size=(batch_size, self.dim_y), random_state=random_state)
            samples = mean + self.sigma_noise * samples_standard_normal
            return samples
        return elfi_simulator


class BimodalGaussianDistractors(BaseSimulator):
    def __init__(self, dim, dim_y, dim_distractors, sigma_noise=0.1, shift=3):
        self.dim_distractors = dim_distractors
        self.sigma_noise = sigma_noise
        self.shift = shift
        super().__init__("bimodal_gaussian", dim, dim_y)

    def sample_numpy(self, theta):
        mode = np.random.choice([0, 1], size=theta.shape[0])
        mean = theta + self.shift
        mean[mode == 1] = theta[mode == 1] - self.shift
        yy = np.random.normal(mean, self.sigma_noise)

        # add dim_distractor samples from a uniform distribution
        yy_distractors = np.random.uniform(
            low=-3,
            high=3,
            size=(theta.shape[0], self.dim_distractors)
        )
        # concatenate informative and distractor parts
        y = np.concatenate([yy, yy_distractors], axis=-1)
        return y

    def sample_jax(self, theta, seed):
        # seed -> prng key for each sample
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        mode = jax.random.randint(subkey, shape=(1,), minval=0, maxval=2)
        mean = jnp.where(mode == 0, theta - self.shift, theta + self.shift)
        key, subkey = jax.random.split(key)
        yy = jax.random.multivariate_normal(subkey, mean, jnp.eye(self.dim) * self.sigma_noise**2)

        # add dim_distractor samples from a uniform distribution
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        yy_distractors = jax.random.uniform(
            subkey,
            shape=(self.dim_distractors,),
            minval=-3,
            maxval=3
        )
        y = jnp.concatenate([yy, yy_distractors], axis=-1)
        return y

    def sample_pytorch(self, theta):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        mode = torch.randint(0, 2, (theta.shape[0],))
        mean = theta + self.shift
        mean[mode == 1] = theta[mode == 1] - self.shift
        yy = mean + torch.randn_like(theta) * self.sigma_noise

        # add dim_distractor samples from a uniform distribution
        yy_distractors = torch.rand((theta.shape[0], self.dim_distractors)) * 6 - 3 # Uniform distribution in [-3, 3]

        # concatenate informative and distractor parts
        y = torch.cat([yy, yy_distractors], dim=-1)
        return y


class ShiftedBimodalGaussian(BaseSimulator):
    """Asymmetric bimodal Gaussian simulator.

    y | theta ~ 0.5 * N(theta, sigma^2 I) + 0.5 * N(theta + shift, sigma^2 I)

    Sanity check for multi-observation inference:
        obs_1 = 0     ->  single-obs posterior modes at theta=0  and theta=-shift
        obs_2 = shift ->  single-obs posterior modes at theta=shift and theta=0
        cross-filter  ->  only theta=0 is consistent with both observations
    """
    def __init__(self, dim, dim_y, sigma_noise=0.1, shift=1.5):
        self.sigma_noise = sigma_noise
        self.shift = shift
        super().__init__("shifted_bimodal_gaussian", dim, dim_y)

    def sample_numpy(self, theta):
        mode = np.random.choice([0, 1], size=theta.shape[0])
        mean = theta.copy()
        mean[mode == 1] = theta[mode == 1] + self.shift
        return np.random.normal(mean, self.sigma_noise)

    def sample_jax(self, theta, seed):
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        mode = jax.random.randint(subkey, shape=(1,), minval=0, maxval=2)
        mean = jnp.where(mode == 0, theta, theta + self.shift)
        key, subkey = jax.random.split(key)
        return jax.random.multivariate_normal(
            subkey, mean, jnp.eye(self.dim_y) * self.sigma_noise ** 2
        )
