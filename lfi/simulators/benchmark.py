from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_TORCH_MSG = "torch not installed. Install with: pip install 'lfi[torch-cpu]' or 'lfi[torch-gpu]'"

from .base import BaseSimulator


class TwoMoons(BaseSimulator):
    def __init__(self, dim, dim_y):
        super().__init__("TwoMoon", dim, dim_y)


    def sample_numpy(self, theta: np.ndarray):
        """Two-moons simulator. D = 2, D_y = 2.

        Args:
            theta(np.array): Array of shape (N, D)
        Returns:
            x(np.array): Array of shape (N, D_y) with simulated points.
        """

        # Step 1: Sample intermediate variables
        N = theta.shape[0]
        a = np.random.uniform(-np.pi/2, np.pi/2, size=N) # uniform distribution
        r = np.random.normal(0.1, 0.01, size=N) # Gaussian noise for radius

        # Step 2: Compute p
        px = r * np.cos(a) + 0.25
        py = r * np.sin(a)
        p = np.stack([px, py], axis=1)

        # Step 3: Apply shift and rotation based on 0
        shift_x = -np.abs(theta[:, 0] + theta[:, 1]) / np.sqrt(2)
        shift_y = (-theta[:, 0] + theta[:, 1]) / np.sqrt(2)
        shift = np.stack([shift_x, shift_y], axis=1)

        x = p + shift
        return x

    def sample_jax(self, theta, seed):
        # seed -> prng key for each sample
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))

        # Step 1: Sample intermediate variables
        a = jax.random.uniform(subkey, minval=-jnp.pi/2, maxval=jnp.pi/2)
        key, subkey = jax.random.split(key)
        r = jax.random.normal(subkey) * 0.01 + 0.1 # Gaussian noise

        # Step 2: Compute intermediate points p
        px = r*jnp.cos(a) + 0.25
        py = r*jnp.sin(a)
        p = jnp.stack([px,py])

        # Step 3: Apply shift and rotation based on theta
        shift_x = -jnp.abs(theta[0] + theta[1]) / jnp.sqrt(2)
        shift_y = (- theta[0] + theta[1]) / jnp.sqrt(2)
        shift = jnp.stack([shift_x, shift_y])

        # Compute the final simulated points
        x = p + shift
        return x


    def sample_pytorch(self, theta):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        # Step 1: Sample intermediate variables
        a = torch.rand(theta.shape[0]) * (np.pi) - np.pi / 2 # Uniform distribution
        r = torch.randn(theta.shape[0]) * 0.01 + 0.1 # Gausian noise for radius

        px = r * torch.cos(a) + 0.25
        py = r * torch.sin(a)
        p = torch.stack([px, py], dim=1)

        shift_x = -torch.abs(theta[:, 0] + theta [:, 1]) / torch.sqrt(torch.tensor(2.0))
        shift_y = (-theta[:, 0] + theta[:, 1]) / torch.sqrt(torch.tensor(2.0))
        shift = torch.stack([shift_x, shift_y], dim=1)

        # Compute final simulated points
        x = p + shift
        return x


class SLCP(BaseSimulator):
    def __init__(self, dim: int, dim_y: int):
        super().__init__("slcp", dim, dim_y)

    def sample_jax(self, theta: jnp.ndarray, seed: int):
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        theta = jnp.array(theta)
        mu = theta[:2]
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rh0 = jnp.tanh(theta[4])
        eps = 0.000001
        cov = jnp.array([[s1 ** 2 + eps, rh0 * s1 * s2], [rh0 * s1 * s2, s2 ** 2 + eps]])
        y = jax.random.multivariate_normal(key=subkey, mean=mu, cov=cov)
        y = y.reshape(-1)
        return y

    def sample_numpy(self, theta: np.ndarray):
        """SLCP simulator. D = 5, D_y = 2.

        theta: (N, D)
        """
        theta = np.array(theta)
        mu = theta[:, :2]
        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rh0 = np.tanh(theta[:, 4])
        eps = 0.000001
        cov = np.array([[s1 ** 2 + eps, rh0 * s1 * s2], [rh0 * s1 * s2, s2 ** 2 + eps]])
        y = np.array([np.random.multivariate_normal(mean=mu[i], cov=cov[:, :, i]) for i in range(theta.shape[0])])
        return y


class SLCPDistractors(BaseSimulator):
    def __init__(self, dim: int, dim_y: int, dim_distractors: int):
        self.dim_distractors = dim_distractors
        self.reindex = [
            87, 47, 39,  9, 25, 21, 26, 94, 41, 30, 34, 44, 12, 27, 89, 20,  6, 13, 51, 40,
            54,  5,  0,  2, 75, 43, 14, 97, 29, 72, 79, 99, 98,  1, 38, 65, 83, 52, 74, 63,
            19, 70,  4, 36, 96, 81, 35, 49, 31, 76, 84, 28, 11, 66, 37, 85, 56, 60, 48, 10,
            22, 82, 24,  8, 42, 32, 73,  3, 59, 95, 90, 50, 18, 68, 45, 67, 92, 91, 17, 93,
            33, 78, 88, 62, 46, 64, 57, 86, 55, 77,  7, 80, 69, 23, 58, 71, 15, 61, 53, 16
        ]
        super().__init__("slcp_distractor", dim, dim_y)

    def sample_jax(self, theta: jnp.ndarray, seed: int):
        # informative part
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))
        theta = jnp.array(theta)
        mu = theta[:2]
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rh0 = jnp.tanh(theta[4])
        eps = 0.000001
        cov = jnp.array([[s1 ** 2 + eps, rh0 * s1 * s2], [rh0 * s1 * s2, s2 ** 2 + eps]])
        yy = jax.random.multivariate_normal(key=subkey, mean=mu, cov=cov)

        # add dim_distractor samples from a uniform distribution
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))

        # distractors come from a normal distribution with
        # mu coming from normal with mu=0, sigma=15
        # sigma = 3e^a where a comes from normal with mu=0, sigma=1
        mu_distractors = jax.random.normal(subkey, shape=(self.dim_distractors,)) * 15.0
        key, subkey = jax.random.split(key)
        a = jax.random.normal(subkey, shape=(self.dim_distractors,))
        sigma_distractors = jnp.exp(a) * 3.0
        key, subkey = jax.random.split(key)
        yy_distractors = jax.random.normal(
            subkey,
            shape=(self.dim_distractors,)
        ) * sigma_distractors + mu_distractors

        # concatenate informative and distractor parts
        y = jnp.concatenate([yy, yy_distractors], axis=-1)
        return y

    def sample_numpy(self, theta: np.ndarray):
        # theta is (N, D) here
        # informative part
        theta = np.array(theta)
        mu = theta[:, :2]
        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rh0 = np.tanh(theta[:, 4])
        eps = 0.000001
        cov = np.array([[s1 ** 2 + eps, rh0 * s1 * s2], [rh0 * s1 * s2, s2 ** 2 + eps]])
        yy = np.array([np.random.multivariate_normal(mean=mu[i], cov=cov[:, :, i]) for i in range(theta.shape[0])])

        # add dim_distractor samples
        # distractors come from a normal distribution with
        # mu coming from normal with mu=0, sigma=15
        # sigma = 3e^a where a comes from normal with mu=0, sigma=1
        mu_distractors = np.random.normal(0, 15.0, size=(theta.shape[0], self.dim_distractors))
        a = np.random.normal(0, 1.0, size=(theta.shape[0], self.dim_distractors))
        sigma_distractors = np.exp(a) * 3.0
        yy_distractors = np.random.normal(0, 1.0, size=(theta.shape[0], self.dim_distractors)) * sigma_distractors + mu_distractors

        # concatenate informative and distractor parts
        y = np.concatenate([yy, yy_distractors], axis=-1)
        return y
