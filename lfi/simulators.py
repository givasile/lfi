import numpy as np
import torch
import jax
import jax.numpy as jnp
import scipy.stats as ss

class BaseSimulator:
    def __init__(self, name: str, dim: int, dim_y: int):
        self.name = name
        self.dim = dim
        self.dim_y = dim_y

    def sample_numpy(self, theta):
        raise NotImplementedError

    def sample_jax(self, theta, keys):
        raise NotImplementedError

    def sample_pytorch(self, theta):
        raise NotImplementedError

    def return_elfi_callable(self):
        raise NotImplementedError


class GaussianNoise(BaseSimulator):
    # TODO: check if sigma_noise square is needed
    def __init__(self, dim, dim_y, sigma_noise):
        self.sigma_noise = sigma_noise
        super().__init__("gaussian_noise", dim, dim_y)

    def sample_numpy(self, theta):
        return np.random.normal(theta, self.sigma_noise)

    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            return theta + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0, 0))(theta, keys)

    def sample_pytorch(self, theta):
        return theta + torch.randn_like(theta)*self.sigma_noise

    def return_elfi_callable(self):
        def elfi_simulator(*th_params, batch_size=1, random_state=None):
            theta = np.stack(th_params, axis=1)
            samples_standard_normal = ss.norm.rvs(size=(batch_size, self.dim_y), random_state=random_state)
            samples = theta + self.sigma_noise * samples_standard_normal
            return samples
        return elfi_simulator


class BimodalGaussian(BaseSimulator):
    def __init__(self, dim, dim_y, sigma_noise):
        self.sigma_noise = sigma_noise
        super().__init__("bimodal_gaussian", dim, dim_y)

    def sample_numpy(self, theta):
        # for each theta in the batch select either the first or the second mode
        mode = np.random.choice([0, 1], size=theta.shape[0])

        mean = theta + 3
        mean[mode == 1] = theta[mode == 1] - 3
        return np.random.normal(mean, self.sigma_noise)

    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            mode = jax.random.choice(key, shape=(theta.shape[0],), a=jnp.array([0, 1]))
            mean = theta + 3
            mean = jax.ops.index_update(mean, jax.ops.index[mode == 1], theta[mode == 1] - 3)
            return mean + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0, 0))(theta, keys)

    def sample_pytorch(self, theta):
        mode = torch.randint(0, 2, (theta.shape[0],))

        mean = theta + 3
        mean[mode == 1] = theta[mode == 1] - 3
        return mean + torch.randn_like(theta)*self.sigma_noise



