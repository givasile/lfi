import numpy as np
import torch
import jax
from jax import random
import jax.numpy as jnp



class LinearSimulator:
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise

    def simulate_numpy(self, theta, x):
        return np.dot(theta, x) + np.random.normal(0, self.sigma_noise)

    def simulate_jax(self, theta, x, keys):
        """
        Simulate data for multiple thetas and keys

        Parameters:
        - theta: ndarray, the parameters, shape (N, 2)
        - x: ndarray, the input data, shape (2,)
        - keys: ndarray, the random keys, shape (N, 2)
        """
        def simulate_one(theta, x, key):
            return jnp.dot(theta, x) + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0, None, 0))(theta, x, keys)
    
    def sample_pytorch(self, theta):
        return torch.matmul(theta, self.x) + torch.randn_like(theta)*self.sigma_noise
    

class BaseSimulator:
    def __init__(self,):
        pass
    
    def sample_numpy(self, theta):
        raise NotImplementedError
    def sample_jax(self, theta, keys):
        raise NotImplementedError
    def sample_pytorch(self, theta):
        raise NotImplementedError
    

class GaussianNoise(BaseSimulator):
    def __init__(self, sigma_noise, dim, dim_y):
        self.sigma_noise = sigma_noise
        super().__init__()

    def sample_numpy(self, theta):
        return np.random.normal(theta, self.sigma_noise)
    
    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            return theta + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0,0))(theta, keys)

    def sample_pytorch(self, theta):
        return theta + torch.randn_like(theta)*self.sigma_noise


class BimodalGaussian(BaseSimulator):
    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise
        super().__init__()

    def sample_numpy(self, theta):
        # for each theta in the batch select either the first or the second mode
        mode = np.random.choice([0,1], size = theta.shape[0])

        mean = theta + 3
        mean[mode==1] = theta[mode == 1] - 3
        return np.random.normal(mean, self.sigma_noise)
    
    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            mode = jax.random.choice(key, shape=(theta.shape[0],), a=jnp.array([0.1]))
            mean = theta + 3
            mean = jax.ops.index_update(mean, jax.ops.index[mode==1], theta[mode==1]-3)
            return mean + jax.random.normal(key)*self.sigma_noise
        return jax.vmap(simulate_one, in_axes=(0,0))(theta, keys)
                                         
    def sample_pytorch(self, theta):
        mode = torch.randint(0, 2, (theta.shape[0],))

        mean = theta + 3
        mean[mode==1] = theta[mode == 1] - 3
        return mean + torch.randn_like(theta) * self.sigma_noise


class TwoMoon(BaseSimulator):
    def __init__(self):
        super().__init__()

    '''
    Two-moons simulator
    Args:
        theta(np.array): Array of shape (num_samples, D), where each row is a parameter [01, 02].
    Returns:
        x(np.array): Simulated observations of shape (num_samples, D)
    '''
    
    def sample_numpy(self, theta):


        # Step 1: Sample intermediate variables
        a = np.random.uniform(-np.pi/2, np.pi/2, size=theta.shape[0]) # uniform distribution
        r = np.random.normal(0.1, 0.01, size=theta.shape[0]) # Gaussian noise for radius

        # Step 2: Compute p
        px = r * np.cos(a) + 0.25
        py = r * np.sin(a)
        p = np.stack([px, py], axis=1)

        # Step 3: Apply shift and rotation based on 0
        shift_x = -np.abs(theta[:, 0] + theta[:, 1]) / np.sqrt(2)
        shift_y = (-theta[:, 0] + theta[:, 1]) / np.sqrt(2)
        shift = np.stack([px, py], axis=1)

        x = p + shift

        return x

    def sample_jax(self, theta, keys):

        def simulate_one(theta, key):                   

            # Step 1: Sample intermediate varialbles
            key_a, key_r = jax.random.split(key)
            a = jax.random.uniform(key_a, minval=-jnp.pi/2, maxval=jnp.pi/2)
            r = jax.random.normal(key_r) * 0.01 + 0.1 # Gaussian noise
    
            # Step 2: Compute intermediate points p
            px = r*jnp.cos(a) + 0.25
            py = r*jnp.sin(a)
            p = jnp.stack([px,py], axis=1)
    
            # Step 3: Apply shift and rotation based on theta
            shift_x = -jnp.abs(theta[:, 0] + theta[:, 1]) / jnp.sqrt(2)
            shift_y = r * jnp.sin(a) # Use r and a for vertical shift
            shift = jnp.stack([shift_x, shift_y], axis=1)

            # Compute the final simulated points
            x = p + shift
            return x

        # Apply vamp to vectorize simulate_one over the batch of theta and keys
        return jax.vmap(simulate_one)(theta, keys)

    def sample_pytorch(self, theta):

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
        
    

    

