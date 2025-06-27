import numpy as np
import torch
import jax
from typing import Callable
import jax.numpy as jnp
import scipy.stats as ss
from scipy.integrate import solve_ivp
from torchdiffeq import odeint



class BaseSimulator:
    def __init__(self, name:str, dim: int, dim_y: int, **kwargs):
        self.name=name
        self.dim=dim
        self.dim_y = dim_y
        self.informative_dims = None

    def set_informative_dims(self, informative_dims: np.array):
        self.informative_dims = informative_dims.astype(bool)

    def sample_numpy(self, theta: np.ndarray):
        """
        Samples from the simulator using NumPy.

        Args:
            theta: Parameters for the simulation, shape (batch_size, dim).

        Returns:
            np.ndarray: Simulated observations, shape (batch_size, dim_y).
        """
        raise NotImplementedError


    def sample_jax(self, theta: jnp.ndarray, seed: int):
        """
        Samples from the simulator using JAX.

        Args:
            theta: Parameters for the simulation, shape (dim, ).
            seed: Random seed for reproducibility.

        Returns:
            jnp.ndarray: Simulated observations, shape (dim_y).
        """
        raise NotImplementedError

    def sample_pytorch(self, theta: torch.Tensor):
        """
        Samples from the simulator using PyTorch.

        Args:
            theta (torch.Tensor): Parameters for the simulation, shape (batch_size, dim).
        Returns:
            torch.Tensor: Simulated observations, shape (batch_size, dim_y).
        """
        raise NotImplementedError

    def return_elfi_callable(self):
        """
        Returns a callable that can be used with ELFI.
        This is useful for ELFI-based inference methods.
        """
        raise NotImplementedError("This simulator does not support ELFI integration.")

    def return_jax_callable(self):
        return self.sample_jax

    def jax_cr_simulator1(self) -> Callable[[jnp.array, int], jnp.array]:
        # f_1: ((BS_th, D_th), s) -> (BS_th, D_y)
        f_1 = jax.vmap(self.return_jax_callable(), in_axes=(0, None))
        return f_1

    def jax_cr_simulator2(self) -> Callable[[jnp.array, jnp.array], jnp.array]:
        # f_1: ((BS_th, D_th), s) -> (BS_th, D_y)
        f_1 = jax.vmap(self.return_jax_callable(), in_axes=(0, None))
        # f_2: ((BS_th, D_th), (BS_s,)) -> (BS_s, BS_th, D_y)
        f_2 = jax.vmap(f_1, in_axes=(None, 0))
        return f_2

    def jax_cr_jacobian(self):
        return jax.jacobian(self.return_jax_callable())

    def jax_cr_jacobian1(self):
        jacobian = self.jax_cr_jacobian()
        jacobian = jax.vmap(jacobian, in_axes=(0, None))
        return jacobian

    def jax_cr_jacobian2(self):
        jacobian = self.jax_cr_jacobian()
        jacobian = jax.vmap(jacobian, in_axes=(0, None))
        jacobian = jax.vmap(jacobian, in_axes=(None, 0))
        return jacobian

    def jax_cr_distance(self, informative_dims=None):
        gen = self.return_jax_callable()

        def distance(th: jnp.array, s: jnp.array, y_0: jnp.array) -> jnp.array:
            y = gen(th, s)  # y: (D,)
            if self.informative_dims is not None:
                y = y[self.informative_dims] # y: (D1,)
                y_0 = y_0[self.informative_dims] # y_0: (D1,)
            diff = jnp.square(y - y_0)  # diff: (D1,)
            dist = jnp.mean(diff)  # sum: ()
            return dist
        return distance

    def jax_cr_distance1(self):
        gen = self.jax_cr_simulator1()

        def distance(th: jnp.array, s: jnp.array, y_0: jnp.array) -> jnp.array:
            y = gen(th, s) # y: (BS_th, D)
            if self.informative_dims is not None:
                y = y[:, self.informative_dims] # y: (BS_th, D1)
                y_0 = y_0[self.informative_dims] # y_0: (D1,)
            y_0 = jnp.expand_dims(y_0, axis=0) # y_0: (1, D)
            diff = jnp.square(y - y_0) # diff: (BS_th, D)
            dist = jnp.mean(diff, axis=-1) # sum: (BS_th,)
            return dist
        return distance

    def jax_cr_distance2(self):
        gen = self.jax_cr_simulator2()

        def distance(th: jnp.array, s: jnp.array, y_0: jnp.array) -> jnp.array:
            y = gen(th, s) # y: (BS_s, BS_th, D)
            if self.informative_dims is not None:
                y = y[:, :, self.informative_dims] # y: (BS_s, BS_th, D1)
                y_0 = y_0[self.informative_dims] # y_0: (D1,)
            y_0 = jnp.expand_dims(jnp.expand_dims(y_0, axis=0), axis=0) # y_0: (1, 1, D1)
            diff = jnp.square(y - y_0) # diff: (BS_s, BS_th, D1)
            dist = jnp.mean(diff, axis=-1) # sum: (BS_s, BS_th)
            return dist

        return distance

    def jax_cr_distance_grad(self):
        dist = self.jax_cr_distance()
        d_grad = jax.value_and_grad(dist, argnums=0)
        return d_grad

    def jax_cr_distance_grad1(self):
        d_grad = self.jax_cr_distance_grad()
        d_grad = jax.vmap(d_grad, in_axes=(0, None, None))
        return d_grad

    def jax_cr_distance_grad2(self):
        dist = self.jax_cr_distance_grad()
        d_grad = jax.vmap(dist, in_axes=(0, None, None))
        d_grad = jax.vmap(d_grad, in_axes=(None, 0, None))
        return d_grad

    def jax_cr_distance_hess(self):
        dist = self.jax_cr_distance()
        d_hess = jax.hessian(dist, argnums=0)
        return d_hess

    def jax_cr_distance_hess1(self):
        d_hess = self.jax_cr_distance_hess()
        d_hess = jax.vmap(d_hess, in_axes=(0, None, None))
        return d_hess

    def jax_cr_distance_hess2(self):
        dist = self.jax_cr_distance_hess()
        d_hess = jax.vmap(dist, in_axes=(0, None, None))
        d_hess = jax.vmap(d_hess, in_axes=(None, 0, None))
        return d_hess
    

class GaussianNoise(BaseSimulator):
    def __init__(self, dim, dim_y, sigma_noise):
        self.sigma_noise = sigma_noise
        super().__init__("gaussian_noise", dim, dim_y)

    def sample_numpy(self, theta):
        return np.random.normal(theta, self.sigma_noise)
    
    def sample_jax(self, theta, seed):
        # seed -> prng key for each sample
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))

        theta = jnp.asarray(theta)
        y = theta + jax.random.normal(key=subkey, shape=(self.dim, )) * self.sigma_noise
        return y

    def sample_pytorch(self, theta):
        return theta + torch.randn_like(theta)*self.sigma_noise
    
    def return_elfi_callable(self):
        def elfi_simulator(*th_params, batch_size=1, random_state=None):
            theta = np.stack(th_params, axis=1)
            samples_standard_normal = ss.norm.rvs(size=(batch_size, self.dim_y), random_state=random_state)
            samples = theta + self.sigma_noise*samples_standard_normal
            return samples
        return elfi_simulator

class GaussianNoiseDistractor(BaseSimulator):
    def __init__(self,
                 dim,
                 dim_y,
                 sigma_noise,
                 distractor_dim,
                 distractor_scale=None,
                 distractor_mu_min=None,
                 distractor_mu_max=None
                 ):
        self.sigma_noise = sigma_noise
        self.distractor_dim = distractor_dim
        self.distractor_scale = distractor_scale if distractor_scale is not None else 1.0
        self.distractor_mu_min = distractor_mu_min if distractor_mu_min is not None else -5.
        self.distractor_mu_max = distractor_mu_max if distractor_mu_max is not None else 5.
        total_output_dim = dim_y + distractor_dim
        super().__init__("gaussian_noise_with_distractors", dim, total_output_dim)

    def sample_numpy(self, theta):
        # Infromative part
        y_info = np.random.normal(theta, self.sigma_noise)

        # Distractor part
        mu = np.random.uniform(self.distractor_mu_min, self.distractor_mu_max, size=(theta.shape[0], self.distractor_dim))
        y_distractors = np.random.normal(mu, np.sqrt(self.distractor_scale))

        # Concatenate informative and distractor parts
        y = np.concatenate([np.atleast_1d(y_info), y_distractors], axis=-1)
        return y

    # def sample_jax(self, theta, keys):
    #     # keys: one key per sample
    #     def simulate_one(theta_val, key):
    #         key_i, key_m, key_n = jax.random.split(key, 3)
    #
    #         # Informative part
    #         y_info = theta_val + jax.random.normal(key_i)*self.sigma_noise
    #
    #         # Distractor part
    #         mu_val = jax.random.uniform(
    #             key_m,
    #             shape=(self.distractor_dim,),
    #             minval=self.distractor_mu_min,
    #             maxval=self.distractor_mu_max
    #             )
    #
    #         noise = jax.random.normal(key_n, (self.distractor_dim,))
    #         y_distractor = mu_val + noise * jnp.sqrt(self.distractor_scale)
    #
    #         return np.concatenate([np.atleast_1d(y_info), y_distractor])
    #
    #     return jax.vmap(simulate_one)(theta, keys)

    def sample_pytorch(self, theta):
        # Informative part
        y_info = theta + torch.randn_like(theta)*self.sigma_noise

        # Distractor part
        mu = (torch.rand((theta.shape[0], self.distractor_dim)) * (self.distractor_mu_max - self.distractor_mu_min) +
                  self.distractor_mu_min)
        y_distractors = mu + torch.randn_like(mu) * np.sqrt(self.distractor_scale)

        return torch.cat([y_info, y_distractors], dim=-1)
         

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

class MultivariateGaussian(BaseSimulator):
    def __init__(self, dim, dim_y, sigma_noise, shift_value):
        '''
        dim: dimensionality of theta
        dim_y: dimensionality of observation,
        sigma_noise: standard deviation of noise,
        shift_value: shift value for the Gaussian distribution
        '''
        self.sigma_noise = sigma_noise
        self.shift_value = shift_value # shift for the mean
        super().__init__("multivariateGaussian", dim, dim_y)

    def sample_numpy(self, theta):
        # Introduce the shift to the mean
        mean = theta + self.sift_value # Shift the mean by the shift_value
        cov = np.diag([self.sigma_noise**2], theta.shape[0]) # Diagonal covariance matrix
        return np.random.multivariate_normal(mean, cov)
    
    def sample_jax(self, theta, keys):
        def simulate_one(theta, key):
            mean = theta + self.shift_value
            cov = jnp.diag([self.sigma_noise**2]*theta.shape[0])
            return mean + jax.random.multivariate_normal(key, cov)
        return jax.vmap(simulate_one, in_axes=(0,0))(theta, keys)

    def sample_pytorch(self, theta):
        mean = theta + self.shift_value
        #print(f"Mean shape: {mean.shape}")
        cov = torch.diag(torch.full((theta.shape[-1],), self.sigma_noise**2, dtype=torch.float32))
        #print(f"cov shape: {cov.shape}")
        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return mvn.sample() 
    

class TwoMoon(BaseSimulator):
    def __init__(self, dim, dim_y):
        super().__init__("TwoMoon", dim, dim_y)

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
        shift = np.stack([shift_x, shift_y], axis=1)

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
        
    

    




class LotkaVolterraSimulator(BaseSimulator):
    def __init__(self, dim, dim_y, t_span=(0,30), n_steps=100, initial_conditions = (30.0, 5.0)):
        '''
        Parameters:
        - t_span: simulation time range (start, end)
        - n_steps: number of time points
        - initial_conditions: fixed initial [x0, y0] (default: [30, 5])
        '''
        super().__init__("lotka_volterra", dim, dim_y) # dim = 4 (alpha, beta delta, gamma), dim_y = 2

        self.t_span = t_span
        self.n_steps = n_steps
        self.initial_conditions = np.array(initial_conditions, dtype=np.float32)
        self.time_points = np.linspace(t_span[0], t_span[1], n_steps)

    def lotka_volterra_numpy(self, t, z, alpha, beta, delta, gamma):
        ''' Numpy implementation of LV equaitons'''
        x, y = z
        dxdt = x * (alpha-beta*y)
        dydt = delta*x*y - gamma*y
        return [dxdt, dydt]

    def lotka_volterra_torch(self, t, z, theta):
        '''Pytorch implementation of LV equations'''
        x, y = z[..., 0], z[..., 1]
        alpha, beta, delta, gamma = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3]
        dxdt = alpha*x - beta*x*y
        dydt = delta*x*y - gamma*y
        return torch.stack([dxdt, dydt], dim=-1)

    def sample_numpy(self, theta):
        theta = np.asarray(theta)
        single_sample = theta.ndim == 1
        if single_sample:
            theta = theta[np.newaxis, :]
        
        trajectories = np.zeros((theta.shape[0], self.n_steps, 2))

        for i, (alpha, beta, delta, gamma) in enumerate(theta):
            sol = solve_ivp(
                fun = self.lotka_volterra_numpy,
                t_span = self.t_span,
                y0 = self.initial_conditions,
                t_eval = self.time_points,
                args = (alpha, beta, delta, gamma),
                method = 'RK45'
            )

            trajectories [i] = sol.y.T

        return trajectories[0] if single_sample else trajectories

    def sample_pytorch(self, theta):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32)

        single_sample = theta.ndim == 1
        if single_sample:
            theta = theta.unsqueeze(0)

        # Create a modified LV function that captures theta
        def lv_wrapper(t, z):
            return self.lotka_volterra_torch(t, z, theta)


        # Initial conditions (batch_size, 2)
        y0 = torch.tensor(self.initial_conditions,
                            dtype=torch.float32,
                            device=theta.device).expand(theta.size(0), -1)
        
        # Solve ODE
        trajectories = odeint(
            func=lv_wrapper,
            y0 = y0,
            t = torch.linspace(*self.t_span, self.n_steps, device=theta.device),
            method = 'dopri5'      
        ).permute(1,0,2) # -> (batch, time, 2)

        #return trajectories.squeeze(0) if single_sample else trajectories

        # Flatten in 1D
        flattened = trajectories.reshape(trajectories.shape[0], -1) # batch_size, n_steps*2)
        return flattened.squeeze(0) if theta.ndim == 1 else flattened
