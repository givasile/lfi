import numpy as np
import torch
import jax
from typing import Callable
import jax.numpy as jnp
import scipy.stats as ss
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from jax.experimental.ode import odeint


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
        return theta + self.shift + torch.randn_like(theta)*self.sigma_noise


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
        mode = torch.randint(0, 2, (theta.shape[0],))
        mean = theta + self.shift
        mean[mode == 1] = theta[mode == 1] - self.shift
        yy = mean + torch.randn_like(theta) * self.sigma_noise

        # add dim_distractor samples from a uniform distribution
        yy_distractors = torch.rand((theta.shape[0], self.dim_distractors)) * 6 - 3 # Uniform distribution in [-3, 3]

        # concatenate informative and distractor parts
        y = torch.cat([yy, yy_distractors], dim=-1)
        return y


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
        
    
class ImageNoise(BaseSimulator):
    def __init__(self, dim, dim_y, H, W, sigma_blur=1.0, sigma_noise=0.5):
        self.H = H
        self.W = W
        self.sigma_blur = sigma_blur
        self.sigma_noise = sigma_noise
        super().__init__("image_noise", dim, dim_y)

    def sample_jax(self, theta: jax.Array, seed: int) -> jax.Array:
        """
        Simulates sensor output from a clean image by applying blur and noise.
        Args:
            theta: (D,) - Flat clean image.
            seed: Random seed for noise generation.
        Returns:
            (D,) - Flat noisy, blurred image.
        """
        # seed to key
        key = jax.random.PRNGKey(seed)
        theta = jnp.asarray(theta)

        # Reshape to image dimensions
        image = theta.reshape((self.H, self.W))

        # Apply Gaussian blur
        def return_kernel(kernel_type="gaussian", size=5, sigma=1.0):
            if kernel_type == "gaussian":
                ax = jnp.linspace(-(size - 1) / 2., (size - 1) / 2., size)
                xx, yy = jnp.meshgrid(ax, ax)
                kernel = jnp.exp(-(xx**2 + yy**2) / (2. * sigma**2))
                kernel = kernel / jnp.sum(kernel)

            elif kernel_type == "checkerboard":
                # Fixed 2x2 high-frequency kernel
                kernel = jnp.array([[1.0, -1.0],
                                    [-1.0, 1.0]])
                kernel = kernel / jnp.sqrt(jnp.sum(kernel**2))

            elif kernel_type == "sharpen":
                # Simple sharpening kernel
                kernel = jnp.array([[ 0, -1,  0],
                                    [-1,  5, -1],
                                    [ 0, -1,  0]])

            elif kernel_type == "sobel_x":
                kernel = jnp.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]) / 4.0  # normalized

            elif kernel_type == "sobel_y":
                kernel = jnp.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]) / 4.0  # normalized

            elif kernel_type == "identity":
                # Centered identity kernel
                kernel = jnp.zeros((size, size))
                center = size // 2
                kernel = kernel.at[center, center].set(1.0)

            else:
                raise ValueError(f"Unknown kernel_type: {kernel_type}")
            return kernel
        
        kernel = return_kernel(kernel_type="checkerboard", size=5, sigma=self.sigma_blur)
        blurred = jax.scipy.signal.convolve2d(image, kernel, mode='same', boundary='fill')
        # blurred = image

        # Alternative invertible transformations (uncomment to use):

        # 1. Pixel scale compression: [0,1] -> [0.45,0.55]
        # Invertible: y = x/10 + 0.5, inverse: x = 10*(y - 0.5)
        # blurred = blurred / 10.0 + 0.5

        # 2. Gamma correction (power law): y = x^gamma
        # Invertible: y = x^gamma, inverse: x = y^(1/gamma)
        # gamma = 0.8  # < 1 brightens, > 1 darkens
        # blurred = jnp.power(jnp.clip(blurred, 0, 1), gamma)

        # 3. Sigmoid contrast enhancement: y = 1/(1+exp(-k*(x-0.5)))
        # Approximately invertible: x = 0.5 + (1/k)*log(y/(1-y))
        # k = 5.0  # steepness parameter
        # blurred = 1.0 / (1.0 + jnp.exp(-k * (blurred - 0.5)))

        # 4. Linear contrast and brightness: y = a*x + b
        # Invertible: y = a*x + b, inverse: x = (y-b)/a
        # a, b = 0.2, 0.5  # contrast and brightness
        # blurred = a * blurred + b
        # blurred = blurred.at[0, 0].set(0.0)

        # 5. Histogram equalization approximation (piecewise linear)
        # Approximately invertible through lookup table
        # breakpoints = jnp.array([0.0, 0.3, 0.7, 1.0])
        # values = jnp.array([0.0, 0.6, 0.8, 1.0])
        # blurred = jnp.interp(blurred, breakpoints, values)

        # Add Gaussian noise
        key, subkey = jax.random.split(key)
        noise = self.sigma_noise * jax.random.normal(subkey, shape=(self.H, self.W))
        noisy_image = blurred + noise
        
        return noisy_image.flatten()
    
    def sample_numpy(self, theta: np.ndarray):
        """
        Simulates sensor output from a clean image by applying blur and noise.
        Args:
            theta: (N, D) - Flat clean images, where N is the batch size and D is the flattened image size.
        Returns:
            (N, D) - Flat noisy, blurred images.
        """
        N, D = theta.shape
        H = W = int(np.sqrt(D))

        # Reshape all images at once
        images = theta.reshape(N, H, W)

        # Apply Gaussian blur to all images
        blurred = np.array([gaussian_filter(img, sigma=self.sigma_blur, mode='constant') 
                           for img in images])

        # Add Gaussian noise to all images
        noise = np.random.normal(loc=0.0, scale=self.sigma_noise, size=(N, H, W))
        noisy_images = blurred + noise

        return noisy_images.reshape(N, D)


class ImagePixelWiseTransform(BaseSimulator):
    def __init__(self, dim, dim_y, H, W, sigma_noise=0.5):
        self.H = H
        self.W = W
        self.sigma_noise = sigma_noise
        super().__init__("image_noise", dim, dim_y)

    def sample_jax(self, theta: jax.Array, seed: int) -> jax.Array:
        """
        Simulates sensor output from a clean image by applying blur and noise.
        Args:
            theta: (D,) - Flat clean image.
            seed: Random seed for noise generation.
        Returns:
            (D,) - Flat noisy, blurred image.
        """
        # seed to key
        key = jax.random.PRNGKey(seed)
        theta = jnp.asarray(theta)

        # Reshape to image dimensions
        image = theta.reshape((self.H, self.W))

        # Invertible transformations (uncomment to use):

        # 1. Pixel scale compression: [0,1] -> [0.45,0.55]
        # Invertible: y = x/10 + 0.5, inverse: x = 10*(y - 0.5)
        image = image / 10.0 + 0.5

        # 2. Gamma correction (power law): y = x^gamma
        # Invertible: y = x^gamma, inverse: x = y^(1/gamma)
        # gamma = 0.8  # < 1 brightens, > 1 darkens
        # image = jnp.power(jnp.clip(image, 0, 1), gamma)

        # 3. Sigmoid contrast enhancement: y = 1/(1+exp(-k*(x-0.5)))
        # Approximately invertible: x = 0.5 + (1/k)*log(y/(1-y))
        # k = 5.0  # steepness parameter
        # image = 1.0 / (1.0 + jnp.exp(-k * (image - 0.5)))

        # 4. Linear contrast and brightness: y = a*x + b
        # Invertible: y = a*x + b, inverse: x = (y-b)/a
        # a, b = 0.2, 0.5  # contrast and brightness
        # image = a * image + b
        # image = image.at[0, 0].set(0.0)

        # 5. Histogram equalization approximation (piecewise linear)
        # Approximately invertible through lookup table
        # breakpoints = jnp.array([0.0, 0.3, 0.7, 1.0])
        # values = jnp.array([0.0, 0.6, 0.8, 1.0])
        # image = jnp.interp(image, breakpoints, values)

        # Add Gaussian noise
        key, subkey = jax.random.split(key)
        noise = self.sigma_noise * jax.random.normal(subkey, shape=(self.H, self.W))
        noisy_image = image + noise

        return noisy_image.flatten()

    def sample_pytorch(self, theta: torch.Tensor):
        """
        Simulates sensor output from a clean image by applying blur and noise.
        Args:
            theta: (N, D) - Flat clean images, where N is the batch size and D is the flattened image size.
        Returns:
            (N, D) - Flat noisy, blurred images.
        """
        N, D = theta.shape
        H = W = int(np.sqrt(D))

        # Reshape all images at once
        images = theta.reshape(N, H, W)

        # Change the pixel values
        images = images / 10.0 + 0.5

        # Add Gaussian noise to all images
        noise = torch.randn((N, H, W)) * self.sigma_noise
        noisy_images = images + noise

        return noisy_images.reshape(N, D)

    def sample_numpy(self, theta: np.ndarray):
        """
        Simulates sensor output from a clean image by applying blur and noise.
        Args:
            theta: (N, D) - Flat clean images, where N is the batch size and D is the flattened image size.
        Returns:
            (N, D) - Flat noisy, blurred images.
        """
        N, D = theta.shape
        H = W = int(np.sqrt(D))

        # Reshape all images at once
        images = theta.reshape(N, H, W)

        # Change the pixel values
        images = images / 10.0 + 0.5

        # Add Gaussian noise to all images
        noise = np.random.normal(loc=0.0, scale=self.sigma_noise, size=(N, H, W))
        noisy_images = images + noise

        return noisy_images.reshape(N, D)


class LotkaVolterra(BaseSimulator):
    def __init__(self,
                 dim,
                 dim_y,
                 t_span=(0, 30),
                 n_steps=100,
                 initial_conditions=(30.0, 5.0)
                 ):
        '''
        Lotka-Volterra simulator.
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

    def sample_jax(self, theta, seed):
        """
        Pure JAX Lotka-Volterra simulator for *one* parameter set.

        Args:
            theta: jnp.array of shape (4,) -> [alpha, beta, gamma, delta]

        Returns:
            jnp.array of shape (20,) -> subsampled log-normal observations
        """

        def lotka_volterra_ode_jax(u, t, params):
            """
            Lotka-Volterra differential equations.
            u: [prey, predator]
            t: time (not used because system is autonomous)
            params: [alpha, beta, gamma, delta]
            """
            x, y = u
            alpha, beta, gamma, delta = params
            dxdt = alpha * x - beta * x * y
            dydt = -gamma * y + delta * x * y
            return jnp.array([dxdt, dydt])

        # seed to key_i
        key, subkey = jax.random.split(jax.random.PRNGKey(seed))

        days = 20.0
        saveat = 0.1
        u0 = jnp.array([30.0, 1.0])  # initial [prey, predator]

        # Full time grid: [0, 0.1, ..., 20.0]
        t_eval = jnp.arange(0.0, days + saveat, saveat)  # shape (201,)

        # Solve ODE
        u_solution = odeint(lotka_volterra_ode_jax, u0, t_eval, theta)  # shape: (201, 2)

        # Transpose to match your shape: (2, timepoints)
        u_solution = u_solution.T  # shape: (2, 201)

        # Subsample every 21st point, take first 10
        subsample_indices = jnp.arange(0, u_solution.shape[1], 21)[:10]
        u_subsampled = u_solution[:, subsample_indices]  # shape: (2, 10)

        # Flatten interleaved [x0,y0,x1,y1,...]
        u_flat = u_subsampled.flatten()  # shape: (20,)

        # Clamp, log, sample from log-normal
        u_clamped = jnp.clip(u_flat, 1e-10, 1e4)
        log_loc = jnp.log(u_clamped)

        # Use PRNG for reproducible noise — pass key externally in practice
        # Here we generate a dummy key for illustration
        key, subkey = jax.random.split(key)
        normal_noise = jax.random.normal(subkey, shape=log_loc.shape) * 0.1
        log_normal_samples = log_loc + normal_noise

        return jnp.exp(log_normal_samples)

    def sample_numpy(self, theta):
        """
        Pure NumPy implementation of Lotka-Volterra simulator

        Args:
            theta: numpy array of shape (num_samples, 4) containing parameters
                    [alpha, beta, gamma, delta] for each sample

        Returns:
            numpy array of shape (num_samples, 20) containing subsampled log-normal observations
        """
        # Constants from original implementation
        days = 20.0 # total time span in days
        saveat = 0.1 # time resolution: save every 0.1 days
        u0 = np.array([30.0, 1.0])  # Initial conditions [prey, predator]
        tspan = (0.0, days) # Time span for integration
        dim_data = 20  # For subsample summary

        # Time points where we want the solution
        t_eval = np.arange(0.0, days + saveat, saveat) # [0.0, 0.1, ..., 20.0]

        num_samples = theta.shape[0]
        results = np.full((num_samples, dim_data), np.nan) # prefils results with NaN

        def lotka_volterra_ode_np(t, u, alpha, beta, gamma, delta):
            """
            Lotka-Volterra differential equation system
            du/dt = [alpha*x - beta*x*y, -gamma*y + delta*x*y]
            where u = [x, y] = [prey, predator]
            """
            x, y = u
            dxdt = alpha * x - beta * x * y
            dydt = -gamma * y + delta * x * y
            return np.array([dxdt, dydt])

        for i in range(num_samples):
            alpha, beta, gamma, delta = theta[i]

            try:
                # Solve the ODE system
                sol = solve_ivp(
                    fun=lambda t, u: lotka_volterra_ode_np(t, u, alpha, beta, gamma, delta),
                    t_span=tspan,
                    y0=u0,
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-8,
                    atol=1e-10
                )

                if sol.success and sol.y.shape[1] == len(t_eval):
                    # Reshape solution: sol.y is (2, n_timepoints) = (2, 201) for 20 days with 0.1 step
                    # We want to subsample every 21st point (::21) like in original
                    u_solution = sol.y  # shape: (2, n_timepoints) = (2, 201)

                    # Subsample every 21st point (equivalent to ::21 in original)
                    # Original had int(dim_data_raw/2) = int(2*(20/0.1+1)/2) = int(202) = 201 timepoints
                    # Subsampling ::21 gives us 10 points per species = 20 total points
                    n_timepoints = u_solution.shape[1]
                    subsample_indices = np.arange(0, n_timepoints, 21)[:10]  # Take first 10 subsampled points

                    if len(subsample_indices) >= 10:
                        u_subsampled = u_solution[:, subsample_indices[:10]]  # shape: (2, 10)

                        u_flat = u_subsampled.flatten()  # shape: (20,) - interleaved [x0,y0,x1,y1,...]

                        # Apply log-normal noise like in original
                        # Original: LogNormal(loc=log(clamp(u, 1e-10, 10000)), scale=0.1)
                        u_clamped = np.clip(u_flat, 1e-10, 10000.0)
                        log_loc = np.log(u_clamped)

                        # Sample from log-normal distribution
                        # LogNormal(loc, scale) in PyTorch corresponds to
                        # exp(Normal(loc, scale)) which is lognormal with log-scale parameterization
                        normal_samples = np.random.normal(loc=log_loc, scale=0.1)
                        results[i] = np.exp(normal_samples)

            except Exception:
                # If integration fails, leave as NaN (already initialized)
                continue

        return results

