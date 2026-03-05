from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_TORCH_MSG = "torch not installed. Install with: pip install 'lfi[torch-cpu]' or 'lfi[torch-gpu]'"

from .base import BaseSimulator


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

        # Pixel scale compression: [0,1] -> [0.45, 0.55]
        image = image / 10.0 + 0.5

        # Add Gaussian noise
        key, subkey = jax.random.split(key)
        noise = self.sigma_noise * jax.random.normal(subkey, shape=(self.H, self.W))
        noisy_image = image + noise

        return noisy_image.flatten()

    def sample_pytorch(self, theta):  # noqa: F821
        """
        Simulates sensor output from a clean image by applying blur and noise.
        Args:
            theta: (N, D) - Flat clean images, where N is the batch size and D is the flattened image size.
        Returns:
            (N, D) - Flat noisy, blurred images.
        """
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
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
