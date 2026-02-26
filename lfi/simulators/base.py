from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable


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
        raise NotImplementedError("This simulator does not support NumPy sampling.")


    def sample_jax(self, theta: jnp.ndarray, seed: int):
        """
        Samples from the simulator using JAX.

        Args:
            theta: Parameters for the simulation, shape (dim, ).
            seed: Random seed for reproducibility.

        Returns:
            jnp.ndarray: Simulated observations, shape (dim_y).
        """
        raise NotImplementedError("This simulator does not support JAX sampling.")

    def sample_pytorch(self, theta):
        """
        Samples from the simulator using PyTorch.

        Args:
            theta (torch.Tensor): Parameters for the simulation, shape (batch_size, dim).
        Returns:
            torch.Tensor: Simulated observations, shape (batch_size, dim_y).
        """
        raise NotImplementedError("This simulator does not support PyTorch sampling.")

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

    def jax_cr_simulator_paired(self) -> Callable[[jnp.array, jnp.array], jnp.array]:
        # f: ((N, D_th), (N,)) -> (N, D_y)  — requires BS_th == BS_s, paired 1-1
        f = jax.vmap(self.return_jax_callable(), in_axes=(0, 0))
        return f

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
