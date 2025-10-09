from typing import Callable
import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np


class BaseSimulator:
    def __init__(self):
        self.informative_dims = None

    def set_informative_dims(self, informative_dims: np.array):
        self.informative_dims = informative_dims.astype(bool)

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        """Creates simulator without any batching

        Returns:
            simulator: callable(th: jnp.array, s: int) -> jnp.array
                th - (D_th,) array
                s - seed

                Returns:
                    y - (D_y,) array
        """
        pass

    def cr_simulator1(self) -> Callable[[jnp.array, int], jnp.array]:
        # f_1: ((BS_th, D_th), s) -> (BS_th, D_y)
        f_1 = jax.vmap(self.cr_simulator(), in_axes=(0, None))
        return f_1

    def cr_simulator2(self) -> Callable[[jnp.array, jnp.array], jnp.array]:
        # f_1: ((BS_th, D_th), s) -> (BS_th, D_y)
        f_1 = jax.vmap(self.cr_simulator(), in_axes=(0, None))
        # f_2: ((BS_th, D_th), (BS_s,)) -> (BS_s, BS_th, D_y)
        f_2 = jax.vmap(f_1, in_axes=(None, 0))
        return f_2

    def cr_jacobian(self):
        return jax.jacobian(self.cr_simulator())

    def cr_jacobian1(self):
        jacobian = self.cr_jacobian()
        jacobian = jax.vmap(jacobian, in_axes=(0, None))
        return jacobian

    def cr_jacobian2(self):
        jacobian = self.cr_jacobian()
        jacobian = jax.vmap(jacobian, in_axes=(0, None))
        jacobian = jax.vmap(jacobian, in_axes=(None, 0))
        return jacobian

    def cr_distance(self, informative_dims=None):
        gen = self.cr_simulator()

        def distance(th: jnp.array, s: jnp.array, y_0: jnp.array) -> jnp.array:
            y = gen(th, s)  # y: (D,)
            if self.informative_dims is not None:
                y = y[self.informative_dims] # y: (D1,)
                y_0 = y_0[self.informative_dims] # y_0: (D1,)
            diff = jnp.square(y - y_0)  # diff: (D1,)
            dist = jnp.mean(diff)  # sum: ()
            return dist
        return distance

    def cr_distance1(self):
        gen = self.cr_simulator1()

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

    def cr_distance2(self):
        gen = self.cr_simulator2()

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

    def cr_distance_grad(self):
        dist = self.cr_distance()
        d_grad = jax.value_and_grad(dist, argnums=0)
        return d_grad

    def cr_distance_grad1(self):
        d_grad = self.cr_distance_grad()
        d_grad = jax.vmap(d_grad, in_axes=(0, None, None))
        return d_grad

    def cr_distance_grad2(self):
        dist = self.cr_distance_grad()
        d_grad = jax.vmap(dist, in_axes=(0, None, None))
        d_grad = jax.vmap(d_grad, in_axes=(None, 0, None))
        return d_grad

    def cr_distance_hess(self):
        dist = self.cr_distance()
        d_hess = jax.hessian(dist, argnums=0)
        return d_hess

    def cr_distance_hess1(self):
        d_hess = self.cr_distance_hess()
        d_hess = jax.vmap(d_hess, in_axes=(0, None, None))
        return d_hess

    def cr_distance_hess2(self):
        dist = self.cr_distance_hess()
        d_hess = jax.vmap(dist, in_axes=(0, None, None))
        d_hess = jax.vmap(d_hess, in_axes=(None, 0, None))
        return d_hess


class Linear(BaseSimulator):
    def __init__(self, dim: int, sigma: float):
        self.dim = dim
        self.sigma = sigma
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))

            # propagate the key and the new subkey
            th = jnp.array(th)
            y = random.normal(key=subkey, shape=(self.dim,)) * self.sigma + th
            return y
        return simulator


class LinearDistractors(BaseSimulator):
    def __init__(self, dim: int, sigma: float, dim_distractors=None):
        self.dim = dim
        if dim_distractors is None:
            self.dim_distractors = 100-dim
        else:
            self.dim_distractors = dim_distractors
        self.sigma = sigma
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))

            # propagate the key and the new subkey
            th = jnp.array(th)
            y1 = random.normal(key=subkey, shape=(self.dim,)) * self.sigma + th

            key, subkey = random.split(key)
            sigma_distractor = 10
            th_distractor = random.uniform(key=subkey, shape=(self.dim_distractors,)) * 20 - 10
            y2 = random.normal(key=subkey, shape=(self.dim_distractors,)) * sigma_distractor + th_distractor
            y = jnp.concatenate([y1, y2])
            return y
        return simulator


class LinearMultiSamples(BaseSimulator):
    def __init__(self, dim: int, sigma: float, nof_samples: int = 2):
        self.dim = dim
        self.sigma = sigma
        self.nof_samples = nof_samples
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            y = (random.normal(subkey, shape=(self.nof_samples,  self.dim)) * self.sigma + th).flatten()
            return y
        return simulator


class Square(BaseSimulator):
    def __init__(self, dim: int, sigma: float):
        self.dim = dim
        self.sigma = sigma
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))

            # propagate the key and the new subkey
            th = jnp.array(th)
            fth = jnp.square(th)
            y = random.multivariate_normal(key=subkey, mean=fth, cov=np.eye(self.dim) * self.sigma**2)
            return y
        return simulator


class SquareDistractors(BaseSimulator):
    def __init__(self, dim: int, sigma: float, dim_distractors=None):
        self.dim = dim
        if dim_distractors is None:
            self.dim_distractors = 100-dim
        else:
            self.dim_distractors = dim_distractors
        self.sigma = sigma
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))

            # propagate the key and the new subkey
            th = jnp.array(th)
            fth = jnp.square(th)
            y1 = random.multivariate_normal(key=subkey, mean=fth, cov=np.eye(self.dim) * self.sigma**2)

            key, subkey = random.split(key)
            sigma_distractor = 10
            th_distractor = random.uniform(key=subkey, shape=(self.dim_distractors,)) * 20 - 10
            y2 = random.normal(key=subkey, shape=(self.dim_distractors,)) * sigma_distractor + th_distractor
            y = jnp.concatenate([y1, y2])
            return y
        return simulator


class SquareMultiSamples(BaseSimulator):
    def __init__(self, dim: int, sigma: float, nof_samples: int = 2):
        self.dim = dim
        self.sigma = sigma
        self.nof_samples = nof_samples
        self.dim_distractors = 100
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            fth = jnp.square(th)
            y1 = (random.normal(subkey, shape=(self.nof_samples,  self.dim)) * self.sigma + fth)

            sigma_distractor = 10
            y2 = random.normal(key=subkey, shape=(self.nof_samples, self.dim_distractors,)) * sigma_distractor
            y = jnp.concatenate([y1, y2], axis=-1).flatten()

            return y
        return simulator


class TwoCasesGaussian(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float):
        self.dim = dim
        self.sigma_1 = sigma_1
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
            key, subkey = random.split(key)
            y = jnp.where(
                u_1 == 0,
                random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),

                random.multivariate_normal(subkey, th+5, np.eye(self.dim) * self.sigma_1**2)
            )
            return y
        return simulator


class TwoCasesGaussianMultiSamples(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, nof_samples=2):
        self.dim = dim
        self.sigma_1 = sigma_1
        self.nof_samples = nof_samples
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            yy = []
            key, subkey = random.split(random.PRNGKey(s))
            for i in range(self.nof_samples):
                key, subkey = random.split(key)
                u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
                key, subkey = random.split(key)
                yy.append(
                    jnp.where(
                    u_1 == 0,
                    random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),

                    random.multivariate_normal(subkey, th+5, np.eye(self.dim) * self.sigma_1**2)
                )
                )
            y = jnp.concatenate(yy)
            return y
        return simulator


class TwoCasesGaussianDistractors(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, dim_distractors=None):
        self.dim = dim
        if dim_distractors is None:
            self.dim_distractors = 100-dim
        else:
            self.dim_distractors = dim_distractors
        self.sigma_1 = sigma_1
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
            key, subkey = random.split(key)
            y1 = jnp.where(
                u_1 == 0,
                random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),

                random.multivariate_normal(subkey, th+5, np.eye(self.dim) * self.sigma_1**2)
            )

            key, subkey = random.split(key)
            sigma_distractor = 10
            th_distractor = random.uniform(key=subkey, shape=(self.dim_distractors,)) * 20 - 10
            y2 = random.normal(key=subkey, shape=(self.dim_distractors,)) * sigma_distractor + th_distractor
            y = jnp.concatenate([y1, y2])
            return y
        return simulator


class TwoCasesGaussianMultiSamplesDistractors(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, nof_samples: int = 2, dim_distractors=None):
        self.dim = dim
        self.sigma_1 = sigma_1
        self.nof_samples = nof_samples
        if dim_distractors is None:
            self.dim_distractors = 100-dim
        else:
            self.dim_distractors = dim_distractors
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            yy = []
            key, subkey = random.split(random.PRNGKey(s))
            for i in range(self.nof_samples):
                key, subkey = random.split(key)
                u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
                key, subkey = random.split(key)
                y1 = jnp.where(
                    u_1 == 0,
                    random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),
                    random.multivariate_normal(subkey, th+5, np.eye(self.dim) * self.sigma_1**2)
                )

                key, subkey = random.split(key)
                sigma_distractor = 10
                th_distractor = random.uniform(key=subkey, shape=(self.dim_distractors,)) * 20 - 10
                y2 = random.normal(key=subkey, shape=(self.dim_distractors,)) * sigma_distractor + th_distractor
                y = jnp.concatenate([y1, y2])
                yy.append(y)
            y = jnp.concatenate(yy)
            return y
        return simulator

class CasesGaussian(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, sigma_2: float):
        self.dim = dim
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
            key, subkey = random.split(key)
            y = jnp.where(
                u_1 == 0,
                random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),

                random.multivariate_normal(subkey, -th, np.eye(self.dim) * self.sigma_2**2)
            )
            return y
        return simulator


class CasesGaussianDistractors(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, sigma_2: float, dim_distractors=None):
        self.dim = dim
        if dim_distractors is None:
            self.dim_distractors = 100-dim
        else:
            self.dim_distractors = dim_distractors
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
            key, subkey = random.split(key)
            y1 = jnp.where(
                u_1 == 0,
                random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),

                random.multivariate_normal(subkey, -th, np.eye(self.dim) * self.sigma_2**2)
            )

            key, subkey = random.split(key)
            sigma_distractor = 10
            th_distractor = random.uniform(key=subkey, shape=(self.dim_distractors,)) * 20 - 10
            y2 = random.normal(key=subkey, shape=(self.dim_distractors,)) * sigma_distractor + th_distractor
            y = jnp.concatenate([y1, y2])
            return y
        return simulator


class CasesGaussianMultiSamples(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, sigma_2: float, nof_samples: int = 2):
        self.dim = dim
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.nof_samples = nof_samples
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            # key, subkey = random.split(random.PRNGKey(s))
            # u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
            # key, subkey = random.split(random.PRNGKey(s))
            # y = jnp.where(
            #     u_1 == 0,
            #     (random.normal(subkey, shape=(self.nof_samples,  self.dim)) * self.sigma_1 + th).flatten(),
            #     (random.normal(subkey, shape=(self.nof_samples,  self.dim)) * self.sigma_2 - th).flatten()
            # )
            yy = []
            key, subkey = random.split(random.PRNGKey(s))
            for i in range(self.nof_samples):
                key, subkey = random.split(key)
                u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
                key, subkey = random.split(key)
                yy.append(jnp.where(
                    u_1 == 0,
                    random.multivariate_normal(
                        subkey, th, np.eye(self.dim) * self.sigma_1**2
                    ),
                    random.multivariate_normal(
                        subkey, -th, np.eye(self.dim) * self.sigma_2**2
                    )
                ))
            y = jnp.concatenate(yy)
            return y
        return simulator


class CasesGaussianMultiSamplesDistractors(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, sigma_2: float, nof_samples: int = 2, dim_distractors=None):
        self.dim = dim
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.nof_samples = nof_samples
        if dim_distractors is None:
            self.dim_distractors = 100-dim
        else:
            self.dim_distractors = dim_distractors
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            yy = []
            key, subkey = random.split(random.PRNGKey(s))
            for i in range(self.nof_samples):
                key, subkey = random.split(key)
                u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
                key, subkey = random.split(key)
                y1 = jnp.where(
                    u_1 == 0,
                    random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),
                    random.multivariate_normal(subkey, -th, np.eye(self.dim) * self.sigma_2**2)
                )

                key, subkey = random.split(key)
                sigma_distractor = 10
                th_distractor = random.uniform(key=subkey, shape=(self.dim_distractors,)) * 20 - 10
                y2 = random.normal(key=subkey, shape=(self.dim_distractors,)) * sigma_distractor + th_distractor
                y = jnp.concatenate([y1, y2])
                yy.append(y)
            y = jnp.concatenate(yy)
            return y
        return simulator


class SLCP(BaseSimulator):
    def __init__(self, dim: int):
        self.dim = dim
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            th = jnp.array(th)
            mu = th[:2]
            s1 = th[2]**2
            s2 = th[3]**2
            rh0 = jnp.tanh(th[4])
            eps = 0.000001
            cov = jnp.array([[s1**2+eps, rh0*s1*s2], [rh0*s1*s2, s2**2+eps]])
            # y = random.multivariate_normal(key=subkey, mean=mu, cov=cov, shape=[4,])
            y = random.multivariate_normal(key=subkey, mean=mu, cov=cov)
            y = y.reshape(-1)
            return y
        return simulator


class TwoMoons(BaseSimulator):
    def __init__(self, dim: int):
        self.dim = dim
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            r = random.normal(key=subkey, shape=(1,)) * 0.01 + 0.1
            key, subkey = random.split(key)
            a = random.uniform(key=subkey, shape=(1,), minval=-jnp.pi/2, maxval=jnp.pi/2)
            t1 = jnp.array([r[0] * jnp.cos(a[0]) + 0.25, r[0] * jnp.sin(a[0])])
            t2 = jnp.array([-jnp.abs(jnp.sum(th))/jnp.sqrt(self.dim), (-th[0] + th[1])/jnp.sqrt(self.dim)])
            y = t1 + t2
            return y
        return simulator


class CasesGaussianSBI(BaseSimulator):
    def __init__(self, dim: int, sigma_1: float, sigma_2: float):
        self.dim = dim
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        super().__init__()

    def cr_simulator(self) -> Callable[[jnp.array, int], jnp.array]:
        def simulator(th: jnp.array, s: int) -> jnp.array:
            key, subkey = random.split(random.PRNGKey(s))
            u_1 = random.randint(key=subkey, minval=0, maxval=2, shape=(1,))
            key, subkey = random.split(random.PRNGKey(s))
            y = jnp.where(
                u_1 == 0,
                random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_1**2),

                random.multivariate_normal(subkey, th, np.eye(self.dim) * self.sigma_2**2)
            )
            return y
        return simulator
