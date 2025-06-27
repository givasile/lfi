from timeit import default_number

import numpy as np
import jax
import timeit
from typing import Callable, Optional, Tuple, Union
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from .base import InferenceBase

import lfi.simulators

Dy: int  # dimension of simulator's output
D: int  # dimension of simulator's input
N: int  # number of samples
S1: int  # number of seeds to generate
S: int  # number of seeds to accept
TH0: int  # number of theta_0 per seed to generate
L: int # number of line search steps


class R2OMC(InferenceBase):
    def __init__(
            self,
            prior: lfi.priors.BasePrior,
            simulator: lfi.simulators.BaseSimulator,
            observation: np.ndarray[Tuple[int]],
    ):
        """

        Args:
            simulator:
            observation:
            prior:
            dim:
        """
        dim = prior.dim
        dim_y = observation.shape[-1]

        # Step: __init__
        self.sim = simulator
        self.sim_1 = jax.jit(simulator.jax_cr_simulator1())
        self.sim_2 = jax.jit(simulator.jax_cr_simulator2())
        self.sim_jac = jax.jit(simulator.jax_cr_jacobian())
        self.sim_jac_1 = jax.jit(simulator.jax_cr_jacobian1())
        self.sim_jac_2 = jax.jit(simulator.jax_cr_jacobian2())
        self.dist_1 = jax.jit(simulator.jax_cr_distance1())
        self.dist_2 = jax.jit(simulator.jax_cr_distance2())
        self.dist_grad_1 = jax.jit(simulator.jax_cr_distance_grad1())
        self.dist_grad_2 = jax.jit(simulator.jax_cr_distance_grad2())
        self.dist_hessian_1 = jax.jit(simulator.jax_cr_distance_hess1())
        self.get_traj = jax.jit(self._optimize_one_seed, static_argnums=(3, 4))
        self.D = dim
        self.Dy = observation.shape[-1]
        self.y_0 = observation[0]
        self.prior = prior

        # Step: find_informative_dims
        self.inf_dims_nof_th: Optional[int] = None
        self.inf_dims_nof_seeds: Optional[int] = None
        self.informative_dims: np.ndarray[bool] = np.ones(self.Dy)

        # Step: create_objective_functions
        self.nof_seeds_total: Optional[int] = None
        self.nof_th0: Optional[int] = None
        self.seeds_init: Optional[np.ndarray[S1]] = None  # (S1,)
        self.th0_init: Optional[np.ndarray[Tuple[S1, TH0, D]]] = None  # (S1, TH0, D)
        self.d0_init: Optional[np.ndarray[Tuple[S1, TH0]]] = None  # (S1, TH0)

        # Step: optimize
        self.nof_gd_steps: Optional[int] = None
        self.alpha: Optional[float] = None
        self.th_star_init: Optional[np.ndarray[Tuple[S1, TH0, D]]] = None
        self.d_star_init: Optional[np.ndarray[Tuple[S1, TH0]]] = None

        # Step: filter_solutions
        self.nof_seeds_accept: Optional[int] = None
        self.eps_1: Optional[float] = None
        self.seeds: Optional[np.ndarray[S]] = None
        self.th0: Optional[np.ndarray[S, TH0, D]] = None
        self.th_star: Optional[np.ndarray[S, TH0, D]] = None
        self.d_star: Optional[np.ndarray[S, TH0]] = None

        # Step: get_directions
        self.hessians: Optional[np.ndarray[S, TH0, D, D]] = None
        self.eig_val: Optional[np.ndarray[S, TH0, D]] = None
        self.eig_vec: Optional[np.ndarray[S, TH0, D, D]] = None

        # Step: get_boxes
        self.eps_2: Optional[float] = None
        self.nof_ls_steps = None
        self.step_size = None
        self.limits: Optional[np.ndarray[S, TH0, D, 2]] = None
        self.volumes: Optional[np.ndarray[S, TH0]] = None
        self.weights: Optional[np.ndarray[S, TH0]] = None

        # Step: weight_sample
        self.eps_3: Optional[float] = None
        self.samples_per_region: Optional[int] = None
        self.samples: Optional[np.ndarray[S, TH0, N, D]] = None
        self.samples_weights: Optional[np.ndarray[S, TH0, N]] = None
        self.samples_flat: Optional[np.ndarray[S * TH0 * N, D]] = None
        self.samples_weights_flat: Optional[np.ndarray[S * TH0 * N]] = None

        super().__init__("R2OMC", prior, simulator, observation, dim, dim_y)

    def find_informative_dims(
            self,
            key: jnp.ndarray = jax.random.PRNGKey(21),
            inf_dims_nof_th: int = 100,
            inf_dims_nof_seeds: int = 5
    ):
        # input
        self.inf_dims_nof_th = inf_dims_nof_th
        self.inf_dims_nof_seeds = inf_dims_nof_seeds

        # function
        key, subkey = jax.random.split(key)
        thetas = self.prior.sample_jax(subkey, shape=[inf_dims_nof_th]) # (inf_dims_nof_th, D)
        key, subkey = jax.random.split(key)
        seeds = jax.random.randint(subkey, (inf_dims_nof_seeds,), 0, 2**31-1) # (inf_dims_nof_seeds,)
        dy_dth = jnp.abs(self.sim_jac_2(thetas, seeds)) # (inf_dims_nof_seeds, inf_dims_nof_th, Dy, D)
        inf_dims = dy_dth.sum(axis=[0, 1, 3]) > 1e-3
        assert inf_dims.shape == (self.Dy,)

        # output
        self.informative_dims = inf_dims
        self.sim.set_informative_dims(self.informative_dims)
        return key

    def create_objective_functions(
            self,
            key,
            nof_seeds_total: int = 500,
            nof_th0: int = 10
    ):
        # input
        self.nof_seeds_total = nof_seeds_total
        self.nof_th0 = nof_th0

        # function
        key, subkey = jax.random.split(key)
        seeds_init = np.array(jax.random.randint(subkey, (nof_seeds_total,), 0, 2**31-1)) # (nof_seeds_total,)
        key, subkey = jax.random.split(key)
        th0_init = self.prior.sample_jax(subkey, [nof_seeds_total, nof_th0]) # (nof_seeds_total, nof_th0, D)
        d0_init = np.array([self.dist_1(th0_init[i], seeds_init[i], self.y_0) for i in range(nof_seeds_total)]) # (nof_seeds_total, nof_th0)

        # output
        self.seeds_init = seeds_init
        self.th0_init = np.array(th0_init)
        self.d0_init = np.array(d0_init)
        self.th_star_init = np.array(th0_init)
        self.d_star_init = np.array(d0_init)
        return key

    def _optimize_one_seed(self, th0: jnp.ndarray, seed: int, y_0: np.ndarray, nof_gd_steps: int = 100, alpha: float = 0.1):
        optimizer = optax.adam(learning_rate=alpha)
        optimizer_state = optimizer.init(th0)
        for i in range(nof_gd_steps):
            d, dth = self.dist_grad_1(th0, seed, y_0)
            updates, optimizer_state = optimizer.update(dth, optimizer_state)
            th0 = optax.apply_updates(th0, updates)
        d, dth = self.dist_grad_1(th0, seed, y_0)
        return th0, d

    def optimize(self, nof_gd_steps=100, alpha=0.01):
        # input
        self.nof_gd_steps = nof_gd_steps
        self.alpha = alpha

        # function
        th_star_init = self.th_star_init
        d_star_init = self.d_star_init
        for i, ss in enumerate(tqdm(self.seeds_init)):
            th_c, d_c = self.get_traj(jnp.array(th_star_init[i]), ss, self.y_0, nof_gd_steps, alpha)
            th_star_init[i] = np.array(th_c)
            d_star_init[i] = np.array(d_c)

        # output
        self.th_star_init = th_star_init
        self.d_star_init = d_star_init

    def filter_solutions(self, nof_seeds_accept: int = 500):
        # input
        assert nof_seeds_accept <= self.nof_seeds_total
        self.nof_seeds_accept = nof_seeds_accept

        # step 1: keep seeds with at least one theta_star inside prior
        th_star_flat = self.th_star_init.reshape((self.nof_seeds_total * self.nof_th0, self.D))
        th_star_inside_prior = self.prior.has_mass(th_star_flat).reshape((self.nof_seeds_total, self.nof_th0))
        seed_inside_prior = np.sum(th_star_inside_prior, axis=1) > 0
        assert np.sum(seed_inside_prior) >= nof_seeds_accept, "Not enough seeds with at least one theta_star inside prior mass"
        seeds = self.seeds_init[seed_inside_prior]
        th_star = self.th_star_init[seed_inside_prior]
        d_star = self.d_star_init[seed_inside_prior]

        # step 2: sort by the best d_star and select the best nof_seeds_accept
        best_dist_per_seed = d_star.min(axis=1) # best solution for each seed
        accepted_indices = np.argsort(best_dist_per_seed)[:nof_seeds_accept] # indices of the best solutions

        # output
        self.eps_1 = best_dist_per_seed[accepted_indices][-1]
        self.seeds = seeds[accepted_indices]
        self.th_star = th_star[accepted_indices]
        self.d_star = d_star[accepted_indices]

    def get_directions(self):
        self.hessians = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, self.D))
        self.eig_val = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D))
        self.eig_vec = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, self.D))
        for i, ss in enumerate(self.seeds):
            self.hessians[i] = np.array(self.dist_hessian_1(self.th_star[i], ss, self.y_0))
            self.eig_val[i], self.eig_vec[i] = np.linalg.eig(self.hessians[i])

    def check_eps_2(self, dx: float = 0.1):
        eps_2 = 0.1
        L = 1
        step_size = dx
        dd = self.get_boxes(eps_2, L, step_size, return_all=True)
        return dd[:, :, :, [0, 2]]

    def _process_limit(self, is_inside_eps, step_size):
        L = is_inside_eps.shape[-1] - 1
        is_outside = is_inside_eps == 0
        all_inside = np.sum(is_outside, -1) == 0

        last_inside = (np.argmax(is_outside, axis=-1) - 1).astype(float)
        last_inside[all_inside] = L # if all are inside, then the max is L
        last_inside[last_inside == 0] = 0.5  # if th_0 + step is false, then the max is 0.5
        last_inside[last_inside == -1] = 0.5  # if th_0 is false, then the max is 0.5
        return last_inside * step_size

    def get_boxes(
            self,
            eps_2: float,
            nof_ls_steps: int = 10,
            step_size: float = 0.1,
            return_all: bool = False
    ):
        self.step_size = step_size
        self.nof_ls_steps = nof_ls_steps
        self.eps_2 = eps_2
        self.limits = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, 2))
        distances = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, 2 * nof_ls_steps + 1))
        for i in tqdm(range(self.nof_seeds_accept)):
            # find distances
            query_points = np.zeros((self.nof_th0, self.D, 2 * nof_ls_steps + 1, self.D))
            th_star = self.th_star[i]
            for dim in range(self.D):
                steps = np.zeros((2 * nof_ls_steps + 1, self.D))
                steps[:, dim] = np.linspace(-nof_ls_steps, nof_ls_steps, 2 * nof_ls_steps + 1) * step_size
                steps_vec = []
                for bs_th in range(self.nof_th0):
                    rotated = np.dot(steps, self.eig_vec[i, bs_th])
                    translation = np.expand_dims(th_star[bs_th, :], 0)
                    affine = rotated + translation
                    steps_vec.append(affine)
                query_points[:, dim, :, :] = np.array(steps_vec)
            query_points_flat = query_points.reshape(self.nof_th0 * self.D * (2 * nof_ls_steps + 1), self.D)
            distances_seed = self.dist_1(query_points_flat, self.seeds[i], self.y_0)
            distances_seed = distances_seed.reshape(self.nof_th0, self.D, 2 * nof_ls_steps + 1)
            distances[i] = distances_seed

            # find min and max translation
            is_inside_seed = np.array(distances_seed <= self.eps_2)
            sliced_is_inside_seed = is_inside_seed[..., :nof_ls_steps + 1]
            reversed_sliced_dd_ind = np.flip(sliced_is_inside_seed, axis=-1)
            self.limits[i, :, :, 0] = -1 * self._process_limit(reversed_sliced_dd_ind, step_size)
            sliced_is_inside_seed = is_inside_seed[..., nof_ls_steps:]
            self.limits[i, :, :, 1] = self._process_limit(sliced_is_inside_seed, step_size)

        self.volumes = np.prod(self.limits[:, :, :, 1] - self.limits[:, :, :, 0], axis=(2))
        self.weights = self.volumes / np.sum(self.volumes)

        if return_all:
            return distances

    def is_inside_region(self, samples, i_seed, i_th, eps_3):
        # # input
        # limits = self.limits[i_seed, i_th]
        # rotation = self.eig_vec[i_seed, i_th]
        # center = self.th_star[i_seed, i_th]
        # volume = self.volumes[i_seed, i_th]
        #
        # # function
        # samples_rot = np.dot(samples, rotation.T) - center
        # term1 = samples_rot >= limits[:, 0]
        # term2 = samples_rot <= limits[:, 1]
        # is_inside_1 = np.all(term1, axis=-1) * np.all(term2, axis=-1)

        is_inside_2 = self.dist_1(samples, self.seeds[i_seed], self.y_0) < eps_3
        weights = self.prior.pdf(samples) * is_inside_2 # * is_inside_1 * volume
        return weights

    def _get_samples_per_region(self, key, nof_samples, i_seed, i_th, eps_3):
        limits = self.limits[i_seed, i_th]
        rotation = self.eig_vec[i_seed, i_th]
        center = self.th_star[i_seed, i_th]
        volume = self.volumes[i_seed, i_th]

        samples_init = jax.random.uniform(key, shape=(nof_samples, self.D), minval=limits[:, 0], maxval=limits[:, 1])
        samples_rot = jax.numpy.dot(samples_init, rotation)
        samples_rot = samples_rot + center

        is_inside = self.dist_1(samples_rot, self.seeds[i_seed], self.y_0) < eps_3
        w_1 = self.prior.pdf(samples_rot) > 0
        # w_1 = self.prior.pdf(samples_rot)
        weights = w_1 * is_inside * volume # * (limits[:,1] - limits[:,0]).mean() # volume
        return samples_rot, weights

    def weight_sample(self, seed, nof_samples, eps_3=10.):
        self.eps_3 = eps_3
        samples_per_region = int(np.ceil(nof_samples / (self.nof_seeds_accept * self.nof_th0)))

        self.samples_per_region = samples_per_region
        key = jax.random.PRNGKey(seed)
        samples = []
        weights = []
        for i_seed in range(self.seeds.shape[0]):
            samples.append([])
            weights.append([])
            for i_th in range(self.nof_th0):
                key, subkey = jax.random.split(key)
                reg_samples, reg_weights = self._get_samples_per_region(
                    subkey,
                    samples_per_region,
                    i_seed,
                    i_th,
                    eps_3
                )
                samples[-1].append(reg_samples)
                weights[-1].append(reg_weights)

        samples = np.array(samples)
        self.samples = samples
        self.samples_flat = samples.reshape((self.nof_seeds_accept * self.nof_th0 * samples_per_region, self.D))

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        self.samples_weights = weights
        self.samples_weights_flat = weights.reshape((self.nof_seeds_accept * self.nof_th0 * samples_per_region,))

        return self.samples_flat, self.samples_weights_flat

    @staticmethod
    def importance_resampling(samples, weights, nof_samples, replace=True):
        if np.sum(weights > 0) < nof_samples:
            print("Not enough samples with positive weight")
            return samples[np.argsort(weights)[::-1]][:nof_samples]
        indices = np.random.choice(
            np.arange(samples.shape[0]),
            size=nof_samples,
            replace=replace,
            p=weights
        )
        return samples[indices]

    def get_proposal_1d(self, th: np.ndarray, x_dim: int, return_indices=False):
        """Return the proposal distribution for a given dimension
        Be careful, we evaluate the proposal distribution on the given dimension only

        Args:
            th: the samples
            x_dim: the dimension to evaluate the proposal distribution
        """
        # vectorized version
        th_res = np.tile(th[:, np.newaxis, np.newaxis], (1, self.seeds.shape[0], self.nof_th0))
        rot_limits = self.eig_vec @ self.limits # (S, TH0, D, 2)
        tr_lims = rot_limits
        tr_lims[:, :, :, 0] += self.th_star
        tr_lims[:, :, :, 1] += self.th_star
        ind = np.logical_and(th_res > tr_lims[:, :, x_dim, 0], th_res < tr_lims[:, :, x_dim, 1])
        y = np.sum(ind, axis=(1, 2))
        y = y > 0
        if return_indices:
            return y, ind
        return y

    def get_proposal_2d(self, th: np.ndarray, x_dims: Tuple[int, int]):
        """Return the proposal distribution on the two given dimensions
        Be careful, we evaluate the proposal distribution on the given dimensions only

        Args:
            th: the points to eval
            x_dims: the dimension to evaluate the proposal distribution
        """
        # vectorized version
        y_1, ind_1 = self.get_proposal_1d(th[:, 0], x_dims[0], return_indices=True)
        y_2, ind_2 = self.get_proposal_1d(th[:, 1], x_dims[1], return_indices=True)
        ind = np.logical_and(ind_1, ind_2)
        y = ind.sum(-1).sum(-1)
        y = y > 0
        return y

    def plot_proposal_2d(self, th_dims, lims, nof_points):
        th1 = np.linspace(lims[0,0], lims[0,1], nof_points)
        th2 = np.linspace(lims[1,0], lims[1,1], nof_points)
        th = np.meshgrid(th1, th2)
        th = np.dstack(th).reshape(-1, 2)
        yy = self.get_proposal_2d(th, th_dims)

        plt.figure()
        plt.title("Proposal region")
        plt.xlim(lims[0,0], lims[0,1])
        plt.ylim(lims[1,0], lims[1,1])
        plt.scatter(
            th[yy > 0, 0],
            th[yy > 0, 1],
            color="r",
            alpha=.5,
        )
        plt.show(block=False)

    def fit(self, budget: int = 1000, fit_kwargs: Optional[dict] = None):
        default_kwargs = {
            "fit_seed": 21,
            "find_informative_dims": True,
            "inf_dims_nof_th": 100,
            "inf_dims_nof_seeds": 50,
            "nof_th0": 1,
            "nof_seeds_total": 1000,
            "nof_seeds_accept": 1000,
            "nof_gd_steps": 50,
            "alpha": 1.0,
            "epochs": 8,
            "dx": 1.0,
            "eps_2": None,  # will be checked
            "nof_ls_steps": 100,
            "step_size": .2,
            # "sample_seed": 71,
            # "nof_samples": 5000,
            # "apply_importance_resampling": True,
            # "nof_samples_final": 1000
        }

        default_kwargs.update(fit_kwargs or {})
        fit_kwargs = default_kwargs

        # assert_fit_kwargs(fit_kwargs)
        key = jax.random.PRNGKey(fit_kwargs["fit_seed"])

        # Step: find informative dimensions
        if fit_kwargs["find_informative_dims"]:
            key, subkey = jax.random.split(key)
            key = self.find_informative_dims(
                key,
                fit_kwargs["inf_dims_nof_th"],
                fit_kwargs["inf_dims_nof_seeds"]
            )

        # Step: create objective functions
        key = self.create_objective_functions(
            key,
            fit_kwargs["nof_seeds_total"],
            fit_kwargs["nof_th0"]
        )

        # Step: optimize
        for i in range(fit_kwargs["epochs"]):
            self.optimize(
                fit_kwargs["nof_gd_steps"],
                fit_kwargs["alpha"]
            )

        # Step: filter_solutions
        self.filter_solutions(
            fit_kwargs["nof_seeds_accept"]
        )

        # Step: get_directions
        self.get_directions()

        # Step: get_boxes
        if "dx" in fit_kwargs.keys():
            eps_2 = self.check_eps_2(fit_kwargs["dx"]).mean()
        elif "eps_2" in fit_kwargs.keys():
            eps_2 = fit_kwargs["eps_2"]
        else:
            raise ValueError("dx or eps_2 must be in fit_kwargs")
        self.get_boxes(eps_2, fit_kwargs["nof_ls_steps"], fit_kwargs["step_size"])

    def sample(self, nof_samples: int = 100, sample_kwargs: Optional[dict] = None):
        default_kwargs = {
            "sample_seed": 71,
            "eps_3": 10.0,
            "nof_initial_samples": 1000,
            "apply_importance_resampling": True
        }
        default_kwargs.update((sample_kwargs or {}))
        sample_kwargs = default_kwargs

        # Step: weight_sample
        self.samples, self.weights = self.weight_sample(
            sample_kwargs["sample_seed"],
            sample_kwargs["nof_initial_samples"],
            sample_kwargs["eps_3"]
        )

        samples_r2omc = self.importance_resampling(
            self.samples, self.weights, nof_samples, False)
        return samples_r2omc
