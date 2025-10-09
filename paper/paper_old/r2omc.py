import numpy as np
import jax
import timeit
from typing import Callable, Optional, Tuple, Union
import simulators
import priors
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

Dy: int  # dimension of simulator's output
D: int  # dimension of simulator's input
N: int  # number of samples
S1: int  # number of seeds to generate
S: int  # number of seeds to accept
TH0: int  # number of theta_0 per seed to generate
L: int # number of line search steps


class R2OMC:
    def __init__(
            self,
            sim: simulators.BaseSimulator,
            y_0: np.ndarray[Tuple[int]],
            prior: priors.BasePrior,
            dim: Optional[int] = None,
    ):
        """

        Args:
            sim:
            y_0:
            prior:
            dim:
        """
        # Step: __init__
        self.sim = sim
        self.sim_1 = jax.jit(sim.cr_simulator1())
        self.sim_2 = jax.jit(sim.cr_simulator2())
        self.sim_jac = jax.jit(sim.cr_jacobian())
        self.sim_jac_1 = jax.jit(sim.cr_jacobian1())
        self.sim_jac_2 = jax.jit(sim.cr_jacobian2())
        self.dist_1 = jax.jit(sim.cr_distance1())
        self.dist_2 = jax.jit(sim.cr_distance2())
        self.dist_grad_1 = jax.jit(sim.cr_distance_grad1())
        self.dist_grad_2 = jax.jit(sim.cr_distance_grad2())
        self.dist_hessian_1 = jax.jit(sim.cr_distance_hess1())
        self.get_traj = jax.jit(self._optimize_one_seed, static_argnums=(3, 4))
        self.D = y_0.shape[-1] if dim is None else dim
        self.Dy = y_0.shape[-1]
        self.y_0 = y_0
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
        thetas = self.prior.sample(subkey, shape=[inf_dims_nof_th]) # (inf_dims_nof_th, D)
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
        th0_init = self.prior.sample(subkey, [nof_seeds_total, nof_th0]) # (nof_seeds_total, nof_th0, D)
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

    def infer(
            self,
            config: dict
    ):
        assert_config(config)
        key = jax.random.PRNGKey(config["fit_seed"])

        # Step: find informative dimensions
        if config["find_informative_dims"]:
            key, subkey = jax.random.split(key)
            key = self.find_informative_dims(
                key,
                config["inf_dims_nof_th"],
                config["inf_dims_nof_seeds"]
            )

        # Step: create objective functions
        key = self.create_objective_functions(
            key,
            config["nof_seeds_total"],
            config["nof_th0"]
        )

        # Step: optimize
        for i in range(config["epochs"]):
            self.optimize(
                config["nof_gd_steps"],
                config["alpha"]
            )

        # Step: filter_solutions
        self.filter_solutions(
            config["nof_seeds_accept"]
        )

        # Step: get_directions
        self.get_directions()

        # Step: get_boxes
        if "dx" in config and not "eps_2" in config:
            eps_2 = self.check_eps_2(config["dx"]).mean()
        elif "eps_2" in config:
            eps_2 = config["eps_2"]
        else:
            raise ValueError("dx or eps_2 must be in config")
        self.get_boxes(eps_2, config["nof_ls_steps"], config["step_size"])

        # Step: weight_sample
        samples, weights = self.weight_sample(
            config["sample_seed"],
            config["nof_samples"],
            config["eps_3"]
        )

        return samples, weights


Ny: int  # number of observations
N_th_k: int  # nof posterior samples per observation
N_th_total: int  # nof posterior samples in total (N_th_k * Ny)
N_th_accept: int  # nof accepted posterior samples
N_th_select: int  # nof selected samples


class IterativeR2OMC:
    def __init__(
            self,
            sim: simulators.BaseSimulator,
            y_0: np.ndarray[Tuple[int, int]],
            prior: priors.BasePrior,
            dim: Optional[int] = None,
    ):
        """

        Args:
            sim: simulator
            y_0: observations
            prior: prior
            dim: dimension of theta
        """
        self.Ny = y_0.shape[0] # number of observations
        self.Dy = y_0.shape[1] # dimension of simulator's output
        self.r2omc_list = [R2OMC(sim, y_0[i], prior, dim) for i in range(self.Ny)]

        self.N_th_per_obs: Optional[int] = None  # nof posterior samples per observation
        self.N_th_total: Optional[int] = None  # nof posterior samples in total (N_th_per_obs * Ny)
        self.N_th_accept: Optional[int] = None  # nof accepted posterior samples
        self.N_th_select: Optional[int] = None  # nof selected samples
        self.th_total: Optional[np.ndarray[N_th_total, D]] = None  # (N_th_total, D)
        self.w_before: Optional[np.ndarray[N_th_total]] = None  # (N_th_total,)
        self.w_after: Optional[np.ndarray[N_th_total]] = None  # (N_th_total,)
        self.th_accepted: Optional[np.ndarray[N_th_accept, D]] = None  # (N_th_accept, D)
        self.th_selected: Optional[np.ndarray[N_th_select, D]] = None  # (N_th_select, D)

    def infer(self, config):
        # input
        Ny = self.Ny # number of observations
        N_s = config["nof_seeds_accept"] # number of seeds to accept
        N_th0 = config["nof_th0"] # number of theta_0 per seed to generate
        N_th_per_obs = config["nof_samples"] # number of posterior samples per observation
        N_th_total = config["nof_samples"] * Ny # number of posterior samples in total (N_th_per_obs * Ny)
        N_th_select = config["nof_samples_to_select"] # number of samples to select from the accepted samples

        if self.th_total is None:
            # new seed for each observation
            seed_per_obs = np.random.randint(0, 2**31-1, (Ny,)) # (Ny,)
            config_list = [config.copy() for i in range(Ny)]
            for i in range(Ny):
                config_list[i]["fit_seed"] = seed_per_obs[i]
            # config_list = [config.copy() for i in range(Ny)]
            th_total, w_before = zip(*(r2omc.infer(config_list[i]) for i, r2omc in enumerate(self.r2omc_list)))

            th_total = np.concatenate(th_total, axis=0) # (N_th_total, D)
        else:
            print("Already inferred")
            th_total = self.th_total
            w_before = self.w_before

        dist_func = self.r2omc_list[0].dist_2
        accept_per_obs = [np.sum(dist_func(th_total, cur.seeds, cur.y_0) < config["eps_3"], axis=0) for i, cur in enumerate(self.r2omc_list)]
        accept_per_obs = np.stack(accept_per_obs, axis=1) # (N_th_total, Ny)

        self.accept_per_obs = accept_per_obs
        # w_from_dist = np.prod(accept_per_obs, axis=1) # (N_th_total,)
        w_from_dist = np.exp(np.sum(np.log(accept_per_obs), axis=1))
        w_from_prior = self.r2omc_list[0].prior.pdf(th_total)  # (N_th_total,)
        N_th_accept = np.sum(w_from_dist*w_from_prior > 0)
        self.accepted_indices = np.where(w_from_dist*w_from_prior > 0)[0]


        # breakpoint()

        # w_before = np.concatenate(w_before, axis=0) # (N_th_total,)
        # w_after = np.zeros((N_th_total,)) # (N_th_total,)
        # is_inside_region = np.zeros((N_th_total, Ny, N_s, N_th0)) # (N_th_total, Ny, N_s, N_th0)
        # for i_y, r2omc in enumerate(self.r2omc_list):
        #     for i_seed in range(r2omc.seeds.shape[0]):
        #         for i_th in range(r2omc.nof_th0):
        #             current_is_inside = r2omc.is_inside_region(th_total, i_seed, i_th, config["eps_3"]) > 0
        #             is_inside_region[:, i_y, i_seed, i_th] = current_is_inside
        #             w_after[current_is_inside] += 1
        # nof_regions_per_obs = np.sum(is_inside_region, axis=(2, 3))
        # is_accepted = np.all(nof_regions_per_obs > 0, axis=1) # (N_th_total,)
        # N_th_accept = np.sum(is_accepted)

        if N_th_accept < N_th_select:
            print(f"Not enough samples to select. Only {N_th_accept} out of {N_th_total} are accepted")
            th_accepted = None
            th_selected = None
            w_after = np.zeros((N_th_total,))
        else:
            print(f"Accepted {N_th_accept} out of {N_th_total} posterior samples")
            w_after = w_from_dist * w_from_prior
            w_after /= w_after.sum() # (N_th_total,)

            # check if there are enough non-zero weights
            if np.sum(w_after > 0) < N_th_select:
                print(f"Not enough samples to select. Only {np.sum(w_after > 0)} out of {N_th_total} are accepted")
                # return all accepted samples
                th_accepted = th_total[w_after > 0]
                th_selected = th_total[w_after > 0]
            else:
                selected_indices = np.random.choice(
                    np.arange(N_th_total),
                    size=N_th_select,
                    replace=False,
                    p=w_after
                ) # (N_th_select,)

                th_accepted = th_total[w_after > 0]
                th_selected = th_total[selected_indices]

        # output
        self.th_accepted = th_accepted
        self.th_selected = th_selected
        self.th_total = th_total
        self.w_before = w_before
        self.w_after = w_after
        self.N_th_per_obs = N_th_per_obs
        self.N_th_total = N_th_total
        self.N_th_accept = N_th_accept
        self.N_th_select = N_th_select
        return th_selected

    def get_proposal_1d(self, th, th_dim):
        y = np.zeros_like(th)
        for i_y, r2omc in enumerate(self.r2omc_list):
            y += r2omc.get_proposal_1d(th, th_dim)
        return y

    def get_proposal_2d(self, th, th_dims):
        y = np.ones(th.shape[0])
        for i_y, r2omc in enumerate(self.r2omc_list):
            y *= r2omc.get_proposal_2d(th, th_dims)
        return y

    def plot_proposal_2d(self, th_dims, lims, nof_points):
        th1 = np.linspace(lims[0,0], lims[0,1], nof_points)
        th2 = np.linspace(lims[1,0], lims[1,1], nof_points)
        th = np.meshgrid(th1, th2)
        th = np.dstack(th).reshape(-1, 2)
        yy = self.get_proposal_2d(th, th_dims)

        plt.figure()
        plt.title("Merged proposal region")
        plt.xlim(lims[0,0], lims[0,1])
        plt.ylim(lims[1,0], lims[1,1])
        plt.scatter(
            th[yy > 0, 0],
            th[yy > 0, 1],
            color="r",
            alpha=.5,
        )
        plt.show(block=False)

    def plot(self, th_dims, obs_ind, xlim, ylim):
        colours = ["r", "b", "y", "m", "c"]
        plt.figure()
        plt.title("Posterior samples")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(f"th_{th_dims[0]}")
        plt.ylabel(f"th_{th_dims[1]}")
        for i, ind in enumerate(obs_ind):
            start = self.N_th_per_obs * ind
            stop = self.N_th_per_obs * (ind + 1)
            plt.scatter(
                self.th_total[start:stop, th_dims[0]],
                self.th_total[start:stop, th_dims[1]],
                # c=self.w_before[start:stop],
                alpha=.5,
                color=colours[i],
                label="observation {}".format(ind+1)
            )
        if self.th_selected is not None:
            plt.scatter(
                self.th_selected[:, th_dims[0]],
                self.th_selected[:, th_dims[1]],
                color="g",
                alpha=.5,
                label="selected"
            )
        # plt.colorbar()
        plt.legend()
        plt.show(block=False)



def assert_config(config):
    assert "fit_seed" in config
    assert "find_informative_dims" in config
    assert "inf_dims_nof_th" in config
    assert "inf_dims_nof_seeds" in config
    assert "nof_seeds_total" in config
    assert "nof_th0" in config
    assert "nof_gd_steps" in config
    assert "alpha" in config
    assert "epochs" in config
    assert "nof_seeds_accept" in config
    assert "eps_2" or "dx" in config
    assert "nof_ls_steps" in config
    assert "step_size" in config
    assert "sample_seed" in config
    assert "nof_samples" in config
    assert "eps_3" in config
