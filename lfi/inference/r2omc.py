import numpy as np
import jax
from typing import Optional, Tuple
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from pyabc.distance.util import log_weights

from .base import InferenceBase

import lfi.simulators


Dy: int  # dimension of simulator's output
D: int  # dimension of simulator's input

S_gen: int  # number of seeds to generate
N_TH0: int  # number of theta_0 per seed to generate

S_accept: int  # number of seeds to accept
N_th0: int  # number of theta_0 per seed to generate

N_per_region: int  # number of samples per region
N: int  # number of samples

class R2OMC(InferenceBase):
    def __init__(
            self,
            prior: lfi.priors.BasePrior,
            simulator: lfi.simulators.BaseSimulator,
            observation: np.ndarray, # (N, Dy)
    ):
        """ R2OMC Inference class for the R2OMC algorithm.
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
        self.informative_dims: np.ndarray = np.ones(self.Dy) # (Dy,)

        # Step: create_objective_functions
        self.seeds_init: Optional[np.ndarray] = None  # (S1,)
        self.th0_init: Optional[np.ndarray] = None  # (S1, TH0, D)
        self.d0_init: Optional[np.ndarray] = None  # (S1, TH0)

        # Step: optimize
        self.nof_gd_steps: Optional[int] = None
        self.alpha: Optional[float] = None
        self.th_star_init: Optional[np.ndarray] = None
        self.d_star_init: Optional[np.ndarray] = None

        # Step: filter_solutions_inside_prior
        self.nof_seeds_total: Optional[int] = None
        self.nof_seeds_inside_prior: Optional[int] = None
        self.seeds_inside_prior: Optional[np.ndarray] = None
        self.th_star_inside_prior: Optional[np.ndarray] = None
        self.d_star_inside_prior: Optional[np.ndarray] = None

        # Step: filter_solutions
        self.nof_seeds_accept: Optional[int] = None
        self.eps_1: Optional[float] = None
        self.seeds: Optional[np.ndarray] = None
        self.th0: Optional[np.ndarray] = None
        self.th_star: Optional[np.ndarray] = None
        self.d_star: Optional[np.ndarray] = None

        # Step: get_directions
        self.hessians: Optional[np.ndarray] = None
        self.eig_val: Optional[np.ndarray] = None
        self.eig_vec: Optional[np.ndarray] = None

        # Step: get_boxes
        self.eps_2: Optional[float] = None
        self.nof_ls_steps = None
        self.step_size = None
        self.limits: Optional[np.ndarray] = None
        self.log_volumes: Optional[np.ndarray] = None
        # self.weights: Optional[np.ndarray] = None

        # Step: weight_sample
        self.samples: Optional[np.ndarray] = None # (S, N_th0, N_per_region, D) samples
        self.samples_weights: Optional[np.ndarray] = None # (S, N_th0, N_per_region) weights
        self.samples_flat: Optional[np.ndarray] = None # (S*N_th0*N_per_region, D) # samples flattened
        self.samples_weights_flat: Optional[np.ndarray] = None # (S*N_th0*N_per_region,) # weights flattened

        # Step: importance_resampling
        self.samples_final: Optional[np.ndarray] = None  # (N, D), # final samples after importance resampling

        # inputs
        self.fit_kwargs: Optional[dict] = None
        self.sample_kwargs: Optional[dict] = None

        super().__init__("R2OMC", prior, simulator, observation, dim, dim_y)

    def find_informative_dims(
            self,
            key: jnp.ndarray = jax.random.PRNGKey(21),
            inf_dims_nof_th: int = 100,
            inf_dims_nof_seeds: int = 5,
            inf_dims_threshold: float = 1e-5
    ):
        key, subkey = jax.random.split(key)
        thetas = self.prior.sample_jax(subkey, inf_dims_nof_th) # (inf_dims_nof_th, D)
        key, subkey = jax.random.split(key)
        seeds = jax.random.randint(subkey, (inf_dims_nof_seeds,), 0, 2**31-1) # (inf_dims_nof_seeds,)
        dy_dth = jnp.abs(self.sim_jac_2(thetas, seeds)) # (inf_dims_nof_seeds, inf_dims_nof_th, Dy, D)
        inf_dims = dy_dth.mean(axis=[0, 1, 3])
        inf_dims = inf_dims > inf_dims_threshold  # (Dy,)
        assert inf_dims.shape == (self.Dy,)

        # set informative dimensions
        self.informative_dims = inf_dims
        self.sim.informative_dims = inf_dims # very important; set that to the simulator object
        return key

    def sample_objective_functions(
            self,
            key,
            nof_seeds_total: int = 500,
            nof_th0: int = 10
    ):
        self.nof_seeds_total = nof_seeds_total
        self.nof_th0 = nof_th0

        # function
        key, subkey = jax.random.split(key)
        seeds_init = np.array(jax.random.randint(subkey, (nof_seeds_total,), 0, 2**31-1)) # (nof_seeds_total,)
        key, subkey = jax.random.split(key)
        NN = nof_seeds_total * nof_th0
        th0_init = self.prior.sample_jax(subkey, NN) # (N, D)
        th0_init = th0_init.reshape((nof_seeds_total, nof_th0, self.D))  # (nof_seeds_total, nof_th0, D)
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
        th_star_init = self.th_star_init
        d_star_init = self.d_star_init
        for i, ss in enumerate(self.seeds_init):
            th_c, d_c = self.get_traj(jnp.array(th_star_init[i]), ss, self.y_0, nof_gd_steps, alpha)
            th_star_init[i] = np.array(th_c)
            d_star_init[i] = np.array(d_c)

        # output
        self.th_star_init = th_star_init
        self.d_star_init = d_star_init

    def filter_solutions_inside_prior(self):
        # step 1: keep seeds with at least one theta_star inside prior
        th_star_flat = self.th_star_init.reshape((self.nof_seeds_total * self.nof_th0, self.D))
        th_star_inside_prior = self.prior.has_mass(th_star_flat).reshape((self.nof_seeds_total, self.nof_th0))
        seed_inside_prior = np.sum(th_star_inside_prior, axis=1) > 0
        nof_seed_inside_prior = np.sum(seed_inside_prior)

        # output
        self.nof_seeds_inside_prior = nof_seed_inside_prior
        self.seeds_inside_prior = self.seeds_init[seed_inside_prior]
        self.th_star_inside_prior = self.th_star_init[seed_inside_prior]
        self.d_star_inside_prior = self.d_star_init[seed_inside_prior]

    def filter_solutions(self, pcg_to_keep: Optional[float] = 0.9, eps_1: Optional[float] = None):
        assert pcg_to_keep is not None or eps_1 is not None, "pcg_to_keep or eps_1 must be provided"

        # step 2: sort by the best d_star
        best_dist_per_seed = self.d_star_inside_prior.min(axis=1)  # best solution for each seed

        if pcg_to_keep is not None:
            nof_seeds_accept = int(np.ceil(self.nof_seeds_total * pcg_to_keep))
            assert nof_seeds_accept <= self.nof_seeds_inside_prior, "Not enough seeds inside prior for pcg_to_keep."
            sorted_indices = np.argsort(best_dist_per_seed)  # indices of the best solutions
            accepted_indices = sorted_indices[:nof_seeds_accept]  # indices of the best solutions
            eps_1 = best_dist_per_seed[accepted_indices][-1]  # eps_1 is the distance of the worst accepted solution
        else:
            # accepted_indices only the ones with best_dist_per_seed <= eps_1
            accepted_indices = np.where(best_dist_per_seed <= eps_1)[0] #
            nof_seeds_accept = len(accepted_indices)

        # step 3: filter solutions
        self.nof_seeds_accept = nof_seeds_accept
        self.eps_1 = eps_1
        self.seeds = self.seeds_inside_prior[accepted_indices]
        self.th_star = self.th_star_inside_prior[accepted_indices]
        self.d_star = self.d_star_inside_prior[accepted_indices]

    def get_directions(self, from_hessian: bool = True):
        self.hessians = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, self.D))
        self.eig_val = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D))
        self.eig_vec = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, self.D))
        if from_hessian:
            for i, ss in enumerate(self.seeds):
                self.hessians[i] = np.array(self.dist_hessian_1(self.th_star[i], ss, self.y_0))
                self.eig_val[i], self.eig_vec[i] = np.linalg.eig(self.hessians[i])
        else:
            for i in range(self.nof_seeds_accept):
                for j in range(self.nof_th0):
                    self.eig_vec[i, j] = np.eye(self.D)
                    self.eig_val[i, j] = np.ones(self.D)

    def _process_limit(self, is_inside_eps, step_size):
        L = is_inside_eps.shape[-1] - 1
        is_outside = is_inside_eps == 0
        all_inside = np.sum(is_outside, -1) == 0

        last_inside = (np.argmax(is_outside, axis=-1) - 1).astype(float)
        last_inside[all_inside] = L # if all are inside, then the max is L
        last_inside[last_inside == 0] = 0.5  # if th_0 + step is false, then the max is 0.5
        last_inside[last_inside == -1] = 0.5  # if th_0 is false, then the max is 0.5
        return last_inside * step_size

    def _get_distances(self, nof_ls_steps: int = 1, step_size: float = 0.1):
        """Get distances for each seed and theta_0, for nof_ls_steps steps in each direction
        with step_size each

        Args:
            nof_ls_steps: how many steps for each direction
            step_size: how much to step in each direction

        Returns:
            distances: np.ndarray of shape (N_seed, N_th0, D, 2 * nof_ls_steps + 1)
        """
        distances = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, 2 * nof_ls_steps + 1))
        for i in range(self.nof_seeds_accept):
            # find query points
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

            # find distances
            distances_seed = self.dist_1(query_points_flat, self.seeds[i], self.y_0)
            distances_seed = distances_seed.reshape(self.nof_th0, self.D, 2 * nof_ls_steps + 1)
            distances[i] = distances_seed
        return distances

    def get_boxes(self, eps_2: float, nof_ls_steps: int = 10, step_size: float = 0.1):
        self.step_size = step_size
        self.nof_ls_steps = nof_ls_steps
        self.eps_2 = eps_2
        self.limits = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, 2))

        distances = self._get_distances(nof_ls_steps, step_size)

        for i in range(self.nof_seeds_accept):
            # find min and max translation
            is_inside_seed = np.array(distances[i] <= self.eps_2)
            sliced_is_inside_seed = is_inside_seed[..., :nof_ls_steps + 1]
            reversed_sliced_dd_ind = np.flip(sliced_is_inside_seed, axis=-1)
            self.limits[i, :, :, 0] = -1 * self._process_limit(reversed_sliced_dd_ind, step_size)
            sliced_is_inside_seed = is_inside_seed[..., nof_ls_steps:]
            self.limits[i, :, :, 1] = self._process_limit(sliced_is_inside_seed, step_size)
        self.log_volumes = np.sum(np.log(self.limits[:, :, :, 1] - self.limits[:, :, :, 0]), axis=2)

    def get_boxes_blind(self, dx):
        # just set limits to be dx away from the th_star
        self.limits = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, 2))
        self.limits[:, :, :, 0] = - dx
        self.limits[:, :, :, 1] = dx
        self.log_volumes = np.sum(np.log(2 * dx * np.ones((self.nof_seeds_accept, self.nof_th0, self.D))), axis=2)

    def _get_samples_per_region(self, key, nof_samples, i_seed, i_th0, eps_3) -> Tuple[np.ndarray, np.ndarray]:
        """Generates nof_samples from the i_seed and i_th0 region.
        Returns the samples and their log weights.
        """
        limits = self.limits[i_seed, i_th0]     # shape (D, 2)
        rotation = self.eig_vec[i_seed, i_th0]  # shape (D, D)
        center = self.th_star[i_seed, i_th0]    # shape (D,)

        # sample uniformly in the limits + rotate + translate
        samples_init = jax.random.uniform(
            key, shape=(nof_samples, self.D), minval=limits[:, 0], maxval=limits[:, 1]) # (N_per_region, D)
        samples_rot = jnp.dot(samples_init, rotation) + center                          # (N_per_region, D)

        # compute weights
        distances = self.dist_1(samples_rot, self.seeds[i_seed], self.y_0) # (N_per_region,)
        log_prior = self.prior.logpdf(samples_rot) # (N_per_region,)
        log_volume = self.log_volumes[i_seed, i_th0] # (N_per_region,)
        is_inside =  distances < eps_3
        log_mask = jnp.where(is_inside, 0.0, -jnp.inf) # (N_per_region,)
        log_weights = log_prior + log_volume + log_mask # (N_per_region,)
        return samples_rot, log_weights

    def weighted_sampling(self, sampling_seed, samples_per_region, eps_3=1.):
        key = jax.random.PRNGKey(sampling_seed)

        # iteration over seeds and theta_0 to draw samples
        samples = []
        log_weights = []
        for i_seed in range(self.seeds.shape[0]):
            seed_samples = []
            seed_log_weights = []
            for i_th in range(self.nof_th0):
                key, subkey = jax.random.split(key)
                reg_samples, reg_log_weights = self._get_samples_per_region(
                    subkey, samples_per_region, i_seed, i_th, eps_3)
                seed_samples.append(reg_samples)
                seed_log_weights.append(reg_log_weights)
            samples.append(seed_samples)
            log_weights.append(seed_log_weights)

        # convert to numpy arrays
        samples = np.array(samples) # (S_accept, N_th0, N_per_region, D)
        log_weights = np.array(log_weights) # (S_accept, N_th0, N_per_region)

        # Normalize weights in log-space for numerical stability
        log_weights -= np.max(log_weights, axis=-1, keepdims=True)  # for numerical stability
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # store the samples and weights
        self.samples = samples
        self.samples_weights = weights

        # flatten the samples and weights
        self.samples_flat = samples.reshape((-1, self.D)) # (S_accept * N_th0 * N_per_region, D)
        self.samples_weights_flat = weights.reshape((-1,)) # (S_accept * N_th0 * N_per_region,)
        return self.samples_flat, self.samples_weights_flat

    def importance_resampling(self, samples: np.ndarray, weights: np.ndarray, nof_samples: int, replace=False):
        if np.sum(weights > 0) < nof_samples:
            print("Not enough samples with positive weight")
            return samples[np.argsort(weights)[::-1]][:nof_samples]
        indices = np.random.choice(np.arange(samples.shape[0]), size=nof_samples, replace=replace, p=weights)
        self.samples_final = samples[indices]
        return self.samples_final

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
            # informative dimensions
            "find_informative_dims": True,
            "inf_dims_nof_th": 100,
            "inf_dims_nof_seeds": 50,
            "inf_dims_threshold": 1e-5,
            # sample objective functions
            "nof_seeds_total": budget,
            "nof_th0": 1,
            # optimize
            "epochs": 4,
            "alpha": 0.1,
            "nof_gd_steps": 50,
            # filter_solutions
            "pcg_to_keep": .8,
            "eps_1": None,  # will be checked
            # get_boxes
            "box_algorithm": "standard", # "standard" or "blind"
            "dx": 0.1,
            "eps_2": None,
            "nof_ls_steps": 100,
            "step_size": .01,
        }

        fit_kwargs = {**default_kwargs, **(fit_kwargs or {})}
        self.fit_kwargs = fit_kwargs

        # inititalize the random key
        key = jax.random.PRNGKey(fit_kwargs["fit_seed"])

        # Step 1: find informative dimensions
        print("Step 1: find informative dimensions")
        print("-----------------------------------")
        if fit_kwargs["find_informative_dims"]:
            key, subkey = jax.random.split(key)
            key = self.find_informative_dims(
                key,
                fit_kwargs["inf_dims_nof_th"],
                fit_kwargs["inf_dims_nof_seeds"],
                fit_kwargs["inf_dims_threshold"]
            )
        print(f"Informative dimensions: {np.sum(self.informative_dims)} out of {self.Dy}")

        # Step 2: optimize objective functions
        print(f"\nStep 2: Optimization ({fit_kwargs['nof_seeds_total']}x{fit_kwargs['nof_th0']})")
        print("---------------------------------")
        key = self.sample_objective_functions(key, fit_kwargs["nof_seeds_total"], fit_kwargs["nof_th0"])

        for epoch in range(fit_kwargs["epochs"]):
            print(
                f"Epoch {epoch + 1}/{fit_kwargs['epochs']}: "
                f"Applying {fit_kwargs['nof_gd_steps']} gradient descent steps with alpha={fit_kwargs['alpha']}"
            )
            self.optimize(fit_kwargs["nof_gd_steps"], fit_kwargs["alpha"])

        print("Statistics:")
        print(
            f"min: {self.d_star_init.min():.5f}, "
            f"max: {self.d_star_init.max():.5f}, "
            f"mean: {self.d_star_init.mean():.5f}, "
            f"std: {self.d_star_init.std():.5f}"
        )

        # Step 3: filter_solutions_inside_prior
        print("\nStep 3: filter solutions inside prior")
        print("---------------------------------")
        self.filter_solutions_inside_prior()
        print(f"Inside prior: {self.seeds_inside_prior.shape[0]}/{self.nof_seeds_total} seeds")

        # Step 4: filter_solutions based on eps_1 or pcg_to_keep
        if fit_kwargs["pcg_to_keep"] is not None:
            print(f"\nStep 4: filter solutions ({fit_kwargs['pcg_to_keep'] * 100}% of seeds)")
        else:
            print(f"\nStep 4: filter solutions (eps_1={fit_kwargs['eps_1']})")
        print("---------------------------------")
        self.filter_solutions(fit_kwargs["pcg_to_keep"], fit_kwargs["eps_1"])
        print(f"Keep ({self.nof_seeds_accept}/{self.seeds_inside_prior.shape[0]} seeds), eps_1={self.eps_1:.5f}")
        print("Statistics:")
        print(
            f"min: {self.d_star.min():.5f}, "
            f"max: {self.d_star.max():.5f}, "
            f"mean: {self.d_star.mean():.5f}, "
            f"std: {self.d_star.std():.5f}"
        )

        # Step 5: find the directions for building the boxes
        print("\nStep 5: get directions")
        print("---------------------------------")
        if fit_kwargs.get("box_algorithm") == "blind":
            self.get_directions(False)
        else:
            self.get_directions(True)
        print("Done!")

        # Step 6: build the bounding boxes
        print(f"\nStep 6: Build bounding boxes - Algorithm: {fit_kwargs['box_algorithm']}")
        print("---------------------------------")

        if fit_kwargs["box_algorithm"] == "standard":
            print(
                f"Input: \n"
                f"- eps_2={fit_kwargs['eps_2']} \n"
                f"- dx={fit_kwargs['dx']} \n"
                f"- nof_ls_steps={fit_kwargs['nof_ls_steps']} \n"
                f"- step_size={fit_kwargs['step_size']}"
            )
            if fit_kwargs.get("dx") is not None:
                eps_2 = self._get_distances(1, fit_kwargs["dx"]).mean()
            elif fit_kwargs.get("eps_2") is not None:
                eps_2 = fit_kwargs["eps_2"]
            else:
                raise ValueError("Either 'dx' or 'eps_2' must be provided in fit_kwargs.")

            self.get_boxes(eps_2, fit_kwargs["nof_ls_steps"], fit_kwargs["step_size"])

            print("Output:")
            print(
                f"Built {self.seeds.shape[0]} x {self.nof_th0} boxes: \n"
                f"- eps_2={self.eps_2:.5f} (criterion)\n"
                f"- dx={fit_kwargs['dx']:.5f}"
            )
        elif fit_kwargs["box_algorithm"] == "blind":
            print(f"Input: \n- dx={fit_kwargs['dx']}")
            self.get_boxes_blind(fit_kwargs["dx"])
            print("Output:")
            print(
                f"Built {self.seeds.shape[0]} x {self.nof_th0} boxes: \n"
                f"- dx={fit_kwargs['dx']:.5f} (criterion)"
            )

    def sample(self, nof_samples: int = 100, sample_kwargs: Optional[dict] = None):
        default_kwargs = {
            "sample_seed": 71,
            "eps_3": 1.0,
            "samples_per_region": 10,
        }
        default_kwargs.update((sample_kwargs or {}))
        sample_kwargs = default_kwargs
        self.sample_kwargs = sample_kwargs

        print("\nStep 7: Sample from the boxes")
        print("-----------------------------")
        print(
            f"Input:\n"
            f"- sample_seed={sample_kwargs['sample_seed']}\n"
            f"- samples_per_region={sample_kwargs['samples_per_region']}\n"
            f"- eps_3={sample_kwargs['eps_3']:.5f}"
        )

        samples, weights = self.weighted_sampling(
            sample_kwargs["sample_seed"],
            sample_kwargs["samples_per_region"],
            sample_kwargs["eps_3"]
        )

        samples_r2omc = self.importance_resampling(samples, weights, nof_samples, False)

        print(f"Output:\n- Returned {samples_r2omc.shape[0]} weighted samples (after resampling)")
        return samples_r2omc
