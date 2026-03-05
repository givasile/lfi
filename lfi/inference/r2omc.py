import warnings
import numpy as np
import jax
from typing import Optional, Tuple
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
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
            key: jax.Array,
            inf_dims_nof_th: int = 100,
            inf_dims_nof_seeds: int = 50,
            inf_dims_threshold: float = 0.01,
            inf_dims_aggregation: str = "mean",
    ):
        """Identify which output dimensions carry information about theta.

        Computes the aggregated absolute Jacobian |dy/dtheta| over a grid of
        (theta, seed) pairs.  A dimension is kept if its aggregated abs Jacobian
        is at least `inf_dims_threshold` times the largest value across all output
        dimensions (relative threshold, scale-invariant).

        Args:
            key: JAX PRNG key for sampling thetas and seeds.
            inf_dims_nof_th: Number of theta samples drawn to estimate the Jacobian.
                Higher → more reliable detection at the cost of compute.
            inf_dims_nof_seeds: Number of CRN seeds drawn to estimate the Jacobian.
                Higher → averages out seed-specific variability.
            inf_dims_threshold: Relative sensitivity threshold in (0, 1).
                A dim is informative if its aggregated abs Jacobian >=
                threshold * max(aggregated abs Jacobian across all dims).
                0.01 keeps any dim with at least 1% of the most sensitive dim's
                sensitivity.  Increase (e.g. 0.1) to drop weakly-informative dims
                more aggressively.
            inf_dims_aggregation: How to reduce the abs Jacobian over the
                (seeds, thetas, input_dims) axes into a single scalar per output
                dim.  One of "mean" (default), "median", or "max".
                "max" is more conservative: keeps a dim if *any* (theta, seed,
                input direction) moves it.  "median" is more robust to outliers.

        Returns:
            key: Updated JAX PRNG key.
        """
        key, subkey = jax.random.split(key)
        thetas = self.prior.sample_jax(subkey, inf_dims_nof_th)  # (inf_dims_nof_th, D)
        key, subkey = jax.random.split(key)
        seeds = jax.random.randint(subkey, (inf_dims_nof_seeds,), 0, 2**31-1)  # (inf_dims_nof_seeds,)
        dy_dth = jnp.abs(self.sim_jac_2(thetas, seeds))  # (n_seeds, n_th, Dy, D)

        # Flatten (n_seeds, n_th, D) into one axis, keeping Dy separate → (Dy, n_seeds*n_th*D)
        flat = dy_dth.transpose(2, 0, 1, 3).reshape(self.Dy, -1)
        if inf_dims_aggregation == "mean":
            aggregated = flat.mean(axis=1)
        elif inf_dims_aggregation == "median":
            aggregated = jnp.median(flat, axis=1)
        elif inf_dims_aggregation == "max":
            aggregated = flat.max(axis=1)
        else:
            raise ValueError(
                f"inf_dims_aggregation must be 'mean', 'median', or 'max', "
                f"got '{inf_dims_aggregation}'"
            )

        max_val = aggregated.max()
        if max_val < 1e-6:  # degenerate: simulator fully insensitive to theta — keep all
            inf_dims = jnp.ones(self.Dy, dtype=bool)
        else:
            inf_dims = aggregated > inf_dims_threshold * max_val  # (Dy,)

        self.informative_dims = inf_dims
        self.sim.informative_dims = inf_dims  # distance functions close over the simulator
        return key

    def sample_objective_functions(
            self,
            key: jax.Array,
            nof_seeds_total: int = 500,
            nof_th0: int = 1,
    ) -> jax.Array:
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
        d = self.dist_1(th0, seed, y_0)
        return th0, d

    def optimize(self, nof_gd_steps: int = 100, alpha: float = 0.01) -> None:
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
            if nof_seeds_accept > self.nof_seeds_inside_prior:
                raise ValueError(
                    f"pcg_to_keep={pcg_to_keep} requires {nof_seeds_accept} seeds but only "
                    f"{self.nof_seeds_inside_prior}/{self.nof_seeds_total} seeds landed inside the prior. "
                    f"Try lowering pcg_to_keep (e.g. <= {self.nof_seeds_inside_prior / self.nof_seeds_total:.2f}) "
                    f"or increasing nof_seeds_total."
                )
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

    def get_directions(self, method: str = "hessian") -> None:
        self.hessians = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, self.D))
        self.eig_val = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D))
        self.eig_vec = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, self.D))
        if method == "hessian":
            for i, ss in enumerate(self.seeds):
                self.hessians[i] = np.array(self.dist_hessian_1(self.th_star[i], ss, self.y_0))
                self.eig_val[i], self.eig_vec[i] = np.linalg.eigh(self.hessians[i])
        elif method == "jacobian":
            for i, ss in enumerate(self.seeds):
                grads = self.dist_grad_1(self.th_star[i], ss, self.y_0)[1]  # (N_th0, D)
                jac = grads[:, :, np.newaxis] @ grads[:, np.newaxis, :]     # (N_th0, D, D)
                self.hessians[i] = np.array(jac)
                self.eig_val[i], self.eig_vec[i] = np.linalg.eigh(self.hessians[i])
        elif method == "blind":
            self.eig_vec[:] = np.eye(self.D)
            self.eig_val[:] = 1.0
        else:
            raise ValueError(
                f"method must be 'hessian', 'jacobian', or 'blind', got '{method}'"
            )

    def _process_limit(self, is_inside_eps: np.ndarray, step_size: float) -> np.ndarray:
        L = is_inside_eps.shape[-1] - 1
        is_outside = is_inside_eps == 0
        all_inside = np.sum(is_outside, -1) == 0

        last_inside = (np.argmax(is_outside, axis=-1) - 1).astype(float)
        last_inside[all_inside] = L    # all points inside → box reaches the last step
        last_inside[last_inside == 0] = 0.5   # th_0 inside but first step already outside → half-step fallback
        last_inside[last_inside == -1] = 0.5  # th_0 itself outside eps_2 → minimal half-step fallback
        return last_inside * step_size

    def _get_distances(self, nof_ls_steps: int = 1, step_size: float = 0.1) -> np.ndarray:
        """Get distances for each seed and theta_0, for nof_ls_steps steps in each direction
        with step_size each

        Args:
            nof_ls_steps: how many steps for each direction
            step_size: how much to step in each direction

        Returns:
            distances: np.ndarray of shape (N_seed, N_th0, D, 2 * nof_ls_steps + 1)
        """
        n_pts = 2 * nof_ls_steps + 1
        distances = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, n_pts))
        for i in range(self.nof_seeds_accept):
            th_star = self.th_star[i]  # (T, D)
            query_points = np.zeros((self.nof_th0, self.D, n_pts, self.D))
            for dim in range(self.D):
                steps = np.zeros((n_pts, self.D))
                steps[:, dim] = np.linspace(-nof_ls_steps, nof_ls_steps, n_pts) * step_size
                rotated = steps[np.newaxis] @ self.eig_vec[i]   # (T, n_pts, D)
                affine = rotated + th_star[:, np.newaxis, :]    # (T, n_pts, D)
                query_points[:, dim, :, :] = affine

            query_points_flat = query_points.reshape(self.nof_th0 * self.D * n_pts, self.D)
            distances_seed = self.dist_1(query_points_flat, self.seeds[i], self.y_0)
            distances[i] = distances_seed.reshape(self.nof_th0, self.D, n_pts)
        return distances

    def get_boxes(self, eps_2: float, nof_ls_steps: int = 10, step_size: float = 0.1) -> None:
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

    def get_boxes_blind(self, dx: float) -> None:
        # just set limits to be dx away from the th_star
        self.limits = np.zeros((self.nof_seeds_accept, self.nof_th0, self.D, 2))
        self.limits[:, :, :, 0] = - dx
        self.limits[:, :, :, 1] = dx
        self.log_volumes = np.sum(np.log(2 * dx * np.ones((self.nof_seeds_accept, self.nof_th0, self.D))), axis=2)

    def _get_samples_per_region(self, key: jax.Array, nof_samples: int, i_seed: int, i_th0: int, eps_3: float) -> Tuple[np.ndarray, np.ndarray]:
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
        log_volume = self.log_volumes[i_seed, i_th0]  # scalar
        is_inside =  distances < eps_3
        log_mask = jnp.where(is_inside, 0.0, -jnp.inf) # (N_per_region,)
        log_weights = log_prior + log_volume + log_mask # (N_per_region,)
        return samples_rot, log_weights

    def weighted_sampling(self, sampling_seed: int, samples_per_region: int, eps_3: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
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

        # Normalise in log-space for numerical stability: shift by global max so
        # the largest weight becomes exp(0)=1, preventing underflow to all-zeros.
        max_lw = log_weights.max()
        if not np.isfinite(max_lw):
            # All samples outside eps_3 — fall back to uniform weights with a warning.
            warnings.warn(
                "All samples have zero weight (all outside eps_3). "
                "Returning uniform weights. Consider increasing eps_3."
            )
            weights = np.ones_like(log_weights, dtype=float) / log_weights.size
        else:
            log_weights -= max_lw
            weights = np.exp(log_weights)
            weights /= np.sum(weights)

        # store the samples and weights
        self.samples = samples
        self.samples_weights = weights

        # flatten the samples and weights
        self.samples_flat = samples.reshape((-1, self.D)) # (S_accept * N_th0 * N_per_region, D)
        self.samples_weights_flat = weights.reshape((-1,)) # (S_accept * N_th0 * N_per_region,)
        return self.samples_flat, self.samples_weights_flat

    def importance_resampling(self, samples: np.ndarray, weights: np.ndarray, nof_samples: int, replace: bool = False) -> np.ndarray:
        if np.sum(weights > 0) < nof_samples:
            warnings.warn(
                f"Only {np.sum(weights > 0)} samples have positive weight, "
                f"but {nof_samples} were requested. Returning top-k by weight."
            )
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

    def fit(self, budget: int = 1000, fit_kwargs: Optional[dict] = None, verbose: int = 1):
        """Fit the R2OMC posterior approximation.

        Runs six sequential steps: detect informative output dims, optimise CRN
        objectives, filter solutions, find local directions, build bounding boxes.
        All hyperparameters have sensible defaults; override via fit_kwargs only
        when the default behaviour is insufficient for your problem.

        Args:
            budget: Total number of CRN seeds to optimise (proxy for compute budget).
            fit_kwargs: Optional dict overriding any of the parameters below.
            verbose: Verbosity level. 0 = silent, 1 = one line per step (default),
                2 = detailed output including per-epoch stats and input/output blocks.

        fit_kwargs reference
        --------------------
        General
            fit_seed (int, default 21)
                Master random seed for reproducibility.

        Step 1 — informative dimension detection
            find_informative_dims (bool, default True)
                Whether to auto-detect which output dims carry signal about theta.
                Set False if all dims are known to be informative (saves compute).
            inf_dims_nof_th (int, default 100)
                Theta samples used to estimate the mean abs Jacobian.
                Higher → more reliable detection, more compute.
            inf_dims_nof_seeds (int, default 50)
                CRN seeds used to estimate the mean abs Jacobian.
                Higher → less seed-specific noise, more compute.
            inf_dims_threshold (float, default 0.01)
                Relative sensitivity threshold. A dim is kept if its aggregated
                abs Jacobian >= threshold * max(aggregated abs Jacobian). 0.01
                keeps any dim with ≥1% of the most sensitive dim's sensitivity.
                Increase to drop weakly-informative dims more aggressively.
            inf_dims_aggregation (str, default "mean")
                How to reduce the abs Jacobian over (seeds, thetas, input_dims)
                into one scalar per output dim. One of "mean", "median", "max".

        Step 2 — objective sampling and optimisation
            nof_seeds_total (int, default budget)
                Total CRN seeds optimised. More seeds → better posterior coverage.
            nof_th0 (int, default 1)
                Random starting points per seed for gradient descent. Higher →
                less likely to miss a solution, more compute. 1 is usually enough
                for unimodal simulators.
            epochs (int, default 4)
                Optimisation passes over all seeds. Each pass runs nof_gd_steps
                Adam steps. More epochs → finer convergence.
            alpha (float, default 0.1)
                Adam learning rate. 0.1 works well when the prior is ~[-3, 3].
                Reduce if the optimiser diverges; increase if convergence is slow.
            nof_gd_steps (int, default 50)
                Adam steps per epoch per seed.
                Total steps per seed = epochs × nof_gd_steps (default: 200).

        Step 3/4 — solution filtering
            pcg_to_keep (float in (0,1], default 0.8)
                Fraction of seeds kept after ranking by d(theta*, s). 0.8 keeps
                the 80% closest solutions. Use this OR eps_1, not both.
            eps_1 (float or None, default None)
                Absolute distance threshold for filtering (alternative to
                pcg_to_keep). Seeds with best d(theta*, s) > eps_1 are dropped.
                Leave None to use pcg_to_keep instead.

        Step 5/6 — bounding box construction
            box_algorithm (str, default "standard")
                "standard"         — Hessian-eigenvector boxes, size from eps_2.
                "standard_jacobian" — same but eigenvectors from Jacobian.
                "blind"            — axis-aligned boxes of fixed half-width dx.
                "eye"              — axis-aligned boxes, size from eps_2.
            dx (float, default 0.1)
                Step size used to auto-compute eps_2 when eps_2=None. eps_2 is
                set to the mean d(theta* + dx·eigvec, s). Increase for wider
                boxes; decrease for tighter ones.
            eps_2 (float or None, default None)
                Direct distance threshold controlling box size (alternative to
                dx). A point is inside a box if d(theta, s) < eps_2.
                Leave None to derive eps_2 automatically from dx.
            nof_ls_steps (int, default 100)
                Line-search steps along each eigenvector to find box edges.
                More steps → more accurate limits.
            step_size (float, default 0.01)
                Physical step size per line-search step in theta space.
                Max box half-width = nof_ls_steps × step_size = 1.0 by default.
                Reduce if theta lives in a small range.
        """
        default_kwargs = {
            "fit_seed": 21,
            # ── Step 1 ────────────────────────────────────────────────────────
            "find_informative_dims": True,
            "inf_dims_nof_th": 100,
            "inf_dims_nof_seeds": 50,
            "inf_dims_threshold": 0.01,
            "inf_dims_aggregation": "mean",
            # ── Step 2 ────────────────────────────────────────────────────────
            "nof_seeds_total": budget,
            "nof_th0": 1,
            "epochs": 4,
            "alpha": 0.1,
            "nof_gd_steps": 50,
            # ── Step 3/4 ──────────────────────────────────────────────────────
            "pcg_to_keep": 0.8,
            "eps_1": None,
            # ── Step 5/6 ──────────────────────────────────────────────────────
            "box_algorithm": "standard",
            "dx": 0.1,
            "eps_2": None,
            "nof_ls_steps": 100,
            "step_size": 0.01,
        }

        fit_kwargs = {**default_kwargs, **(fit_kwargs or {})}
        self.fit_kwargs = fit_kwargs

        # inititalize the random key
        key = jax.random.PRNGKey(fit_kwargs["fit_seed"])

        # Step 1: find informative dimensions
        if verbose >= 2:
            print("Step 1: find informative dimensions")
            print("-----------------------------------")
        if fit_kwargs["find_informative_dims"]:
            key = self.find_informative_dims(
                key,
                fit_kwargs["inf_dims_nof_th"],
                fit_kwargs["inf_dims_nof_seeds"],
                fit_kwargs["inf_dims_threshold"],
                fit_kwargs["inf_dims_aggregation"],
            )
        if verbose >= 1:
            print(f"Step 1: {int(np.sum(self.informative_dims))}/{self.Dy} informative dims")

        # Step 2: optimize objective functions
        if verbose >= 2:
            print(f"\nStep 2: Optimization ({fit_kwargs['nof_seeds_total']}x{fit_kwargs['nof_th0']})")
            print("---------------------------------")
        key = self.sample_objective_functions(key, fit_kwargs["nof_seeds_total"], fit_kwargs["nof_th0"])

        for epoch in range(fit_kwargs["epochs"]):
            if verbose >= 2:
                print(
                    f"Epoch {epoch + 1}/{fit_kwargs['epochs']}: "
                    f"{fit_kwargs['nof_gd_steps']} GD steps, alpha={fit_kwargs['alpha']}"
                )
            self.optimize(fit_kwargs["nof_gd_steps"], fit_kwargs["alpha"])
            if verbose >= 2:
                print(
                    f"  dist: {self.d_star_init.mean():.5f} ± {self.d_star_init.std():.5f}"
                    f"  [{self.d_star_init.min():.5f}, {self.d_star_init.max():.5f}]"
                )
        if verbose >= 1:
            print(
                f"Step 2: {fit_kwargs['nof_seeds_total']}×{fit_kwargs['nof_th0']} seeds optimised "
                f"({fit_kwargs['epochs']} epochs) | "
                f"dist {self.d_star_init.mean():.4f} ± {self.d_star_init.std():.4f}"
            )

        # Step 3: filter_solutions_inside_prior
        if verbose >= 2:
            print("\nStep 3: filter solutions inside prior")
            print("---------------------------------")
        self.filter_solutions_inside_prior()
        if verbose >= 1:
            print(f"Step 3: {self.nof_seeds_inside_prior}/{self.nof_seeds_total} seeds inside prior")

        # Step 4: filter_solutions based on eps_1 or pcg_to_keep
        if verbose >= 2:
            if fit_kwargs["pcg_to_keep"] is not None:
                print(f"\nStep 4: filter solutions (pcg_to_keep={fit_kwargs['pcg_to_keep']})")
            else:
                print(f"\nStep 4: filter solutions (eps_1={fit_kwargs['eps_1']})")
            print("---------------------------------")
        self.filter_solutions(fit_kwargs["pcg_to_keep"], fit_kwargs["eps_1"])
        if verbose >= 1:
            print(f"Step 4: {self.nof_seeds_accept} seeds accepted | eps_1={self.eps_1:.4f}")
        if verbose >= 2:
            print(
                f"  dist: min={self.d_star.min():.5f}, max={self.d_star.max():.5f}, "
                f"mean={self.d_star.mean():.5f}, std={self.d_star.std():.5f}"
            )

        # Step 5: find the directions for building the boxes
        if verbose >= 2:
            print("\nStep 5: get directions")
            print("---------------------------------")
        if fit_kwargs.get("box_algorithm") in ["blind", "eye"]:
            directions_method = "blind"
        elif fit_kwargs.get("box_algorithm") == "standard_jacobian":
            directions_method = "jacobian"
        else:
            directions_method = "hessian"
        self.get_directions(directions_method)
        if verbose >= 1:
            print(f"Step 5: directions computed ({directions_method})")

        # Step 6: build the bounding boxes
        if verbose >= 2:
            print(f"\nStep 6: Build bounding boxes - Algorithm: {fit_kwargs['box_algorithm']}")
            print("---------------------------------")

        if fit_kwargs["box_algorithm"] in ["standard", "eye", "standard_jacobian"]:
            if verbose >= 2:
                print(
                    f"Input: \n"
                    f"- eps_2={fit_kwargs['eps_2']} \n"
                    f"- dx={fit_kwargs['dx']} \n"
                    f"- nof_ls_steps={fit_kwargs['nof_ls_steps']} \n"
                    f"- step_size={fit_kwargs['step_size']}"
                )
            if fit_kwargs.get("eps_2") is not None:
                eps_2 = fit_kwargs["eps_2"]
                if verbose >= 2:
                    print(f"Using provided eps_2={eps_2:.5f}")
            elif fit_kwargs.get("dx") is not None:
                eps_2 = self._get_distances(1, fit_kwargs["dx"]).mean()
                if verbose >= 2:
                    print(f"Computed eps_2={eps_2:.5f} from dx={fit_kwargs['dx']:.5f}")
            else:
                raise ValueError("Either 'dx' or 'eps_2' must be provided in fit_kwargs.")

            self.get_boxes(eps_2, fit_kwargs["nof_ls_steps"], fit_kwargs["step_size"])
            if verbose >= 1:
                print(f"Step 6: {self.seeds.shape[0]}×{self.nof_th0} boxes built | eps_2={self.eps_2:.4f}")

        elif fit_kwargs["box_algorithm"] == "blind":
            if verbose >= 2:
                print(f"Input: \n- dx={fit_kwargs['dx']}")
            self.get_boxes_blind(fit_kwargs["dx"])
            if verbose >= 1:
                print(f"Step 6: {self.seeds.shape[0]}×{self.nof_th0} blind boxes built | dx={fit_kwargs['dx']:.4f}")

    def sample(self, nof_samples: int = 100, sample_kwargs: Optional[dict] = None, verbose: int = 1) -> np.ndarray:
        """Draw posterior samples using weighted sampling + importance resampling.

        Args:
            nof_samples: Number of posterior samples to return.
            sample_kwargs: Optional dict overriding any of the parameters below.
            verbose: Verbosity level. 0 = silent, 1 = one line per step (default),
                2 = detailed output including input/output blocks.

        sample_kwargs reference
        -----------------------
            sample_seed (int, default 71)
                Random seed for reproducible sampling.
            eps_3 (float, default 1.0)
                Distance threshold for the within-box accept/reject mask.
                Only samples with d(theta, s) < eps_3 receive positive weight.
                Increase to accept more samples; decrease for tighter filtering.
            samples_per_region (int, default 10)
                Candidate samples drawn per (seed, theta*) box before resampling.
                Total candidates = nof_seeds_accept * nof_th0 * samples_per_region.
                Increase for better coverage at the cost of memory and compute.
            replace (bool, default False)
                Whether importance resampling draws with replacement.
                False avoids duplicate samples but requires enough positive-weight
                candidates (>= nof_samples). Set True if the pool is small.

        Returns:
            samples: np.ndarray of shape (nof_samples, D).
        """
        default_kwargs = {
            "sample_seed": 71,
            "eps_3": 1.0,
            "samples_per_region": 10,
            "replace": False,
        }
        sample_kwargs = {**default_kwargs, **(sample_kwargs or {})}
        self.sample_kwargs = sample_kwargs

        if verbose >= 2:
            print("\nStep 7: Sample from the boxes")
            print("-----------------------------")
            print(
                f"Input:\n"
                f"- sample_seed={sample_kwargs['sample_seed']}\n"
                f"- samples_per_region={sample_kwargs['samples_per_region']}\n"
                f"- eps_3={sample_kwargs['eps_3']:.5f}\n"
                f"- replace={sample_kwargs['replace']}"
            )

        samples, weights = self.weighted_sampling(
            sample_kwargs["sample_seed"],
            sample_kwargs["samples_per_region"],
            sample_kwargs["eps_3"]
        )

        samples_r2omc = self.importance_resampling(samples, weights, nof_samples, sample_kwargs["replace"])

        if verbose >= 1:
            print(f"Step 7: {samples_r2omc.shape[0]} samples returned")
        self.samples = samples_r2omc
        return samples_r2omc


class R2OMCMultiObs(InferenceBase):
    supports_multiple_observations: bool = True

    def __init__(
            self,
            prior: lfi.priors.BasePrior,
            simulator: lfi.simulators.BaseSimulator,
            observation: np.ndarray, # (N_obs, Dy)
    ):
        """ R2OMC Inference class for the R2OMC algorithm with multiple observations.
        """
        self.N_obs = observation.shape[0]
        self.r2omc_list = [R2OMC(prior, simulator, observation[i:i+1]) for i in range(self.N_obs)]
        super().__init__("r2omc_multi_obs", prior, simulator, observation, prior.dim, observation.shape[-1])

        # state variables
        self.N_th_per_obs: Optional[int] = None  # nof posterior samples per observation
        self.N_th_total: Optional[int] = None  # nof posterior samples in total (N_th_per_obs * Ny)
        self.N_th_accept: Optional[int] = None  # nof accepted posterior samples
        self.N_th_select: Optional[int] = None  # nof selected samples

        self.th_total: Optional[np.ndarray] = None     # (N_th_total, D)
        self.w_before: Optional[np.ndarray] = None     # (N_th_total,)
        self.w_after: Optional[np.ndarray] = None      # (N_th_total,)
        self.th_accepted: Optional[np.ndarray] = None  # (N_th_accept, D)
        self.th_selected: Optional[np.ndarray] = None  # (N_th_select, D)

    def fit(self, budget: int = 1000, fit_kwargs: Optional[dict] = None, verbose: int = 1):
        """Fit R2OMC independently for each observation, then combine.

        Delegates to R2OMC.fit() for each observation.  All fit_kwargs are
        forwarded unchanged — see R2OMC.fit() docstring for the full parameter
        reference.

        Args:
            budget: CRN seeds per observation.
            fit_kwargs: Optional dict overriding any R2OMC.fit() parameter.
            verbose: Verbosity level. 0 = silent, 1 = one line per step (default),
                2 = detailed output. Passed through to each R2OMC.fit() call.
        """
        default_kwargs = {
            "fit_seed": 21,
            # ── Step 1 ────────────────────────────────────────────────────────
            "find_informative_dims": True,
            "inf_dims_nof_th": 100,
            "inf_dims_nof_seeds": 50,
            "inf_dims_threshold": 0.01,
            "inf_dims_aggregation": "mean",
            # ── Step 2 ────────────────────────────────────────────────────────
            "nof_seeds_total": budget,
            "nof_th0": 1,
            "epochs": 4,
            "alpha": 0.1,
            "nof_gd_steps": 50,
            # ── Step 3/4 ──────────────────────────────────────────────────────
            "pcg_to_keep": 0.8,
            "eps_1": None,
            # ── Step 5/6 ──────────────────────────────────────────────────────
            "box_algorithm": "standard",
            "dx": 0.1,
            "eps_2": None,
            "nof_ls_steps": 100,
            "step_size": 0.01,
        }

        fit_kwargs = {**default_kwargs, **(fit_kwargs or {})}
        self.fit_kwargs = fit_kwargs

        # fit each R2OMC instance
        np.random.seed(fit_kwargs["fit_seed"])
        seeds = np.random.randint(0, 2**31-1, size=self.N_obs)
        for i, r2omc in enumerate(self.r2omc_list):
            if verbose >= 1:
                print(f"\nFitting observation {i+1}/{self.N_obs}")
            fit_kwargs["fit_seed"] = int(seeds[i])
            r2omc.fit(budget, fit_kwargs, verbose=verbose)

    def sample(self, nof_samples: int = 100, sample_kwargs: Optional[dict] = None, verbose: int = 1):
        default_kwargs = {
            "sample_seed":       71,
            "eps_3":             1.0,   # passed through to single-obs R2OMC sampling
            "samples_per_region": 10,
            "nof_samples_per_obs": nof_samples,  # candidates drawn per observation
            "quantile":          0.5,   # fraction kept by step (i); None = skip step (i)
            "temperature":       None,  # step (ii) softmax scale; None = auto (median of distances)
        }
        sample_kwargs = {**default_kwargs, **(sample_kwargs or {})}
        self.sample_kwargs = sample_kwargs

        nof_samples_per_obs = sample_kwargs["nof_samples_per_obs"]
        quantile            = sample_kwargs["quantile"]
        n_total             = nof_samples_per_obs * self.N_obs
        n_after_filter      = int(np.floor(n_total * quantile)) if quantile is not None else n_total

        # ── Sanity-check summary ───────────────────────────────────────────────
        if verbose >= 1:
            print(f"\n── R2OMCMultiObs sampling plan ──────────────────────────────")
            print(f"  Observations          : {self.N_obs}")
            print(f"  Samples per obs       : {nof_samples_per_obs}  "
                  f"→  total candidate pool : {n_total}")
            if quantile is not None:
                print(f"  Step (i)  quantile    : {quantile}  "
                      f"→  after minimax filter : ~{n_after_filter}")
            else:
                print(f"  Step (i)              : skipped (quantile=None)")
            print(f"  Step (ii) resampling  : {n_after_filter} → {nof_samples} final samples")
            if n_after_filter < nof_samples:
                warnings.warn(
                    f"Expected survivors after step (i) (~{n_after_filter}) is less than "
                    f"nof_samples ({nof_samples}). Consider increasing `nof_samples_per_obs` "
                    f"(currently {nof_samples_per_obs}) or raising `quantile` "
                    f"(currently {quantile})."
                )
            print(f"─────────────────────────────────────────────────────────────\n")

        # ── Draw candidates from each per-observation R2OMC ───────────────────
        th_list = []
        for i, r2omc_cur in enumerate(self.r2omc_list):
            th_i = r2omc_cur.sample(
                nof_samples=nof_samples_per_obs,
                sample_kwargs=sample_kwargs,
                verbose=verbose,
            )
            th_list.append(th_i)
        self.th_total = np.vstack(th_list)  # (N_total, D)

        # ── Compute distance matrix ────────────────────────────────────────────
        # distances[n, θ, seed] = MSE(sim(θ, seed), y_0[n])
        dist_func      = self.r2omc_list[0].dist_2
        nof_seeds      = self.r2omc_list[0].seeds.shape[0]
        distances      = np.zeros((self.N_obs, self.th_total.shape[0], nof_seeds))
        for n, r2omc_cur in enumerate(self.r2omc_list):
            d = dist_func(self.th_total, r2omc_cur.seeds, r2omc_cur.y_0)  # (N_seeds, N_total)
            distances[n] = d.T                                              # (N_total, N_seeds)
        self.distances = distances  # (N_obs, N_total, N_seeds)

        # ── Step (i): minimax quantile filter (optional) ──────────────────────
        # score(θ) = max_n  min_seed  distances[n, θ, seed]
        d_min  = distances.min(axis=2)          # (N_obs, N_total)  best seed per obs
        score  = d_min.max(axis=0)              # (N_total,)        worst obs per theta
        self.score = score

        if quantile is not None:
            threshold     = np.quantile(score, quantile)
            mask_filter   = score <= threshold   # (N_total,)
        else:
            mask_filter   = np.ones(self.th_total.shape[0], dtype=bool)

        th_filtered       = self.th_total[mask_filter]   # (N_filtered, D)
        distances_filtered = distances[:, mask_filter, :] # (N_obs, N_filtered, N_seeds)
        self.th_filtered  = th_filtered

        if verbose >= 1:
            print(f"Step (i) : {mask_filter.sum()}/{self.th_total.shape[0]} candidates kept "
                  f"(quantile={quantile})")

        # ── Step (ii): exponential weighted resampling ────────────────────────
        # weight_inside(θ) = ∏_n  Σ_seed  exp(−distances[n, θ, seed] / temperature)
        temperature = sample_kwargs["temperature"]
        if temperature is None:
            d_min_filtered = distances_filtered.min(axis=2)          # (N_obs, N_filtered)
            eps_1_mean     = float(np.mean([r.eps_1 for r in self.r2omc_list]))
            temperature    = float(max(d_min_filtered.mean(), eps_1_mean))
        self.temperature = temperature

        # log-sum-exp per observation for numerical stability, then sum logs across obs
        # log_w_inside[θ] = Σ_n  log Σ_seed exp(−d[n,θ,seed] / T)
        log_w_inside = np.sum(
            np.log(np.sum(np.exp(-distances_filtered / temperature), axis=2) + 1e-300),
            axis=0,
        )  # (N_filtered,)

        log_prior    = self.r2omc_list[0].prior.logpdf(th_filtered)  # (N_filtered,)
        log_w        = log_prior + log_w_inside
        log_w       -= log_w.max()          # shift for numerical stability
        w            = np.exp(log_w)
        w_norm       = w / w.sum()

        self.weight_inside = np.exp(log_w_inside)
        self.weight_prior  = np.exp(log_prior)
        self.w_norm        = w_norm

        # resample
        n_positive = int(np.sum(w_norm > 0))
        if n_positive < nof_samples:
            warnings.warn(
                f"Only {n_positive} candidates have positive weight after step (ii) "
                f"(need {nof_samples}). Returning top-{nof_samples} by weight. "
                f"To improve: increase `nof_samples_per_obs` (currently {nof_samples_per_obs}) "
                f"or raise `quantile` (currently {quantile})."
            )
            th_selected = th_filtered[np.argsort(w_norm)[::-1]][:nof_samples]
        else:
            indices     = np.random.choice(len(th_filtered), size=nof_samples,
                                           replace=False, p=w_norm)
            th_selected = th_filtered[indices]

        if verbose >= 1:
            ess = 1.0 / np.sum(w_norm ** 2)
            print(f"Step (ii): {nof_samples} samples returned  "
                  f"(ESS={ess:.1f}, temperature={temperature:.4e})")

        self.th_accepted = th_filtered[w_norm > 0]
        self.th_selected = th_selected
        self.samples     = th_selected
        return th_selected


