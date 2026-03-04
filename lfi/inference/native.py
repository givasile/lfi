import warnings
import numpy as np
import jax
import jax.numpy as jnp
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from .base import InferenceBase
from typing import Optional


class ABCRejection(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
    ):
        # Compute dimensions from the prior and the observation
        dim = prior.dim
        dim_y = observation.shape[1]

        # Intialize the base class
        super().__init__("ABC Rejection", prior, simulator, observation, dim, dim_y)

        self.posterior = None

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None, verbose: int = 1):
        default_kwargs = {
            "eps": None,
            "quantile": 0.1,
        }
        default_kwargs.update(fit_kwargs or {})
        eps = default_kwargs["eps"]
        quantile = default_kwargs["quantile"]

        # Sample from the prior
        thetas = self.prior.sample_numpy(budget)

        # Compute the simulated data
        sim = self.simulator.sample_numpy(thetas)

        # Compute the distances
        distances = np.linalg.norm(self.observation - sim, axis=1)

        if eps is not None:
            # Identify accepted samples (satisfy the distance criterion)
            accepted_indices = np.where(distances < eps)[0]
            accepted_samples = thetas[accepted_indices]
        elif quantile is not None:
            num_top_samples = int(budget * quantile)
            best_indices = np.argsort(distances)
            accepted_samples = thetas[best_indices[:num_top_samples]]
        else:
            raise ValueError("one of eps or quantile has to be passed in fit_kwargs")

        self.posterior = accepted_samples
        if verbose >= 1:
            print(f"ABCRejection: {len(accepted_samples)}/{budget} samples accepted")

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None, verbose: int = 1):
        '''
        Draw samples from the posterior. If fewer samples than requested
        were accepted, returns all available samples.
        '''
        if self.posterior is None:
            raise ValueError("Posterior is not computed yet.")

        available_samples = self.posterior.shape[0]
        if nof_samples > available_samples:
            warnings.warn(f"Only {available_samples} accepted samples available. Returning all.")
            samples = self.posterior
        else:
            samples = self.posterior[:nof_samples]
        self.samples = samples
        if verbose >= 1:
            print(f"ABCRejection: {len(samples)} samples returned")
        return samples


class SMCInference(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
    ):
        dim = prior.dim
        dim_y = observation.shape[1]
        super().__init__("SMC Inference", prior, simulator, observation, dim, dim_y)

        self.tolerance_sequence = None
        self.posterior = None # To store the final posterior particles
        self.all_particles = [] # To store the accepted particles for plotting

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None, verbose: int = 1):
        """
        ``budget`` = **total** simulator calls, split equally across rounds:
            n_particles = budget // nof_rounds   (per-round particle count)

        Three scheduling modes (checked in priority order):

        1. ``quantile_sequence`` — explicit adaptive schedule. At each round the
           threshold is set to the given quantile of the *current* distance
           distribution.  E.g. ``[1.0, 0.5, 0.2, 0.1]`` keeps 100 % → 50 % →
           20 % → 10 % of particles per round.

        2. ``tolerance_sequence`` — explicit fixed absolute thresholds.

        3. Auto (default) — provide ``nof_rounds`` (default 4) and
           ``nof_samples`` (should match the value passed to ``sample()``).
           A geometric schedule from 1.0 down to ``nof_samples / n_particles``
           is computed, so the final round produces exactly ``nof_samples``
           particles:

               n_particles     = budget // nof_rounds
               final_quantile  = nof_samples / n_particles
               quantile_sequence = geomspace(1.0, final_quantile, nof_rounds)
        """
        default_kwargs = {
            "nof_rounds":        4,
            "nof_samples":       max(1, budget // 10),  # default: 10 % of budget
            "quantile_sequence": None,
            "tolerance_sequence": None,
        }
        default_kwargs.update(fit_kwargs or {})
        quantile_sequence  = default_kwargs["quantile_sequence"]
        tolerance_sequence = default_kwargs["tolerance_sequence"]

        # budget is split equally across rounds
        nof_rounds   = len(quantile_sequence or tolerance_sequence or [None] * default_kwargs["nof_rounds"])
        n_particles  = budget // nof_rounds

        # determine schedule (priority: quantile_sequence > tolerance_sequence > auto)
        if quantile_sequence is not None:
            use_quantiles = True
            schedule = quantile_sequence
        elif tolerance_sequence is not None:
            use_quantiles = False
            schedule = tolerance_sequence
        else:
            use_quantiles = True
            final_q  = default_kwargs["nof_samples"] / n_particles
            schedule = list(np.geomspace(1.0, final_q, nof_rounds))

        # Initialize particles from the prior
        self.particles = self.prior.sample_numpy(n_particles)
        self.weights = np.ones(n_particles) / n_particles

        if verbose >= 2:
            mode = "quantile" if use_quantiles else "tolerance"
            print(f"Starting SMCInference | budget={budget} | {nof_rounds} rounds × {n_particles} particles | {mode} schedule: {[round(s,3) for s in schedule]}")

        for i, sched_val in enumerate(schedule):
            # Simulate data for each particle
            sim = self.simulator.sample_numpy(self.particles)
            distances = np.linalg.norm(self.observation - sim, axis=1)

            # Determine threshold for this round
            threshold = np.quantile(distances, sched_val) if use_quantiles else sched_val

            if verbose >= 2:
                print(f"Round {i+1}/{nof_rounds}, threshold={threshold:.4f}")

            # Accept particles within threshold
            accepted_indices = np.where(distances < threshold)[0]
            accepted_particles = self.particles[accepted_indices]
            self.all_particles.append(accepted_particles)

            if accepted_particles.size == 0:
                warnings.warn(f"No accepted particles at threshold {threshold:.4f}. Stopping early.")
                break

            # Update weights and resample
            weights = np.exp(-distances[accepted_indices] / threshold)
            weights /= np.sum(weights)
            resample_indices = np.random.choice(
                len(accepted_particles), size=n_particles, replace=True, p=weights)
            resampled_particles = accepted_particles[resample_indices]

            # Perturb
            perturbation_std = threshold * 0.5
            self.particles = resampled_particles + np.random.normal(
                0, perturbation_std, size=resampled_particles.shape)
            self.weights = np.ones(n_particles) / n_particles

            if verbose >= 2:
                print(f"Round {i+1} complete. Accepted particles: {len(accepted_particles)}")

        # Set posterior to accepted particles from the last round
        self.posterior = self.all_particles[-1]

        if verbose >= 1:
            print(f"SMCInference: {len(self.posterior)} posterior particles after {len(self.all_particles)} rounds "
                  f"({nof_rounds * n_particles} total simulator calls)")

        return self.all_particles

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None, verbose: int = 1):
        '''
        Draw samples from the posterior. If fewer samples than requested
        were accepted, returns all available samples.
        '''
        if self.posterior is None:
            raise ValueError("Posterior is not completed yet.")

        if nof_samples > len(self.posterior):
            warnings.warn(f"Only {len(self.posterior)} accepted samples available. Returning all.")
            samples = self.posterior
        else:
            samples = self.posterior[:nof_samples]
        self.samples = samples
        if verbose >= 1:
            print(f"SMCInference: {len(samples)} samples returned")
        return samples


class ABCRejectionJAX(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray,  # (1, Dy)
    ):
        dim = prior.dim
        dim_y = observation.shape[1]
        super().__init__("ABC Rejection JAX", prior, simulator, observation, dim, dim_y)
        self.posterior = None

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None, verbose: int = 1):
        default_kwargs = {
            "key": jax.random.PRNGKey(0),
            "eps": None,
            "quantile": 0.1,
        }
        default_kwargs.update(fit_kwargs or {})
        key = default_kwargs["key"]
        eps = default_kwargs["eps"]
        quantile = default_kwargs["quantile"]

        key_prior, _ = jax.random.split(key)

        # Sample thetas from prior
        thetas = self.prior.sample_jax(key_prior, budget)  # (budget, dim)

        # Simulate with per-theta integer seeds
        seeds = jnp.arange(budget)
        sim_fn = self.simulator.jax_cr_simulator_paired()
        sim = sim_fn(thetas, seeds)  # (budget, dim_y)

        # Compute L2 distances (observation broadcasts over batch)
        distances = jnp.linalg.norm(sim - jnp.array(self.observation), axis=1)  # (budget,)

        if eps is not None:
            accepted_indices = jnp.where(distances < eps)[0]
            accepted_samples = thetas[accepted_indices]
        elif quantile is not None:
            num_top = int(budget * quantile)
            best_indices = jnp.argsort(distances)[:num_top]
            accepted_samples = thetas[best_indices]
        else:
            raise ValueError("one of eps or quantile has to be passed in fit_kwargs")

        self.posterior = accepted_samples
        if verbose >= 1:
            print(f"ABCRejectionJAX: {len(accepted_samples)}/{budget} samples accepted")

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None, verbose: int = 1):
        if self.posterior is None:
            raise ValueError("Posterior is not computed yet.")

        available = self.posterior.shape[0]
        if nof_samples > available:
            warnings.warn(f"Only {available} accepted samples available. Returning all.")
            samples = self.posterior
        else:
            samples = self.posterior[:nof_samples]
        self.samples = samples
        if verbose >= 1:
            print(f"ABCRejectionJAX: {len(samples)} samples returned")
        return samples


class SMCInferenceJAX(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray,  # (1, Dy)
    ):
        dim = prior.dim
        dim_y = observation.shape[1]
        super().__init__("SMC Inference JAX", prior, simulator, observation, dim, dim_y)

        self.tolerance_sequence = None
        self.posterior = None
        self.all_particles = []

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None, verbose: int = 1):
        """
        Three scheduling modes (checked in priority order):

        1. ``quantile_sequence`` — explicit adaptive schedule (quantile of
           current distances used as threshold each round).

        2. ``tolerance_sequence`` — explicit fixed absolute thresholds.

        3. Auto (default) — provide ``nof_rounds`` (default 4) and
           ``nof_samples`` (should match the value passed to ``sample()``).
           Computes a geometric schedule from 1.0 down to
           ``nof_samples / budget``:

               quantile_sequence = geomspace(1.0, nof_samples/budget, nof_rounds)

           Rule of thumb: ``budget × final_quantile == nof_samples``.
        """
        default_kwargs = {
            "key":               jax.random.PRNGKey(0),
            "nof_rounds":        4,
            "nof_samples":       max(1, budget // 10),  # default: 10 % of budget
            "quantile_sequence": None,
            "tolerance_sequence": None,
        }
        default_kwargs.update(fit_kwargs or {})
        key                = default_kwargs["key"]
        quantile_sequence  = default_kwargs["quantile_sequence"]
        tolerance_sequence = default_kwargs["tolerance_sequence"]

        # budget is split equally across rounds
        nof_rounds  = len(quantile_sequence or tolerance_sequence or [None] * default_kwargs["nof_rounds"])
        n_particles = budget // nof_rounds

        # determine schedule (priority: quantile_sequence > tolerance_sequence > auto)
        if quantile_sequence is not None:
            use_quantiles = True
            schedule = quantile_sequence
        elif tolerance_sequence is not None:
            use_quantiles = False
            schedule = tolerance_sequence
        else:
            use_quantiles = True
            final_q  = default_kwargs["nof_samples"] / n_particles
            schedule = list(np.geomspace(1.0, final_q, nof_rounds))

        sim_fn = self.simulator.jax_cr_simulator_paired()
        obs = jnp.array(self.observation)  # (1, dim_y) — broadcasts over batch

        # Initialize particles from the prior
        key, subkey = jax.random.split(key)
        particles = self.prior.sample_jax(subkey, n_particles)

        if verbose >= 2:
            mode = "quantile" if use_quantiles else "tolerance"
            print(f"Starting SMCInferenceJAX | budget={budget} | {nof_rounds} rounds × {n_particles} particles | {mode} schedule: {[round(s,3) for s in schedule]}")

        for i, sched_val in enumerate(schedule):
            seeds = jnp.arange(n_particles)
            sim = sim_fn(particles, seeds)  # (n_particles, dim_y)
            distances = jnp.linalg.norm(sim - obs, axis=1)  # (n_particles,)

            # Determine threshold for this round
            threshold = float(jnp.quantile(distances, sched_val)) if use_quantiles else sched_val

            if verbose >= 2:
                print(f"Round {i+1}/{nof_rounds}, threshold={threshold:.4f}")

            accepted_indices = jnp.where(distances < threshold)[0]
            accepted_particles = particles[accepted_indices]
            self.all_particles.append(np.array(accepted_particles))

            if accepted_particles.shape[0] == 0:
                warnings.warn(f"No accepted particles at threshold {threshold:.4f}. Stopping early.")
                break

            weights = jnp.exp(-distances[accepted_indices] / threshold)
            weights = weights / jnp.sum(weights)

            # Resample
            key, subkey = jax.random.split(key)
            resample_indices = jax.random.choice(
                subkey, accepted_particles.shape[0], shape=(n_particles,), replace=True, p=weights)
            resampled = accepted_particles[resample_indices]

            # Perturb
            key, subkey = jax.random.split(key)
            perturbation_std = threshold * 0.5
            particles = resampled + jax.random.normal(subkey, shape=resampled.shape) * perturbation_std

            if verbose >= 2:
                print(f"Round {i+1} complete. Accepted particles: {accepted_particles.shape[0]}")

        self.posterior = self.all_particles[-1]

        if verbose >= 1:
            print(f"SMCInferenceJAX: {len(self.posterior)} posterior particles after {len(self.all_particles)} rounds "
                  f"({nof_rounds * n_particles} total simulator calls)")

        return self.all_particles

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None, verbose: int = 1):
        if self.posterior is None:
            raise ValueError("Posterior is not completed yet.")

        if nof_samples > len(self.posterior):
            warnings.warn(f"Only {len(self.posterior)} accepted samples available. Returning all.")
            samples = self.posterior
        else:
            samples = self.posterior[:nof_samples]
        self.samples = samples
        if verbose >= 1:
            print(f"SMCInferenceJAX: {len(samples)} samples returned")
        return samples
