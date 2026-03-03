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
        default_kwargs = {
            "tolerance_sequence": [1.0, 0.5, 0.25, 0.1],
        }
        default_kwargs.update(fit_kwargs or {})
        self.tolerance_sequence = default_kwargs["tolerance_sequence"]

        # Initialize particles from the prior
        self.particles = self.prior.sample_numpy(budget)

        # Initialize weights uniformly
        self.weights = np.ones(budget) / budget

        if verbose >= 2:
            print("Starting Sequential Monte Carlo Inference")

        for i, tolerance in enumerate(self.tolerance_sequence):
            if verbose >= 2:
                print(f"Round {i+1}/{len(self.tolerance_sequence)}, Tolerance: {tolerance}")

            # Simulate data for each particle
            sim = self.simulator.sample_numpy(self.particles)

            # Compute distances from the observation
            distances = np.linalg.norm(self.observation - sim, axis=1)

            # Accept particles where the distance is within the tolerance
            accepted_indices = np.where(distances < tolerance)[0]
            accepted_particles = self.particles[accepted_indices]
            self.all_particles.append(accepted_particles)

            # if no particles are accepted, break the loop
            if accepted_particles.size == 0:
                warnings.warn(f"No accepted particles at tolerance {tolerance}. Stopping early.")
                break

            # Update particles and weights
            weights = np.exp(-distances[accepted_indices] / tolerance)
            weights /= np.sum(weights)

            # Resample particles
            resample_indices = np.random.choice(
                len(accepted_particles), size=budget, replace=True, p=weights)
            resampled_particles = accepted_particles[resample_indices]

            # Perturb the resampled particles
            perturbation_std = tolerance * 0.5
            particles = resampled_particles + np.random.normal(
                0, perturbation_std, size=resampled_particles.shape)

            self.particles = particles
            self.weights = np.ones(budget) / budget

            if verbose >= 2:
                print(f"Round {i+1} complete. Accepted particles: {len(accepted_particles)}")

        # Set posterior to accepted particles from the last round
        self.posterior = self.all_particles[-1]

        if verbose >= 1:
            print(f"SMCInference: {len(self.posterior)} posterior particles after {len(self.all_particles)} rounds")

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
        default_kwargs = {
            "key": jax.random.PRNGKey(0),
            "tolerance_sequence": [1.0, 0.5, 0.25, 0.1],
        }
        default_kwargs.update(fit_kwargs or {})
        key = default_kwargs["key"]
        self.tolerance_sequence = default_kwargs["tolerance_sequence"]

        sim_fn = self.simulator.jax_cr_simulator_paired()
        obs = jnp.array(self.observation)  # (1, dim_y) — broadcasts over batch

        # Initialize particles from the prior
        key, subkey = jax.random.split(key)
        particles = self.prior.sample_jax(subkey, budget)  # (budget, dim)

        if verbose >= 2:
            print("Starting Sequential Monte Carlo Inference (JAX)")

        for i, tolerance in enumerate(self.tolerance_sequence):
            if verbose >= 2:
                print(f"Round {i+1}/{len(self.tolerance_sequence)}, Tolerance: {tolerance}")

            seeds = jnp.arange(budget)
            sim = sim_fn(particles, seeds)  # (budget, dim_y)

            distances = jnp.linalg.norm(sim - obs, axis=1)  # (budget,)

            accepted_indices = jnp.where(distances < tolerance)[0]
            accepted_particles = particles[accepted_indices]
            self.all_particles.append(np.array(accepted_particles))

            if accepted_particles.shape[0] == 0:
                warnings.warn(f"No accepted particles at tolerance {tolerance}. Stopping early.")
                break

            weights = jnp.exp(-distances[accepted_indices] / tolerance)
            weights = weights / jnp.sum(weights)

            # Resample
            key, subkey = jax.random.split(key)
            resample_indices = jax.random.choice(
                subkey, accepted_particles.shape[0], shape=(budget,), replace=True, p=weights)
            resampled = accepted_particles[resample_indices]

            # Perturb
            key, subkey = jax.random.split(key)
            perturbation_std = tolerance * 0.5
            particles = resampled + jax.random.normal(subkey, shape=resampled.shape) * perturbation_std

            if verbose >= 2:
                print(f"Round {i+1} complete. Accepted particles: {accepted_particles.shape[0]}")

        self.posterior = self.all_particles[-1]

        if verbose >= 1:
            print(f"SMCInferenceJAX: {len(self.posterior)} posterior particles after {len(self.all_particles)} rounds")

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
