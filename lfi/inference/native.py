import numpy as np
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

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):
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

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        '''
        Draw samples from the posterior. If fewer samples than requested
        were accepted, returns all available samples.
        '''
        if self.posterior is None:
            raise ValueError("Posterior is not computed yet.")

        available_samples = self.posterior.shape[0]
        if nof_samples > available_samples:
            print(f"Only {available_samples} accepted samples available. Returning all")
            samples = self.posterior
        else:
            samples = self.posterior[:nof_samples]
        self.samples = samples
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

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):
        default_kwargs = {
            "tolerance_sequence": [1.0, 0.5, 0.25, 0.1],
        }
        default_kwargs.update(fit_kwargs or {})
        self.tolerance_sequence = default_kwargs["tolerance_sequence"]

        # Initialize particles from the prior
        self.particles = self.prior.sample_numpy(budget)

        # Initialize weights uniformly
        self.weights = np.ones(budget) / budget

        print("Starting Sequential Monte Carlo Inference")

        for i, tolerance in enumerate(self.tolerance_sequence):
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
                print(f"No accepted particles at tolerance {tolerance}")
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

            print(f"Round {i+1} complete. Accepted particles: {len(accepted_particles)}")

        # Set posterior to accepted particles from the last round
        self.posterior = self.all_particles[-1]

        return self.all_particles

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        '''
        Draw samples from the posterior. If fewer samples than requested
        were accepted, returns all available samples.
        '''
        if self.posterior is None:
            raise ValueError("Posterior is not completed yet.")

        if nof_samples > len(self.posterior):
            print(f"Only {len(self.posterior)} accepted samples available. Return all")
            samples = self.posterior
        else:
            samples = self.posterior[:nof_samples]
            print(f"Final posterior: {samples.shape}")
        self.samples = samples
        return samples
