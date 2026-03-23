import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from .base import InferenceBase

# Create Sequential Monte Carlo Inference (SMC) model
class SMCInference(InferenceBase):
    def __init__(
            self, 
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy),
            tolerance_sequence: list[float],
    ):
        
        # Compute the dimesnions for the prior and the observation
        dim = prior.dim
        dim_y = observation.shape[1]

        # Intialize the base class
        super().__init__("SMC Inference", prior, simulator, observation, dim, dim_y)
        
        # SMC parameters

        self.tolerance_sequence = tolerance_sequence # list of the tolerance sequence
        self.posterior = None # To store the final posterior particles 
        self.all_particles = [] # To store the accepted particles for plotting

    def fit(self, budget: int = 1_000):
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
            self.all_particles.append(accepted_particles) # Store for plotting
            #accepted_weights = self.weights[accepted_indices]

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
            #noise = np.random.normal(0, 0.01, size=resampled_particles.shape) # Small noise
            #particles += noise
            
            # Reinitialize particles 
            self.particles = particles

            # Reinitialize weights
            weights = np.ones(budget) / budget

            print(f"Round {i+1} complete. Accepted particles: {len(accepted_particles)}")

        # Set posterior to accepted particles from the last round
        self.posterior = self.all_particles[-1]
        #print(f"posterior shape: {self.posterior.shape}")

        return self.all_particles

    def sample(self, nof_samples: int = 100):
        '''
        Draw samples from the posterior. if fewer samples than requested
        were accepted, return all available samples.
        '''
        if self.posterior is None:
            raise ValueError("Posterior is not completed yet.")
        
        # Handle the case where fewer samples were accepted
        if nof_samples > len(self.posterior):
            print(f"Only {len(self.posterior)} accepted samples available. Return all")
            samples = self.posterior
            return samples
        else:
            # Sample from the posterior
            samples = self.posterior[:nof_samples]
            print(f"Final posterior: {samples.shape}")
            return samples
        


