import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from .base import InferenceBase
import typing
from typing import Optional 


# Create Rejection abc class
class ABCRejection(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
            eps: Optional[float]=None,
            quantile: Optional[float]=None,
    ):
        # Compute dimensions from the prior and the observation
        dim = prior.dim
        dim_y = observation.shape[1]

        # Intialize the base class
        super().__init__("ABC Rejection", prior, simulator, observation, dim, dim_y)

        # ABC rejection tolerance
        self.eps = eps
        self.quantile = quantile

        self.posterior = None

    def fit(self, budget: int = 1_000):
        # Sample from the prior
        thetas = self.prior.sample_numpy(budget)
        
        # Compute the simulated data 
        sim = self.simulator.sample_numpy(thetas)       

        # Compute the distances
        distances = np.linalg.norm(self.observation - sim, axis=1)      

        if self.eps is not None:
            # Identify accepted samples (satisfy the distance criterion)
            accepted_indices = np.where(distances < self.eps)[0]
            #print("First 10 accepted distances: ", distances[best_indices][:10])
            accepted_samples = thetas[accepted_indices]
        elif self.quantile is not None:
            num_top_samples = int(budget * self.quantile)
            best_indices = np.argsort(distances)
            accepted_samples = thetas[best_indices[:num_top_samples]]
        else:
            raise ValueError("one of epsilon or quantile has to be passed")

        self.posterior = accepted_samples

        
    
    def sample(self, nof_samples: int = 100):
        '''
        Draw samples from the posterior. If fewer samples than requested 
        were accepted, returns all available samples.
        '''

        if self.posterior is None:
            raise ValueError("Posterior is not computed yet.")
        
        # Handle case where fewer than requested samples were accepted
        available_samples = self.posterior.shape[0]
        if nof_samples > available_samples:
            print(f"Only {available_samples} accepted samples available. Returning all")
            samples = self.posterior
            return samples
        else:
            # Sample from the posterior
            samples = self.posterior[:nof_samples]
            return samples
        