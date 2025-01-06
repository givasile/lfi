import sbi 
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import MCABC, SMCABC
import matplotlib.pyplot as plt
import torch
import sbi.analysis
import numpy as np
import typing
import pandas as pd
import timeit
from .base import InferenceBase
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from torch.distributions import Distribution

class ABCBase(InferenceBase):
    def __init__(
            self,
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
    ):
        
        self.name = name

        # Prepare prior and simulator
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior)
        print("Sbi_prior type: ", type(sbi_prior))
        sim = process_simulator(simulator, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)

        # Paremeters of the prior
        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__(name, sbi_prior, sim, observation, dim, dim_y)
        print("Sbi_prior inferencebase: ", type(self.prior))
        print("Sbi_prior sample: ", hasattr(sbi_prior, 'sample'))

    def fit(self, budget: int = 1_000, *args, **kwargs):
        raise NotImplementedError
        
    def sample(self, nof_samples: int = 100, *args, **kwargs):
        raise NotImplementedError
    
    def fit_and_sample(self, budget, num_samples):
        tic = timeit.default_timer()
        # self.fit(budget)
        samples = self.sample(num_samples, budget)
        toc = timeit.default_timer()
        print(f"\nSampling time: {toc - tic:.2f} seconds")
        return samples, toc-tic
        
class SBI_MCABC(ABCBase):
    def __init__(self, prior, simulator, observation):
        print("Prior type passed to SBI_MCABC :", type(prior))
        print("Prior :", prior)
        super().__init__("MCABC", prior, simulator, observation)
        print("SBI_MCABC.sample type: ", type(self.prior))
        print("SBI_MCABC.sample Does self.prior has sample?: ", hasattr(self.prior, 'sample'))
        print("SBI_MCABC.sample self.prior: ", self.prior)

    def fit(self, budget: int=1_000):
            pass

    def sample(self, nof_samples: int=100, budget: int = 1000,):
        print("Prior:", type(self.prior))
        print("Inside SBI_MCABC.sample")
        print("self.prior type:", type(self.prior))
        print("Does self.prior have sample?:", hasattr(self.prior, 'sample'))
        print("Type of self.prior.sample:", type(self.prior.sample))
        print("Is self.prior.sample callable?:", callable(self.prior.sample))
        self.inference = MCABC(self.prior, self.simulator)
        print("SBI_MCABC.sample after inference: ", type(self.prior))
        print("SBI_MCABC.sample after inference: ", hasattr(self.prior, 'sample'))

        quantile = nof_samples / budget
        self.posterior = self.inference(
             self.observation, 
             num_simulations=budget,
             quantile=quantile
             )
        return self.posterior
    

class SBI_SMCABC(ABCBase):
    def __init__(self, prior, simulator, observation):
        super.__init__("SMCABC", prior, simulator, observation)

    def fit(self, budget: int = 1000):
        pass

    def sample(self, nof_samples: int = 100, budget: int = 1000, epsilon_decay: float = 0.1):
        self.inference = SMCABC(self.prior, self.simulator)
        num_initial_pop = budget / 2
        self.posterior = self.inference(
            self.observation,
            num_particles = nof_samples,
            num_initial_pop = num_initial_pop,
            num_simulations = budget,
            epsilon_decay = epsilon_decay,
            )
        
        return self.posterior
        

