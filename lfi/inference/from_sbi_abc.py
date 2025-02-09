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

class SBI_MCABC(InferenceBase):
    def __init__(
            self,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
    ):
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        print(type(sbi_prior))
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)

        #sbi_prior = prior.return_sbi_object()

        # Parameters of the prior
        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__("SBI_MCABC", prior, simulator, observation, dim, dim_y)

    def fit(self, *args, **kwargs):
        print(type(self.prior,))
        print(type(self.simulator))
        self.inference_method = MCABC(self.prior, self.simulator)

    def sample(self, budget: int=1000, nof_samples: int = 100, *args, **kwargs):
        self.budget = budget
        self.nof_samples = nof_samples
        quantile = self.nof_samples / self.budget
        self.posterior = self.inference_method(
            self.observation,
            num_simulations = self.budget,
            quantile=quantile
        )
        return self.posterior

        


    

# class SBI_SMCABC(ABCBase):
#     def __init__(self, prior, simulator, observation):
#         super.__init__("SMCABC", prior, simulator, observation)

#     def fit(self, budget: int = 1000):
#         pass

#     def sample(self, nof_samples: int = 100, budget: int = 1000, epsilon_decay: float = 0.1):
#         self.inference = SMCABC(self.prior, self.simulator)
#         num_initial_pop = budget / 2
#         self.posterior = self.inference(
#             self.observation,
#             num_particles = nof_samples,
#             num_initial_pop = num_initial_pop,
#             num_simulations = budget,
#             epsilon_decay = epsilon_decay,
#             )
        
#         return self.posterior
        

