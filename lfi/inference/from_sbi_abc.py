import sbi 
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import MCABC, SMCABC
import matplotlib.pyplot as plt
import torch
import sbi.analysis
import numpy as np
import typing
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
        # prepare prior and simulator
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)

        # Parameters dimension
        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__("sbi_mcabc", prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1000, fit_kwargs: dict=None):
        
        self.budget = budget

        #prepare arguments
        default_kwargs = {
            "distance": 'l2'
        }
        default_kwargs.update(fit_kwargs or {})

        self.inference_method = MCABC(self.prior.return_sbi_object(), 
                                      self.simulator.sample_pytorch,
                                      distance = default_kwargs["distance"]
                                      )

    def sample(self, nof_samples: int = 100, sample_kwargs: dict=None):
        budget = self.budget # Retrieve the budget stored in fit()
        quantile = nof_samples / budget
        self.posterior = self.inference_method(
            self.observation,
            num_simulations = budget,
            quantile = quantile
        )
        return self.posterior.numpy()

    

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
        

