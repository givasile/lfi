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

        self.inference_method = MCABC(self.simulator.sample_pytorch,
                                      self.prior.return_sbi_object(),
                                      distance = default_kwargs["distance"]
                                      )

    def sample(self, nof_samples: int = 100, sample_kwargs: dict=None):
        quantile = nof_samples / self.budget
        self.posterior = self.inference_method(
            x_o = torch.as_tensor(self.observation),
            num_simulations = self.budget,
            quantile=quantile
        )
        return self.posterior.numpy()

    

class SBI_SMCABC(InferenceBase):
    def __init__(
            self,
            prior:BasePrior,
            simulator:BaseSimulator,
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
        super().__init__("sbi_smcabc", prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1000, fit_kwargs: dict=None):

        self.budget = budget

        # prepare arguments
        default_kwargs = {
            "distance": 'l2'
            }
        default_kwargs.update(fit_kwargs or {})

        self.inference_method = SMCABC(self.simulator.sample_pytorch,
                                       self.prior.return_sbi_object(),
                                       distance = default_kwargs["distance"]
                                       )
        
    def sample(self, nof_samples: int = 100, sample_kwargs: dict=None):
        
        default_kwargs = {
            "num_initial_pop": int(0.1 * self.budget),
            "epsilon_decay": 0.7,
            "distance_based_decay": True
           }
        default_kwargs.update(sample_kwargs or {})

        self.posterior = self.inference_method(
            x_o = torch.as_tensor(self.observation),
            num_particles = nof_samples,
            num_initial_pop = default_kwargs["num_initial_pop"],
            num_simulations = self.budget,
            epsilon_decay = default_kwargs["epsilon_decay"],
            distance_based_decay = default_kwargs["distance_based_decay"]
        )
        return self.posterior.numpy()