import sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE_C, FMPE, NPE_A
import matplotlib.pyplot as plt
import torch
import sbi.analysis
import sbi.neural_nets
import numpy as np
import jax
import jax.numpy as jnp
import typing
import pandas as pd
import seaborn as sns
import timeit
from .base import InferenceBase
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator

class NPEBase(InferenceBase):
    def __init__(
            self, 
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator, 
            observation: np.ndarray, # (1, Dy).
    ):
        self.name = name

        # prepare prior and simulator
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior)
        sim = process_simulator(simulator, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)
        # Parameters of the prior
        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__(name, prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1_000, *args, **kwargs):
        raise NotImplementedError

    def sample(self, nof_samples: int = 100, *args, **kwargs):
        if self.posterior is None:
            raise ValueError("Posterior is not trained yet.")
        return self.posterior.sample((nof_samples,), x=torch.Tensor(self.observation))

    def fit_and_sample(self, budget, num_samples):
        tic = timeit.default_timer()
        self.fit(budget)
        samples = self.sample(num_samples)
        toc = timeit.default_timer()
        print(f"\nTraining/Sampling time: {toc - tic:.2f} seconds")
        return samples, toc - tic

    def plot_training_summary(self, budget, savefig=None, num_components=None):
        fig, ax = plt.subplots()

        # Create the base title
        title = f"{self.name}: D={self.dim}, budget={budget}"
        # Add num_components to the title if provided
        if num_components is not None:
            title += f", num_components ={num_components}"
        ax.set_title(title)
        ax.plot(self.inference_method.summary["training_loss"], ".-", label = "tr")
        ax.plot(self.inference_method.summary["validation_loss"], ".-", label = "val")
        ax.set_xlim(1,1000),
        ax.set_xlabel("Epoch"),
        ax.set_ylabel("Loss = Negative Log Likelihood"),
        ax.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()
        return fig, ax
    

class NPEASingleRound(NPEBase):
    def __init__(self, prior, simulator, observation):       
        super().__init__("NPE-A (single round)", prior, simulator, observation)

    def fit(self, budget: int = 1_000, num_components=10):
        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations=budget)

        self.inference_method = NPE_A(self.prior, num_components=num_components)
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=500,
            max_num_epochs=1000,
            final_round=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior
    
    def fit_and_sample(self, budget, num_samples, num_components = 10):
        tic = timeit.default_timer()
        self.fit(budget, num_components)
        samples = self.sample(num_samples)
        toc = timeit.default_timer()
        print(f"\nTraining/Sampling time: {toc - tic:.2f} seconds")
        return samples, toc - tic
    
class NPECSingleRound(NPEBase):
    def __init__(self, prior, simulator, observation):
        super().__init__("NPE-C (single round)", prior, simulator, observation)

    def fit(self, budget, density_estimator=None):
        theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations=budget)

        if density_estimator is None:
            density_estimator = sbi.neural_nets.posterior_nn(
                model='nsf',
                hidden_features = 100,
                num_transforms=8,
                z_score_x = "independent",
                z_score_theta = "independent",
            )

        self.inference_method = NPE_C(self.prior, density_estimator=density_estimator)
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=500,
            max_num_epochs=1000,
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior
    
    def fit_and_sample(self, budget, num_samples, density_estimator=None):
        tic = timeit.default_timer()
        self.fit(budget, density_estimator)
        samples = self.sample(num_samples)
        toc = timeit.default_timer()
        print(f"\nTraining/Samling time: {toc - tic:.2f} seconds")
        return samples, toc - tic
    
class FMPESingleRound(NPEBase):
    def __init__(self, prior, simulator, observation):
        super().__init__("FMPE (single round)", prior, simulator, observation)

    def fit(self, budget):
        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator, self.prior, num_simulations=budget)

        self.inference_method = FMPE(self.prior)
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=500,
            max_num_epochs=1000
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior
    
                
    


