import numpy as np
import typing
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
import lfi

class InferenceBase:
    def __init__(
            self,
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
            dim: int,
            dim_y: int,
        ):
        self.name = name
        self.prior = prior
        self.simulator = simulator
        self.observation = observation # (1, Dy)

        self.dim_y = dim_y
        self.dim = dim

        self.posterior = None

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):
        """Estimate the posterior distribution.

        Args:
            budget: The simulation budget for training the posterior
            fit_kwargs: Any additional arguments passed to the fit method
        """
        raise NotImplementedError

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        """Sample from the posterior distribution.

        Args:
            nof_samples: The number of samples to draw from the posterior
            sample_kwargs: Any additional arguments passed to the sample method
        """
        raise NotImplementedError

    def fit_and_sample(
            self,
            budget: int,
            nof_samples: int,
            fit_kwargs: dict = None,
            sample_kwargs: dict = None
    ):
        self.fit(budget, (fit_kwargs or {}))
        return self.sample(nof_samples, (sample_kwargs or {}))

    @staticmethod
    def plot_posterior_samples(
            samples: np.ndarray, # (N, Dy)
            samples_gt: typing.Union[None, np.ndarray] = None, # (N, Dy)
            subset_dims: typing.Union[None, list] = None,
            limits: typing.Union[None, list] = None,
            savefig: typing.Union[None, str] = None,
    ):
        g = lfi.visualization.plot_pairwise_posterior(
            samples=samples,
            subset_dims=subset_dims,
            limits=limits,
            savefig=savefig,
            samples_gt=samples_gt
        )
        plt.show()
        return g

    @staticmethod
    def store(samples, path):
        samples_df = pd.DataFrame(
            samples,
            columns=[f"x_{i + 1}" for i in range(samples.shape[1])],
        )
        samples_df.to_csv(path, index=False)



    



