import numpy as np
import typing
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
        self.samples = None

        self._assertions()

    def _assertions(self):
        th = self.prior.sample_numpy(10)
        y = self.simulator.sample_numpy(th)
        assert th.shape[1] == self.dim, f"Expected {self.dim} dimensions in the prior, got {th.shape[1]}."
        assert y.shape[1] == self.dim_y, f"Expected {self.dim_y} dimensions in the output, got {y.shape[1]}."

    def fit(self, budget: int = 1_000, fit_kwargs: typing.Optional[dict] = None):
        """Fit the posterior distribution to data.
        This step may do nothing if the inference method does not require fitting, e.g. ABC methods.
        Otherwise, it sets the self.posterior attribute with the fitted posterior distribution.

        Args:
            budget: Number of training samples to generate for fitting.
            fit_kwargs: Method-specific arguments for modeling and fitting the posterior distribution.
        """
        raise NotImplementedError

    def sample(self, nof_samples: int = 100, sample_kwargs: typing.Optional[dict] = None):
        """Sample from the posterior distribution.
        This step sets the self.samples attribute with the sampled posterior distribution.

        Args:
            nof_samples: The number of samples to draw from the posterior
            sample_kwargs: Method-specific arguments for sampling from the posterior distribution.
        """
        raise NotImplementedError

    def fit_and_sample(
            self,
            budget: int,
            nof_samples: int,
            fit_kwargs: dict = None,
            sample_kwargs: dict = None
    ):
        """Fit the posterior distribution and sample from it.

        Args:
            budget: Number of training samples to generate for fitting.
            nof_samples: The number of samples to draw from the posterior
            fit_kwargs: Method-specific arguments for modeling and fitting the posterior distribution.
            sample_kwargs: Method-specific arguments for sampling from the posterior distribution.
        """
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
