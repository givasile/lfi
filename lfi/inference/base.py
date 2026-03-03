import numpy as np
import typing
import pandas as pd
import matplotlib.pyplot as plt
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from lfi.visualization import plot_pairwise_posterior

class InferenceBase:
    supports_multiple_observations: bool = False

    def __init__(
            self,
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy) or (N_obs, Dy)
            dim: int,
            dim_y: int,
        ):
        self.name = name
        self.prior = prior
        self.simulator = simulator
        self.observation = observation

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
        N_obs = self.observation.shape[0]
        if not self.supports_multiple_observations:
            assert N_obs == 1, (
                f"{self.name} only supports a single observation (shape (1, D_y)), got {N_obs}."
            )

    def fit(self, budget: int = 1_000, fit_kwargs: typing.Optional[dict] = None, verbose: int = 1):
        """Fit the posterior distribution to data.
        This step may do nothing if the inference method does not require fitting, e.g. ABC methods.
        Otherwise, it sets the self.posterior attribute with the fitted posterior distribution.

        Args:
            budget: Number of (a) unique calls to simulator or (b) number of training samples to generate for fitting,
                whatever is more costly depending on the inference method.
                If the method does not require fitting, it is (a).
                If the method requires fitting, it is (b).
            fit_kwargs: Method-specific arguments for modeling and fitting the posterior distribution.
            verbose: Verbosity level. 0 = silent, 1 = one line per step, 2 = detailed output.
        """
        raise NotImplementedError

    def sample(self, nof_samples: int = 100, sample_kwargs: typing.Optional[dict] = None, verbose: int = 1):
        """Sample from the posterior distribution.
        This step sets the self.samples attribute with the sampled posterior distribution.

        Args:
            nof_samples: The number of samples to draw from the posterior
            sample_kwargs: Method-specific arguments for sampling from the posterior distribution.
            verbose: Verbosity level. 0 = silent, 1 = one line per step, 2 = detailed output.
        """
        raise NotImplementedError

    def fit_and_sample(
            self,
            budget: int,
            nof_samples: int,
            fit_kwargs: dict = None,
            sample_kwargs: dict = None,
            verbose: int = 1,
    ):
        """Fit the posterior distribution and sample from it.

        Args:
            budget: Number of training samples to generate for fitting.
            nof_samples: The number of samples to draw from the posterior
            fit_kwargs: Method-specific arguments for modeling and fitting the posterior distribution.
            sample_kwargs: Method-specific arguments for sampling from the posterior distribution.
            verbose: Verbosity level. 0 = silent, 1 = one line per step, 2 = detailed output.
        """
        self.fit(budget, (fit_kwargs or {}), verbose=verbose)
        return self.sample(nof_samples, (sample_kwargs or {}), verbose=verbose)

    def plot_posterior_samples(
            self,
            samples: typing.Union[None, np.ndarray] = None, # (N, Dy)
            samples_gt: typing.Union[None, np.ndarray] = None, # (N, Dy)
            subset_dims: typing.Union[None, list] = None,
            limits: typing.Union[None, list] = None,
            savefig: typing.Union[None, str] = None,
            show: bool = True,
            title: typing.Union[None, str] = None,
    ):
        if samples is None:
            samples = self.samples
        if title is None:
            title = f"Posterior samples — {self.name}"
        g = plot_pairwise_posterior(
            samples=samples,
            subset_dims=subset_dims,
            limits=limits,
            savefig=savefig,
            samples_gt=samples_gt,
            title=title,
        )
        if show:
            plt.show(block=False)
        return g

    @staticmethod
    def store(samples, path):
        samples_df = pd.DataFrame(
            samples,
            columns=[f"x_{i + 1}" for i in range(samples.shape[1])],
        )
        samples_df.to_csv(path, index=False)
