import torch
import pandas as pd
from typing import List
import numpy as np

class BaseGroundTruth:
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim

    def sample(self, nof_samples: int):
        pass

class Gaussian(BaseGroundTruth):
    def __init__(self, dim: int, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        super().__init__("gaussian", dim=dim)

    def return_samples(self, nof_samples: int):
        """numpy code"""
        return np.random.normal(self.mu, self.sigma, (nof_samples, self.dim))


class GaussianMixture(BaseGroundTruth):
    def __init__(
            self,
            dim: int,
            mu: List[float],
            sigma: List[float],
            weights: List[float]
    ):
        """
        Initializes a Gaussian Mixture Model with specified parameters.

        Args:
            dim (int): Dimensionality of each Gaussian component.
            mu (List[float]): List of mean values, one per component.
            sigma (List[float]): List of standard deviations (diagonal values), one per component.
            weights (List[float]): Mixture weights for each component.

        Raises:
            ValueError: If the lengths of `mu`, `sigma`, or `weights` are inconsistent.
        """
        super().__init__("gaussian_mixture", dim=dim)

        # Validate inputs
        if len(mu) != len(sigma) or len(mu) != len(weights):
            raise ValueError("The lengths of `mu`, `sigma`, and `weights` must match.")

        # Assign attributes
        self.dim = dim
        self.mu = mu
        self.sigma = sigma
        self.weights = weights

        # Repeat `mu` and `sigma` values along the dimension
        means = torch.tensor([[m] * dim for m in mu], dtype=torch.float32)  # Repeat each mean `dim` times
        covs = torch.stack([torch.diag(torch.tensor([s] * dim, dtype=torch.float32)) for s in sigma])  # Repeat each sigma `dim` times

        # Initialize the mixture model
        self.distribution = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(probs=torch.tensor(weights, dtype=torch.float32)),
            component_distribution=torch.distributions.MultivariateNormal(
                loc=means,
                covariance_matrix=covs
            )
        )

    def return_samples(self, nof_samples: int):
        return self.distribution.sample((nof_samples,)).numpy()


# class FromSamples(BaseGroundTruth):
#     def __init__(self, path: str):
#         self.path = path
#         super().__init__("from_samples")
#
#     def return_samples(self, nof_samples: int):
#         samples = torch.tensor(pd.read_csv(self.path).values, dtype=torch.float32)
#         assert samples.shape[0] >= nof_samples, f"Number of samples requested ({nof_samples}) exceeds the number of samples in the file ({samples.shape[0]})"
#         return samples[:nof_samples]

