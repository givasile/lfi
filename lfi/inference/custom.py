import torch
import torch.nn as nn
import torch.distributions as dist
from lfi.inference.base import InferenceBase

class MDN(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.dim = dim
        self.K = K

        self.net = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.dim * K)  # Output for means and log-sigmas
        )

    def forward(self, x):
        phi = self.net(x)  # Shape: [batch_size, K * 2 * D]
        phi = phi.view(-1, self.K, self.dim, 2)  # Reshape into [batch_size, K, D, 2]

        # Split into mean and log(sigma)
        mu = phi[..., 0]  # [batch_size, K, D]
        sigma = torch.exp(phi[..., 1]) + 0.1  # Stabilize variance with small constant

        # Independent Normal distributions for each component
        component_distributions = dist.Independent(
            dist.Normal(loc=mu, scale=sigma),
            reinterpreted_batch_ndims=1
        )

        # Uniform mixture weights
        mixture_weights = torch.ones(x.size(0), self.K, device=x.device) / self.K
        mixture = dist.MixtureSameFamily(
            mixture_distribution=dist.Categorical(probs=mixture_weights),
            component_distribution=component_distributions
        )
        return mixture

    def logprob(self, th, x):
        return self.forward(x).log_prob(th)

    def loss(self, th, x):
        return -self.logprob(th, x).mean()

    def sample(self, nof_samples, x):
        return self.forward(x).sample((nof_samples,))


class MixtureDensityNetwork(InferenceBase):
    def __init__(self, prior, simulator, observation):
        name = "mdn"
        self.net = None
        dim = prior.dim
        dim_y = simulator.sample_numpy(prior.sample_numpy(1)).shape[1]
        super().__init__(name, prior, simulator, observation, dim, dim_y)


    def fit(
            self,
            budget: int = 1_000,
            fit_kwargs: dict = None,
    ):
        default_kwargs = {
            "nof_components": 10,
            "nof_epochs": 1000,
        }
        default_kwargs.update(fit_kwargs or {})

        # prepare dataset
        theta = self.prior.sample_pytorch(budget) # Shape: [budget, D]
        x = self.simulator.sample_pytorch(theta) # Shape: [budget, Dy]

        # Initialize the network
        self.net = MDN(self.dim, default_kwargs["nof_components"])

        # training loop
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_list = []
        for i in range(default_kwargs["nof_epochs"]):
            optimizer.zero_grad()
            loss = self.net.loss(theta, x)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {i}, Loss: {loss.item()}")

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        samples = self.net.sample(nof_samples, torch.Tensor(self.observation))
        return samples.squeeze().detach().numpy()
