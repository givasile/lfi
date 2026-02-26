from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from .base import InferenceBase
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import sbi
    import sbi.neural_nets
    import sbi.neural_nets.embedding_nets
    from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
    from sbi.inference import simulate_for_sbi, NPE_C, FMPE, NPE_A, NPE, MCABC, SMCABC
    from sbi.utils import RestrictedPrior, get_density_thresholder
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_TORCH_MSG = "torch/sbi not installed. Install with: pip install 'lfi[torch-cpu]' or 'lfi[torch-gpu]'"


# ── Neural Posterior Estimation (NPE) ─────────────────────────────────────────
#
# Implemented:
#   NPEASingleRound  — Fast ε-free inference with Bayesian conditional density estimation
#   NPECSingleRound  — Automatic Posterior Transformation (APT / NPE-C)
#   NPECMultiRound   — Sequential NPE-C with proposal refinement
#   FMPESingleRound  — Flow Matching Posterior Estimation
#   BayesFlow        — NPE with learned FC embedding network
#
# Not implemented:
#   TSNPE            — Truncated proposals (unstable in practice)
#   All-in-one SBI   — https://arxiv.org/abs/2404.09636
#   Compositional Score Modeling for SBI
#   Sequential Neural Score Estimation


class NPEBase(InferenceBase):
    def __init__(
            self,
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray, # (1, Dy)
            embedding_net: Optional[nn.Module] = None
    ):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        self.name = name
        self.embedding_net = embedding_net

        # prepare prior and simulator
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)
        self.sbi_prior = sbi_prior

        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__(name, prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        if self.posterior is None:
            raise ValueError("Posterior is not trained yet.")
        y = self.posterior.sample((nof_samples,), x=torch.Tensor(self.observation))
        self.samples = y.detach().numpy()
        return self.samples

    def plot_training_summary(self, budget, savefig=None, num_components=None):
        fig, ax = plt.subplots()
        title = f"{self.name}: D={self.dim}, budget={budget}"
        if num_components is not None:
            title += f", num_components ={num_components}"
        ax.set_title(title)
        ax.plot(self.inference_method.summary["training_loss"], ".-", label="tr")
        ax.plot(self.inference_method.summary["validation_loss"], ".-", label="val")
        ax.set_xlim(1, 1000)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss = Negative Log Likelihood")
        ax.legend()
        if savefig:
            plt.savefig(savefig)
        plt.show()
        return fig, ax


class NPEASingleRound(NPEBase):
    """
    Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation.
    Papamakarios, Murray. Uses the NPE-A implementation from the SBI library.
    """
    def __init__(self, prior, simulator, observation, embedding_net=None):
        super().__init__("npe_a_single_round", prior, simulator, observation, embedding_net)

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):
        default_kwargs = {
            "num_components": 10,
            "training_batch_size": 500,
            "max_num_epochs": 1000,
        }
        default_kwargs.update(fit_kwargs or {})

        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        if self.embedding_net is not None:
            x = self.embedding_net(x)

        self.inference_method = NPE_A(self.sbi_prior, num_components=default_kwargs["num_components"])

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            final_round=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


class NPECSingleRound(NPEBase):
    """
    Automatic Posterior Transformation (APT / NPE-C) for likelihood-free inference.
    https://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf
    """
    def __init__(self, prior, simulator, observation, embedding_net: Optional[nn.Module] = None):
        super().__init__("npe_c_single_round", prior, simulator, observation, embedding_net)

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):
        default_kwargs = {
            "model": "nsf",
            "hidden_features": 100,
            "num_transforms": 8,
            "num_bins": 10,
            "z_score_x": "independent",
            "z_score_theta": "independent",
            "training_batch_size": 500,
            "max_num_epochs": 1000,
            "force_first_round": True,
        }
        default_kwargs.update(fit_kwargs or {})

        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        if self.embedding_net is not None:
            x = self.embedding_net(x)

        density_estimator = sbi.neural_nets.posterior_nn(
            model=default_kwargs["model"],
            hidden_features=default_kwargs["hidden_features"],
            num_transforms=default_kwargs["num_transforms"],
            num_bins=default_kwargs["num_bins"],
            z_score_x=default_kwargs["z_score_x"],
            z_score_theta=default_kwargs["z_score_theta"],
        )

        self.inference_method = NPE_C(self.sbi_prior, density_estimator=density_estimator)

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


class NPECMultiRound(NPEBase):
    """
    Multi-round NPE-C with sequential proposal refinement.
    https://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf
    """
    def __init__(self, prior, simulator, observation, embedding_net: Optional[nn.Module] = None):
        super().__init__("npe_c_multi_round", prior, simulator, observation, embedding_net)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        default_kwargs = {
            "model": "nsf",
            "hidden_features": 100,
            "num_transforms": 8,
            "z_score_x": "independent",
            "z_score_theta": "independent",
            "training_batch_size": 500,
            "max_num_epochs": 1000,
            "num_rounds": 10,
        }
        default_kwargs.update(fit_kwargs or {})

        density_estimator = sbi.neural_nets.posterior_nn(
            model=default_kwargs["model"],
            hidden_features=default_kwargs["hidden_features"],
            num_transforms=default_kwargs["num_transforms"],
            z_score_x=default_kwargs["z_score_x"],
            z_score_theta=default_kwargs["z_score_theta"]
        )

        inference = NPE_C(self.sbi_prior, density_estimator=density_estimator)
        proposal = self.prior.return_sbi_object()
        nof_new_samples = budget // default_kwargs["num_rounds"]
        for _ in range(default_kwargs["num_rounds"]):
            theta = proposal.sample((nof_new_samples,))
            x = self.simulator.sample_pytorch(theta)
            _ = inference.append_simulations(theta, x, proposal).train(
                training_batch_size=default_kwargs["training_batch_size"],
                max_num_epochs=default_kwargs["max_num_epochs"],
                force_first_round_loss=True
            )
            posterior = inference.build_posterior().set_default_x(torch.Tensor(self.observation))
            proposal = posterior
        self.posterior = posterior
        return self.posterior


class FMPESingleRound(NPEBase):
    """
    Flow Matching Posterior Estimation (FMPE).
    https://proceedings.neurips.cc/paper_files/paper/2023/file/3663ae53ec078860bb0b9c6606e092a0-Paper-Conference.pdf
    """
    def __init__(self, prior, simulator, observation):
        super().__init__("fmpe_single_round", prior, simulator, observation)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        default_kwargs = {
            "vf_estimator": "mlp",
            "training_batch_size": 500,
            "max_num_epochs": 1000,
        }
        default_kwargs.update(fit_kwargs or {})

        theta, x = simulate_for_sbi(
            self.simulator.sample_pytorch,
            self.prior.return_sbi_object(),
            num_simulations=budget
        )

        self.inference_method = FMPE(self.sbi_prior, vf_estimator=default_kwargs["vf_estimator"])

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


class BayesFlow(NPEBase):
    """
    BayesFlow: Learning complex stochastic models with invertible neural networks.
    https://arxiv.org/pdf/2003.06281
    NPE with a learned FC embedding network.
    """
    def __init__(self, prior, simulator, observation):
        super().__init__("bayes_flow", prior, simulator, observation)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        default_kwargs = {
            "embedding_net_output_dim": 20,
            "embedding_net_num_layers": 2,
            "embedding_net_num_hiddens": 50,
            "model": "nsf",
            "hidden_features": 100,
            "num_transforms": 8,
            "z_score_x": "independent",
            "z_score_theta": "independent",
            "training_batch_size": 500,
            "max_num_epochs": 1000,
            "force_first_round": True,
        }
        default_kwargs.update(fit_kwargs or {})

        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        embedding_net = sbi.neural_nets.embedding_nets.FCEmbedding(
            input_dim=self.dim_y,
            output_dim=default_kwargs["embedding_net_output_dim"],
            num_layers=default_kwargs["embedding_net_num_layers"],
            num_hiddens=default_kwargs["embedding_net_num_hiddens"]
        )

        density_estimator = sbi.neural_nets.posterior_nn(
            model=default_kwargs["model"],
            hidden_features=default_kwargs["hidden_features"],
            num_transforms=default_kwargs["num_transforms"],
            z_score_x=default_kwargs["z_score_x"],
            z_score_theta=default_kwargs["z_score_theta"],
            embedding_net=embedding_net,
        )

        self.inference_method = sbi.inference.NPE(self.sbi_prior, density_estimator=density_estimator)

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


# ── ABC methods via SBI ────────────────────────────────────────────────────────

class SBI_MCABC(InferenceBase):
    def __init__(self, prior: BasePrior, simulator: BaseSimulator, observation: np.ndarray):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)

        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__("sbi_mcabc", prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        self.budget = budget
        default_kwargs = {"distance": "l2"}
        default_kwargs.update(fit_kwargs or {})
        self.inference_method = MCABC(
            self.simulator.sample_pytorch,
            self.prior.return_sbi_object(),
            distance=default_kwargs["distance"]
        )

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        quantile = nof_samples / self.budget
        self.posterior = self.inference_method(
            x_o=torch.as_tensor(self.observation),
            num_simulations=self.budget,
            quantile=quantile
        )
        self.samples = self.posterior.numpy()
        return self.samples


class SBI_SMCABC(InferenceBase):
    def __init__(self, prior: BasePrior, simulator: BaseSimulator, observation: np.ndarray):
        if not _HAS_TORCH:
            raise ImportError(_TORCH_MSG)
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)

        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__("sbi_smcabc", prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        self.budget = budget
        default_kwargs = {"distance": "l2"}
        default_kwargs.update(fit_kwargs or {})
        self.inference_method = SMCABC(
            self.simulator.sample_pytorch,
            self.prior.return_sbi_object(),
            distance=default_kwargs["distance"]
        )

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        default_kwargs = {
            "num_initial_pop": int(0.1 * self.budget),
            "epsilon_decay": 0.7,
            "distance_based_decay": True
        }
        default_kwargs.update(sample_kwargs or {})
        self.posterior = self.inference_method(
            x_o=torch.as_tensor(self.observation),
            num_particles=nof_samples,
            num_initial_pop=default_kwargs["num_initial_pop"],
            num_simulations=self.budget,
            epsilon_decay=default_kwargs["epsilon_decay"],
            distance_based_decay=default_kwargs["distance_based_decay"]
        )
        self.samples = self.posterior.numpy()
        return self.samples
