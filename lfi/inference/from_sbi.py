import sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE_C, FMPE, NPE_A
import matplotlib.pyplot as plt
import torch
import sbi.neural_nets
import sbi.neural_nets.embedding_nets
import numpy as np
from .base import InferenceBase
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator
from typing import Optional
import torch.nn as nn
from sbi.inference import NPE
from sbi.utils import RestrictedPrior, get_density_thresholder


# Implemented methods:
# NPE-A: Fast epsilon-free inference of simulation models with Bayesian conditional density estimation (done)
# NPE-C: Automatic posterior transformation for likelihood-free inference (done)
# Sequential version of NPE-C -> should be treated with care for being stable enough (done)
# BayesFlow: Learning complex stochastic models with invertible neural networks (done)
# FMPESingleRound: Flow Matching Posterior Estimation (done)


# Comment: Not implemented yet
# Truncated proposals for scalable and hassle-free simulation-based inference -> does not seem to be working
# All in one simultion-based inference: https://arxiv.org/abs/2404.09636
# Compositional Score Modeling for Simulation-Based Inference
# Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models

class NPEBase(InferenceBase):
    def __init__(
            self, 
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator, 
            observation: np.ndarray, # (1, Dy).
            embedding_net: Optional[nn.Module] = None
    ):
        self.name = name
        self.embedding_net = embedding_net

        # prepare prior and simulator
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)
        self.sbi_prior = sbi_prior

        # Parameters of the prior
        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__(name, prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int = 1000, fit_kwargs: dict = None):
        """Fit the posterior distribution to data.
        This step may do nothing if the inference method does not require fitting, e.g. ABC methods.
        Otherwise, it sets the self.posterior attribute with the fitted posterior distribution.

        Args:
            budget: Number of training samples to generate for fitting.
            fit_kwargs: Method-specific arguments for modeling and fitting the posterior distribution.
                Here can go arguments that will be used in one of:
                    - <neural method>.__init__() for defining the inference method
                    - <neural method>.train() for training the inference method
                    - <neural method>.build_posterior() for building the posterior distribution
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        if self.posterior is None:
            raise ValueError("Posterior is not trained yet.")
        y = self.posterior.sample((nof_samples,), x=torch.Tensor(self.observation))
        return  y.detach().numpy()

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
    """
    From the paper:
    Fast \epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation
    Papamakarios, Murray

    Uses the NPE-A implementation from the SBI library.
    """
    def __init__(self, prior, simulator, observation, embedding_net=None):
        super().__init__("npe_a_single_round", prior, simulator, observation, embedding_net)

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):

        # prepare arguments
        default_kwargs = {
            "num_components": 10,
            "training_batch_size": 500,
            "max_num_epochs": 1000,            
        }
        default_kwargs.update(fit_kwargs or {})

        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        if self.embedding_net is not None:
            x = self.embedding_net(x) 

        # fit the model
        self.inference_method = NPE_A(self.sbi_prior,
                                      num_components=default_kwargs["num_components"],
                                      )

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            final_round=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior

class NPECSingleRound(NPEBase):
    """
    The Automatic posterior transformation (APT) for likelihood-free inference (NPE-C) inference method:
    https://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf
    Uses the NPE-C implementation from the SBI library.
    """
    def __init__(self, prior, simulator, observation, embedding_net: Optional[nn.Module] = None):
        super().__init__("npe_c_single_round", prior, simulator, observation, embedding_net)

    def fit(self, budget: int = 100,
            fit_kwargs: dict=None
            ):
        # default arguments
        default_kwargs = {
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

        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        if self.embedding_net is not None:
            x = self.embedding_net(x)

        # define the density estimator
        density_estimator = sbi.neural_nets.posterior_nn(
            model=default_kwargs["model"],
            hidden_features=default_kwargs["hidden_features"],
            num_transforms=default_kwargs["num_transforms"],
            z_score_x = default_kwargs["z_score_x"],
            z_score_theta = default_kwargs["z_score_theta"],
        )

        # fit the model
        self.inference_method = NPE_C(self.sbi_prior,
                                      density_estimator=density_estimator
                                      )

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior

class FMPESingleRound(NPEBase):
    """
    The Flow Matching Posterior Estimation (FMPE) inference method:
    https://proceedings.neurips.cc/paper_files/paper/2023/file/3663ae53ec078860bb0b9c6606e092a0-Paper-Conference.pdf
    Uses the FMPE implementation from the SBI library.
    """
    def __init__(self, prior, simulator, observation):
        super().__init__("fmpe_single_round", prior, simulator, observation)

    def fit(self,
            budget: int=1000,
            fit_kwargs: dict = None
            ):

        # default arguments
        default_kwargs = {
            "vf_estimator": "mlp", # ["mlp", "ada_mlp", "transformer", "transformer_cross_attn"] or a Callable that builds a vf_estimator
            "training_batch_size": 500, # batch size for training the density estimator
            "max_num_epochs": 1000, # maximum number of epochs for training the density estimator
        }
        default_kwargs.update(fit_kwargs or {})

        # prepare dataset
        theta, x = simulate_for_sbi(
            self.simulator.sample_pytorch,
            self.prior.return_sbi_object(),
            num_simulations=budget
        )

        # fit the model
        self.inference_method = FMPE(
            self.sbi_prior,
            vf_estimator=default_kwargs["vf_estimator"],
        )

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


class BayesFlow(NPEBase):
    """
    The BayesFlow inference method:
    https://arxiv.org/pdf/2003.06281
    Uses the BayesFlow implementation from the SBI library.
    """
    def __init__(self, prior, simulator, observation):
        super().__init__("bayes_flow", prior, simulator, observation)

    def fit(self,
            budget: int = 1000,
            fit_kwargs: dict = None
            ):
        # default arguments
        default_kwargs = {
            "embedding_net_output_dim": 20,  # Output dimension of the embedding network
            "embedding_net_num_layers": 2,  # Number of layers in the embedding network
            "embedding_net_num_hiddens": 50,  # Number of hidden layers in the embedding network
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

        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        # fit the model
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

        self.inference_method = sbi.inference.NPE(
            self.sbi_prior,
            density_estimator=density_estimator,
        )

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            force_first_round_loss=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


# class TSNPE(NPEBase):
#     """
#     The Truncated Sequential Neural Posterior Estimation (TSNPE) inference method:
#     https://arxiv.org/abs/2404.09636
#     Uses the TSNPE implementation from the SBI library.
#     """
#     def __init__(self, prior, simulator, observation):
#         super().__init__("tsnpe", prior, simulator, observation)
#
#     def fit(self,
#             budget: int = 1000,
#             fit_kwargs: dict = None
#             ):
#         # default arguments
#         default_kwargs = {
#             "num_rounds": 10,  # Number of rounds for sequential inference
#         }
#         default_kwargs.update(fit_kwargs or {})
#
#         inference = NPE(self.sbi_prior)
#         proposal = self.prior.return_sbi_object()
#         nof_new_samples = budget // default_kwargs["num_rounds"]
#         for _ in range(default_kwargs["num_rounds"]):
#             theta = proposal.sample((nof_new_samples,))
#             x = self.simulator.sample_pytorch(theta)
#             _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
#             posterior = inference.build_posterior().set_default_x(torch.Tensor(self.observation))
#
#             accept_reject_fn = get_density_thresholder(posterior, quantile=1e-3)
#             proposal = RestrictedPrior(self.prior.return_sbi_object(), accept_reject_fn, sample_with="rejection")
#
#         self.posterior = inference.build_posterior().set_default_x(torch.Tensor(self.observation))
#         return self.posterior


class NPECMultiRound(NPEBase):
    """
    The NPE-C multi-round inference method:
    https://proceedings.mlr.press/v97/greenberg19a/greenberg19a.pdf
    Uses the NPE-C implementation from the SBI library.
    This is a multi-round version of the NPE-C method.
    """
    def __init__(self, prior, simulator, observation, embedding_net: Optional[nn.Module] = None):
        super().__init__("npe_c_multi_round", prior, simulator, observation, embedding_net)

    def fit(self, budget: int = 1000,
            fit_kwargs: dict=None
            ):
        # default arguments
        default_kwargs = {
            "model": "nsf",
            "hidden_features": 100,
            "num_transforms": 8,
            "z_score_x": "independent",
            "z_score_theta": "independent",
            "training_batch_size": 500,
            "max_num_epochs": 1000,
            "num_rounds": 10,  # Number of rounds for sequential inference
        }

        default_kwargs.update(fit_kwargs or {})

        # define the density estimator
        density_estimator = sbi.neural_nets.posterior_nn(
            model=default_kwargs["model"],
            hidden_features=default_kwargs["hidden_features"],
            num_transforms=default_kwargs["num_transforms"],
            z_score_x = default_kwargs["z_score_x"],
            z_score_theta = default_kwargs["z_score_theta"]
        )

        inference = NPE_C(self.sbi_prior,
                          density_estimator=density_estimator
                          )
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

