import sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE_C, FMPE, NPE_A
import matplotlib.pyplot as plt
import torch
import sbi.analysis
import sbi.neural_nets
import numpy as np
from .base import InferenceBase
from lfi.priors import BasePrior
from lfi.simulators import BaseSimulator


class NPEBase(InferenceBase):
    def __init__(
            self,
            name: str,
            prior: BasePrior,
            simulator: BaseSimulator,
            observation: np.ndarray,  # (1, Dy)
        ):
        self.name = name

        # prepare prior and simulator
        sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior.return_sbi_object())
        sim = process_simulator(simulator.sample_pytorch, sbi_prior, prior_returns_numpy)
        check_sbi_inputs(sim, sbi_prior)
        dim = num_parameters
        dim_y = observation.shape[1]

        self.inference_method = None
        self.posterior = None
        super().__init__(name, prior, simulator, observation, dim, dim_y)

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        if self.posterior is None:
            raise ValueError("Posterior is not trained yet.")
        return self.posterior.sample((nof_samples,), x=torch.Tensor(self.observation))

    def plot_training_summary(self, budget=None, savefig=None):
        fig, ax = plt.subplots()
        budget = "not specified" if budget is None else str(budget)
        ax.set_title("%s: D=%d, budget=%s" % (self.name, self.dim, budget))
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


class NPE_A_SingleRound(NPEBase):
    def __init__(self, prior, simulator, observation):
        super().__init__("npe_a_single_round", prior, simulator, observation)

    def fit(
            self,
            budget: int = 1_000,
            fit_kwargs: dict = None
    ):
        # prepare arguments
        default_kwargs = {
            "num_components": 10,
            "training_batch_size": 500,
            "max_num_epochs": 1000,
        }
        default_kwargs.update(fit_kwargs or {})

        # prepare dataset
        theta, x = simulate_for_sbi(
            self.simulator.sample_pytorch,
            self.prior.return_sbi_object(),
            num_simulations=budget
        )

        # fit the model
        self.inference_method = NPE_A(
            self.prior.return_sbi_object(),
            num_components=default_kwargs["num_components"]
        )

        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"],
            final_round=True
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior


class NPE_C_SingleRound(NPEBase):
    def __init__(self, prior, simulator, observation):
        super().__init__("npe_c_single_round", prior, simulator, observation)

    def fit(
            self,
            budget: int = 1_000,
            fit_kwargs: dict = None
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
            "force_first_round_loss": True,
        }
        default_kwargs.update(fit_kwargs or {})

        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        # define the density estimator
        density_estimator = sbi.neural_nets.posterior_nn(
            model=default_kwargs["model"],
            hidden_features=default_kwargs["hidden_features"],
            num_transforms=default_kwargs["num_transforms"],
            z_score_x=default_kwargs["z_score_x"],
            z_score_theta=default_kwargs["z_score_theta"],
        )

        # fit the model
        self.inference_method = NPE_C(
            self.prior.return_sbi_object(),
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
    def __init__(self, prior, simulator, observation):
        super().__init__("fmpe_single_round", prior, simulator, observation)

    def fit(
            self,
            budget: int = 1_000,
            fit_kwargs: dict = None
    ):
        # default arguments
        default_kwargs = {
            "training_batch_size": 500,
            "max_num_epochs": 1000,
        }
        default_kwargs.update(fit_kwargs or {})

        # prepare dataset
        theta, x = simulate_for_sbi(self.simulator.sample_pytorch, self.prior.return_sbi_object(), num_simulations=budget)

        # fit the model
        self.inference_method = FMPE(self.prior.return_sbi_object())
        _ = self.inference_method.append_simulations(theta, x).train(
            training_batch_size=default_kwargs["training_batch_size"],
            max_num_epochs=default_kwargs["max_num_epochs"]
        )

        self.posterior = self.inference_method.build_posterior().set_default_x(torch.Tensor(self.observation))
        return self.posterior