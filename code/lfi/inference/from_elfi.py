import elfi
from .base import InferenceBase

class RejectionSampling(InferenceBase):
    def __init__(self, prior, simulator, observation):
        # set a new "clean" model as the default
        elfi.new_model() # # This sets a new default model and returns it
        
        thetas = prior.return_elfi_objects()
        elfi_callable = simulator.return_elfi_callable()
        self.sim = elfi.Simulator(elfi_callable, *thetas, observed=observation)
        self.d = elfi.Distance('euclidean', self.sim)
        dim = len(thetas)
        dim_y = observation.shape[1]
        super().__init__('elfi_rejection_sampling', prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int=1000, fit_kwargs: dict = None):
        self.budget = budget
        default_kwargs = {
            "batch_size": 1_000
        }
        default_kwargs.update((fit_kwargs or {}))
        self.inference_method = elfi.Rejection(self.d, batch_size=default_kwargs["batch_size"])

    def sample(self, nof_samples: int=100, sample_kwargs: dict = None):
        quantile = nof_samples / self.budget
        samples_res = self.inference_method.sample(nof_samples, quantile=quantile)
        return samples_res.samples_array


class SMCRejection(InferenceBase):
    def __init__(self, prior, simulator, observation):
        # set a new "clean" model as the default
        elfi.new_model() # # This sets a new default model and returns it

        elfi_callable = simulator.return_elfi_callable()
        thetas = prior.return_elfi_objects()
        self.sim = elfi.Simulator(elfi_callable, *thetas, observed=observation)
        self.d = elfi.Distance('euclidean', self.sim)
        dim = len(thetas)
        dim_y = observation.shape[0]
        super().__init__('elfi_smc_rejection', prior, simulator, observation, dim, dim_y)

    def fit(self, budget: int=1000, fit_kwargs: dict=None):
        self.budget = budget
        default_kwargs = {
            "batch_size": 1_000
        }
        default_kwargs.update((fit_kwargs or {}))
        self.inference_method = elfi.SMC(self.d, batch_size=default_kwargs["batch_size"])

    def sample(self, nof_samples: int=100, sample_kwargs: dict=None):
        default_kwargs = {
            "nof_iterations": 4
            }
        default_kwargs.update((sample_kwargs or {}))
        quantiles = [1 - i / default_kwargs["nof_iterations"] 
                        for i in range(default_kwargs["nof_iterations"])]

        default_kwargs.update((sample_kwargs or {}))
        samples_res = self.inference_method.sample(nof_samples, quantiles=quantiles)
        return samples_res.samples_array
