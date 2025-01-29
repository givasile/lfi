import elfi
from .base import InferenceBase

class RejectionSampling(InferenceBase):
    def __init__(self, prior, simulator, observation):

        elfi_sim = simulator.return_elfi_callable()

        thetas = prior.return_elfi_objects()
        self.sim = elfi.Simulator(elfi_sim, *thetas, observed=observation)
        self.d = elfi.Distance('euclidean', self.sim)
        super().__init__('elfi_rejection_sampling', prior, simulator, observation, dim=simulator.dim, dim_y=simulator.dim_y)

    def fit(self, budget: int = 1_000, fit_kwargs: dict = None):
        self.budget = budget
        default_kwargs = {
            "batch_size": 1_000
        }
        default_kwargs.update((fit_kwargs or {}))
        self.inference_method = elfi.Rejection(self.d, batch_size=default_kwargs['batch_size'])

    def sample(self, nof_samples: int = 100, sample_kwargs: dict = None):
        quantile = nof_samples / self.budget
        samples_res = self.inference_method.sample(nof_samples, quantile=quantile)
        return samples_res.samples_array
