from .base import InferenceBase

try:
    import elfi
    _HAS_ELFI = True
except ImportError:
    _HAS_ELFI = False

_ELFI_MSG = "elfi not installed. Install with: pip install 'lfi[elfi]'"


class RejectionSampling(InferenceBase):
    def __init__(self, prior, simulator, observation):
        if not _HAS_ELFI:
            raise ImportError(_ELFI_MSG)
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
        self.samples = samples_res.samples_array
        return self.samples


class SMCRejection(InferenceBase):
    def __init__(self, prior, simulator, observation):
        if not _HAS_ELFI:
            raise ImportError(_ELFI_MSG)
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
        self.samples = samples_res.samples_array
        return self.samples
