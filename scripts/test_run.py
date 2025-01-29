import elfi

import lfi
import numpy as np

# define experiment
dim = 3
prior = lfi.priors.UniformPrior(low=-3, high=3, dim=dim)
simulator= lfi.simulators.GaussianNoise(sigma_noise=0.1, dim=dim, dim_y=dim)
observation = np.zeros((1, dim)) + 2

# define inference
inference = lfi.inference.from_elfi.RejectionSampling(
    prior=prior,
    simulator=simulator,
    observation=observation
)

# fit
inference.fit(
    budget=30_000,
    fit_kwargs={
        "batch_size": 1000,
    }
)

samples = inference.sample(nof_samples=100)

# # sample
# posterior_samples = inference.sample(nof_samples=100)
#
# # plot
# inference.plot_training_summary()
#
# # plot
# inference.plot_posterior_samples(samples=posterior_samples)