"""
Tests lfi method using a simple Gaussian noise simulator, of D=5.
All methods must succeed with this test.
"""

import lfi
import numpy as np
import matplotlib.pyplot as plt

# modeling parameters
low = -5
high = 5
dim = 5
sigma_noise = 0.1
dim_y = 5
observation_nof_obs = 1
shift_value = 2

# inference parameters
budget_runs = 10_000
batch_size = 5_000
budget = 10_000 # batch_size * budget_runs
training_batch_size = 1000

nof_samples = 100

# Modeling
prior = lfi.priors.UniformPrior(
    low=low,
    high=high,
    dim=dim
)

simulator = lfi.simulators.GaussianNoise(
    dim=dim,
    dim_y=dim_y,
    sigma_noise=sigma_noise
)

obs = lfi.observations.Zeros(
    dim_y = dim_y,
    nof_observations=observation_nof_obs
)
observation = obs.sample() - 3.

# SBI Inference
# inference = lfi.inference.from_elfi.RejectionSampling(
#     prior=prior,
#     simulator=simulator,
#     observation=observation
# )

# inference = lfi.inference.from_elfi.SMCRejection(
#     prior=prior,
#     simulator=simulator,
#     observation=observation
# )

# inference = lfi.inference.from_sbi.FMPESingleRound(
#     prior=prior,
#     simulator=simulator,
#     observation=observation
# )

inference = lfi.inference.from_sbi.TSNPE(
    prior=prior,
    simulator=simulator,
    observation=observation,
)


# fit
inference.fit(
    budget=budget,
    fit_kwargs = {
        "num_rounds": 3
    }
)

# sample
samples = inference.sample(nof_samples=nof_samples)

# Analysis

# Generate 100 samples
mean = observation[0]
cov = np.eye(dim_y)*sigma_noise**2
samples_gt = np.random.multivariate_normal(mean, cov, 100)

g = lfi.visualization.plot_pairwise_posterior(
    samples,
    limits = [low, high],
    samples_gt = samples_gt,
    max_dims_to_plot=2
)
plt.show()

# Evaluation
# c2st = lfi.evaluation.c2st(samples, samples_gt)
