import sys
import os

# Add the parent directory (lfi's location) to sys.path




import lfi
import numpy as np
import matplotlib.pyplot as plt

# modeling parameters 
prior_low = -5
prior_high = 5
prior_dim = 2
simulator_sigma_noise = 0.1
simulator_dim_y = 2
observation_nof_obs = 1
shift_value = 2

# inference parameters
budget = 20_000
num_components = 5
batch_size = 1_000
max_num_epochs = 1_000
nof_samples = 100

# define experiment
prior = lfi.priors.UniformPrior(
    low=prior_low,
    high=prior_high,
    dim=prior_dim
)

simulator = lfi.simulators.GaussianNoise(
    dim=prior_dim,
    dim_y=simulator_dim_y,
    sigma_noise=simulator_sigma_noise
)

# simulator = lfi.simulators.MultivariateGaussian(
#     dim=prior_dim,
#     dim_y = simulator_dim_y,
#     sigma_noise=simulator_sigma_noise,
#     shift_value=shift_value
# )

obs = lfi.observations.Zeros(
    dim_y = simulator_dim_y,
    nof_observations=observation_nof_obs
)
observation = obs.sample()

# define inference
# inference = lfi.inference.from_elfi.RejectionSampling(
#     prior=prior,
#     simulator=simulator,
#     observation=observation
# )

inference = lfi.inference.from_elfi.SMCRejection(
    prior=prior,
    simulator=simulator,
    observation=observation
)

# fit
inference.fit(
    budget=budget,
    fit_kwargs = {
        "batch_size": batch_size,
        "num_components": num_components
    }
)

samples = inference.sample(nof_samples=nof_samples)

# analysis
# Mean and covariance 
#mean= np.array([3,3])
#cov = np.eye(2)*simulator_sigma_noise**2
mean = np.array([0,0])
cov = np.eye(2)*simulator_sigma_noise**2


# Generate 100 samples
samples_gt = np.random.multivariate_normal(mean, cov, 100)

g = lfi.visualization.plot_pairwise_posterior(
    samples,
    limits = [prior_low, prior_high],
    samples_gt = samples_gt
)
# plt.show()

# Evaluation
# c2st = lfi.evaluation.c2st(samples, samples_gt)

#simulator = lfi.simulators.GaussianNoise(sigma_noise=simulator_sigma_noise, dim=prior_dim, dim_y=simulator_dim_y)
