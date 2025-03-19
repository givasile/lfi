import sys
import os


# Add the parent directory (lfi) to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import lfi
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# modeling parameters
prior_low=-5
prior_high=5
prior_dim=2
simulator_sigma_noise=0.1
simulator_dim_y = 2
observation_nof_obs = 1

# inference parameters
budget = 1000
nof_samples=100

# define experiment
prior = lfi.priors.UniformPrior(prior_low, prior_high, dim=prior_dim)
simulator = lfi.simulators.GaussianNoise(sigma_noise=simulator_sigma_noise, dim=prior_dim, dim_y = simulator_dim_y)
obs=lfi.observations.Zeros(dim_y = simulator_dim_y, nof_observations=observation_nof_obs)
observation = obs.sample()

# define inference
inference = lfi.inference.from_sbi_abc.SBI_MCABC(
prior=prior,
simulator=simulator,
observation=observation
)

# run inference
inference.fit()

samples = inference.sample(budget = budget, nof_samples=nof_samples)


# analysis
# Mean and covariance 
mean= np.array([3,3])
cov = np.eye(2)*simulator_sigma_noise**2

# Generate 100 samples
samples_gt = np.random.multivariate_normal(mean, cov, 100)

lfi.visualization.plot_pairwise_posterior(
    samples,
    limits = [prior_low, prior_high],
    samples_gt = samples_gt
)

# Evaluation
c2st = lfi.evaluation.c2st(samples.detach().numpy(), samples_gt)



