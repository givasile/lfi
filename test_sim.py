import lfi
import torch
import numpy as np

# modeling
prior = lfi.priors.UniformPrior(dim=2, low=-3, high=3)
simulator = lfi.simulators.GaussianNoise(dim=2, dim_y=2, sigma_noise=0.1)
observation = np.array([[1.5, 1.5]])

# inference
method = lfi.inference.from_sbi.NPECSingleRound(
    prior=prior,
    simulator=simulator,
    observation=observation
)
samples_inferred = method.fit_and_sample(budget=1000, nof_samples=100)

# analysis
method.plot_posterior_samples(limits=(-3, 3))

# # evaluation
# samples_gt = lfi.ground_truth.Gaussian(
#     dim=2,
#     mu=np.array([0.5, 0.5]),
#     sigma=np.array([[0.1, 0.1]])
# ).sample(100)

# lfi.evaluation.c2st(samples_inferred, samples_gt)


method_2 = lfi.inference.r2omc.R2OMC(
    prior=prior,
    simulator=simulator,
    observation=observation
)

samples_inferred_2 = method_2.fit_and_sample(budget=1000, nof_samples=100)
method_2.plot_posterior_samples(limits=(-3, 3))
