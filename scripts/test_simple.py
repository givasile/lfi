import lfi
import torch
import numpy as np

# set seed
np.random.seed(42)
torch.manual_seed(42)

# modeling
prior = lfi.priors.UniformPrior(dim=2, low=-1, high=1)
simulator = lfi.simulators.GaussianNoise(dim=2, dim_y= 2, sigma_noise=0.1)
observation = np.array([[0.5, 0.5]])

# inference
method = lfi.inference.from_sbi.NPE_C_SingleRound(
    prior=prior,
    simulator=simulator,
    observation=observation
)
samples_inferred = method.fit_and_sample(budget=1000, nof_samples=100)

# analysis
method.plot_posterior_samples(samples_inferred)

# evaluation
samples_gt = lfi.ground_truth.Gaussian(
    dim=2,
    mu=np.array([0.5, 0.5]),
    sigma=np.array([[0.1, 0.1]])
).sample(100)
lfi.evaluation.c2st(samples_inferred, samples_gt)
# 0.54
