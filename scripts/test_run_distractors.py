"""
Tests lfi method using a simple Gaussian noise simulator, of D=5 with distractors.
All methods must succeed with this test.
"""

import lfi
import numpy as np
import matplotlib.pyplot as plt

# modeling parameters
low = -5
high = 5
dim = 5
sigma_noise = 1
dim_y = 5
distractor_dim = 50
distractor_scale = 1.0
distractor_mu_min = -10.0
distractor_mu_max = 10.0
observation_nof_obs = 1

# inference parameters
budget_runs = 10_000
batch_size = 5_000
budget = 5_000 # batch_size * budget_runs
training_batch_size = 1000

nof_samples = 100

# Modeling
prior = lfi.priors.UniformPrior(
    low=low,
    high=high,
    dim=dim
)

simulator = lfi.simulators.GaussianNoiseDistractors(
    dim=dim,
    dim_y=dim_y,
    sigma_noise=sigma_noise,
    distractor_dim=distractor_dim,
    distractor_scale=distractor_scale,
    distractor_mu_min=distractor_mu_min,
    distractor_mu_max=distractor_mu_max
)

obs = lfi.observations.Zeros(
    dim_y = dim_y + distractor_dim,
    nof_observations=observation_nof_obs
)
observation = obs.sample() - 1.

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

# inference = lfi.inference.from_sbi.NPECSingleRound(
#     prior=prior,
#     simulator=simulator,
#     observation=observation,
# )

# inference = lfi.inference.from_sbi.TSNPE(
#     prior=prior,
#     simulator=simulator,
#     observation=observation,
# )

inference = lfi.inference.from_sbi.BayesFlow(
    prior=prior,
    simulator=simulator,
    observation=observation,
)

# fit
inference.fit(
    budget=budget,
    fit_kwargs = {
        "embedding_net_output_dim": 15,
        "embedding_net_num_hiddens": 50,
        "embedding_net_num_layers": 4,
    }
)

# sample
samples = inference.sample(nof_samples=nof_samples)

# Analysis

# Generate 100 samples
mean = observation[0][:dim]
cov = np.eye(dim)*sigma_noise**2
samples_gt = np.random.multivariate_normal(mean, cov, 100)

g = lfi.visualization.plot_pairwise_posterior(
    samples,
    limits = [low, high],
    samples_gt = samples_gt,
    # max_dims_to_plot=5
)
plt.show()

# Evaluation
c2st = lfi.evaluation.c2st(samples, samples_gt)
print(f"C2ST: {c2st}")