import lfi
import numpy as np

# modeling parameters
prior_low = -5
prior_high = 5
prior_dim = 2
simulator_sigma_noise = 1
simulator_dim_y = 2
observation_nof_obs = 1

# inference parameters
budget = 2_000
num_components = 5
batch_size = 1_000
max_num_epochs = 1_000
nof_samples=100

# define experiment
prior = lfi.priors.UniformPrior(low=prior_low, high=prior_high, dim=prior_dim)
simulator= lfi.simulators.GaussianNoise(sigma_noise=simulator_sigma_noise, dim=prior_dim, dim_y=simulator_dim_y)
obs = lfi.observations.Zeros()
observation = obs.sample(nof_obs=observation_nof_obs, dim_y=simulator_dim_y) + 3


# define inference
inference = lfi.inference.from_sbi.NPE_A_SingleRound(
    prior=prior,
    simulator=simulator,
    observation=observation
)

# fit
inference.fit(
    budget=budget,
    fit_kwargs={
        "batch_size": batch_size,
        "num_components": num_components
    }
)

samples = inference.sample(nof_samples=nof_samples)

# analysis
# Mean and covariance
mean = np.array([3, 3])            # Center of the Gaussian
cov = np.eye(2)*simulator_sigma_noise**2  # Covariance matrix

# Generate 100 samples
samples_gt = np.random.multivariate_normal(mean, cov, 100)

lfi.visualization.plot_pairwise_posterior(
    samples,
    limits=[prior_low, prior_high],
    samples_gt=samples_gt
)

# Evaluation
c2st = lfi.evaluation.c2st(samples, samples_gt)
