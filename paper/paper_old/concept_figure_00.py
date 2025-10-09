import matplotlib.pyplot as plt
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal
import priors_sbi
import numpy as np
import simulators
from sbi.analysis import pairplot
import sbi.inference
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import c2st
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from scipy.stats import multivariate_normal

# from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
# from sbi.neural_nets import posterior_nn

# Seeding
torch.manual_seed(1)
nof_tr_samples = 10000
theta_dim = 10


# sbi prior
sbi_prior, num_parameters, prior_returns_numpy = process_prior(
    priors_sbi.get_uniform(prior_limits=[-15, 15], D=10)
    )




# Train SNLE.
inferer = sbi.inference.SNLE(sbi_prior, show_progress_bars=True, density_estimator="mdn")

sim = simulators.CasesGaussian(dim=10, sigma_1=1.0, sigma_2=1.0)
sim_func = sim.cr_simulator()

theta = np.array(sbi_prior.sample((nof_tr_samples, )))
seeds = np.random.randint(0, 10000000, nof_tr_samples)
x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples)])
theta = torch.Tensor(theta)
x = torch.Tensor(x)

inferer.append_simulations(theta, x).train(training_batch_size=1000)


# Obtain posterior samples for different number of iid xos.
nle_samples = []
num_samples = 1000

mcmc_parameters = dict(
    num_chains=50,
    thin=5,
    warmup_steps=30,
    init_strategy="proposal",
)
mcmc_method = "slice_np_vectorized"

posterior = inferer.build_posterior(
    mcmc_method=mcmc_method,
    mcmc_parameters=mcmc_parameters,
)




num_trials = [1, 5, 15, 20]
theta_o = zeros(1, theta_dim)

# Generate multiple x_os with increasing number of trials.
obs = torch.Tensor(
    [
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
    ]
)

# Generate samples with MCMC given the same set of x_os as above.
nle_samples.append(posterior.sample(sample_shape=(num_samples,), x=obs))



import matplotlib.pyplot as plt

th_star = np.array([5., 5.])
plt.figure()
plt.scatter(nle_samples[0][:, 0], nle_samples[0][:, 1],
            color="cyan",
            marker=".",
            label="xo = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]")
# ground truth
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
rv = multivariate_normal(mean=th_star, cov=np.eye(2)/np.sqrt(4))
plt.contour(X, Y, rv.pdf(pos), levels=5, linestyle="--")
rv = multivariate_normal(mean=-th_star, cov=np.eye(2)/np.sqrt(4))
plt.contour(X, Y, rv.pdf(pos), levels=5, linestyle="--")
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.show(block=False)
