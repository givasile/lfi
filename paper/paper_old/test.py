import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import simulators
import priors
import priors_sbi
import utils
import jax
import torch
import matplotlib.pyplot as plt

simulator = simulators.CasesGaussian(dim=2, sigma_1=1., sigma_2=1.)
prior = priors.Uniform(low=-10., high=10., dim=2)
y_0 = np.array([5., 5.])
y_0_sbi = torch.tensor(y_0).float()

N = 4000
method_name = "SNPE-A"
sbi_prior = priors_sbi.get_uniform(prior_limits=[-10., 10.], D=2)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
theta = np.concatenate([
    np.ones((int(N/2), 2)) * 5 + jax.random.normal(subkey, shape=(int(N/2), 2)),
    np.ones((int(N/2), 2)) * -5 + jax.random.normal(subkey, shape=(int(N/2), 2))
])

# plt.figure()
# plt.scatter(theta[:, 0], theta[:, 1], c="darkmagenta", label=r"$\mathtt{GT}$")
# plt.show()

key, subkey = jax.random.split(key)
sim_func = simulator.cr_simulator()
seeds = np.random.randint(0, 10000000, N)
x = np.array([sim_func(theta[i], seeds[i]) for i in range(N)])

# plt.figure()
# plt.scatter(x[:, 0], x[:, 1], c="darkmagenta", label=r"$\mathtt{GT}$")
# plt.show()

samples_npe = utils.posterior_sbi(
    method_name,
    sbi_prior,
    theta,
    x,
    y_0_sbi,
    nof_samples=1000,
)

plt.figure()
plt.scatter(samples_npe[:, 0], samples_npe[:, 1], c="darkmagenta", label=r"$\mathtt{SNPE-A}$")
plt.legend()
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
