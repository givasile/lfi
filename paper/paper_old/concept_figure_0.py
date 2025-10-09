import priors
import simulators
import jax
import jax.numpy as jnp
import r2omc
import numpy as np
import matplotlib.pyplot as plt
import torch
import priors_sbi
from scipy.stats import multivariate_normal
import sbi
import sbi.inference


def plot_posterior_samples(
        samples,
        title,
        color,
        savefig=False,
):
    th_star = np.array([5., 5.])
    plt.figure()
    plt.title(title)
    plt.scatter(
        samples[:100, 0],
        samples[:100, 1],
        color=color,
        marker="o",
        )
    # ground truth
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=th_star, cov=np.eye(2)/np.sqrt(4))
    plt.contour(X, Y, rv.pdf(pos), levels=5)
    rv = multivariate_normal(mean=-th_star, cov=np.eye(2)/np.sqrt(4))
    plt.contour(X, Y, rv.pdf(pos), levels=5)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if savefig:
        plt.savefig(f"./paper/figures/concept_image_multi_obs_{title}.pdf", bbox_inches="tight")
    plt.show(block=False)

# set experimental variables
np.random.seed(21)
torch.manual_seed(21)
key = jax.random.key(seed=21)
key, subkey = jax.random.split(key)

# Method 1: ROMC
# (i) define experiment and parameters
dim = 10
simulator = simulators.CasesGaussianMultiSamples(
    dim=dim,
    sigma_1=1.,
    sigma_2=1.,
    nof_samples=4
)
y_0 = np.concatenate([np.ones(dim)*5, np.ones(dim)*-5, np.ones(dim)*5, np.ones(dim)*-5])
prior = priors.Uniform(low=-15., high=15., dim=dim)

# (ii) perform inference
romc = r2omc.R2OMC(simulator, y_0, prior, dim)
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 500,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .1,
    "epochs": 8,
    "nof_seeds_accept": 100,
    "eps_2": 20.,
    "nof_ls_steps": 50,
    "step_size": .1,
    "sample_seed": 21,
    "nof_samples": 1000,
    "eps_3": 15.,
}
samples, weight = romc.infer(config)

selected_indices = np.random.choice(
    np.arange(samples.shape[0]),
    size=100,
    replace=False,
    p=np.ones(samples.shape[0])/samples.shape[0]
)
samples_romc = samples[selected_indices]

# (iii) plot
plot_posterior_samples(samples_romc, "ROMC", "magenta", True)

# Method 2: R2OMC
# (i): define experiment and parameters
dim = 10
simulator = simulators.CasesGaussian(
    dim=dim,
    sigma_1=1.,
    sigma_2=1.,
)
y_0 = np.array([np.ones(dim)*5, np.ones(dim)*-5, np.ones(dim)*5, np.ones(dim)*-5])
prior = priors.Uniform(low=-15., high=15., dim=dim)

# (ii) perform inference
iter_r2omc = r2omc.IterativeR2OMC(simulator, y_0, prior, dim=dim)
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 10,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 500,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .2,
    "epochs": 4,
    "nof_seeds_accept": 100,
    "eps_2": .3,
    "nof_ls_steps": 100,
    "step_size": .1,
    "sample_seed": 21,
    "nof_samples": 200,
    "eps_3": 1.,
    "nof_samples_to_select": 100,
}
samples_r2omc = iter_r2omc.infer(config)

# (iii) plot
plot_posterior_samples(samples_r2omc, "R2OMC", "darkmagenta", True)

# Method 3: SNLE
# (i) define experiment and parameters
nof_tr_samples = 5_000
tr_batch_size = 1000
dim = 10
y_0 = np.array([np.ones(dim)*5, np.ones(dim)*-5, np.ones(dim)*5, np.ones(dim)*-5])

sbi_prior, num_parameters, prior_returns_numpy = sbi.utils.user_input_checks.process_prior(
    priors_sbi.get_uniform(
        prior_limits=[-15, 15],
        D=dim
    )
)

theta = np.array(sbi_prior.sample((nof_tr_samples, )))
sim = simulators.CasesGaussian(dim=dim, sigma_1=1.0, sigma_2=1.0)
sim_func = sim.cr_simulator()
seeds = np.random.randint(0, 10000000, nof_tr_samples)
x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples)])
theta = torch.Tensor(theta)
x = torch.Tensor(x)

# (ii) inference
inferer = sbi.inference.SNLE(sbi_prior, show_progress_bars=True, density_estimator="mdn")
inferer.append_simulations(theta, x)
inferer.train(training_batch_size=tr_batch_size)

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

nle_samples = posterior.sample(sample_shape=(100,), x=torch.Tensor(y_0))

plot_posterior_samples(
    nle_samples,
    "SNLE",
    "darkturquoise",
    True
)

# Method 4: SNRE
# (i) define experiment and parameters
dim = 10
y_0 = np.concatenate([np.ones(dim)*5, np.ones(dim)*-5, np.ones(dim)*5, np.ones(dim)*-5])
nof_tr_samples = 10_000
tr_batch_size = 1000

sbi_prior, num_parameters, prior_returns_numpy = sbi.utils.user_input_checks.process_prior(
    priors_sbi.get_uniform(
        prior_limits=[-15, 15],
        D=dim
    )
)

theta = np.array(sbi_prior.sample((nof_tr_samples, )))
sim = simulators.CasesGaussianMultiSamples(dim=dim, sigma_1=1.0, sigma_2=1.0, nof_samples=4)
sim_func = sim.cr_simulator()
seeds = np.random.randint(0, 10000000, nof_tr_samples)
x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples)])
theta = torch.Tensor(theta)
x = torch.Tensor(x)

# (ii) inference
inferer = sbi.inference.SNPE(prior=sbi_prior)
inferer.append_simulations(theta, x)
density_estimator = inferer.train(training_batch_size=tr_batch_size)
posterior = inferer.build_posterior(density_estimator)
samples_npe = posterior.sample((100,), x=torch.Tensor(y_0))

# (iii) plot
plot_posterior_samples(
    samples_npe,
    "SNPE",
    "dodgerblue",
    True
)
