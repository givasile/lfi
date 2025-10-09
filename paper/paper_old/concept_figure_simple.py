import priors
import simulators
import jax
import r2omc
import numpy as np
import matplotlib.pyplot as plt
import torch
import priors_sbi
import sbi.inference
import utils
import time

# Plot ground truth
utils.plot_gt_2d(savefig="paper/figures/concept_image_simple_gt.pdf")

# Method 1: ROMC
np.random.seed(2143453)
torch.manual_seed(21436)
key = jax.random.key(seed=2154672)
key, subkey = jax.random.split(key)

# (i) define experiment and parameters
dim = 5
simulator = simulators.CasesGaussian(
    dim=dim,
    sigma_1=1.,
    sigma_2=1.,
)
y_0 = np.ones((dim,)) * 5
prior = priors.Uniform(low=-15., high=15., dim=dim)

# (ii) perform inference
romc = r2omc.R2OMC(simulator, y_0, prior, dim)
config = {
    "find_informative_dims": True,
    "fit_seed": 2567671,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 2_000,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .1,
    "epochs": 8,
    "nof_seeds_accept": 2_000,
    "eps_2": .00001,
    "nof_ls_steps": 50,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 1_000,
    "eps_3": .0001,
}
tic = time.time()
samples, weight = romc.infer(config)
time_romc = time.time() - tic
print(f"ROMC took {time_romc} seconds.")

selected_indices = np.random.choice(
    np.arange(samples.shape[0]),
    size=1_000,
    replace=False,
    p=np.ones(samples.shape[0])/samples.shape[0]
)
samples_romc = samples[selected_indices]
utils.plot_posterior_kde_2D(samples_romc, savefig="paper/figures/concept_image_simple_romc_kde.pdf", bw_method=.2)

# Method 2: R2OMC
np.random.seed(2143453)
torch.manual_seed(21436)
key = jax.random.key(seed=2154672)
key, subkey = jax.random.split(key)

# (i): define experiment and parameters
dim = 5
simulator = simulators.CasesGaussian(
    dim=dim,
    sigma_1=1.,
    sigma_2=1.,
)
y_0 = np.ones((dim,)) * 5
prior = priors.Uniform(low=-15., high=15., dim=dim)

# (ii) perform inference
rromc = r2omc.R2OMC(simulator, y_0, prior, dim)
config = {
    "find_informative_dims": True,
    "fit_seed": 2451,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 2_000,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .1,
    "epochs": 8,
    "nof_seeds_accept": 2_000,
    "eps_2": .00000001,
    "nof_ls_steps": 50,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 1_000,
    "eps_3": .0001,
}
tic = time.time()
samples, weight = rromc.infer(config)
time_r2omc = time.time() - tic
print(f"R2OMC took {time_r2omc} seconds.")

selected_indices = np.random.choice(
    np.arange(samples.shape[0]),
    size=1_000,
    replace=False,
    p=np.ones(samples.shape[0])/samples.shape[0]
)
samples_rromc = samples[selected_indices]
utils.plot_posterior_kde_2D(samples_rromc, savefig="paper/figures/concept_image_simple_rromc_kde.pdf", bw_method=.2)

# Method 3: NLE
np.random.seed(2145453)
torch.manual_seed(21436)
key = jax.random.key(seed=2154672)
key, subkey = jax.random.split(key)

# (i) define experiment and parameters
nof_tr_samples = 10_000
tr_batch_size = 1000
dim = 5
y_0 = np.concatenate([np.ones((dim,)) * 5])

sbi_prior, num_parameters, prior_returns_numpy = sbi.utils.user_input_checks.process_prior(
    priors_sbi.get_uniform(
        prior_limits=[-15, 15],
        D=dim
    )
)

tic = time.time()
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
samples_nle = posterior.sample(sample_shape=(5000,), x=torch.Tensor(y_0))
time_nle = time.time() - tic
print(f"NLE took {time_nle} seconds.")

utils.plot_posterior_kde_2D(samples_nle, savefig="paper/figures/concept_image_simple_nle_kde.pdf", bw_method=.2)

## Method 4: SNPE
# set experimental variables
np.random.seed(21355)
torch.manual_seed(21)
key = jax.random.key(seed=21)
key, subkey = jax.random.split(key)

# (i) define experiment and parameters
dim = 5
nof_epochs = 2
nof_tr_samples_per_epoch = 5_000
tr_batch_size = 1_000
y_0 = np.ones((dim,)) * 5

sbi_prior, num_parameters, prior_returns_numpy = sbi.utils.user_input_checks.process_prior(
    priors_sbi.get_uniform(
        prior_limits=[-15, 15],
        D=dim
    )
)

tic = time.time()
inferer = sbi.inference.SNPE(prior=sbi_prior)
proposal = sbi_prior
for _ in range(nof_epochs):
    # propose samples
    theta = np.array(proposal.sample((nof_tr_samples_per_epoch, )))
    plt.figure()
    plt.plot(theta[:, 0], theta[:, 1], "o")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show(block=False)
    sim = simulators.CasesGaussian(dim=dim, sigma_1=1.0, sigma_2=1.0)
    sim_func = sim.cr_simulator()
    seeds = np.random.randint(0, 10000000, nof_tr_samples_per_epoch)
    x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples_per_epoch)])

    # x = simulator(theta)
    _ = inferer.append_simulations(torch.Tensor(theta), torch.Tensor(x), proposal=proposal).train(training_batch_size=tr_batch_size)
    posterior = inferer.build_posterior().set_default_x(y_0)
    proposal = posterior

samples_npe = posterior.sample((1000,), x=torch.Tensor(y_0))
time_npe = time.time() - tic
print(f"SNPE took {time_npe} seconds.")

utils.plot_posterior_kde_2D(samples_npe, savefig="paper/figures/concept_image_simple_npe_kde.pdf", bw_method=.2)

# Plot 1D
utils.plot_posterior_kde_1d(
    "Simple",
    samples_rromc,
    samples_romc,
    samples_npe,
    samples_nle,
    savefig="paper/figures/concept_image_simple.pdf"
)
