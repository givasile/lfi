import simulators
import numpy as np
import priors
import r2omc
import timeit
import matplotlib.pyplot as plt
import priors_sbi
from scipy.stats import multivariate_normal
import utils
import torch
import jax
import sbi
import time
import pandas as pd
import os
import ast
from scipy.stats import gaussian_kde

conf_load = True

conf_D_list = [20]
conf_run_r2omc = False
conf_run_nle = False
conf_run_romc = False
conf_run_npe = False

conf_save = True
conf_save_plot = True


def plot_posterior_samples_2D(
        samples,
        savefig,
        ):
    fig, ax = plt.subplots()
    samples = samples[:, :2]
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        color="blue",
        marker=".",
        alpha=0.5,
    )
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")

    plt.show(block=False)


def plot_posterior_kde_2D(samples, savefig=False):
    fig, ax = plt.subplots()
    # Create grid for KDE evaluation
    x = np.linspace(-15, 15, 500)
    y = np.linspace(-15, 15, 500)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Calculate 2D KDE using scipy's gaussian_kde
    samples = samples[:, :2]
    kde = gaussian_kde(samples.T, bw_method=.3)
    Z = np.reshape(kde(positions).T, X.shape)  # Evaluate KDE on grid and reshape

    # Plot KDE as filled contour plot
    plt.contourf(X, Y, Z)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # Remove axis lines (spines) and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ax.set_xticks([])  # Remove x ticks
    # ax.set_yticks([])  # Remove y ticks

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")

    plt.show(block=False)


def plot_c2st(metrics, savefig):
    data = {}

    # Loop through the dictionary and organize data by method
    for key, value in metrics.items():
        if 'runtime' not in key:  # Skip runtime keys
            method, N = key.split('_N_')
            N = int(N)

            # Collect data for each method
            if method not in data:
                data[method] = {'N': [], 'scores': []}

            data[method]['N'].append(N)
            data[method]['scores'].append(value)

    # Create the plot
    fig, ax = plt.subplots()

    # Plot data for each method
    for method, values in data.items():
        if method == 'nle':
            values['N'], values['scores'] = zip(*sorted(zip(values['N'], values['scores'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1), '--o', label="NLE", color='darkturquoise')
        if method == 'npe':
            values['N'], values['scores'] = zip(*sorted(zip(values['N'], values['scores'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1), '--o', label="NPE", color='dodgerblue')
        elif method == "romc":
            values['N'], values['scores'] = zip(*sorted(zip(values['N'], values['scores'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1), '--o', label="ROMC", color='magenta')
        elif method == 'r2omc':
            values['N'], values['scores'] = zip(*sorted(zip(values['N'], values['scores'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1), '--o', label="R2OMC", color='darkmagenta')

    ax.set_xlabel(r'Observations (N)', fontsize=15)
    ax.set_ylabel("Score (C2ST)", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_yticks([.4, .6, .8, 1.])
    ax.set_ylim([.4, 1.1])

    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(fontsize=16)
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")
    plt.show(block=False)

def plot_runtime(metrics, savefig):
    data = {}

    # Loop through the dictionary and organize data by method
    for key, value in metrics.items():
        if 'runtime' in key:  # Skip runtime keys
            method, N = key.split('_N_')
            N = N.replace("_runtime", "")
            N = int(N)

            # Collect data for each method
            if method not in data:
                data[method] = {'N': [], 'scores': []}

            data[method]['N'].append(N)
            data[method]['scores'].append(value)

    # Create the plot
    fig, ax = plt.subplots()

    # Plot data for each method
    for method, values in data.items():
        if method == 'nle':
            # values['scores'], values['N'] = zip(*sorted(zip(values['scores'], values['N'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1)/60, '--o', label="NLE", color='darkturquoise')
        elif method == "npe":
            # values['scores'], values['N'] = zip(*sorted(zip(values['scores'], values['N'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1)/60, '--o', label="NPE", color='dodgerblue')
        elif method == "romc":
            # values['scores'], values['N'] = zip(*sorted(zip(values['scores'], values['N'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1)/60, '--o', label="ROMC", color='magenta')
        elif method == 'r2omc':
            # values['scores'], values['N'] = zip(*sorted(zip(values['scores'], values['N'])))
            ax.plot(values['N'], np.array(values['scores']).mean(1)/60, '--o', label="R2OMC", color='darkmagenta')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_yscale('log')
    ax.set_xlabel(r'Observations (N)', fontsize=15)
    ax.set_ylabel("Runtime (Minutes)", fontsize=16)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim([.05, 60.])
    ax.set_yticks([.1, 1., 10., 60.])
    ax.set_yticklabels(['$0.1$', '$1$', '$10$', '$60$'])

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")
    plt.show(block=False)


def posterior_gt(N, y_0, D, nof_samples, sigma=1.):
    covariance_matrix = np.diag([sigma ** 2] * D) / np.sqrt(nof_samples)
    center = y_0[:D] if y_0.ndim == 1 else y_0[0, :]
    return np.random.multivariate_normal(center, covariance_matrix, size=(N,))


def plot_posterior_gt(N_y, savefig):
    fig, ax = plt.subplots()
    x = np.linspace(-15, 15, 500)
    y = np.linspace(-15, 15, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=[5, 5], cov=np.eye(2)/np.sqrt(N_y))
    plt.contourf(X, Y, rv.pdf(pos))

    plt.xlim(0, 10)
    plt.ylim(0, 10)

    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")
    plt.show(block=False)


def save_metrics(metrics):
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df.to_csv("results/exp_multi_observation_base.csv", index=False)


def load_metrics():
    # if file exists, load it, otherwise do nothing
    if os.path.exists("results/exp_multi_observation_base.csv"):
        df = pd.read_csv("results/exp_multi_observation_base.csv")
        metrics = {row["Metric"]: row["Value"] for index, row in df.iterrows()}
        for key in metrics.keys():
            metrics[key] = ast.literal_eval(metrics[key])
        return metrics
    return None


# set seed
np.random.seed(21)

if conf_load:
    metrics = load_metrics()
else:
    metrics = {}

for i, N_y in enumerate(conf_D_list): #   ([2, 5, 10, 15, 20]):
    dim = 10
    prior = priors.Uniform(low=-15, high=15, dim=dim)

    sim = simulators.Linear(dim=dim, sigma=1.0)
    sim_flat = simulators.LinearMultiSamples(dim=dim, sigma=1.0, nof_samples=N_y)

    y_0 = np.ones((N_y, dim)) * 5
    y_0_flat = y_0.flatten()
    y_0_sbi = torch.Tensor(y_0_flat).unsqueeze(0)

    # ground truth
    gt_samples = posterior_gt(1000, y_0, dim, N_y, sigma=1.)
    plot_posterior_samples_2D(gt_samples,
                              "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_gt_samples.pdf")
    plot_posterior_gt(N_y, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_gt_posterior.pdf")


    # R2OMC
    if conf_run_r2omc:
        metrics["r2omc_N_%d" % N_y] = []
        metrics["r2omc_N_%d_runtime" % N_y] = []
        tic = timeit.default_timer()
        iter_omc = r2omc.IterativeR2OMC(sim, y_0, prior, dim=dim)
        config = {
            "find_informative_dims": True,
            "fit_seed": 21,
            "inf_dims_nof_th": 10,
            "inf_dims_nof_seeds": 50,
            "nof_seeds_total": 2000,
            "nof_th0": 1,
            "nof_gd_steps": 30,
            "alpha": .1,
            "epochs": 8,
            "nof_seeds_accept": 2000,
            "dx": .3,
            "nof_ls_steps": 100,
            "step_size": .01,
            "sample_seed": 21,
            "nof_samples": 2000,
            "eps_3": .5,
            "nof_samples_to_select": 1000
        }
        samples = iter_omc.infer(config)
        toc = timeit.default_timer() - tic
        print(f"R2OMC took {toc:.2f} seconds.")

        plot_posterior_samples_2D(samples, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_r2omc_samples.pdf")
        plot_posterior_kde_2D(samples, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_r2omc_kde.pdf")
        c2st_r2omc = utils.evaluate(gt_samples, samples)[0].item()

        metrics["r2omc_N_%d" % N_y].append(c2st_r2omc)
        metrics["r2omc_N_%d_runtime" % N_y].append(toc)

    # Method 2: Naive ROMC
    if conf_run_romc:
        metrics["romc_N_%d" % N_y] = []
        metrics["romc_N_%d_runtime" % N_y] = []
        tic = timeit.default_timer()
        romc = r2omc.R2OMC(sim_flat, y_0_flat, prior, dim=dim)
        config = {
            "find_informative_dims": False,
            "fit_seed": 21,
            "inf_dims_nof_th": 10,
            "inf_dims_nof_seeds": 50,
            "nof_seeds_total": 5000,
            "nof_th0": 1,
            "nof_gd_steps": 30,
            "alpha": .1,
            "epochs": 8,
            "nof_seeds_accept": 5000,
            "dx": 1.,
            "nof_ls_steps": 100,
            "step_size": .1,
            "sample_seed": 21,
            "nof_samples": 5_000,
            "eps_3": 3.,
        }
        samples, weight = romc.infer(config)
        selected_indices = np.random.choice(
            np.arange(samples.shape[0]),
            size=1_000,
            replace=False,
            p=np.ones(samples.shape[0])/samples.shape[0]
        )
        samples_romc = samples[selected_indices]

        toc = timeit.default_timer() - tic
        print(f"ROMC took {toc:.2f} seconds.")

        plot_posterior_samples_2D(samples_romc, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_romc_samples.pdf")
        plot_posterior_kde_2D(samples_romc, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_romc_kde.pdf")
        c2st_romc = utils.evaluate(gt_samples, samples_romc)[0].item()

        metrics["romc_N_%d" % N_y].append(c2st_romc)
        metrics["romc_N_%d_runtime" % N_y].append(toc)

    # Method 3: NLE
    if conf_run_nle:
        metrics["nle_N_%d" % N_y] = []
        metrics["nle_N_%d_runtime" % N_y] = []

        np.random.seed(2145453)
        torch.manual_seed(21436)
        key = jax.random.key(seed=2154672)
        key, subkey = jax.random.split(key)

        # (i) define experiment and parameters
        nof_tr_samples = 10_000
        tr_batch_size = 1000

        sbi_prior, num_parameters, prior_returns_numpy = sbi.utils.user_input_checks.process_prior(
            priors_sbi.get_uniform(
                prior_limits=[-15, 15],
                D=dim
            )
        )

        tic = time.time()
        theta = np.array(sbi_prior.sample((nof_tr_samples, )))
        sim = simulators.Linear(dim=dim, sigma=1.0)
        sim_func = sim.cr_simulator()
        seeds = np.random.randint(0, 10000000, nof_tr_samples)
        x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples)])
        theta = torch.Tensor(theta)
        x = torch.Tensor(x)

        # (ii) inference
        inferer = sbi.inference.SNLE(sbi_prior, show_progress_bars=True, density_estimator="mdn")
        inferer.append_simulations(theta, x)
        inferer.train(training_batch_size=tr_batch_size, max_num_epochs=500)

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

        samples_nle = posterior.sample(sample_shape=(1000,), x=torch.Tensor(y_0))
        time_nle = time.time() - tic
        print(f"NLE took {time_nle} seconds.")

        plot_posterior_samples_2D(samples_nle, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_nle_samples.pdf")
        plot_posterior_kde_2D(samples_nle, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_nle_kde.pdf")
        c2st_nle = utils.evaluate(gt_samples, samples_nle)[0].item()

        metrics["nle_N_%d" % N_y].append(c2st_nle)
        metrics["nle_N_%d_runtime" % N_y].append(time_nle)

    # Method 4: NPE
    if conf_run_npe:
        metrics["npe_N_%d" % N_y] = []
        metrics["npe_N_%d_runtime" % N_y] = []
        # set experimental variables
        np.random.seed(21355)
        torch.manual_seed(21)
        key = jax.random.key(seed=21)
        key, subkey = jax.random.split(key)

        # (i) define experiment and parameters
        nof_epochs = 1
        nof_tr_samples_per_epoch = 10_000
        tr_batch_size = 1_000
        # y_0 = np.concatenate([np.ones(dim) * 5, np.ones(dim) * -5, np.ones(dim) * 5, np.ones(dim) * -5])

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
            # plt.figure()
            # plt.plot(theta[:, 0], theta[:, 1], "o")
            # plt.xlim(-15, 15)
            # plt.ylim(-15, 15)
            # plt.show(block=False)
            sim = simulators.CasesGaussianMultiSamples(dim=dim, sigma_1=1.0, sigma_2=1.0, nof_samples=N_y)
            sim_func = sim.cr_simulator()
            seeds = np.random.randint(0, 10000000, nof_tr_samples_per_epoch)
            x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples_per_epoch)])

            # x = simulator(theta)
            _ = inferer.append_simulations(torch.Tensor(theta), torch.Tensor(x), proposal=proposal).train(max_num_epochs=1000, training_batch_size=tr_batch_size)
            posterior = inferer.build_posterior().set_default_x(y_0_flat)
            proposal = posterior

        samples_npe = posterior.sample((1000,), x=torch.Tensor(y_0_flat))
        time_npe = time.time() - tic
        print(f"NPE took {time_npe} seconds.")

        plot_posterior_samples_2D(samples_npe, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_npe_samples.pdf")
        plot_posterior_kde_2D(samples_npe, "paper/figures/exp_multi_observation_base_N_" + str(N_y) + "_npe_kde.pdf")
        c2st_npe = utils.evaluate(gt_samples, samples_npe)[0].item()

        metrics["npe_N_%d" % N_y].append(c2st_npe)
        metrics["npe_N_%d_runtime" % N_y].append(time_npe)

# save
if conf_save:
    save_metrics(metrics)

# plot
if conf_save_plot:
    plot_c2st(metrics, "paper/figures/exp_multi_observation_base_c2st.pdf")
    plot_runtime(metrics, "paper/figures/exp_multi_observation_base_runtime.pdf")
else:
    plot_c2st(metrics, None)
