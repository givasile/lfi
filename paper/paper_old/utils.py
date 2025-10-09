import numpy as np
import simulators
import sbi.inference
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, check_sbi_inputs
import torch
import matplotlib.pyplot as plt
from sbi import utils
from sbibm.metrics import c2st, mmd, ppc
from sbi.utils.get_nn_models import posterior_nn
from scipy.stats import gaussian_kde, norm, multivariate_normal


def posterior_sbi(method_name, prior_sbi, theta, y, y_0_sbi, nof_samples):
    # set up inference method
    sbi_prior, num_parameters, prior_returns_numpy = process_prior(prior_sbi)
    method = method_name_to_sbi(method_name)

    density_estimator_fun = posterior_nn(
        model='nsf',
        hidden_features=50,
        z_score_x="independent",
        z_score_theta="independent"
    )

    inference = method(prior=sbi_prior, density_estimator=density_estimator_fun)

    # train density estimator
    inference.append_simulations(torch.Tensor(theta), torch.Tensor(np.array(y)))
    density_estimator = inference.train(
        num_atoms=10,
        training_batch_size=theta.shape[0],
        show_train_summary=True
    )
    posterior = inference.build_posterior(density_estimator, sbi_prior)
    th_snpe = posterior.sample((nof_samples,), x=y_0_sbi)
    return th_snpe


def posterior_sbi_sequential(method_name, prior_sbi, nof_tr_samples, y_0_sbi, nof_samples, epochs, sim, divide=False):
    nof_tr_samples = int(nof_tr_samples / epochs) if divide else nof_tr_samples
    sbi_prior, num_parameters, prior_returns_numpy = sbi.utils.user_input_checks.process_prior(prior_sbi)
    method = method_name_to_sbi(method_name)
    inference = method(prior=sbi_prior)
    # sim = simulators.Linear(D, sigma)
    sim_func = sim.cr_simulator()
    theta = np.array(sbi_prior.sample_n(nof_tr_samples, ))
    seeds = np.random.randint(0, 10000000, nof_tr_samples)
    x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples)])
    density_estimator = inference.append_simulations(torch.Tensor(theta), torch.Tensor(np.array(x))).train()
    for i in range(epochs-1):
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(y_0_sbi)

        theta = proposal.sample((nof_tr_samples,))
        seeds = np.random.randint(0, 10000000, nof_tr_samples)
        x = np.array([sim_func(theta[i], seeds[i]) for i in range(nof_tr_samples)])
        density_estimator = inference.append_simulations(torch.Tensor(theta), torch.Tensor(np.array(x)), proposal=proposal).train()

    th_snpe = posterior.sample((nof_samples,), x=y_0_sbi)

    return th_snpe


def evaluate(gt, pred):
    c2st_romc = c2st(torch.Tensor(gt), torch.Tensor(pred))
    mmd_romc = mmd(torch.Tensor(gt), torch.Tensor(pred))
    meddist_romc = ppc.median_distance(torch.Tensor(gt), torch.Tensor(pred))
    print("C2ST:", c2st_romc, "MMD:", mmd_romc, "Median distance:", meddist_romc)
    return c2st_romc, mmd_romc, meddist_romc


def method_name_to_sbi(method_name):
    if method_name == "SNPE-A":
        return sbi.inference.SNPE_A
    elif method_name == "SNPE-C":
        return sbi.inference.SNPE_C
    elif method_name == "SNLE-A":
        return sbi.inference.SNLE_A
    elif method_name == "SNRE-A":
        return sbi.inference.SNRE_A
    elif method_name == "SNRE-B":
        return sbi.inference.SNRE_B
    elif method_name == "SNRE-C":
        return sbi.inference.SNRE_C
    else:
        raise ValueError("Invalid method name")


def plot_gt_2d(savefig, nof_obs=1):
    th_star = np.array([5., 5.])
    fig, ax = plt.subplots()
    x = np.linspace(-15, 15, 500)
    y = np.linspace(-15, 15, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv1 = multivariate_normal(mean=-th_star, cov=np.eye(2)/np.sqrt(nof_obs))
    rv2 = multivariate_normal(mean=th_star, cov=np.eye(2)/np.sqrt(nof_obs))
    plt.contourf(X, Y, (rv1.pdf(pos) + rv2.pdf(pos))/2)

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # Remove axis lines (spines) and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")
    plt.show(block=False)


def plot_posterior_kde_1d(title, samples_rromc, samples_romc, samples_npe, sample_nle, savefig=False):

    # Plot the scatter (dots)
    fig, ax = plt.subplots()
    # plt.scatter(samples_i, np.zeros(len(samples_i)), alpha=0.5, label='Samples (Dots)', color='blue')

    labels = ['R2OMC', 'ROMC', 'SNPE', 'SNLE']
    colors = ['darkmagenta', 'magenta', 'dodgerblue', 'darkturquoise']
    x_range = np.linspace(-15, 15, 1000)
    # Estimate the density using Kernel Density Estimation (KDE)
    for i, samples in enumerate([samples_rromc, samples_romc, samples_npe, sample_nle]):
        if samples is not None:
            label = labels[i]
            color = colors[i]
            samples_i = samples[:, 0]
            kde = gaussian_kde(samples_i, bw_method=0.1)
            kde_values = kde(x_range)
            dx = x_range[1] - x_range[0]
            kde_values_normalized = kde_values / np.sum(kde_values * dx)
            ax.plot(x_range, kde_values_normalized, label=label, color=color)

    # Create ground truth: Mixture of two Gaussians
    mu1, mu2 = -5, 5  # Means of the two Gaussians
    sigma = 1  # Unit covariance (standard deviation)
    gaussian1 = norm.pdf(x_range, mu1, sigma)
    gaussian2 = norm.pdf(x_range, mu2, sigma)
    mixture = 0.5 * gaussian1 + 0.5 * gaussian2
    mixture_normalized = mixture / np.sum(mixture * dx)
    ax.plot(x_range, mixture_normalized, label='gt', color='red', linestyle='dashed')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks
    plt.legend()

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")
    plt.show(block=False)

def plot_posterior_kde_2D(
        samples,
        savefig=False,
        bw_method=0.15,
):
    fig, ax = plt.subplots()

    # Create grid for KDE evaluation
    x = np.linspace(-15, 15, 500)
    y = np.linspace(-15, 15, 500)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Calculate 2D KDE using scipy's gaussian_kde
    samples = samples[:, :2]
    kde = gaussian_kde(samples.T, bw_method=bw_method)
    Z = np.reshape(kde(positions).T, X.shape)  # Evaluate KDE on grid and reshape

    # Plot KDE as filled contour plot
    plt.contourf(X, Y, Z)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    # Remove axis lines (spines) and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")

    plt.show(block=False)


def plot_posterior_samples_2D(
        samples,
        savefig=False,
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
    # Remove axis lines (spines) and ticks
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    # ax.set_xticks([])  # Remove x ticks
    # ax.set_yticks([])  # Remove y ticks

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.savefig(savefig.replace(".pdf", ".png"), bbox_inches="tight")

    plt.show(block=False)

def plot_posterior_samples_1d(
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
        color=color,
        marker="o",
        )
    # ground truth
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=th_star, cov=np.eye(2))
    plt.contour(X, Y, rv.pdf(pos), levels=5)
    rv = multivariate_normal(mean=-th_star, cov=np.eye(2))
    plt.contour(X, Y, rv.pdf(pos), levels=5)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if savefig:
        plt.savefig(f"./paper/figures/concept_image_distractors_{title}.pdf", bbox_inches="tight")
    plt.show(block=False)
