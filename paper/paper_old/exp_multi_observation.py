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


def plot_ground_truth(N, case=1, savefig=False):
    plt.figure()
    plt.title(r"Case " + str(case) + r", $N = " + str(N) + r"$")

    # Create a grid of points
    X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    pos = np.dstack((X, Y))

    if case == 1:
        rv1 = multivariate_normal(np.ones([2, ]) * -5, np.eye(2) / N)
        pdf1 = rv1.pdf(pos)
        levels1 = np.linspace(pdf1.max() * 0.05, pdf1.max(), 5)  # 10% to 100% of max
        plt.contour(X, Y, pdf1, levels=levels1, linestyles="--")

    rv2 = multivariate_normal(np.zeros([2, ]), np.eye(2) / N)
    pdf2 = rv2.pdf(pos)
    levels2 = np.linspace(pdf2.max() * 0.05, pdf2.max(), 5)  # 10% to 100% of max
    plt.contour(X, Y, pdf2, levels=levels2, linestyles="--")

    plt
    plt.xlabel(r"$\theta_i$")
    plt.ylabel(r"$\theta_j$")
    plt.xticks([-10, 0, 10])
    plt.yticks([-10, 0, 10])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    if savefig:
        plt.savefig("./paper/figures/multi_obs_case_" + str(case) + "_gt_N_" + str(N) + ".pdf", bbox_inches="tight", pad_inches=0)
    plt.show(block=False)


# # Plot ground truth for both cases
# for N in [1, 4, 10, 20]:
#     plot_ground_truth(N, case=1, savefig=True)
#     plot_ground_truth(N, case=2, savefig=True)

# set seed
np.random.seed(21)

# Define simulator
dim = 4
N_y = 4
prior = priors.Uniform(low=-15, high=15, dim=dim)

sim = simulators.TwoCasesGaussian(dim=dim, sigma_1=1.0)
sim_flat = simulators.TwoCasesGaussianMultiSamples(dim=dim, sigma_1=1.0, nof_samples=4)

y_0 = np.zeros((N_y, dim))
y_0_flat = y_0.flatten()
y_0_sbi = torch.Tensor(y_0_flat).unsqueeze(0)

omc = r2omc.R2OMC(sim_flat, y_0_flat, prior, dim=dim)
# Simple OMC
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 10,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 2500,
    "nof_th0": 1,
    "nof_gd_steps": 30,
    "alpha": .1,
    "epochs": 7,
    "nof_seeds_accept": 2000,
    "dx": .2,
    "nof_ls_steps": 100,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 1000,
    "eps_3": 10.,
}

samples, weights = omc.infer(config)

plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], color="magenta", marker=".", label=r"$\mathtt{ROMC}$")
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
pos = np.dstack((X, Y))
rv1 = multivariate_normal([-5, -5], [[1/4, 0], [0, 1/4]])
plt.contour(X, Y, rv1.pdf(pos), linestyles="-")
rv2 = multivariate_normal([0, 0], [[1/4, 0], [0, 1/4]])
plt.contour(X, Y, rv2.pdf(pos), linestyles="-")
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
# plt.savefig("./paper/figures/multi_obs_case_1_romc.pdf", bbox_inches="tight", pad_inches=0)
plt.show(block=False)


# Iterative R2OMC
iter_omc = r2omc.IterativeR2OMC(sim, y_0, prior, dim=dim)
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 10,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 1200,
    "nof_th0": 1,
    "nof_gd_steps": 25,
    "alpha": .2,
    "epochs": 10,
    "nof_seeds_accept": 1000,
    "dx": .1,
    "nof_ls_steps": 100,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 1000,
    "eps_3": .1,
    "nof_samples_to_select": 1000,
}
samples = iter_omc.infer(config)

th1 = np.linspace(-10, 10, 30)
th2 = np.linspace(-10, 10, 30)
th = np.meshgrid(th1, th2)
th = np.dstack(th).reshape(-1, 2)
yy_list = []
for i in range(y_0.shape[0]):
    yy_list.append(iter_omc.r2omc_list[i].get_proposal_2d(th, x_dims=[0, 1]))

# Plot
plt.figure()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
yy = iter_omc.get_proposal_2d(th, th_dims=[0, 1])
colors = ["g", "g", "g", "g"]
label_list = [r"$\mathbf{y}_{1}$", r"$\mathbf{y}_{2}$", r"$\mathbf{y}_{3}$", r"$\mathbf{y}_{4}$"]
for i, y_cur in enumerate(yy_list):
    plt.scatter(
        th[y_cur > 0, 0],
        th[y_cur > 0, 1],
        color=colors[i],
        alpha=.4,
        label=label_list[i]
    )
plt.scatter(
    th[yy > 0, 0],
    th[yy > 0, 1],
    color="darkmagenta",
    alpha=.8,
    label=r"Intersection"
)
plt.legend()
plt.savefig("./paper/figures/multi_obs_case_1_iterative_romc_proposal.pdf", bbox_inches="tight", pad_inches=0)
plt.show(block=False)

# Plot samples
plt.figure()
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
pos = np.dstack((X, Y))
rv1 = multivariate_normal([-5, -5], [[1/4, 0], [0, 1/4]])
plt.contour(X, Y, rv1.pdf(pos), linestyles="-")
rv1 = multivariate_normal([0, 0], [[1/4, 0], [0, 1/4]])
plt.contour(X, Y, rv1.pdf(pos), linestyles="-", label="Ground truth")
plt.scatter(samples[:, 0], samples[:, 1], marker=".", color="darkmagenta", label=r"$\mathtt{R2OMC}$")
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.savefig("./paper/figures/multi_obs_case_1_iterative_romc.pdf", bbox_inches="tight", pad_inches=0)
plt.show(block=False)

# # NPE
# tic = timeit.default_timer()
# method_name = "SNPE-C"
# training_samples = 5_000
# nof_samples = 1000
# sbi_prior = priors_sbi.get_uniform(prior_limits=[-15, 15], D=dim)
# sim = sim_flat
# sim_func = sim.cr_simulator()
# theta = np.array(sbi_prior.sample_n(training_samples, ))
# seeds = np.random.randint(0, 10000000, training_samples)
# x = np.array([sim_func(theta[i], seeds[i]) for i in range(training_samples)])
# samples_npe = utils.posterior_sbi(method_name, sbi_prior, theta, x, y_0_sbi, nof_samples)
# toc = timeit.default_timer() - tic
# print(f"Time for NPE: {toc:.2f} seconds")

# plt.figure()
# plt.scatter(samples_npe[:, 0], samples_npe[:, 1], color="dodgerblue", label=r"$\mathtt{NPE}$", marker=".")
# X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
# pos = np.dstack((X, Y))
# rv1 = multivariate_normal([-5, -5], [[1/4, 0], [0, 1/4]])
# plt.contour(X, Y, rv1.pdf(pos), linestyles="-")
# rv2 = multivariate_normal([0., 0.], [[1/4, 0], [0, 1/4]])
# plt.contour(X, Y, rv2.pdf(pos), linestyles="-")
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
# plt.xlabel(r"$\theta_1$")
# plt.ylabel(r"$\theta_2$")
# plt.legend()
# plt.savefig("./paper/figures/multi_obs_case_1_npe.pdf", bbox_inches="tight", pad_inches=0)
# plt.show(block=False)

# NLE

# set seed
np.random.seed(21)

# Define simulator
dim = 4
N_y = 4
prior = priors.Uniform(low=-15, high=15, dim=dim)

sim = simulators.TwoCasesGaussian(dim=dim, sigma_1=1.0)
sim_flat = simulators.TwoCasesGaussianMultiSamples(dim=dim, sigma_1=1.0, nof_samples=4)

y_0 = np.zeros((N_y, dim))
y_0[1::2, :] = 5
y_0_flat = y_0.flatten()
y_0_sbi = torch.Tensor(y_0_flat).unsqueeze(0)

omc = r2omc.R2OMC(sim_flat, y_0_flat, prior, dim=dim)

# Simple OMC
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 10,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 2500,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .1,
    "epochs": 4,
    "nof_seeds_accept": 2000,
    "dx": .2,
    "nof_ls_steps": 100,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 1000,
    "eps_3": 10.,
}

samples, weights = omc.infer(config)

plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], color="magenta", marker=".", label=r"$\mathtt{ROMC}$")
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
pos = np.dstack((X, Y))
rv1 = multivariate_normal([0, 0], [[1/4, 0], [0, 1/4]])
plt.contour(X, Y, rv1.pdf(pos), linestyles="-")
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.savefig("./paper/figures/multi_obs_case_2_romc.pdf", bbox_inches="tight", pad_inches=0)
plt.show(block=False)


# Iterative R2OMC
iter_omc = r2omc.IterativeR2OMC(sim, y_0, prior, dim=dim)
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 10,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 2500,
    "nof_th0": 1,
    "nof_gd_steps": 20,
    "alpha": .2,
    "epochs": 8,
    "nof_seeds_accept": 2000,
    "dx": .5,
    "nof_ls_steps": 100,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 2000,
    "eps_3": .05,
    "nof_samples_to_select": 1000,
}
samples = iter_omc.infer(config)

th1 = np.linspace(-10, 10, 30)
th2 = np.linspace(-10, 10, 30)
th = np.meshgrid(th1, th2)
th = np.dstack(th).reshape(-1, 2)
yy_list = []
for i in range(y_0.shape[0]):
    yy_list.append(iter_omc.r2omc_list[i].get_proposal_2d(th, x_dims=[0, 1]))

# Plot
plt.figure()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
yy = iter_omc.get_proposal_2d(th, th_dims=[0, 1])
colors = ["g", "y", "g", "y"]
label_list = [r"$\mathbf{y}_{1}$", r"$\mathbf{y}_{2}$", r"$\mathbf{y}_{3}$", r"$\mathbf{y}_{4}$"]
for i, y_cur in enumerate(yy_list):
    plt.scatter(
        th[y_cur > 0, 0],
        th[y_cur > 0, 1],
        color=colors[i],
        alpha=.4,
        label=label_list[i]
    )
plt.scatter(
    th[yy > 0, 0],
    th[yy > 0, 1],
    color="darkmagenta",
    alpha=.8,
    label=r"Intersection"
)
plt.legend()
plt.savefig("./paper/figures/multi_obs_case_2_iterative_romc_proposal.pdf", bbox_inches="tight", pad_inches=0)
plt.show(block=False)

# Plot samples
plt.figure()
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
pos = np.dstack((X, Y))
rv1 = multivariate_normal([0, 0], [[1/4, 0], [0, 1/4]])
plt.contour(X, Y, rv1.pdf(pos), linestyles="-", label="Ground truth")
plt.scatter(samples[:, 0], samples[:, 1], marker=".", color="darkmagenta", label=r"$\mathtt{R2OMC}$")
plt.legend()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.savefig("./paper/figures/multi_obs_case_2_iterative_romc.pdf", bbox_inches="tight", pad_inches=0)
plt.show(block=False)

# # NPE
# tic = timeit.default_timer()
# method_name = "SNPE-C"
# training_samples = 5_000
# nof_samples = 1000
# sbi_prior = priors_sbi.get_uniform(prior_limits=[-15, 15], D=dim)
# sim = sim_flat
# sim_func = sim.cr_simulator()
# theta = np.array(sbi_prior.sample_n(training_samples, ))
# seeds = np.random.randint(0, 10000000, training_samples)
# x = np.array([sim_func(theta[i], seeds[i]) for i in range(training_samples)])
# samples_npe = utils.posterior_sbi(method_name, sbi_prior, theta, x, y_0_sbi, nof_samples)
# toc = timeit.default_timer() - tic
# print(f"Time for NPE: {toc:.2f} seconds")
#
# plt.figure()
# plt.scatter(samples_npe[:, 0], samples_npe[:, 1], color="dodgerblue", label=r"$\mathtt{NPE}$", marker=".")
# X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
# pos = np.dstack((X, Y))
# rv1 = multivariate_normal([0, 0], [[1/4, 0], [0, 1/4]])
# plt.contour(X, Y, rv1.pdf(pos), linestyles="-")
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
# plt.xlabel(r"$\theta_1$")
# plt.ylabel(r"$\theta_2$")
# plt.legend()
# plt.savefig("./paper/figures/multi_obs_case_2_npe.pdf", bbox_inches="tight", pad_inches=0)
# plt.show(block=False)
