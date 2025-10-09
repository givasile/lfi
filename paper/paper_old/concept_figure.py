import priors
import simulators
import jax
import r2omc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# set seed
np.random.seed(21)
key = jax.random.key(seed=21)
key, subkey = jax.random.split(key)

th_star = np.array([2., 2.])
y_0 = np.array([[0., 0.], [5, 5]])
y_0_flat = y_0.flatten()

# Case 1: flat output
prior = priors.Uniform(low=-15., high=15., dim=2)
simulator = simulators.TwoCasesGaussianMultiSamples(
    dim=2,
    sigma_1=1.,
    nof_samples=2
)
omc = r2omc.R2OMC(simulator, y_0_flat, prior, 2)
config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 300,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .1,
    "epochs": 4,
    "nof_seeds_accept": 200,
    "dx": .3,
    "nof_ls_steps": 30,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 1000,
    "eps_3": .5,
}
samples, weight = omc.infer(config)

# Plot
plt.figure()
plt.title(r"$\mathbf{\theta} \sim p^{ROMC}(\mathbf{\theta} | \mathbf{y})$")

samples = omc.samples_flat[np.random.permutation(omc.samples_flat.shape[0])]
plt.scatter(
    samples[:300, 0],
    samples[:300, 1],
    color="b", label=r"$\mathtt{ROMC}$", alpha=.5, marker="o")

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

mean1 = [0., 0.]
cov = [[1., 0], [0, 1.]]

rv1 = multivariate_normal(mean1, cov)
plt.plot(0, 0, "x", color="black")
plt.contour(X, Y, rv1.pdf(pos), linestyles="-")

plt.xlim(-7.5, 7.5)
plt.ylim(-7.5, 7.5)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.legend()
plt.grid()
plt.savefig("./paper/figures/concept_image_romc.png", bbox_inches="tight")
plt.show(block=False)


# Iterative R2OMC
simulator = simulators.TwoCasesGaussian(dim=2, sigma_1=1.)
iter_r2omc = r2omc.IterativeR2OMC(simulator, y_0, prior, 2)

config = {
    "find_informative_dims": True,
    "fit_seed": 21,
    "inf_dims_nof_th": 10,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 250,
    "nof_th0": 2,
    "nof_gd_steps": 50,
    "alpha": .2,
    "epochs": 4,
    "nof_seeds_accept": 250,
    "dx": .5,
    "nof_ls_steps": 40,
    "step_size": .1,
    "sample_seed": 21,
    "nof_samples": 1000,
    "eps_3": 10,
    "nof_samples_to_select": 300,
}
samples = iter_r2omc.infer(config)

# Plot for concept image
th1 = np.linspace(-10, 10, 20)
th2 = np.linspace(-10, 10, 20)
th = np.meshgrid(th1, th2)
th = np.dstack(th).reshape(-1, 2)

# prior
plt.figure(figsize=(3, 2))
plt.plot(th[:, 0], th[:, 1], "rx", alpha=.5)
plt.xlim(-11, 11)
plt.ylim(-11, 11)
plt.xticks([])
plt.yticks([])
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.savefig("./paper/figures/concept_image_1_slide.png", bbox_inches="tight")
plt.show(block=False)


yy_list = []
for i in range(2):
    color = "b" if i == 0 else "g"
    yy = iter_r2omc.r2omc_list[i].get_proposal_2d(th, x_dims=[0, 1])
    yy_list.append(yy)

    plt.figure(figsize=(3, 2))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.scatter(
        th[yy > 0, 0],
        th[yy > 0, 1],
        color=color,
        marker="x",
        alpha=.5,
    )
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"./paper/figures/concept_image_{i + 2}_slide.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)

# Plot
plt.figure(figsize=(4, 3))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
yy = iter_r2omc.get_proposal_2d(th, th_dims=[0, 1])
plt.scatter(
    th[yy_list[0] > 0, 0],
    th[yy_list[0] > 0, 1],
    color="b",
    marker="x",
    alpha=.4,
    label=r"$\mathbf{y}_0$"
)
plt.scatter(
    th[yy_list[1] > 0, 0],
    th[yy_list[1] > 0, 1],
    marker="x",
    color="g",
    alpha=.4,
    label=r"$\mathbf{y}_1$"
)
plt.scatter(
    th[yy > 0, 0],
    th[yy > 0, 1],
    color="r",
    marker="x",
    alpha=.8,
    label=r"Intersection"
)
plt.xticks([])
plt.yticks([])
plt.savefig("./paper/figures/concept_image_4_slide.png", dpi=300, bbox_inches="tight")
plt.show(block=False)

plt.figure(figsize=(4, 3))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

# X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
# pos = np.dstack((X, Y))
# rv1 = multivariate_normal([0, 0], [[1, 0], [0, 1]])
# plt.contour(X, Y, rv1.pdf(pos), linestyles="-")

plt.scatter(
    samples[:100, 0],
    samples[:100, 1],
    color="b", label=r"$\mathtt{ROMC}$", alpha=.5, marker="o")
plt.xticks([])
plt.yticks([])
plt.savefig("./paper/figures/concept_image_5_slide.png", dpi=300, bbox_inches="tight")
plt.show(block=False)



# Plot for real image
th1 = np.linspace(-10, 10, 50)
th2 = np.linspace(-10, 10, 50)
th = np.meshgrid(th1, th2)
th = np.dstack(th).reshape(-1, 2)

# prior
plt.figure()
plt.title(r"$p(\mathbf{\theta})$")
plt.plot(th[:, 0], th[:, 1], "rx", alpha=.5)
plt.xlim(-10.5, 10.5)
plt.ylim(-10.5, 10.5)
plt.xticks([])
plt.yticks([])
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.savefig("./paper/figures/concept_image_1.png", bbox_inches="tight")
plt.show(block=False)


yy_list = []
for i in range(2):
    color = "b" if i == 0 else "g"
    yy = iter_r2omc.r2omc_list[i].get_proposal_2d(th, x_dims=[0, 1])
    yy_list.append(yy)

    plt.figure()
    plt.title(r"Proposal for $\mathbf{y}_{" + str(i) + "}$")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.scatter(
        th[yy > 0, 0],
        th[yy > 0, 1],
        color=color,
        alpha=.5,
    )
    plt.grid()
    plt.savefig(f"./paper/figures/concept_image_{i + 2}.png", bbox_inches="tight")
    plt.show(block=False)



# Plot
plt.figure()
plt.title(r"Intersection of Proposals")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
yy = iter_r2omc.get_proposal_2d(th, th_dims=[0, 1])
plt.scatter(
    th[yy_list[0] > 0, 0],
    th[yy_list[0] > 0, 1],
    color="b",
    alpha=.4,
    label=r"$\mathbf{y}_0$"
)
plt.scatter(
    th[yy_list[1] > 0, 0],
    th[yy_list[1] > 0, 1],
    color="g",
    alpha=.4,
    label=r"$\mathbf{y}_1$"
)
plt.scatter(
    th[yy > 0, 0],
    th[yy > 0, 1],
    color="r",
    alpha=.8,
    label=r"Intersection"
)
plt.grid()
plt.legend()
plt.savefig("./paper/figures/concept_image_4.png", bbox_inches="tight", pad_inches=0)
plt.show(block=False)


plt.figure()
plt.title(r"Samples from $p^{ROMC}(\mathbf{\theta} | \mathbf{y})$")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

# X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
# pos = np.dstack((X, Y))
# rv1 = multivariate_normal([0, 0], [[1, 0], [0, 1]])
# plt.contour(X, Y, rv1.pdf(pos), linestyles="-")

plt.scatter(
    samples[:100, 0],
    samples[:100, 1],
    color="b", label=r"$\mathtt{ROMC}$", alpha=.5, marker="o")
plt.grid()
plt.savefig("./paper/figures/concept_image_5_slide.png", dpi=300, bbox_inches="tight")
plt.show(block=False)
