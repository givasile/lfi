import simulators
import priors
import experiments
import numpy as np
import r2omc
import matplotlib.pyplot as plt

import utils

np.random.seed(21)

exp = experiments.SbiExperiment(
    task_name="slcp",
    exp_num=1
)

obs = 0
y_0 = np.zeros((4, 2))
for i in range(4):
    y_0[i] = exp.y_0[2*obs:2*obs+2]
    obs += 1

metrics = []
samples = []

# Simulation budget: 1000
rromc = r2omc.IterativeR2OMC(
    simulators.SLCP(dim=5),
    y_0,
    priors.Uniform(low=-3., high=3., dim=5),
    dim=5
)

config = {
    "find_informative_dims": True,
    "fit_seed": 2451,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 1000,
    "nof_th0": 50,
    "nof_gd_steps": 20,
    "alpha": .1,
    "epochs": 10,
    "nof_seeds_accept": 200,
    "dx": .01,
    "nof_ls_steps": 200,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 50*200 * 5,
    "eps_3": .2,
    "nof_samples_to_select": 1000,
}

rromc_samples = rromc.infer(config)
samples.append(rromc_samples)

# select 1000 gt samples at random
np.random.seed(21)
ind = np.random.choice(np.arange(exp.samples_gt.shape[0]), 1000, replace=False)
samples_gt = exp.samples_gt[ind]

metrics.append(utils.evaluate(samples_gt, rromc_samples)[0].item())

# Simulation budget: 10_000
rromc = r2omc.IterativeR2OMC(
    simulators.SLCP(dim=5),
    y_0,
    priors.Uniform(low=-3., high=3., dim=5),
    dim=5
)

config = {
    "find_informative_dims": True,
    "fit_seed": 2451,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 10_000,
    "nof_th0": 30,
    "nof_gd_steps": 20,
    "alpha": .2,
    "epochs": 5,
    "nof_seeds_accept": 1_000,
    "dx": .01,
    "nof_ls_steps": 100,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 30_000,
    "eps_3": .2,
    "nof_samples_to_select": 1000,
}

rromc_samples = rromc.infer(config)
samples.append(rromc_samples)

# select 1000 gt samples at random
np.random.seed(21)
ind = np.random.choice(np.arange(exp.samples_gt.shape[0]), 1000, replace=False)
samples_gt = exp.samples_gt[ind]

metrics.append(utils.evaluate(samples_gt, rromc_samples))

# SIMULATION BUDGET: 30_000
rromc = r2omc.IterativeR2OMC(
    simulators.SLCP(dim=5),
    y_0,
    priors.Uniform(low=-3., high=3., dim=5),
    dim=5
)


config = {
    "find_informative_dims": True,
    "fit_seed": 2451,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_seeds_total": 30_000,
    "nof_th0": 10,
    "nof_gd_steps": 20,
    "alpha": .2,
    "epochs": 5,
    "nof_seeds_accept": 3_000,
    "dx": .01,
    "nof_ls_steps": 100,
    "step_size": .01,
    "sample_seed": 21,
    "nof_samples": 30_000,
    "eps_3": .1,
    "nof_samples_to_select": 1000,
}

rromc_samples = rromc.infer(config)
samples.append(rromc_samples)

# select 1000 gt samples at random
np.random.seed(21)
ind = np.random.choice(np.arange(exp.samples_gt.shape[0]), 1000, replace=False)
samples_gt = exp.samples_gt[ind]

metrics.append(utils.evaluate(samples_gt, rromc_samples))



# plot
save_path = "paper/figures/"
task_name = "slcp"
metrics_others = {
            "REJ-ABC":
                {
                    "c2st": [0.99, 0.97, 0.95],
                },
            "NLE":
                {
                    "c2st": [0.94, 0.78, 0.7],
                },
            "NPE":
                {
                    "c2st": [0.98, 0.9, 0.82],
                },
            "NRE":
                {
                    "c2st": [0.98, 0.95, 0.91],
                },
            "SMC-ABC":
                {
                    "c2st": [0.98, 0.97, 0.96],
                },
            "SNLE":
                {
                    "c2st": [0.91, 0.71, 0.59],
                },
            "SNPE":
                {
                    "c2st": [0.97, 0.84, 0.67],
                },
            "SNRE":
                {
                    "c2st": [0.97, 0.91, 0.72],
                }
        }

x = [1_000, 10_000, 100_000]

fig, ax = plt.subplots()
y = metrics
ax.plot(x, y, "-o", label="R2OMC", color="darkmagenta")
ax.set_ylabel("C2ST")
ax.set_xlabel("Simulation Budget (Log)")
ax.set_xscale('log')
ax.set_xticks(x)
ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
ax.set_ylim([0.4, 1.1])
ax.spines[['right', 'top']].set_visible(False)
if save_path is not None:
    plt.savefig(save_path + task_name + "_romc.pdf", bbox_inches='tight')
plt.show(block=False)

fig, ax = plt.subplots()
ii = 0
# create 8 colors
colors = ["green", "darkturquoise", "dodgerblue", "orange", "darkgreen", "cadetblue", "blue", "darkorange"]
for key, value in metrics_others.items():
    ax.plot(x, value["c2st"], 'o-', label=key, color=colors[ii])
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ii += 1

# add R2OMC
ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_yticks([.4, .6, .8, 1.])
y = metrics
ax.plot(x, y, "--x", label="R2OMC", color="darkmagenta")
ax.set_ylabel(r'Score (C2ST)', fontsize=14)
ax.set_xlabel(r'Simulation Budget (Log)', fontsize=13)
ax.set_ylim([0.4, 1.1])
ax.spines[['right', 'top']].set_visible(False)
ax.legend(fontsize=14, loc='lower left', ncol=3)
if save_path is not None:
    plt.savefig(save_path + task_name + ".pdf", bbox_inches='tight')
plt.show(block=False)
