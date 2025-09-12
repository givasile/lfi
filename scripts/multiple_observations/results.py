import numpy as np
import os
import matplotlib.pyplot as plt


def other_methods():
    metrics_others = {
        "REJ-ABC": {
            "two_moons_c2st": [0.99, 0.97, 0.95],
            "slcp_c2st": [1., 1., 1.],
            "sclp_distractor_c2st": [1., .97, .95],
            },
        "NLE": {
            "two_moons_c2st": [1., 0.97, 0.9],
             "slcp_c2st": [.95, .78, .7],
             "sclp_distractor_c2st": [1., .97, .9],
            },
        "NPE": {
            "two_moons_c2st": [0.98, 0.9, 0.82],
            "slcp_c2st": [.99, .9, .82],
            "sclp_distractor_c2st": [.99, .99, .88],
            },
        "NRE": {
            "two_moons_c2st": [0.98, 0.95, 0.91],
            "slcp_c2st": [.98, .95, .92],
            "sclp_distractor_c2st": [.99, .98, .95]
            },
        "SMC-ABC": {
            "two_moons_c2st": [0.98, 0.97, 0.96],
            "slcp_c2st": [1., .99, .98],
            "sclp_distractor_c2st": [1., 1., 1.]
            },
        "SNLE": {
            "two_moons_c2st": [0.91, 0.71, 0.59],
            "slcp_c2st": [0.92, 0.71, 0.59],
        "sclp_distractor_c2st": [1., 0.94, 0.88]
            },
        "SNPE": {
            "two_moons_c2st": [0.97, 0.84, 0.67],
            "slcp_c2st": [0.98, 0.83, 0.68],
            "sclp_distractor_c2st": [0.98, 0.92, 0.79]
            },
        "SNRE": {
            "two_moons_c2st": [0.97, 0.91, 0.72],
            "slcp_c2st": [0.97, 0.92, 0.72],
            "sclp_distractor_c2st": [0.98, 0.97, 0.78]
        }
    }
    return metrics_others


def gather_r2omc_results(path):
    experiment_runs = os.listdir(path)
    c2st = {"accepted": [], "selected": [], "total": []}
    runtime = {"accepted": [], "selected": [], "total": []}
    for run in experiment_runs:
        for type in ["accepted", "selected", "total"]:
            c2st[type].append(np.loadtxt(os.path.join(path, run, f"c2st_{type}.csv"), delimiter=","))
            runtime[type].append(np.loadtxt(os.path.join(path, run, f"time_{type}.csv"), delimiter=","))

    for type in ["accepted", "selected", "total"]:
        c2st[type] = np.array(c2st[type])
        runtime[type] = np.array(runtime[type])
    return c2st, runtime



path = "./../../results/multiple_observations/slcp/"
c2st, runtime = gather_r2omc_results(path)

save_path = "./../../results/multiple_observations/plots/"

fig, ax = plt.subplots()
x = [1_000, 10_000, 100_000]
metrics = other_methods()

colormapping = {
    "REJ-ABC": "green",
    "NLE": "darkturquoise",
    "NPE": "dodgerblue",
    "NRE": "orange",
    "SMC-ABC": "darkgreen",
    "SNLE": "cadetblue",
    "SNPE": "blue",
    "SNRE": "darkorange"
}

task_name = "slcp"
key = task_name + "_c2st"
for method in metrics.keys():
    res = metrics[method][key]
    ax.plot(x, res, 'o-', label=method, color=colormapping[method])
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_yticks([.4, .6, .8, 1.])

# add R2OMC
x = [1_500]
y = c2st["selected"].mean()
ax.plot(x, y, "--x", label="R2OMC", color="darkmagenta")
ax.set_ylabel("C2ST", fontsize=14)
ax.set_xlabel("Simulation Budget (Log)", fontsize=13)
ax.set_xscale('log')
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=13, loc="lower center", ncols=3)
plt.show(block=False)
