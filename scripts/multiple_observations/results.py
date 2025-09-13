import numpy as np
import os
import matplotlib.pyplot as plt


def other_methods():
    metrics_others = {
        "REJ-ABC": {
            "two_moons_c2st": [0.99, 0.97, 0.95],
            "slcp_c2st": [1., 1., 1.],
            "slcp_runtime": [0.018*60, 0.02*60, .25*60],  # in seconds
            "slcp_distractors_c2st": [1., .97, .95],
            "slcp_distractors_runtime": [0.037*60, 0.2*60, 1.89*60]  # in seconds
            },
        "SMC-ABC": {
            "two_moons_c2st": [0.98, 0.97, 0.96],
            "slcp_c2st": [1., .99, .98],
            "slcp_runtime": [0.019*60, .041*60, .13*60],  # in seconds
            "slcp_distractors_c2st": [1., 1., 1.],
            "slcp_distractors_runtime": [0.04*60, 0.23*60, 2.37*60]  # in seconds
        },
        "NPE": {
            "two_moons_c2st": [0.98, 0.9, 0.82],
            "slcp_c2st": [.99, .9, .82],
            "slcp_runtime": [0.21*60, 7.84*60, 60.36*60],  # in seconds
            "slcp_distractors_c2st": [.99, .99, .88],
            "slcp_distractors_runtime": [0.323*60, 4.76*60, 62.32*60]  # in seconds
            },
        "SNPE": {
            "two_moons_c2st": [0.97, 0.84, 0.67],
            "slcp_c2st": [0.98, 0.83, 0.68],
            "slcp_runtime": [8.58*60, 28.5*60, 647*60],  # in seconds
            "slcp_distractors_c2st": [0.98, 0.92, 0.79],
            "slcp_distractors_runtime": [2.16*60, 328*60, 704*60]  # in seconds
        },
        "NLE": {
            "two_moons_c2st": [1., 0.97, 0.9],
             "slcp_c2st": [.95, .78, .7],
            "slcp_runtime": [6.48*60, 16.09*60, 40.93*60],  # in seconds
             "slcp_distractors_c2st": [1., .97, .9],
             "slcp_distractors_runtime": [11.09*60, 18*60, 135*60]  # in seconds
            },
        "SNLE": {
            "two_moons_c2st": [0.91, 0.71, 0.59],
            "slcp_c2st": [0.92, 0.71, 0.59],
            "slcp_runtime": [14.65*60, 18.36*60, 77.32*60],  # in seconds
            "slcp_distractors_c2st": [1., 0.94, 0.88],
            "slcp_distractors_runtime": [26*60, 33*60, 223*60]  # in seconds
        },
        "NRE": {
            "two_moons_c2st": [0.98, 0.95, 0.91],
            "slcp_c2st": [.98, .95, .92],
            "slcp_runtime": [2.81*60, 4.16*60, 183.*60],  # in seconds
            "slcp_distractors_c2st": [.99, .98, .95],
            "slcp_distractors_runtime": [2.94*60, 4.5*60, 241*60]  # in seconds
            },
        "SNRE": {
            "two_moons_c2st": [0.97, 0.91, 0.72],
            "slcp_c2st": [0.97, 0.92, 0.72],
            "slcp_runtime": [7.18*60, 9.65*60, 170*60],  # in seconds
            "slcp_distractors_c2st": [0.98, 0.97, 0.78],
            "slcp_distractors_runtime": [7.35*60, 9.95*60, 164*60]  # in seconds
        },
    }
    return metrics_others


def gather_r2omc_results(path):
    # find all budgets
    dir_names = os.listdir(path)
    budgets = [int(dir_name.split("_")[-1]) for dir_name in dir_names]
    # init dicts
    budgets.sort()
    c2st = {}
    runtime = {}
    for i, budget in enumerate(budgets):
        # find all experiment runs
        path_budget = os.path.join(path, f"budget_{budget}")
        experiment_runs = os.listdir(path_budget)
        c2st[budget] = {"accepted": [], "selected": [], "total": []}
        runtime[budget] = []
        for run in experiment_runs:
            for type in ["accepted", "selected", "total"]:
                c2st[budget][type].append(np.loadtxt(os.path.join(path_budget, run, f"c2st_{type}.csv"), delimiter=","))
            runtime[budget].append(np.loadtxt(os.path.join(path_budget, run, f"time_selected.csv"), delimiter=","))

        for type in ["accepted", "selected", "total"]:
            c2st[budget][type] = np.array(c2st[budget][type])
        runtime[budget] = np.array(runtime[budget])
    return c2st, runtime


path = "./../../results/multiple_observations/"
task_name = "slcp_distractors"
plot_type = "runtime"  # "c2st" or "runtime"
path = os.path.join(path, task_name)
key = task_name + "_" + plot_type

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

if plot_type == "c2st":
    for method in metrics.keys():
        res = metrics[method][key]
        ax.plot(x, res, 'o-', label=method, color=colormapping[method])
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_yticks([.4, .6, .8, 1.])
else:
    for method in metrics.keys():
        res = np.array(metrics[method][key])  # convert to minutes
        ax.plot(x, res, 'o-', label=method, color=colormapping[method])
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
    ax.tick_params(axis='both', which='major', labelsize=13)

# add R2OMC
c2st, runtime = gather_r2omc_results(path)
x = c2st.keys()
if plot_type == "c2st":
    y = [current["selected"].mean() for current in c2st.values()]
    ax.plot(x, y, "--x", label="R2OMC", color="darkmagenta")
    ax.set_ylabel("C2ST", fontsize=14)
    ax.set_xlabel("Budget (Log)", fontsize=13)
    ax.set_xscale('log')
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=13, loc="lower center", ncols=3)
    plt.show(block=False)
else:
    y = [current.mean() for current in runtime.values()]
    ax.plot(x, y, "--x", label="R2OMC", color="darkmagenta")
    ax.set_ylabel("Runtime (seconds)", fontsize=14)
    ax.set_xlabel("Budget (Log)", fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=13, loc="upper left", ncols=3)
    plt.show(block=False)
