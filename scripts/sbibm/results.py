import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


def other_methods():
    metrics_others = {
        "REJ-ABC": {
            "two_moons_c2st": [0.99, 0.97, 0.95],
            "two_moons_runtime": [0.021*60, 0.022*60, 0.023*60],  # in seconds
            "slcp_c2st": [1., 1., 1.],
            "slcp_runtime": [0.018*60, 0.02*60, .25*60],  # in seconds
            "slcp_distractors_c2st": [1., .97, .95],
            "slcp_distractors_runtime": [0.037*60, 0.2*60, 1.89*60]  # in seconds
            },
        "SMC-ABC": {
            "two_moons_c2st": [0.98, 0.97, 0.96],
            "two_moons_runtime": [0.024*60, 0.035*60, 0.12*60],  # in seconds
            "slcp_c2st": [1., .99, .98],
            "slcp_runtime": [0.019*60, .041*60, .13*60],  # in seconds
            "slcp_distractors_c2st": [1., 1., 1.],
            "slcp_distractors_runtime": [0.04*60, 0.23*60, 2.37*60]  # in seconds
        },
        "NPE": {
            "two_moons_c2st": [0.98, 0.9, 0.82],
            "two_moons_runtime": [0.51*60, 6.2*60, 24.5*60],  # in seconds
            "slcp_c2st": [.99, .9, .82],
            "slcp_runtime": [0.21*60, 7.84*60, 60.36*60],  # in seconds
            "slcp_distractors_c2st": [.99, .99, .88],
            "slcp_distractors_runtime": [0.323*60, 4.76*60, 62.32*60]  # in seconds
            },
        "SNPE": {
            "two_moons_c2st": [0.97, 0.84, 0.67],
            "two_moons_runtime": [2.5*60, 16.2*60, 372*60],  # in seconds
            "slcp_c2st": [0.98, 0.83, 0.68],
            "slcp_runtime": [8.58*60, 28.5*60, 647*60],  # in seconds
            "slcp_distractors_c2st": [0.98, 0.92, 0.79],
            "slcp_distractors_runtime": [2.16*60, 328*60, 704*60]  # in seconds
        },
        "NLE": {
            "two_moons_c2st": [1., 0.97, 0.9],
            "two_moons_runtime": [3.15*60, 6.61*60, 16.1*60],  # in seconds
             "slcp_c2st": [.95, .78, .7],
            "slcp_runtime": [6.48*60, 16.09*60, 40.93*60],  # in seconds
             "slcp_distractors_c2st": [1., .97, .9],
             "slcp_distractors_runtime": [11.09*60, 18*60, 135*60]  # in seconds
            },
        "SNLE": {
            "two_moons_c2st": [0.91, 0.71, 0.59],
            "two_moons_runtime": [6.62*60, 8.79*60, 43.19*60],  # in seconds
            "slcp_c2st": [0.92, 0.71, 0.59],
            "slcp_runtime": [14.65*60, 18.36*60, 77.32*60],  # in seconds
            "slcp_distractors_c2st": [1., 0.94, 0.88],
            "slcp_distractors_runtime": [26*60, 33*60, 223*60]  # in seconds
        },
        "NRE": {
            "two_moons_c2st": [0.98, 0.95, 0.91],
            "two_moons_runtime": [1.19*60, 2.44*60, 141*60],  # in seconds
            "slcp_c2st": [.98, .95, .92],
            "slcp_runtime": [2.81*60, 4.16*60, 183.*60],  # in seconds
            "slcp_distractors_c2st": [.99, .98, .95],
            "slcp_distractors_runtime": [2.94*60, 4.5*60, 241*60]  # in seconds
            },
        "SNRE": {
            "two_moons_c2st": [0.97, 0.91, 0.72],
            "two_moons_runtime": [2.8*60, 5.59*60, 97.57*60],  # in seconds
            "slcp_c2st": [0.97, 0.92, 0.72],
            "slcp_runtime": [7.18*60, 9.65*60, 170*60],  # in seconds
            "slcp_distractors_c2st": [0.98, 0.97, 0.78],
            "slcp_distractors_runtime": [7.35*60, 9.95*60, 164*60]  # in seconds
        },
    }
    return metrics_others


def gather_r2omc_results_slcp(path):
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


def gather_r2omc_results_two_moons(path):
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

        c2st[budget] = []
        runtime[budget] = []
        for run in experiment_runs:
            c2st[budget].append(np.loadtxt(os.path.join(path_budget, run, f"c2st_score.csv"), delimiter=","))
            runtime[budget].append(np.loadtxt(os.path.join(path_budget, run, f"runtime.csv"), delimiter=","))
        c2st[budget] = np.array(c2st[budget])
        runtime[budget] = np.array(runtime[budget])
    return c2st, runtime



def plot_results(task_name, plot_type, r2omc_c2st, r2omc_runtime, add_legend=False, savefig=False):
     # Plotting
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
    key = f"{task_name}_{plot_type}"
    if plot_type == "c2st":
        for method in metrics.keys():
            res = metrics[method][key]
            ax.plot(x, res, 'o--', label=method, color=colormapping[method])
            ax.set_xscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_yticks([.4, .6, .8, 1.])
    else:
        for method in metrics.keys():
            res = np.array(metrics[method][key])  # convert to minutes
            ax.plot(x, res, 'o--', label=method, color=colormapping[method])
            ax.set_xscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
        ax.tick_params(axis='both', which='major', labelsize=13)

    # add R2OMC
    x = r2omc_c2st.keys()
    if plot_type == "c2st":
        if task_name == "two_moons":
            y = [current.mean() for current in r2omc_c2st.values()]
        else:
            y = [current["selected"].mean() for current in r2omc_c2st.values()]

        ax.plot(x, y, "--x", label="R2OMC", color="darkmagenta")
        ax.set_ylabel("C2ST", fontsize=14)
        ax.set_xlabel("Budget (Log)", fontsize=13)
        ax.set_xscale('log')
        ax.set_ylim(0.4, 1.05)
        if add_legend:
            ax.legend(fontsize=13, loc="lower center", ncols=3)
    else:
        if task_name == "two_moons":
            y = [current.mean() for current in r2omc_runtime.values()]
        else:
            y = [current.mean() for current in r2omc_runtime.values()]
        ax.plot(x, y, "--x", label="R2OMC", color="darkmagenta")
        ax.set_ylabel("Runtime (seconds)", fontsize=14)
        ax.set_xlabel("Budget (Log)", fontsize=13)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if add_legend:
            ax.legend(fontsize=13, loc="upper center", ncols=3)

    if savefig:
        os.makedirs(f"./../../paper/figures/sbibm/{task_name}", exist_ok=True)
        plt.savefig(f"./../../paper/figures/sbibm/{task_name}/{plot_type}.png", bbox_inches='tight')
        plt.savefig(f"./../../paper/figures/sbibm/{task_name}/{plot_type}.pdf", bbox_inches='tight')
    else:
        plt.show(block=False)


# ------- Main -------
# ------- SLCP Part ------- #
for task_name in ["slcp", "slcp_distractors"]:
    path = f"./../../results/sbibm/{task_name}"
    r2omc_c2st, r2omc_runtime = gather_r2omc_results_slcp(path)
    for plot_type in ["c2st", "runtime"]:
        plot_results(task_name, plot_type, r2omc_c2st, r2omc_runtime, add_legend=False, savefig=True)

# copy pairwise posterior plots to paper folder
exp_num = 5
budget = 10_000
path_from = f"./../../results/sbibm/slcp/budget_{budget}/exp_{exp_num}"
path_to = f"../../paper/figures/sbibm/slcp"
for type in ["accepted", "selected", "total"]:
    os.makedirs(f"../../paper/figures/sbibm/slcp/", exist_ok=True)
    shutil.copyfile(
        os.path.join(path_from, f"pairwise_posterior_{type}.png"),
        os.path.join(path_to, f"pairwise_posterior_{type}.png")
    )

exp_num = 5
budget = 10_000
path_from = f"./../../results/sbibm/slcp_distractors/budget_{budget}/exp_{exp_num}"
path_to = f"../../paper/figures/sbibm/slcp_distractors"
for type in ["accepted", "selected", "total"]:
    os.makedirs(f"../../paper/figures/sbibm/slcp_distractors/", exist_ok=True)
    shutil.copyfile(
        os.path.join(path_from, f"pairwise_posterior_{type}.png"),
        os.path.join(path_to, f"pairwise_posterior_{type}.png")
    )

# ------- Two Moons Part ------- #
task_name = "two_moons"
path = f"./../../results/sbibm/{task_name}"
r2omc_c2st, r2omc_runtime = gather_r2omc_results_two_moons(path)
for plot_type in ["c2st", "runtime"]:
    plot_results(task_name, plot_type, r2omc_c2st, r2omc_runtime, add_legend=False, savefig=True)

# copy pairwise posterior plots to paper folder
exp_num = 3
budget = 10_000
path_from = f"./../../results/sbibm/two_moons/budget_{budget}/exp_{exp_num}"
path_to = f"../../paper/figures/sbibm/two_moons"
os.makedirs(f"../../paper/figures/sbibm/two_moons/", exist_ok=True)
shutil.copyfile(
    os.path.join(path_from, f"pairwise_posterior.png"),
    os.path.join(path_to, f"pairwise_posterior.png")
)