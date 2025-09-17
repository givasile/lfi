import matplotlib.pyplot as plt
import numpy as np
import os


def plot_samples(
        task,
        budget = None,
        method_name = None,
        run_id = None,
):
    base_path = "./../../results/concept_figure/"
    gt_samples_path = os.path.join(base_path, task, "gt_samples.csv")
    gt_samples = np.loadtxt(gt_samples_path, delimiter=',')

    if method_name is not None and budget is not None and run_id is not None:
        dir_path = os.path.join(
            base_path,
            task,
            f"{method_name}_{selected_experiments[budget][task][method_name]['budget']}",
            f"run_{run_id}"
        )
        samples = np.loadtxt(os.path.join(dir_path, "samples.csv"), delimiter=',')
        runtime = float(np.loadtxt(os.path.join(dir_path, "runtime.csv"), delimiter=','))
        c2st = float(np.loadtxt(os.path.join(dir_path, "c2st.csv"), delimiter=','))

    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('white')
    plt.scatter(gt_samples[:, 0], gt_samples[:, 1], alpha=0.2, color='red')
    if method_name is not None and budget is not None and run_id is not None:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, color='blue')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    if method_name is not None:
        plt.text(
            0.97,
            0.05,
            f"Time: {runtime:.0f}s",
            transform=plt.gca().transAxes,
            fontsize=35, color='black',
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none')
        )
        plt.text(
            0.97,
            0.2,
            f"c2st: {c2st:.2f}",
            transform=plt.gca().transAxes,
            fontsize=35, color='black',
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none')
        )

        if selected_experiments[budget][task][method_name]['budget'] >= 1000:
            budget_str = f"{int((selected_experiments[budget][task][method_name]['budget'] / 1000))}k"
        elif selected_experiments[budget][task][method_name]['budget'] >= 100:
            budget_str = f"{(selected_experiments[budget][task][method_name]['budget'] / 100):.1f}k"
        else:
            budget_str = str(selected_experiments[budget][task][method_name]['budget'])

        plt.text(
            0.03,
            0.95,
            f"Budget: {budget_str}",
            transform=plt.gca().transAxes,
            fontsize=35, color='black',
            horizontalalignment='left', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none')
        )
        # save at
        base_path = "./../../paper/figures/concept_figure/"
        save_path = os.path.join(
            base_path,
            task,
        )
        os.makedirs(save_path, exist_ok=True)
        if method_name in ["npec", "bayes_flow", "flow_matching"]:
            plt.savefig(os.path.join(save_path, f"{method_name}_{budget}.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_path, f"{method_name}_{budget}.pdf"), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_path, f"{method_name}.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_path, f"{method_name}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # save at
        base_path = "./../../paper/figures/concept_figure/"
        save_path = os.path.join(
            base_path,
            task,
        )
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"gt.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, f"gt.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

# -------- Main Part --------
selected_experiments = {
    "low": {
        "simple": {
            "npec": {"budget": 1000, "run_id": 2},
            "bayes_flow": {"budget": 1000, "run_id": 0},
            "flow_matching": {"budget": 1000, "run_id": 4},
            "r2omc": {"budget": 1000, "run_id": 0}
        },
        "distractors": {
            "npec": {"budget": 1000, "run_id": 1},
            "bayes_flow": {"budget": 1000, "run_id": 0},
            "flow_matching": {"budget": 1000, "run_id": 3},
            "r2omc": {"budget": 1000, "run_id": 0}
        },
        "high_dim": {
            "npec": {"budget": 1000, "run_id": 0},
            "bayes_flow": {"budget": 1000, "run_id": 0},
            "flow_matching": {"budget": 1000, "run_id": 0},
            "r2omc": {"budget": 1000, "run_id": 0}
        }
    },
    "high": {
        "simple": {
            "npec": {"budget": 10_000, "run_id": 0},
            "bayes_flow": {"budget": 10_000, "run_id": 4},
            "flow_matching": {"budget": 30_000, "run_id": 0},
            "r2omc": {"budget": 1000, "run_id": 0}
        },
        "distractors": {
            "npec": {"budget": 30_000, "run_id": 2},
            "bayes_flow": {"budget": 10_000, "run_id": 1},
            "flow_matching": {"budget": 10_000, "run_id": 1},
            "r2omc": {"budget": 1000, "run_id": 0}
        },
        "high_dim": {
            "npec": {"budget": 50_000, "run_id": 2},
            "bayes_flow": {"budget": 50_000, "run_id": 3},
            "flow_matching": {"budget": 50_000, "run_id": 3},
            "r2omc": {"budget": 1000, "run_id": 0}
        }
    }
}


for budget in ["low", "high"]:
    for task in ["simple", "distractors", "high_dim"]:
        plot_samples(task)
        for method in ["npec", "bayes_flow", "flow_matching", "r2omc"]:
            run_id = selected_experiments[budget][task][method]['run_id']
            if run_id is not None:
                plot_samples(task, budget, method, run_id)
