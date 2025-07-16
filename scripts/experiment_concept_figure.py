import lfi
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------- #
# Global config
# ----------------------------- #

np.random.seed(42)
figure_path = "./../paper/figures/sbibm/concept_figure"
os.makedirs(figure_path, exist_ok=True)

# Problem setup
low, high = -3, 3
dim = 2
dim_y = 2
nof_observations = 1
nof_samples = 1000

# ----------------------------- #
# Shared objects
# ----------------------------- #

prior = lfi.priors.UniformPrior(low=low, high=high, dim=dim)

simulator = lfi.simulators.BimodalGaussian(
    dim=dim,
    dim_y=dim_y,
    sigma_noise=0.2,
    shift=1
)

obs = lfi.observations.Zeros(
    dim_y=dim_y,
    nof_observations=nof_observations,
)
observation = obs.sample()

gt_posterior = lfi.ground_truth.GaussianMixture(
    dim=dim,
    mu=[-1., 1.],
    sigma=[0.2, 0.2],
    weights=[0.5, 0.5]
)
samples_gt = gt_posterior.sample(1000)

# ----------------------------- #
# Utility functions
# ----------------------------- #

def run_inference(method_name, inference_class, budget, fit_kwargs, sample_kwargs=None):
    """Run inference, sample, plot, and save C2ST."""
    np.random.seed(42)
    torch.manual_seed(42)

    inference = inference_class(
        prior=prior,
        simulator=simulator,
        observation=observation,
    )

    inference.fit(budget=budget, fit_kwargs=fit_kwargs)

    samples = inference.sample(
        nof_samples=nof_samples,
        sample_kwargs=sample_kwargs or {}
    )

    plot_and_save(samples, method_name)
    c2st_score = lfi.evaluation.c2st(samples, samples_gt)
    save_c2st(c2st_score, method_name)
    print(f"[{method_name}] C2ST: {c2st_score:.4f}")

def plot_and_save(samples, method_name):
    """Plot posterior samples and save figure."""
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, color='red')
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.savefig(os.path.join(figure_path, f"posterior_{method_name}.png"))
    plt.show(block=False)

def save_c2st(score, method_name):
    """Save C2ST score as CSV."""
    out_file = os.path.join(figure_path, f"c2st_{method_name}.csv")
    np.savetxt(out_file, [score], delimiter=",")

# ----------------------------- #
# Experiments
# ----------------------------- #

# ROMC
run_inference(
    method_name="r2omc_100",
    inference_class=lfi.inference.r2omc.R2OMC,
    budget=100,
    fit_kwargs={"pcg_to_keep": 1., "box_algorithm": "standard", "dx": 0.3},
    sample_kwargs={"samples_per_region": 10},
)

# NPEC Single Round - 1,000
run_inference(
    method_name="npec_1000",
    inference_class=lfi.inference.from_sbi.NPECSingleRound,
    budget=1000,
    fit_kwargs={"batch_size": 100, "training_batch_size": 100},
)

# NPEC Single Round - 10,000
run_inference(
    method_name="npec_10000",
    inference_class=lfi.inference.from_sbi.NPECSingleRound,
    budget=10000,
    fit_kwargs={"batch_size": 100, "training_batch_size": 100},
)
