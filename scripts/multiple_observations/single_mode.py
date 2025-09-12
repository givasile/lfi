import lfi
import numpy as np
import torch
import os
import exp_utils
import time
import matplotlib.pyplot as plt


# ----------------------------- #
# Global config
# ----------------------------- #

nof_observations_list = [5] # [2, 5, 10]  # Number of observations to consider
seed_list = [42] # , 48930, 1234] # , 123456, 98765]
budget = 1_000
dim = 10
dir_path = "./../../results/multiple_observations/single_mode"
os.makedirs(dir_path, exist_ok=True)

# ----------------------------- #
# Experiments
# ----------------------------- #
np.random.seed(42)
torch.manual_seed(42)

for nof_observations in nof_observations_list:
    dir_path_current = os.path.join(dir_path, f"obs_{nof_observations}")
    os.makedirs(dir_path_current, exist_ok=True)

    dim_y = dim
    low, high = -15, 15
    nof_samples = 1000

    # Problem setup
    prior = lfi.priors.UniformPrior(low=low, high=high, dim=dim)
    simulator = lfi.simulators.GaussianNoise(
        dim=dim,
        dim_y=dim_y,
        sigma_noise=1,
        shift=5
    )
    obs = lfi.observations.Zeros(
        dim_y=dim_y,
        nof_observations=nof_observations,
    )
    observation = obs.sample()

    gt_posterior = lfi.ground_truth.Gaussian(
        dim=dim,
        mu=-5,
        sigma=1 / np.sqrt(nof_observations)
    )
    samples_gt = gt_posterior.sample(1000)

    out_file = os.path.join(dir_path_current, f"gt_samples.csv")
    np.savetxt(out_file, samples_gt, delimiter=",")

    # Run experiments
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)

        r2omc = lfi.inference.r2omc.R2OMCMultiObs(
            prior=prior,
            simulator=simulator,
            observation=observation,
        )

        tic = time.time()
        r2omc.fit(
            budget=budget,
            fit_kwargs={"pcg_to_keep": 1., "box_algorithm": "standard", "dx": 0.2, "epochs": 10, "alpha": 0.1},
        )

        samples = r2omc.sample(
            nof_samples=nof_samples,
            sample_kwargs={"samples_per_region": 3,
                           "nof_samples_per_obs": 3000,
                           "eps_3": 0.5, "eps_4": 0.3},
        )
        runtime = time.time() - tic

        # Extract samples
        samples_total = r2omc.th_total
        samples_accepted = r2omc.th_accepted
        samples_selected = r2omc.th_selected

        # Visualization
        g = lfi.visualization.plot_pairwise_posterior(
            samples_selected,
            limits=[-3., 3.],
            samples_gt=samples_gt,
            savefig=None
        )
        plt.show(block=False)

        c2st = lfi.evaluation.c2st(samples, samples_gt)
        print(f"[r2omc] C2ST: {c2st:.4f}, Runtime: {runtime:.2f} seconds")
        exp_utils.save_stats(dir_path_current, samples, samples_gt, f"r2omc_{budget}", None, c2st, runtime, seed=seed)
