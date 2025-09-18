import lfi
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# set seed
np.random.seed(42)

# figure_path = "./../../paper/figures/sbibm/two_moons"
results_path = "./../../results/sbibm/two_moons/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# modeling parameters
low = -1
high = 1
dim = 2
dim_y = 2
observation_nof_obs = 1

# inference parameters
budget = 1000
nof_samples = 1000

for budget in [5_000, 10_000]: # [1000, 5000, 10000]:
    for exp_num in range(1, 9):
        path = os.path.join(results_path, f"budget_{budget}", f"exp_{exp_num}")
        if not os.path.exists(path):
            os.makedirs(path)

        # Modeling
        prior = lfi.priors.UniformPrior(
            low=low,
            high=high,
            dim=dim
        )

        simulator = lfi.simulators.TwoMoons(
            dim=dim,
            dim_y=dim_y
        )

        obs = lfi.observations.FromSBIBM(
            task_name="two_moons",
            exp_num=exp_num
        )
        observation = obs.sample()

        # SBI Inference
        inference = lfi.inference.r2omc.R2OMC(
            prior=prior,
            simulator=simulator,
            observation=observation,
        )
        tic = time.time()
        if exp_num == 5:
            fit_kwargs = {"pcg_to_keep": 0.3, "box_algorithm": "standard", "dx": 0.01}
            sample_kwargs = {"samples_per_region": 10}
        else:
            fit_kwargs = {"pcg_to_keep": 0.8, "box_algorithm": "standard", "dx": 0.01}
            sample_kwargs = {"samples_per_region": 5}


        inference.fit(
            budget=budget,
            fit_kwargs=fit_kwargs
        )

        # # sample
        samples = inference.sample(
            nof_samples=nof_samples,
            sample_kwargs=sample_kwargs
        )
        toc = time.time()

        # Generate 100 samples
        samples_gt = lfi.ground_truth.FromSBIBM(
            task_name="two_moons",
            exp_num=exp_num
        ).sample(nof_samples)

        g = lfi.visualization.plot_pairwise_posterior(
            samples,
            limits=[low, high],
            samples_gt=samples_gt,
            savefig=os.path.join(path, "pairwise_posterior.png")
        )
        plt.show(block=False)

        # # Evaluation
        c2st_score = lfi.evaluation.c2st(samples, samples_gt)
        np.savetxt(os.path.join(path, "c2st_score.csv"), np.array([c2st_score]), delimiter=",")

        runtime = toc - tic
        np.savetxt(os.path.join(path, "runtime.csv"), np.array([runtime]), delimiter=",")


    # print(f"C2ST: {np.mean(c2st_scores):.3f} ± {np.std(c2st_scores):.3f}")
    # store C2ST score as csv
    # np.savetxt(
    #     os.path.join(figure_path, "c2st_scores.csv"),
    #     c2st_scores,
    #     delimiter=",",
    #     header="C2ST scores for R2OMC on Two Moons"
    # )
