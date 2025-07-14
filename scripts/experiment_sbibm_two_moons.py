import lfi
import jax
import sbibm
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    # set seed
    np.random.seed(42)

    figure_path = "./../paper/figures/sbibm/two_moons"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # modeling parameters
    low = -1
    high = 1
    dim = 2
    dim_y = 2
    observation_nof_obs = 1
    exp_num = 3

    # inference parameters
    budget = 1000
    nof_samples = 1000

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

    c2st_scores = []
    for ii in range(nof_repetitions := 5):
        # SBI Inference
        inference = lfi.inference.r2omc.R2OMC(
            prior=prior,
            simulator=simulator,
            observation=observation,
        )

        inference.fit(
            budget=budget,
            fit_kwargs={"box_algorithm": "standard", "dx": 0.01}
        )

        # # sample
        samples = inference.sample(
            nof_samples=nof_samples,
            sample_kwargs={
                "samples_per_region": 10
            }
        )

        # Generate 100 samples
        samples_gt = lfi.ground_truth.FromSBIBM(
            task_name="two_moons",
            exp_num=exp_num
        ).sample(nof_samples)

        g = lfi.visualization.plot_pairwise_posterior(
            samples,
            limits=[low, high],
            samples_gt=samples_gt,
            savefig=os.path.join(figure_path, f"run_{ii}_posterior_pairwise.png"),
        )
        plt.show(block=False)

        # # Evaluation
        c2st_scores.append(lfi.evaluation.c2st(samples, samples_gt))

    print(f"C2ST: {np.mean(c2st_scores):.3f} ± {np.std(c2st_scores):.3f}")
    # store C2ST score as csv
    np.savetxt(
        os.path.join(figure_path, "c2st_scores.csv"),
        c2st_scores,
        delimiter=",",
        header="C2ST scores for R2OMC on Two Moons"
    )
