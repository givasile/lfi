import lfi
import sbibm
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# set seed
np.random.seed(42)

figure_path = "./../../results/multiple_observations/slcp_distractors/"
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

for i in range(1, 10):
    # ------ Parameters ------
    # modeling parameters
    dim = 5
    dim_y = 2
    observation_nof_obs = 4
    exp_num = i

    # create dir
    exp_path = os.path.join(figure_path, f"exp_{exp_num}")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # inference parameters
    budget = 1500
    nof_samples = 100

    # ------ Experiment Setup ------
    # Prior
    prior = lfi.priors.UniformPrior(
        low=-3.0,
        high=3.0,
        dim=dim,
    )

    # Run simulator
    simulator = lfi.simulators.SLCPDistractors(
        dim=5,
        dim_y=2,
        dim_distractors=23,
    )

    # Observation
    obs = lfi.observations.FromSBIBM(
        task_name="slcp_distractors",
        exp_num=exp_num
    )
    p = simulator.reindex
    q = np.empty_like(p)
    q[p] = np.arange(len(p))

    observation_full = obs.sample()
    observation_full = observation_full[:, q]

    observation_inf = observation_full[:, :8].reshape(-1, 2)
    observation_dist = observation_full[:, 8:].reshape(-1, 23)
    observation = np.concatenate([observation_inf, observation_dist], axis=1)

    # check
    task = sbibm.get_task("slcp_distractors")
    gt_theta = task.get_true_parameters(exp_num).numpy()

    # ------ Inference ------
    inference = lfi.inference.r2omc.R2OMCMultiObs(
        prior=prior,
        simulator=simulator,
        observation=observation,
    )

    tic = time.time()
    inference.fit(
        budget=budget,
        fit_kwargs={
            "epochs": 20,
            "nof_gd_steps": 10,
            "alpha": 0.01,
            "pcg_to_keep": .05,
            "box_algorithm": "standard",
            "dx": 0.01,
        }
    )

    inference.sample(
        nof_samples=50,
        sample_kwargs={
            "samples_per_region": 1,
            "eps_3": 10.0,
        }
    )
    toc = time.time()
    print(f"Total time: {toc - tic:.2f} seconds")

    # Extract samples
    samples_total = inference.th_total
    samples_accepted = inference.th_accepted
    samples_selected = inference.th_selected
    # store samples as csv
    np.savetxt(os.path.join(exp_path, "samples_total.csv"), samples_total, delimiter=",")
    np.savetxt(os.path.join(exp_path, "samples_accepted.csv"), samples_accepted, delimiter=",")
    np.savetxt(os.path.join(exp_path, "samples_selected.csv"), samples_selected, delimiter=",")

    # ------ Evaluate ------
    # Generate 50 samples
    samples_gt = lfi.ground_truth.FromSBIBM(
        task_name="slcp_distractors",
        exp_num=exp_num
    ).sample(50)

    for i, samples in enumerate([samples_total, samples_accepted, samples_selected]):
        print(f"--- Sample set {i} ---")
        names = ["total", "accepted", "selected"]

        # Visualization
        g = lfi.visualization.plot_pairwise_posterior(
            samples,
            limits=[-3., 3.],
            samples_gt=samples_gt,
            savefig=os.path.join(exp_path, f"pairwise_posterior_{names[i]}.png")
        )
        plt.show(block=False)

        # Evaluation
        c2st_score = np.array([lfi.evaluation.c2st(samples, samples_gt)]) # float
        print(f"C2ST of {names[i]}: {np.mean(c2st_score):.3f} ± {np.std(c2st_score):.3f}")
        # store as csv
        np.savetxt(os.path.join(exp_path, f"c2st_{names[i]}.csv"), c2st_score, delimiter=",")

        # save time to a txt
        elapsed_time = np.array([toc - tic])
        np.savetxt(os.path.join(exp_path, f"time_{names[i]}.csv"), elapsed_time, delimiter=",")
