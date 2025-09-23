import lfi
import sbibm
import numpy as np
import matplotlib.pyplot as plt
import os

# set seed
np.random.seed(42)

figure_path = "../../results/sbibm/lotka_volterra"
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

# modeling parameters
dim = 4
prior_mean = np.array([-0.125, -3.0, -0.125, -3.0])
prior_std = np.array([0.5, 0.5, 0.5, 0.5])
dim_y = 20
observation_nof_obs = 1


for exp_num in [4]: # range(1, 9):
    # inference parameters
    budget = 1000
    nof_samples = 100

    path = f"../../results/sbibm/lotka_volterra/budget_{budget}/exp_{exp_num}"
    if not os.path.exists(path):
        os.makedirs(path)


    # Prior
    prior = lfi.priors.LogNormal(
        dim=4,
        mean=prior_mean,
        std=prior_std
    )

    # Run simulator
    simulator = lfi.simulators.LotkaVolterra(
        dim=4,
        dim_y=20
    )

    # Observation
    obs = lfi.observations.FromSBIBM(
        task_name="lotka_volterra",
        exp_num=exp_num
    )
    observation = obs.sample()

    inference = lfi.inference.r2omc.R2OMC(
        prior=prior,
        simulator=simulator,
        observation=observation,
    )


    # check
    task = sbibm.get_task("lotka_volterra")
    gt_theta = task.get_true_parameters(exp_num).numpy()

    inference.fit(
        budget=budget,
        fit_kwargs={
            "find_informative_dims": False,
            "inf_dims_nof_th": 30,
            "inf_dims_nof_seeds": 10,
            "inf_dims_threshold": 1e-1,
            "epochs": 15,
            "nof_gd_steps": 20,
            "alpha": 0.001,
            "pcg_to_keep": .1,
            "box_algorithm": "blind",
            "dx": 0.005,
        }
    )

    samples = inference.sample(
        nof_samples=100,
        sample_kwargs={
            "samples_per_region": 100,
            "eps_3": 2.0,
        }
    )
    np.savetxt(f"{path}/samples.csv", samples, delimiter=",")
    # samples = inference.th_star[:100, 0]

    # Generate 100 samples
    samples_gt = lfi.ground_truth.FromSBIBM(
        task_name="lotka_volterra",
        exp_num=exp_num
    ).sample(100)

    g = lfi.visualization.plot_pairwise_posterior(
        samples,
        limits=[-0., 1.5],
        samples_gt=samples_gt,
        savefig=f"{path}/pairwise_posterior.png",
    )
    plt.show(block=True)

    # Evaluation
    c2st_score = lfi.evaluation.c2st(samples, samples_gt)
    np.savetxt(f"{path}/c2st_score.csv", np.array([c2st_score]), delimiter=",")
    print(f"C2ST: {c2st_score}")
