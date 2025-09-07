import lfi
import numpy as np
import torch
import os
import exp_utils

# ----------------------------- #
# Global config
# ----------------------------- #

D_list = [2, 5, 10, 15, 20]
budget_list = [1_000, 5_000, 10_000] # [10_000, 30_000]
seed_list = [42, 48930, 1234] # , 123456, 98765]

r2omc = True
npe = True
snpe = False
bayes_flow = True
flow_matching = True

dir_path = "./../../results/mog_benchmark/two_modes_distractors"
os.makedirs(dir_path, exist_ok=True)

# ----------------------------- #
# Experiments
# ----------------------------- #
for dim in D_list:
    np.random.seed(42)
    torch.manual_seed(42)

    dir_path_current = os.path.join(dir_path, f"D_{dim}")
    os.makedirs(dir_path_current, exist_ok=True)

    dim_distractors = 18
    dim_y = dim + dim_distractors
    low, high = -3, 3
    nof_observations = 1
    nof_samples = 1000

    # Problem setup
    prior = lfi.priors.UniformPrior(low=low, high=high, dim=dim)
    simulator = lfi.simulators.BimodalGaussianDistractors(
        dim=dim,
        dim_y=dim_y,
        dim_distractors=dim_distractors,
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

    out_file = os.path.join(dir_path_current, f"gt_samples.csv")
    np.savetxt(out_file, samples_gt, delimiter=",")

    # Run experiments
    for budget in budget_list:
        for seed in seed_list:
            print(f"Running experiments for D={dim}, budget={budget}, seed={seed}...")

            if r2omc:
                if budget == 1_000:
                    print("Running R2OMC...")
                    c2st, runtime, samples = exp_utils.run_inference(
                        prior=prior,
                        simulator=simulator,
                        observation=observation,
                        samples_gt=samples_gt,
                        method_name=f"r2omc_{budget}",
                        inference_class=lfi.inference.r2omc.R2OMC,
                        budget=budget,
                        nof_samples=nof_samples,
                        fit_kwargs={"pcg_to_keep": 1., "box_algorithm": "standard", "dx": 0.2, "epochs": 10, "alpha": 0.1},
                        sample_kwargs={"samples_per_region": 2},
                        seed=seed,
                    )
                    exp_utils.save_stats(dir_path_current, samples, samples_gt, f"r2omc_{budget}", None, c2st, runtime, seed=seed)

            if npe:
                print("Running NPE...")
                c2st, runtime, samples = exp_utils.run_inference(
                    prior=prior,
                    simulator=simulator,
                    observation=observation,
                    samples_gt=samples_gt,
                    method_name=f"npec_{budget}",
                    inference_class=lfi.inference.from_sbi.NPECSingleRound,
                    budget=budget,
                    nof_samples=nof_samples,
                    fit_kwargs={
                        "num_transforms": 16,
                        "num_bins": 16,
                        "hidden_features": 200,
                        "training_batch_size": 500
                    },
                    sample_kwargs=None,
                    seed=seed,
                )
                exp_utils.save_stats(dir_path_current, samples, samples_gt, f"npec_{budget}", None, c2st, runtime, seed=seed)

            # S-NPE
            if snpe:
                c2st, runtime, samples = exp_utils.run_inference(
                    prior=prior,
                    simulator=simulator,
                    observation=observation,
                    samples_gt=samples_gt,
                    method_name=f"snpec_{budget}",
                    inference_class=lfi.inference.from_sbi.NPECMultiRound,
                    budget=budget,
                    nof_samples=nof_samples,
                    fit_kwargs={"batch_size": 100, "training_batch_size": 100, "num_rounds": 3},
                    sample_kwargs=None,
                    seed=seed
                    )
                exp_utils.save_stats(dir_path_current, samples, samples_gt, f"snpec_{budget}", None, c2st, runtime, seed=seed)

            # Bayes Flow
            if bayes_flow:
                c2st, runtime, samples = exp_utils.run_inference(
                    prior=prior,
                    simulator=simulator,
                    observation=observation,
                    samples_gt=samples_gt,
                    method_name=f"bayes_flow_{budget}",
                    inference_class=lfi.inference.from_sbi.BayesFlow,
                    budget=budget,
                    nof_samples=nof_samples,
                    fit_kwargs={
                        "batch_size": 500,
                        "training_batch_size": 500,
                        "embedding_net_output_dim": dim,
                        "embedding_net_num_layers": 2,
                        "embedding_net_num_hiddens": 32,
                    },
                    sample_kwargs=None,
                    seed=seed
                )
                exp_utils.save_stats(dir_path_current, samples, samples_gt, f"bayes_flow_{budget}", None, c2st, runtime,
                                     seed=seed)

            # Flow Matching
            if flow_matching:
                c2st, runtime, samples = exp_utils.run_inference(
                    prior=prior,
                    simulator=simulator,
                    observation=observation,
                    samples_gt=samples_gt,
                    method_name=f"flow_matching_{budget}",
                    inference_class=lfi.inference.from_sbi.FMPESingleRound,
                    budget=budget,
                    nof_samples=nof_samples,
                    fit_kwargs={"vf_estimator": "mlp", "batch_size": 100, "training_batch_size": 100},
                    sample_kwargs=None,
                    seed=seed
                )
                exp_utils.save_stats(dir_path_current, samples, samples_gt, f"flow_matching_{budget}", None, c2st,
                                     runtime, seed=seed)
