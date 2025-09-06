import lfi
import numpy as np
import torch
import os
import exp_utils

# ----------------------------- #
# Global config
# ----------------------------- #

budget_list = [20_000]
seed_list = [42, 48930, 1234, 123456, 98765]

r2omc = False
npe = True
snpe = False
bayes_flow = False
flow_matching = False

np.random.seed(42)
torch.manual_seed(42)

dir_path = "./../../results/concept_figure/high_dim"
os.makedirs(dir_path, exist_ok=True)

# Problem setup
low, high = -3, 3
dim = 20
dim_y = 20
nof_observations = 1
nof_samples = 1000

# ----------------------------- #
# Problem setup
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
# Experiments
# ----------------------------- #
# plot and save ground truth samples
exp_utils.plot_samples(dir_path, None, samples_gt, "gt", title=None, runtime=None)
out_file = os.path.join(dir_path, "gt_samples.csv")
np.savetxt(out_file, samples_gt, delimiter=",")

# ROMC
if r2omc:
    for budget in [1_000]:
        for seed in seed_list:
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
            exp_utils.save_stats(dir_path, samples, samples_gt, f"r2omc_{budget}", None, c2st, runtime, seed=seed)

# NPE
if npe:
    for budget in budget_list:
        for seed in seed_list:
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
            exp_utils.save_stats(dir_path, samples, samples_gt, f"npec_{budget}", None, c2st, runtime, seed=seed)

# S-NPE
if snpe:
    for budget in budget_list:
        for seed in seed_list:
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
            exp_utils.save_stats(dir_path, samples, samples_gt, f"snpec_{budget}", None, c2st, runtime, seed=seed)

# Bayes Flow
if bayes_flow:
    for budget in budget_list:
        for seed in seed_list:
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
                    "batch_size": 100,
                    "training_batch_size": 100,
                    "embedding_net_output_dim": 30,
                    "embedding_net_num_layers": 2,
                    "embedding_net_num_hiddens": 32,
                },
                sample_kwargs=None,
                seed=seed
            )
            exp_utils.save_stats(dir_path, samples, samples_gt, f"bayes_flow_{budget}", None, c2st, runtime, seed=seed)

# Flow Matching
if flow_matching:
    for budget in budget_list:
        for seed in seed_list:
            c2st, runtime, samples = exp_utils.run_inference(
                prior=prior,
                simulator=simulator,
                observation=observation,
                samples_gt=samples_gt,
                method_name=f"flow_matching_{budget}",
                inference_class=lfi.inference.from_sbi.FMPESingleRound,
                budget=budget,
                nof_samples=nof_samples,
                fit_kwargs={"vf_estimator": "transformer", "batch_size": 100, "training_batch_size": 100},
                sample_kwargs=None,
                seed=seed
            )
            exp_utils.save_stats(dir_path, samples, samples_gt, f"flow_matching_{budget}", None, c2st, runtime, seed=seed)
