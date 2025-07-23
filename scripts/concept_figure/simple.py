import lfi
import numpy as np
import os
import exp_utils

# ----------------------------- #
# Global config
# ----------------------------- #

budget_list = [10_000, 30_000]
r2omc = False
npe = True
snpe = True
bayes_flow = True
flow_matching = True


np.random.seed(42)
dir_path = "./../../paper/figures/concept_figure/simple"
os.makedirs(dir_path, exist_ok=True)

# Problem setup
low, high = -3, 3
dim = 2
dim_y = 2
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

# ROMC
if r2omc:
    c2st, runtime, samples = exp_utils.run_inference(
        prior=prior,
        simulator=simulator,
        observation=observation,
        samples_gt=samples_gt,
        method_name="r2omc_1000",
        inference_class=lfi.inference.r2omc.R2OMC,
        budget=1000,
        nof_samples=nof_samples,
        fit_kwargs={"pcg_to_keep": 1., "box_algorithm": "standard", "dx": 0.2},
        sample_kwargs={"samples_per_region": 2},
    )
    exp_utils.save_stats(dir_path, samples, samples_gt, "r2omc_1000", None, c2st, runtime)

# NPE
if npe:
    for budget in budget_list:
        c2st, runtime, samples = exp_utils.run_inference(
            prior=prior,
            simulator=simulator,
            observation=observation,
            samples_gt=samples_gt,
            method_name=f"npec_{budget}",
            inference_class=lfi.inference.from_sbi.NPECSingleRound,
            budget=budget,
            nof_samples=nof_samples,
            fit_kwargs={"batch_size": 100, "training_batch_size": 100},
            sample_kwargs=None,
        )
        exp_utils.save_stats(dir_path, samples, samples_gt, f"npec_{budget}", None, c2st, runtime)

# S-NPE
if snpe:
    for budget in budget_list:
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
        )
        exp_utils.save_stats(dir_path, samples, samples_gt, f"snpec_{budget}", None, c2st, runtime)

# Bayes Flow
if bayes_flow:
    for budget in budget_list:
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
                "embedding_net_output_dim": 2,
                "embedding_net_num_layers": 1,
                "embedding_net_num_hiddens": 32,
            },
            sample_kwargs=None,
        )
        exp_utils.save_stats(dir_path, samples, samples_gt, f"bayes_flow_{budget}", None, c2st, runtime)

# Flow Matching
if flow_matching:
    for budget in budget_list:
        c2st, runtime, samples = exp_utils.run_inference(
            prior=prior,
            simulator=simulator,
            observation=observation,
            samples_gt=samples_gt,
            method_name=f"flow_matching_{budget}",
            inference_class=lfi.inference.from_sbi.FMPESingleRound,
            budget=budget,
            nof_samples=nof_samples,
            fit_kwargs={"batch_size": 100, "training_batch_size": 100},
            sample_kwargs=None,
        )
        exp_utils.save_stats(dir_path, samples, samples_gt, f"flow_matching_{budget}", None, c2st, runtime)
