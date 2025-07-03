import lfi
import jax
import sbibm
import numpy as np
import matplotlib.pyplot as plt

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

# SBI Inference
inference = lfi.inference.r2omc.R2OMC(
    prior=prior,
    simulator=simulator,
    observation=observation,
)

inference.fit(
    budget=budget,
    fit_kwargs={"box_algorithm": "blind", "dx": 0.01}
)

# # sample
samples = inference.sample(
    nof_samples=nof_samples,
    sample_kwargs={
        "samples_per_region": 10
    }
)

# Analysis

# Generate 100 samples
samples_gt = lfi.ground_truth.FromSBIBM(
    task_name="two_moons",
    exp_num=exp_num
).sample(nof_samples)

g = lfi.visualization.plot_pairwise_posterior(
    samples,
    limits=[low, high],
    samples_gt=samples_gt,
    # max_dims_to_plot=5
)
plt.show()

# # Evaluation
c2st = lfi.evaluation.c2st(samples, samples_gt)
print(f"C2ST: {c2st}")
#
