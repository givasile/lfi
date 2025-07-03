import lfi
import jax
import numpy as np
import matplotlib.pyplot as plt

# modeling parameters
low = -5
high = 5
dim = 1000
sigma_noise = .1
dim_y = 1000
observation_nof_obs = 1

# inference parameters
budget = 100 # batch_size * budget_runs
nof_samples = 100

# Modeling
prior = lfi.priors.ImageDatasetPrior(
    dataset_name="mnist",
    split="train",
    nof_samples=budget,
    logpdf_method="kde",
    bandwidth=0.1,
)

simulator = lfi.simulators.ImageNoise(
    dim=prior.dim,
    dim_y=prior.dim,
    H=28,
    W=28,
    sigma_blur=1.,
    sigma_noise=0.01
)

observation_clean = prior.sample_jax(jax.random.PRNGKey(945), observation_nof_obs)
observation = np.array(simulator.sample_jax(observation_clean, 42)).reshape(1, -1)


# SBI Inference
inference = lfi.inference.r2omc.R2OMC(
    prior=prior,
    simulator=simulator,
    observation=observation,
)

inference.fit(
    budget=budget,
    fit_kwargs={
        # informative dimensions
        "find_informative_dims": True,
        "inf_dims_nof_th": 1,
        "inf_dims_nof_seeds": 5,
        "inf_dims_threshold": 1e-5,
        # training
        "alpha": .01,
        "epochs": 8,
        "eps_1": 0.5,
        "pcg_to_keep": 1.,
        "box_algorithm": "blind",
        "dx": 0.01,
        # "eps_2": None,
        # "nof_ls_steps": 20,
        # "step_size": 0.02
        }
)

posterior_samples = inference.sample(
    nof_samples=nof_samples,
    sample_kwargs={
        "samples_per_region": 3
    }
)

# show 5 random samples from posterior_samples
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    tmp = posterior_samples[i].reshape(28, 28)
    tmp[tmp < 0.15] = 0
    plt.imshow(tmp, cmap='gray')
    plt.axis('off')
    plt.title(f"Sample {i+1}")
plt.tight_layout()
plt.show(block=False)


plt.figure(figsize=(5, 5))
tmp = posterior_samples.mean(axis=0).reshape(28, 28)
tmp[tmp < 0.4] = 0
plt.imshow(tmp, cmap='gray')
plt.title("Posterior Samples Mean")
plt.axis('off')
plt.show(block=False)

plt.figure(figsize=(5, 5))
plt.imshow(posterior_samples.std(axis=0).reshape(28, 28), cmap='gray')
plt.title("Posterior Samples Std")
# set color scale from 0 to 1
plt.clim(0, 1)
plt.colorbar()
plt.axis('off')
plt.show(block=False)

plt.figure(figsize=(5, 5))
plt.imshow(observation[0].reshape(28, 28), cmap='gray')
plt.title("Observation")
plt.axis('off')
plt.show(block=False)


plt.figure(figsize=(5, 5))
plt.imshow(observation_clean.reshape(28, 28), cmap='gray')
plt.title("Observation (clean)")
plt.axis('off')
plt.show(block=False)
