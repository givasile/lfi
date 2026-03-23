import lfi
import jax
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # general settings
    savefig = True
    dir_path = "../../results/images/image_pixelwise"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # modeling parameters
    low = -5.
    high = 5.
    dim = 28 * 28  # 784
    sigma_noise = .01
    dim_y = 28 * 28  # 784
    observation_nof_obs = 1

    # inference parameters
    budget = 100 # batch_size * budget_runs
    nof_samples = 100

    key, subkey = jax.random.split(jax.random.PRNGKey(21))
    for i in range(10):
        key, subkey = jax.random.split(key)

        prior = lfi.priors.ImageDatasetPrior(
            dataset_name="mnist",
            split="train",
            nof_samples=budget,
            bandwidth=0.5,
        )

        observation_clean = prior.sample_jax(subkey, observation_nof_obs)

        prior = lfi.priors.UniformPrior(
            low=low,
            high=high,
            dim=28 * 28,
        )

        simulator = lfi.simulators.ImagePixelWiseTransform(
            dim=prior.dim,
            dim_y=prior.dim,
            H=28,
            W=28,
            sigma_noise=sigma_noise
        )

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
                "epochs": 16,
                "eps_1": 0.5,
                "pcg_to_keep": 1.,
                "box_algorithm": "blind",
                "dx": 0.1,
                # "eps_2": None,
                # "nof_ls_steps": 20,
                # "step_size": 0.02
                }
        )

        posterior_samples = inference.sample(
            nof_samples=nof_samples,
            sample_kwargs={
                "samples_per_region": 10
            }
        )

        # how to call the posterior_samples where all pixels < 0.4 are set to 0?
        posterior_samples_mask = np.copy(posterior_samples)
        posterior_samples_mask = np.where(posterior_samples_mask < 0.2, 0, posterior_samples_mask)

        # # show 5 random samples from posterior_samples
        # plt.figure(figsize=(10, 10))
        # for i in range(5):
        #     plt.subplot(1, 5, i + 1)
        #     tmp = posterior_samples[i].reshape(28, 28)
        #     tmp[tmp < 0.4] = 0
        #     plt.imshow(tmp, cmap='gray', vmin=0, vmax=1)
        #     plt.axis('off')
        #     plt.title(f"Sample {i+1}")
        # plt.tight_layout()
        # plt.show(block=False)

        # outputs
        plt.figure(figsize=(5, 5))
        plt.imshow(posterior_samples_mask.mean(axis=0).reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        plt.clim(0, 1)
        plt.axis('off')
        if savefig:
            plt.savefig(f"{dir_path}/fig_{i}_posterior_mean.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)

        plt.figure(figsize=(5, 5))
        plt.imshow(posterior_samples_mask.std(axis=0).reshape(28, 28), cmap='gray_r', vmin=0, vmax=1)
        plt.clim(0, 1)
        plt.axis('off')
        if savefig:
            plt.savefig(f"{dir_path}/fig_{i}_posterior_std.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)

        plt.figure(figsize=(5, 5))
        plt.imshow(observation[0].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        plt.clim(0, 1)
        plt.axis('off')
        if savefig:
            plt.savefig(f"{dir_path}/fig_{i}_image_observation.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)

        plt.figure(figsize=(5, 5))
        plt.imshow(observation_clean.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        plt.clim(0, 1)
        plt.axis('off')
        if savefig:
            plt.savefig(f"{dir_path}/fig_{i}_image_clean.png", bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)
