# Example on how to run an experiment from a configuration file.
import lfi


config = {
    "seed": 21,
    "prior": {
        "name": "uniform",
        "params": {
            "dim": 4,
            "low": -5.,
            "high": 5.
        }
    },
    "simulator": {
        "name": "bimodal_gaussian",
        "params": {
            "sigma_noise": 1.,
        }
    },
    "observation": {
        "name": "zeros",
        "params": {
            "nof_obs": 1,
            "dim_y": 4
        }
    },
    "inference": {
        "name": "mdn",
        "train_and_sample": {
            "budget": 2_000,
            "nof_samples": 100,
            "fit_kwargs": {
                "nof_epochs": 5_000,
                "nof_components": 4

            },
            "sample_kwargs": {}
        }
    }
}

# problem definition
prior = lfi.utils.PRIOR_TO_CLASS[config['prior']['name']](**config['prior']['params'])
simulator = lfi.utils.SIMULATOR_TO_CLASS[config['simulator']['name']](**config['simulator']['params'])
observation = lfi.utils.OBSERVATION_TO_CLASS[config['observation']['name']]().sample(**config['observation']['params'])

# inference
inference = lfi.utils.INFERENCE_TO_CLASS[config['inference']['name']](
    prior=prior,
    simulator=simulator,
    observation=observation,
)

# run inference
samples, time = inference.fit_and_sample(
    budget=config['inference']['train_and_sample']['budget'],
    nof_samples=config['inference']['train_and_sample']['nof_samples'],
    fit_kwargs=config['inference']['train_and_sample']['fit_kwargs'],
    sample_kwargs=config['inference']['train_and_sample']['sample_kwargs']
)

# # plot training summary
# inference.plot_training_summary(
#     budget=config['inference']['train_and_sample']['budget']
# )

# plot posterior samples
inference.plot_posterior_samples(
    samples=samples
)
