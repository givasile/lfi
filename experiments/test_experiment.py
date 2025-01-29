import lfi
import experiment

# with open('config.yml', 'r') as file:
#     config = yaml.safe_load(file)
#
# D = config["prior"]["params"]["dim"]
#
#
# exp = lfi.experiment.SingleRun(config=config)
# exp.run()
#
# exp.inference.plot_training_summary(
#     budget=1000
# )
#
# posterior_modes = np.zeros((2, D))
# posterior_modes[0, :] = -3.
# posterior_modes[1, :] = 3.
#
#
#
# exp.inference.plot_posterior_samples(
#     samples=exp.samples,
#     budget=1000,
#     posterior_modes=posterior_modes,
#     subset_dims=[0, 1]
# )

# create config here
config = {
    "seed": 21,
    "prior": {
        "name": "uniform",
        "params": {
            "dim": 2,
            "low": -10.,
            "high": 10.
        }
    },
    "simulator": {
        "name": "bimodal_gaussian",
        "params": {
            "sigma_noise": 0.1,
        }
    },
    "observation": {
        "name": "zeros",
        "params": {
            "nof_obs": 1,
            "dim_y": 2
        }
    },
    "inference": {
        "name": "npe_a_single_round",
        "train_and_sample": {
            "budget": 500,
            "nof_samples": 500
        }
    }
}

exp = experiment.SingleRun(
    config=config,
    use_mlflow=True,
    experiment_name="test_experiment"
)

exp.infer()

exp.plot(limits=[-10., 10.])


gt = lfi.ground_truth.GaussianMixture(
    dim=2,
    mu=[3, -3],
    sigma=[.1, .1],
    weights=[.5, .5]
)
samples=gt.return_samples(500)

c2st = exp.evaluate(gt_samples=samples)

exp.end_experiment()

