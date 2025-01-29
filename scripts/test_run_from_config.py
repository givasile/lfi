# Example on how to run an experiment from a configuration file.
import lfi
import yaml

config = yaml.safe_load(open("test_run_config.yaml", "r"))

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

# create grounf truth
ground_truth = lfi.utils.GROUND_TRUTH_TO_CLASS[config['ground_truth']['name']](**config['ground_truth']['params'])
ground_truth_samples= ground_truth.return_samples(nof_samples=config["inference"]["train_and_sample"]["nof_samples"])

# evaluation
metric = lfi.utils.EVALUATION_TO_CLASS[config['evaluation']['name']](samples, ground_truth_samples)