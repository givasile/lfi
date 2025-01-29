import numpy as np
import yaml
import random
import torch
import os
import json
import hashlib
import logging
import mlflow
import lfi.utils
import lfi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleRun:
    def __init__(self, config, use_mlflow=True, experiment_name=None):
        self.use_mlflow = use_mlflow
        logger.info(f"Initializing experiment with config:")
        logger.info(config)

        self.config = config

        self.prior = lfi.utils.PRIOR_TO_CLASS[self.config['prior']['name']](**self.config['prior']['params'])
        self.simulator = lfi.utils.SIMULATOR_TO_CLASS[self.config['simulator']['name']](**self.config['simulator']['params'])
        self.observation = lfi.utils.OBSERVATION_TO_CLASS[self.config['observation']['name']]().sample(**self.config['observation']['params'])

        self.inference = lfi.utils.INFERENCE_TO_CLASS[self.config['inference']['name']](
            prior=self.prior,
            simulator=self.simulator,
            observation=self.observation,
        )
        self.path = self._create_experiment_path()
        # store the config as yml
        with open(os.path.join(self.path, "config.yml"), "w") as f:
            yaml.dump(self.config, f)

        self.samples = None
        self.time = None

        # evaluation
        self.c2st = None
        self.metrics = {}

        if self.use_mlflow:
            if experiment_name is not None:
                mlflow.set_experiment(experiment_name)
            mlflow.start_run()
        logger.info(f"Experiment initialized")

    def _set_seed(self):
        seed = self.config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    def infer(self):
        logger.info(f"Running inference")
        self._set_seed()
        self.samples, self.time = self.inference.fit_and_sample(
            budget=self.config['inference']['train_and_sample']['budget'],
            nof_samples=self.config['inference']['train_and_sample']['nof_samples'],
        )
        logger.info(f"Inference finished")

        if self.use_mlflow:
            mlflow.log_params(lfi.utils.flatten_config(self.config))

            mlflow.log_metric("time", self.time)

            # store the samples in a file
            samples_path = os.path.join(self.path, "samples.csv")
            self.inference.store(self.samples, samples_path)
            mlflow.log_artifact(samples_path)

    def plot(self, limits=None):
        # if plot_training_summary is not implemented, skip
        if hasattr(self.inference, "plot_training_summary"):
            self.inference.plot_training_summary(
                budget=self.config['inference']['train_and_sample']['budget'],
                savefig=os.path.join(self.path, "training_summary.png")
            )
            mlflow.log_artifact(os.path.join(self.path, "training_summary.png"))

        dim = self.config['prior']['params']['dim']
        self.inference.plot_posterior_samples(
            samples=self.samples,
            subset_dims=[i for i in range(dim)] if dim < 10 else [i for i in range(10)],
            limits=limits,
            savefig=os.path.join(self.path, "posterior_samples.png")
        )
        mlflow.log_artifact(os.path.join(self.path, "posterior_samples.png"))

    def evaluate(self, gt_samples):
        self.c2st = lfi.evaluation.c2st(self.samples, gt_samples)
        self.metrics["c2st"] = self.c2st
        # save dict to file
        with open(os.path.join(self.path, "metrics.json"), "w") as f:
            json.dump(self.metrics, f)
        logger.info(f"C2ST: {self.c2st}")
        mlflow.log_metric("c2st", self.c2st)
        return self.c2st

    def run(self, gt_samples, limits):
        self.infer()
        self.plot(limits)
        self.evaluate(gt_samples)
        self.end_experiment()

    def end_experiment(self):
        if self.use_mlflow:
            mlflow.end_run()

    @staticmethod
    def _generate_run_id(base_path):
        existing_runs = [
            d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
        ]
        return f"run_{len(existing_runs) + 1:02d}"

    @staticmethod
    def _generate_unique_id(config):
        # Create a JSON string with sorted keys to ensure consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _create_experiment_path(self):
        unique_id = self._generate_unique_id(self.config)
        base_path = os.path.join(
            "./../experiment_runs",
            self.config['simulator']['name'],
            self.config['prior']['name'],
            self.config['inference']['name'],
            unique_id
        )
        os.makedirs(base_path, exist_ok=True)

        # Generate a run ID for this specific run
        run_id = self._generate_run_id(base_path)
        run_path = os.path.join(base_path, run_id)
        os.makedirs(run_path, exist_ok=True)

        return run_path


