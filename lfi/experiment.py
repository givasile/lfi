import numpy as np
import pandas as pd
import time
import random
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
import os
import json
import hashlib
import logging
import lfi.utils
import lfi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleRun:
    def __init__(
            self,
            config,
            analyze=True,
            evaluate=True,
            store=True,
            experiment_name=None
    ):
        logger.info(f"Initializing experiment with config:")
        logger.info(config)

        self.config = config
        self.analyze = analyze
        self.evaluate = evaluate
        self.store = store
        self.experiment_name = experiment_name

        # init prior
        init_params = self.config['prior']['params']
        init_params['dim'] = self.config['experiment_run']['dim']
        self.prior = lfi.utils.PRIOR_TO_CLASS[self.config['prior']['name']](**init_params)

        # init simulator
        init_params = self.config['simulator']['params']
        init_params['dim'] = self.config['experiment_run']['dim']
        init_params['dim_y'] = self.config['experiment_run']['dim_y']
        self.simulator = lfi.utils.SIMULATOR_TO_CLASS[self.config['simulator']['name']](**init_params)

        # init observation
        init_params = self.config['observation']['params']
        init_params['dim_y'] = self.config['experiment_run']['dim_y']
        self.observation = lfi.utils.OBSERVATION_TO_CLASS[self.config['observation']['name']](**init_params).sample()

        # init inference
        self.inference_method = lfi.utils.INFERENCE_TO_CLASS[self.config['inference']['name']](
            prior=self.prior,
            simulator=self.simulator,
            observation=self.observation,
        )

        if evaluate:
            init_params = self.config['evaluation']['ground_truth']['params']
            init_params['dim'] = self.config['experiment_run']['dim']
            self.ground_truth = lfi.utils.GROUND_TRUTH_TO_CLASS[
                self.config['evaluation']['ground_truth']['name']
                ](**init_params)
            self.gt_samples = self.ground_truth.sample(
                nof_samples = self.config["inference"]["params"]["nof_samples"]
            )


        self.store_dir = None

        # inference staff
        self.samples = None
        self.time = None

        # evaluation staff
        self.c2st = None
        self.metrics = {}

    def start_experiment(self):
        if self.store:
            self.store_dir = self._create_experiment_path()

        # set seed
        seed = self.config['experiment_run']["seed"]
        np.random.seed(seed)
        if _HAS_TORCH:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        random.seed(seed)

        logger.info(f"Experiment initialized")

    def inference(self):
        logger.info(f"Running inference")
        tic = time.time()
        self.samples = self.inference_method.fit_and_sample(
            budget=self.config['inference']['params']['budget'],
            nof_samples=self.config['inference']['params']['nof_samples'],
            fit_kwargs=self.config['inference']['params']['fit_kwargs'],
            sample_kwargs=self.config['inference']['params']['sample_kwargs']
        )
        toc = time.time()
        self.time = toc - tic
        logger.info(f"Inference finished")

        if self.store:
            # store time
            with open(os.path.join(self.path, "time.json"), "w") as f:
                json.dump({"time": self.time}, f)

            # store samples
            samples_path = os.path.join(self.path, "inferred_samples.csv")
            self.inference_method.store(self.samples, samples_path)

    def analysis(self):
        # if plot_training_summary is not implemented, skip
        if hasattr(self.inference, "plot_training_summary"):
            savefig = os.path.join(self.path, "training_summary.png") if self.store else None
            self.inference_method.plot_training_summary(
                budget=self.config['inference']['train_and_sample']['budget'],
                savefig=savefig
            )

        dim = self.config['prior']['params']['dim']
        savefig = os.path.join(self.path, "posterior_samples.png") if self.store else None
        self.inference_method.plot_posterior_samples(
            samples=self.samples,
            samples_gt=None,
            subset_dims=[i for i in range(dim)] if dim < 10 else [i for i in range(10)],
            limits=None,
            savefig=savefig,
        )

    def evaluation(self):
        self.c2st = lfi.evaluation.c2st(self.samples, self.gt_samples)

        # save dict to file
        if self.store:
            with open(os.path.join(self.path, "metrics.json"), "w") as f:
                json.dump(self.c2st, f)
        logger.info(f"C2ST: {self.c2st}")

        # plot
        dim = self.config['prior']['params']['dim']
        savefig = os.path.join(self.path, "posterior_samples.png") if self.store else None
        self.inference_method.plot_posterior_samples(
            samples=self.samples,
            samples_gt=self.gt_samples,
            subset_dims=[i for i in range(dim)] if dim < 10 else [i for i in range(10)],
            limits=None,
            savefig=savefig,
        )

    def end_experiment(self):
        pass

    def run(self):
        # start experiment
        self.start_experiment()

        # infer
        self.inference()

        # analyze
        if self.analyze:
            self.analysis()

        # evaluate
        if self.evaluate:
            self.evaluation()

        self.end_experiment()

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
        base_path = os.path.join("../experiment_runs", )
        if self.experiment_name is not None:
            base_path = os.path.join(base_path, self.experiment_name)
        unique_id = self._generate_unique_id(self.config)

        base_path = os.path.join(
            base_path,
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
        self.path = run_path
        return run_path

