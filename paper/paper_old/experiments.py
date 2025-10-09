import sbibm
import torch
import pandas as pd
import sbibm.tasks
import priors_sbi
import numpy as np
import jax.numpy as jnp
import r2omc
import utils
import timeit
import matplotlib.pyplot as plt
import simulators
import priors
import ast
import os
import sbi
from itertools import product
from sklearn.mixture import GaussianMixture


class HighDimensionalExperiment:
    def __init__(self, name, D_list, nof_repetitions):
        self.name = name
        self.D_list = D_list
        self.nof_repetitions = nof_repetitions
        self.metrics = {}

    def get_gt_posterior_samples(self, N, D):
        if "linear_gaussian" in self.name:
            def posterior_gt(N, y_0, sigma=1.5):
                covariance_matrix = np.diag([sigma ** 2] * D)
                center = y_0[:D] if y_0.ndim == 1 else y_0[0, :]
                return np.random.multivariate_normal(center, covariance_matrix, size=(N,))
            return posterior_gt(N, self.y_0, 1.5)
        elif "MoG_same_sigma" in self.name or "MoG_diff_sigma" in self.name:
            def posterior_gt(nof_samples, y_0, sigma_1, sigma_2):
                y_0 = y_0[:D] if y_0.ndim == 1 else y_0[0, :]
                centers = [y_0, -y_0]
                n_components = len(centers)

                gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
                gmm.means_ = np.array(centers)
                gmm.covariances_ = np.array([np.eye(D) * sigma_1**2, np.eye(D) * sigma_2**2])
                gmm.weights_ = np.ones(n_components) / n_components
                samples, labels = gmm.sample(nof_samples)
                return samples
            if "MoG_same_sigma" in self.name:
                return posterior_gt(N, self.y_0, .1, .1)
            elif "MoG_diff_sigma" in self.name:
                return posterior_gt(N, self.y_0, .1, 1.)
            else:
                raise ValueError("Unknown MoG name")
        elif "square_gaussian" in self.name:
            def posterior_gt(N, y_0, sigma=.1):
                y_0 = y_0[:D] if y_0.ndim == 1 else y_0[0, :]
                possible_values = [np.sqrt(y_0[0]), -np.sqrt(y_0[0])]
                centers = list(product(possible_values, repeat=D))
                n_components = len(centers)
                gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
                gmm.means_ = np.array(centers)
                gmm.covariances_ = np.array([np.eye(D) * sigma**2 for _ in range(n_components)])
                gmm.weights_ = np.ones(n_components) / n_components
                samples, labels = gmm.sample(N)
                return samples[np.random.permutation(samples.shape[0])]
            return posterior_gt(N, self.y_0, .8)

    def set_up_experiment(self, D):
        # prior is always the same
        self.prior_limits = [-15, 15]
        self.prior = priors.Uniform(-15, 15, D)

        # set up the observation
        if "iid" in self.name:
            if "MoG" in self.name:
                self.y_0 = np.ones((4, D)) * 5
                self.y_0[[1,3], :] = -5
            elif "square" in self.name:
                self.y_0 = np.ones((2, D)) * 5
                self.y_0 = np.concatenate([self.y_0, np.zeros((2, 100))], axis=1)
            else:
                self.y_0 = np.ones((4, D)) * 5
        else:
            self.y_0 = np.ones(D) * 5
            if "distractors" in self.name:
                self.y_0 = np.concatenate([self.y_0, np.zeros(100)])
        self.y_0_flat = self.y_0.flatten()
        self.y_0_sbi = torch.Tensor(self.y_0_flat).unsqueeze(0)

        # set up simulator
        simulator_mapping = {
            "linear_gaussian": lambda: (
                simulators.Linear(dim=D, sigma=1.5),
                None
            ),
            "linear_gaussian_distractors": lambda: (
                simulators.LinearDistractors(dim=D, sigma=1.5, dim_distractors=100),
                None
            ),
            "linear_gaussian_iid": lambda: (
                simulators.Linear(dim=D, sigma=1.5),
                simulators.LinearMultiSamples(dim=D, sigma=1.5, nof_samples=4)
            ),
            "MoG_same_sigma": lambda: (
                simulators.CasesGaussian(dim=D, sigma_1=.1, sigma_2=.1),
                None
            ),
            "MoG_same_sigma_distractors": lambda: (
                simulators.CasesGaussianDistractors(dim=D, sigma_1=.1, sigma_2=.1, dim_distractors=100),
                None
            ),
            "MoG_same_sigma_iid": lambda: (
                simulators.CasesGaussian(dim=D, sigma_1=.1, sigma_2=.1),
                simulators.CasesGaussianMultiSamples(dim=D, sigma_1=.1, sigma_2=.1, nof_samples=4)
            ),
            "MoG_diff_sigma": lambda: (
                simulators.CasesGaussian(dim=D, sigma_1=.1, sigma_2=1.0),
                None
            ),
            "MoG_diff_sigma_distractors": lambda: (
                simulators.CasesGaussianDistractors(dim=D, sigma_1=.1, sigma_2=1.0, dim_distractors=100),
                None
            ),
            "MoG_diff_sigma_iid": lambda: (
                simulators.CasesGaussian(dim=D, sigma_1=.1, sigma_2=1.0),
                simulators.CasesGaussianMultiSamples(dim=D, sigma_1=.1, sigma_2=1.0, nof_samples=4)
            ),
            "square_gaussian": lambda: (
                simulators.Square(dim=D, sigma=.8),
                None
            ),
            "square_gaussian_distractors": lambda: (
                simulators.SquareDistractors(dim=D, sigma=.8, dim_distractors=100),
                None
            ),
            "square_gaussian_iid": lambda: (
                simulators.SquareDistractors(dim=D, sigma=.8, dim_distractors=100),
                simulators.SquareMultiSamples(dim=D, sigma=.8, nof_samples=2)
            ),
        }

        if self.name in simulator_mapping:
            self.sim, self.sim_mo = simulator_mapping[self.name]()
        else:
            raise ValueError(f"Unknown simulator name: {self.name}")

    def run_romc(self, config):
        config["find_informative_dims"] = False
        for D in self.D_list:
            self.metrics["ROMC_D_%d" % D] = []
            self.metrics["ROMC_D_%d_runtime" % D] = []
            for _ in range(self.nof_repetitions):
                self.set_up_experiment(D)
                samples_gt = self.get_gt_posterior_samples(
                    config["nof_samples_final"], D
                )

                tic = timeit.default_timer()
                sim = self.sim if self.y_0.ndim == 1 else self.sim_mo
                r2omc_method = r2omc.R2OMC(sim, self.y_0_flat, self.prior, D)
                self.romc_method = r2omc_method
                samples_romc, weight = r2omc_method.infer(config)
                toc = timeit.default_timer() - tic
                if config["apply_importance_resampling"]:
                    samples_romc = r2omc_method.importance_resampling(
                        samples_romc,
                        weight,
                        config["nof_samples_final"], False)
                    samples_romc = samples_romc[np.random.permutation(samples_romc.shape[0])]
                # Evaluation
                self.samples_romc = samples_romc
                self.metrics["ROMC_D_%d" % D].append(
                    utils.evaluate(samples_gt, samples_romc)[0].item()
                )
                self.metrics["ROMC_D_%d_runtime" % D].append(toc)

    def run_r2omc(self, config):
        config["find_informative_dims"] = True
        for D in self.D_list:
            self.metrics["R2OMC_D_%d" % D] = []
            self.metrics["R2OMC_D_%d_runtime" % D] = []
            for i in range(self.nof_repetitions):
                self.set_up_experiment(D)
                samples_gt = self.get_gt_posterior_samples(config["nof_samples_final"], D)

                tic = timeit.default_timer()
                if self.y_0.ndim == 1:
                    r2omc_method = r2omc.R2OMC(self.sim, self.y_0, self.prior, D)
                    samples_r2omc, weight = r2omc_method.infer(config)
                    if config["apply_importance_resampling"]:
                        samples_r2omc = r2omc_method.importance_resampling(
                            samples_r2omc, weight, config["nof_samples_final"], False)
                else:
                    config["nof_samples_to_select"] = config["nof_samples_final"]
                    r2omc_method = r2omc.IterativeR2OMC(self.sim, self.y_0, self.prior, D)
                    samples_r2omc = r2omc_method.infer(config)
                toc = timeit.default_timer() - tic
                self.r2omc_method = r2omc_method

                # Evaluation
                self.samples_r2omc = samples_r2omc
                self.metrics["R2OMC_D_%d" % D].append(
                    utils.evaluate(samples_gt, samples_r2omc)[0].item()
                )
                self.metrics["R2OMC_D_%d_runtime" % D].append(toc)

    def run_npe(self, config):
        for D in self.D_list:
            self.metrics["NPE_D_%d" % D] = []
            self.metrics["NPE_D_%d_runtime" % D] = []
            for _ in range(self.nof_repetitions):
                self.set_up_experiment(D)
                samples_gt = self.get_gt_posterior_samples(config["nof_samples_final"], D)

                # NPE
                tic = timeit.default_timer()
                method_name = "SNPE-C"
                training_samples = config["training_samples"]
                sbi_prior = priors_sbi.get_uniform(self.prior_limits, D)
                sim = self.sim if self.y_0.ndim == 1 else self.sim_mo
                sim_func = sim.cr_simulator()
                theta = np.array(sbi_prior.sample_n(training_samples,))
                seeds = np.random.randint(0, 10000000, training_samples)
                x = np.array([sim_func(theta[i], seeds[i]) for i in range(training_samples)])
                samples_npe = utils.posterior_sbi(method_name, sbi_prior, theta, x, self.y_0_sbi, config["nof_samples_final"])
                toc = timeit.default_timer() - tic
                print(f"Time for NPE: {toc:.2f} seconds")

                # Evaluation
                self.samples_npe = samples_npe
                self.metrics["NPE_D_%d" % D].append(utils.evaluate(samples_gt, samples_npe)[0].item())
                self.metrics["NPE_D_%d_runtime" % D].append(toc)

    def run_nle(self, config):
        for D in self.D_list:
            self.metrics["NLE_D_%d" % D] = []
            self.metrics["NLE_D_%d_runtime" % D] = []
            for _ in range(self.nof_repetitions):
                self.set_up_experiment(D)
                samples_gt = self.get_gt_posterior_samples(config["nof_samples_final"], D)

                # NPE
                tic = timeit.default_timer()
                training_samples = config["training_samples"]
                sbi_prior = priors_sbi.get_uniform(self.prior_limits, D)
                sim = self.sim if self.y_0.ndim == 1 else self.sim_mo
                sim_func = sim.cr_simulator()
                theta = np.array(sbi_prior.sample_n(training_samples,))
                seeds = np.random.randint(0, 10000000, training_samples)
                x = np.array([sim_func(theta[i], seeds[i]) for i in range(training_samples)])

                theta = torch.Tensor(theta)
                x = torch.Tensor(x)

                inferer = sbi.inference.SNLE(sbi_prior, show_progress_bars=True, density_estimator="mdn")
                inferer.append_simulations(theta, x)
                inferer.train(training_batch_size=config["tr_batch_size"])

                mcmc_parameters = dict(
                    num_chains=50,
                    thin=5,
                    warmup_steps=30,
                    init_strategy="proposal",
                )
                mcmc_method = "slice_np_vectorized"

                posterior = inferer.build_posterior(
                    mcmc_method=mcmc_method,
                    mcmc_parameters=mcmc_parameters,
                )
                samples_nle = posterior.sample(sample_shape=(config["nof_samples_final"],), x=torch.Tensor(self.y_0_sbi))

                # samples_npe = utils.posterior_sbi(method_name, sbi_prior, theta, x, self.y_0_sbi, config["nof_samples_final"])

                toc = timeit.default_timer() - tic
                print(f"Time for NPE: {toc:.2f} seconds")

                # Evaluation
                self.samples_nle = samples_nle
                self.metrics["NLE_D_%d" % D].append(utils.evaluate(samples_gt, samples_nle)[0].item())
                self.metrics["NLE_D_%d_runtime" % D].append(toc)

    def run_snpe(self, config):
        for D in self.D_list:
            self.metrics["SNPE_D_%d" % D] = []
            self.metrics["SNPE_D_%d_runtime" % D] = []
            for _ in range(self.nof_repetitions):
                self.set_up_experiment(D)
                samples_gt = self.get_gt_posterior_samples(config["nof_samples_final"], D)

                if D < 100:
                    tic = timeit.default_timer()
                    method_name = "SNPE-C"
                    training_samples = config["training_samples"]
                    sbi_prior = priors_sbi.get_uniform(self.prior_limits, D)
                    sim = self.sim if self.y_0.ndim == 1 else self.sim_mo
                    sim_func = sim.cr_simulator()
                    try:
                        samples_snpe = utils.posterior_sbi_sequential(
                            method_name,
                            sbi_prior,
                            training_samples,
                            self.y_0_sbi,
                            config["nof_samples_final"],
                            config["epochs"],
                            sim,
                            True
                        )
                        toc = timeit.default_timer() - tic
                        print(f"Time for SNPE: {toc:.2f} seconds")

                        # Evaluation
                        self.metrics["SNPE_D_%d" % D].append(utils.evaluate(samples_gt, samples_snpe)[0].item())
                        self.metrics["SNPE_D_%d_runtime" % D].append(toc)
                    except:
                        print("SNPE failed")

    def save(self):
        df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
        df.to_csv("results/high_dim_" + self.name + ".csv")

    def load(self):
        # if file exists, load it, otherwise do nothing
        if os.path.exists("results/high_dim_" + self.name + ".csv"):
            df = pd.read_csv("results/high_dim_" + self.name + ".csv")
            self.metrics = {row["Metric"]: row["Value"] for index, row in df.iterrows()}
            for key in self.metrics.keys():
                self.metrics[key] = ast.literal_eval(self.metrics[key])

    def plot(self, title, savefig=False):
        xx = self.D_list
        fig, ax = plt.subplots()
        # ax.set_title(title)
        if "NPE" in [key[:3] for key in self.metrics.keys()]:
            yy_npe = [np.mean(self.metrics["NPE_D_%d" % D]) for D in self.D_list]
            ax.plot(xx, yy_npe, "--o", label="NPE", color="dodgerblue")
        if "NLE" in [key[:3] for key in self.metrics.keys()]:
            yy_npe = [np.mean(self.metrics["NLE_D_%d" % D]) for D in self.D_list]
            ax.plot(xx, yy_npe, "--o", label="NLE", color="darkturquoise")
        if "ROMC" in [key[:4] for key in self.metrics.keys()]:
            yy_romc = [np.mean(self.metrics["ROMC_D_%d" % D]) for D in self.D_list]
            ax.plot(xx, yy_romc, "--o", label="NAIVE-ROMC", color="magenta")
        if "R2OMC" in [key[:5] for key in self.metrics.keys()]:
            yy_r2omc = [np.mean(self.metrics["R2OMC_D_%d" % D]) for D in self.D_list]
            ax.plot(xx, yy_r2omc, "--o", label="R2OMC", color="darkmagenta")

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('Dimensions (D)', fontsize=16)
        ax.set_ylabel(r'Score (C2ST)', fontsize=16)
        ax.set_yticks([.4, .6, .8, 1.])

        ax.set_ylim([.4, 1.1])
        ax.spines[['right', 'top']].set_visible(False)
        # if plot_title is "Linear Gaussian" then legend
        if title in ["Linear Gaussian", 'MoG \Sigma_1 not \Sigma_2']:
            ax.legend(fontsize=16)
        if savefig:
            plt.savefig("paper/figures/experiments/" + "high_dim_" + self.name + ".pdf", bbox_inches='tight')
            plt.savefig("paper/figures/experiments/" + "high_dim_" + self.name + ".png", bbox_inches='tight')
        plt.show(block=False)

    def plot_runtime(self, title, savefig=False):
        xx = self.D_list
        fig, ax = plt.subplots()
        # ax.set_title(title)
        if "NPE" in [key[:3] for key in self.metrics.keys()]:
            yy_npe = [np.mean(self.metrics["NPE_D_%d_runtime" % D])/60 for D in self.D_list]
            ax.plot(xx, yy_npe, "--o", label="NPE", color="dodgerblue")
        if "NLE" in [key[:3] for key in self.metrics.keys()]:
            yy_npe = [np.mean(self.metrics["NLE_D_%d_runtime" % D])/60 for D in self.D_list]
            ax.plot(xx, yy_npe, "--o", label="NLE", color="darkturquoise")
        if "ROMC" in [key[:4] for key in self.metrics.keys()]:
            yy_romc = [np.mean(self.metrics["ROMC_D_%d_runtime" % D])/60 for D in self.D_list]
            ax.plot(xx, yy_romc, "--o", label="NAIVE-ROMC", color="magenta")
        if "R2OMC" in [key[:5] for key in self.metrics.keys()]:
            yy_r2omc = [np.mean(self.metrics["R2OMC_D_%d_runtime" % D])/60 for D in self.D_list]
            ax.plot(xx, yy_r2omc, "--o", label="R2OMC", color="darkmagenta")
        # if "SNPE" in [key[:4] for key in self.metrics.keys()]:
        #     yy_snpe = [np.mean(self.metrics["SNPE_D_%d_runtime" % D])/60 for D in self.D_list]
        #     ax.plot(xx, yy_snpe, "--o", label="SNPE", color="lightskyblue")

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel(r'Dimensions (D)', fontsize=16)
        ax.set_ylabel(r'Runtime (Minutes)', fontsize=16)
        ax.set_yscale('log')
        ax.set_ylim([.05, 60.])
        ax.set_yticks([.1, 1., 10., 60.])
        ax.set_yticklabels(['$0.1$', '$1$', '$10$', '$60$'])

        ax.spines[['right', 'top']].set_visible(False)
        if savefig:
            plt.savefig("paper/figures/experiments/" + "high_dim_" + self.name + "_runtimes" + ".pdf", bbox_inches='tight')
            plt.savefig("paper/figures/experiments/" + "high_dim_" + self.name + "_runtimes" + ".png", bbox_inches='tight')
        plt.show(block=False)


class SbiExperiment:
    def __init__(
            self,
            task_name,
            exp_num,
    ):
        self.task_name = task_name
        self.exp_num = exp_num

        task = sbibm.get_task(task_name)
        samples_gt = task.get_reference_posterior_samples(exp_num)
        self.samples_gt = samples_gt[np.random.permutation(samples_gt.shape[0])]
        self.y_0_pt = task.get_observation(exp_num)
        self.th_star = task.get_true_parameters(exp_num)
        self.y_0 = jnp.array(self.y_0_pt.flatten())
        self.task = task
        self.D = task.dim_parameters

        self.metrics_romc = {"c2st": [], "mmd": [], "meddist": [], "runtime": []}
        self.metrics_romc_npe = {"c2st": [], "mmd": [], "meddist": [], "runtime": []}
        self.metrics_others = results_other_methods[task_name]

    def run_romc(
            self,
            prior,
            sim,
            pcg_of_added_seeds,
            config,
            simulation_budget = [1_000, 10_000]
    ):
        for sim_budget in simulation_budget:
            # nof samples based on the simulation budget
            nof_seeds_total = int((1 + pcg_of_added_seeds)*sim_budget)
            nof_seeds_accept = 1_000 # sim_budget if sim_budget < 10_000 else 10_000
            nof_samples = 1_000  # nof_seeds_accept

            config["nof_seeds_total"] = nof_seeds_total
            config["nof_seeds_accept"] = nof_seeds_accept
            config["nof_samples"] = nof_samples

            tic = timeit.default_timer()
            r2omc_method = r2omc.R2OMC(sim, self.y_0, prior, self.D)
            self.romc_method = r2omc_method
            samples, weight = r2omc_method.infer(config)
            toc = timeit.default_timer() - tic

            if self.samples_gt.shape[0] > samples.shape[0]:
                N = samples.shape[0]
                metrics = utils.evaluate(self.samples_gt[:N], samples)
            else:
                metrics = utils.evaluate(self.samples_gt, samples)
            self.inf = r2omc_method

            self.metrics_romc["c2st"].append(metrics[0].item())
            self.metrics_romc["mmd"].append(metrics[1].item())
            self.metrics_romc["meddist"].append(metrics[2].item())
            self.metrics_romc["runtime"].append(toc)

    def save(self, which="all"):
        if which == "all":
            pd.DataFrame(self.metrics_romc).to_csv("results/" + self.task_name + "_romc.csv")
            pd.DataFrame(self.metrics_romc_npe).to_csv("results/" + self.task_name + "_romc_npe.csv")
        elif which == "romc":
            pd.DataFrame(self.metrics_romc).to_csv("results/" + self.task_name + "_romc.csv")

    def load(self):
        metrics_romc = pd.read_csv("results/" + self.task_name + "_romc.csv")
        self.metrics_romc = {key: metrics_romc[key].to_list() for key in metrics_romc.keys()}

    def plot(self, save_path=None):
        x = [1_000, 10_000, 100_000]

        fig, ax = plt.subplots()
        y = self.metrics_romc["c2st"]
        ax.plot(x, y + [None], "-o", label="R2OMC", color="darkmagenta")
        ax.set_ylabel("C2ST")
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
        ax.set_ylim([0.4, 1.1])
        ax.spines[['right', 'top']].set_visible(False)
        if save_path is not None:
            plt.savefig(save_path + self.task_name + "_romc.pdf", bbox_inches='tight')
        plt.show(block=False)

        fig, ax = plt.subplots()
        ii = 0
        # create 8 colors
        colors = ["green", "darkturquoise", "dodgerblue", "orange", "darkgreen", "cadetblue", "blue", "darkorange"]
        for key, value in self.metrics_others.items():
            ax.plot(x, value["c2st"], 'o-', label=key, color=colors[ii])
            ax.set_xscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
            ii += 1

        # add R2OMC
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_yticks([.4, .6, .8, 1.])
        y = self.metrics_romc["c2st"]
        ax.plot(x, y + [None], "--x", label="R2OMC", color="darkmagenta")
        ax.set_ylabel(r'Score (C2ST)', fontsize=14)
        ax.set_xlabel(r'Simulation Budget (Log)', fontsize=13)
        ax.set_ylim([0.4, 1.1])
        ax.spines[['right', 'top']].set_visible(False)
        if self.task_name == "gaussian_mixture":
            ax.legend(fontsize=14, loc='upper right', ncol=3)
        if save_path is not None:
            plt.savefig(save_path + self.task_name + ".pdf", bbox_inches='tight')
        plt.show(block=False)

        # num_figures = 1 + len(self.metrics_others)
        # # create a row with num figures
        # x = [1_000, 10_000, 100_000]

        # fig, axs = plt.subplots(1, num_figures, sharey=True, figsize=(12., 2.))
        # y = self.metrics_romc["c2st"]
        # axs[0].plot(x, y + [None], "-o", label="ROMC", color="red")
        # axs[0].set_ylabel("C2ST")
        # axs[0].set_xscale('log')
        # axs[0].set_xticks(x)
        # axs[0].set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
        # axs[0].set_title("ROMC")

        # ii = 1
        # for key, value in self.metrics_others.items():
        #     axs[ii].plot(x, value["c2st"], 'o-', label=key)
        #     axs[ii].set_xscale('log')
        #     axs[ii].set_xticks(x)
        #     axs[ii].set_xticklabels(['$10^3$', '$10^4$', '$10^5$'])
        #     axs[ii].set_title(key)
        #     ii += 1

        # if save_path is not None:
        #     plt.savefig(save_path + self.task_name + ".pdf", bbox_inches='tight')
        # plt.show()


results_other_methods = {
    "two_moons":
    {
        "REJ-ABC":
            {
                "c2st": [0.95, 0.85, 0.66],
            },
        "NLE":
            {
                "c2st": [0.77, 0.71, 0.66],
            },
        "NPE":
            {
                "c2st": [0.72, 0.60, 0.54],
            },
        "NRE":
            {
                "c2st": [0.82, 0.76, 0.62],
            },
        "SMC-ABC":
            {
                "c2st": [0.92, 0.7, 0.66],
            },
        "SNLE":
            {
                "c2st": [0.65, 0.57, 0.58],
            },
        "SNPE":
            {
                "c2st": [0.64, 0.55, 0.52],
            },
        "SNRE":
            {
                "c2st": [0.65, 0.58, 0.56],
            }
    },
    "gaussian_linear_uniform":
    {
        "REJ-ABC":
            {
                "c2st": [0.97, 0.94, 0.91],
            },
        "NLE":
            {
                "c2st": [0.72, 0.54, 0.51],
            },
        "NPE":
            {
                "c2st": [0.69, 0.55, 0.50],
            },
        "NRE":
            {
                "c2st": [0.78, 0.70, 0.63],
            },
        "SMC-ABC":
            {
                "c2st": [0.97, 0.92, 0.79],
            },
        "SNLE":
            {
                "c2st": [0.63, 0.52, 0.51],
            },
        "SNPE":
            {
                "c2st": [0.63, 0.52, 0.51],
            },
        "SNRE":
            {
                "c2st": [0.68, 0.60, 0.54],
            }
    },
    "gaussian_mixture":
    {
        "REJ-ABC":
            {
                "c2st": [0.88, 0.79, 0.77],
            },
        "NLE":
            {
                "c2st": [0.81, 0.73, 0.75],
            },
        "NPE":
            {
                "c2st": [0.73, 0.66, 0.55],
            },
        "NRE":
            {
                "c2st": [0.78, 0.75, 0.73],
            },
        "SMC-ABC":
            {
                "c2st": [0.79, 0.74, 0.66],
            },
        "SNLE":
            {
                "c2st": [0.70, 0.70, 0.62],
            },
        "SNPE":
            {
                "c2st": [0.69, 0.58, 0.53],
            },
        "SNRE":
            {
                "c2st": [0.72, 0.66, 0.54],
            }
    },
    "slcp":
    {
        "REJ-ABC":
            {
                "c2st": [0.95, 0.85, 0.66],
            },
        "NLE":
            {
                "c2st": [0.77, 0.71, 0.66],
            },
        "NPE":
            {
                "c2st": [0.72, 0.60, 0.54],
            },
        "NRE":
            {
                "c2st": [0.82, 0.76, 0.62],
            },
        "SMC-ABC":
            {
                "c2st": [0.92, 0.7, 0.66],
            },
        "SNLE":
            {
                "c2st": [0.65, 0.57, 0.58],
            },
        "SNPE":
            {
                "c2st": [0.64, 0.55, 0.52],
            },
        "SNRE":
            {
                "c2st": [0.65, 0.58, 0.56],
            }
    }
}
