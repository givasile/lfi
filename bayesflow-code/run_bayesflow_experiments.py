"""
Run modern BayesFlow (flow matching, no summary network) on the same
two_modes_distractors settings as the paper:
  - theta dimension D in [2, 5, 10, 15, 20]
  - distractors fixed at 18
  - budgets: [5_000, 10_000, 20_000, 50_000, 100_000]
  - 3 seeds per (D, budget) cell

Results saved to ../code/results/mog_benchmark/two_modes_distractors/D_<D>/
in the same directory structure as existing baselines, so results.py can
pick them up automatically.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score

import bayesflow as bf

# ------------------------------------------------------------------ #
# Simulator (matches lfi.simulators.BimodalGaussianDistractors)
# ------------------------------------------------------------------ #
LOW, HIGH = -3.0, 3.0
SHIFT = 1.0
SIGMA_NOISE = 0.2
DIM_DISTRACTORS = 18


def make_prior(dim_theta):
    def prior(rng=None):
        rng = np.random.default_rng(rng)
        return {"parameters": rng.uniform(LOW, HIGH, size=dim_theta)}
    return prior


def make_simulator(dim_theta):
    def simulator(parameters, rng=None):
        rng = np.random.default_rng(rng)
        theta = parameters
        mode = rng.integers(0, 2)
        sign = 1.0 if mode == 0 else -1.0
        y_informative = rng.normal(loc=theta + sign * SHIFT, scale=SIGMA_NOISE, size=dim_theta)
        y_distractors = rng.uniform(LOW, HIGH, size=DIM_DISTRACTORS)
        return {"obs": np.concatenate([y_informative, y_distractors])}
    return simulator


def ground_truth_posterior_samples(dim_theta, n_samples, seed=0):
    """GMM posterior at observation=zeros: two modes at ±SHIFT."""
    rng = np.random.default_rng(seed)
    modes = rng.integers(0, 2, size=n_samples)
    means = np.where(modes[:, None] == 0, -SHIFT, SHIFT) * np.ones((n_samples, dim_theta))
    return rng.normal(loc=means, scale=SIGMA_NOISE)


def c2st(X, Y, seed=1, n_folds=5):
    """Paper-standard C2ST: matches lfi.evaluation.c2st exactly."""
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - X_mean) / X_std
    Y = (Y - X_mean) / X_std
    ndim = X.shape[1]
    clf = MLPClassifier(activation="relu", hidden_layer_sizes=(10*ndim, 10*ndim),
                        max_iter=10000, solver="adam", random_state=seed)
    data = np.concatenate([X, Y])
    target = np.array([0]*len(X) + [1]*len(Y), dtype=float)
    scores = cross_val_score(clf, data, target,
                             cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
                             scoring="accuracy")
    return float(np.mean(scores))


# ------------------------------------------------------------------ #
# Experiment config
# ------------------------------------------------------------------ #
D_LIST = [2, 5, 10, 15, 20]
BUDGET_LIST = [5_000, 10_000, 20_000, 50_000, 100_000]
SEEDS = [42, 48930, 1234]
NOF_SAMPLES = 1000
EPOCHS = 50
BATCH_SIZE = 256

RESULTS_ROOT = os.path.join(
    os.path.dirname(__file__),
    "../code/results/mog_benchmark/two_modes_distractors"
)
METHOD_NAME = "real_bayesflow"


def run_one(dim_theta, budget, seed):
    np.random.seed(seed)

    prior_fn = make_prior(dim_theta)
    sim_fn = make_simulator(dim_theta)
    simulator = bf.make_simulator([prior_fn, sim_fn])

    # observation: zeros for informative dims, random uniform for distractors
    rng = np.random.default_rng(seed)
    obs_informative = np.zeros((1, dim_theta))
    obs_distractors = rng.uniform(LOW, HIGH, size=(1, DIM_DISTRACTORS))
    obs = np.concatenate([obs_informative, obs_distractors], axis=1)

    train_data = simulator.sample(budget)
    val_size = max(200, budget // 10)
    val_data = simulator.sample(val_size)

    workflow = bf.BasicWorkflow(
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (128,) * 3}),
        inference_variables=["parameters"],
        inference_conditions=["obs"],
    )

    t0 = perf_counter()
    workflow.fit_offline(
        train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_data,
        verbose=0,
    )
    runtime = perf_counter() - t0

    samples = workflow.sample(conditions={"obs": obs}, num_samples=NOF_SAMPLES)
    samples = samples["parameters"][0]

    samples_gt = ground_truth_posterior_samples(dim_theta, NOF_SAMPLES, seed=seed)
    score = c2st(samples, samples_gt, seed=seed)

    return score, runtime, samples, samples_gt


def save(dir_path, samples, samples_gt, score, runtime, seed):
    run_dir = os.path.join(dir_path, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    np.savetxt(os.path.join(run_dir, "c2st.csv"), [score])
    np.savetxt(os.path.join(run_dir, "runtime.csv"), [runtime])
    np.savetxt(os.path.join(run_dir, "samples.csv"), samples, delimiter=",")
    if not os.path.exists(os.path.join(dir_path, "..", "gt_samples.csv")):
        np.savetxt(os.path.join(dir_path, "..", "gt_samples.csv"), samples_gt, delimiter=",")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for dim in D_LIST:
        d_dir = os.path.join(RESULTS_ROOT, f"D_{dim}")
        os.makedirs(d_dir, exist_ok=True)

        for budget in BUDGET_LIST:
            method_dir = os.path.join(d_dir, f"{METHOD_NAME}_{budget}")
            os.makedirs(method_dir, exist_ok=True)

            for seed in SEEDS:
                out_dir = os.path.join(method_dir, f"seed_{seed}")
                if os.path.exists(os.path.join(out_dir, "c2st.csv")):
                    print(f"[skip] D={dim}, budget={budget}, seed={seed}")
                    continue

                print(f"Running D={dim}, budget={budget}, seed={seed} ...", flush=True)
                try:
                    score, runtime, samples, samples_gt = run_one(dim, budget, seed)
                    save(method_dir, samples, samples_gt, score, runtime, seed)
                    print(f"  -> C2ST={score:.4f}, runtime={runtime:.1f}s")
                except Exception as e:
                    print(f"  ERROR: {e}")
