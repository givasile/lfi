"""
Run modern BayesFlow (flow matching, no summary network) on the same
MoG settings as the paper, for BOTH the distractor-free and distractor cases:

  - theta dimension D in [2, 5, 10, 15, 20]
  - distractors in {0, 18}   (0 -> two_modes/, 18 -> two_modes_distractors/)
  - budgets: [5_000, 10_000, 20_000, 50_000, 100_000]
  - 3 seeds per (D, budget) cell

Config matches the reviewer's setup exactly:
  bf.BasicWorkflow + FlowMatching(widths=(128,)*3), no summary network.

Results are written incrementally (one cell at a time) to
  ../code/results/mog_benchmark/<two_modes | two_modes_distractors>/D_<D>/
in the same directory structure as existing baselines, so results.py can
pick them up automatically. Re-running is idempotent: a cell whose
c2st.csv already exists is skipped, so the run is resumable.

Reproducibility: keras.utils.set_random_seed(seed) pins python / numpy /
backend RNG (network init etc.) per cell; ground-truth and C2ST classifier
are seeded too. Posterior samples are saved, so analysis is fully
reproducible regardless of minor data-gen nondeterminism.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pandas as pd
from time import perf_counter, strftime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score

import keras
import bayesflow as bf

# ------------------------------------------------------------------ #
# Simulator (matches lfi.simulators.BimodalGaussian[Distractors])
# ------------------------------------------------------------------ #
LOW, HIGH = -3.0, 3.0
SHIFT = 1.0
SIGMA_NOISE = 0.2


def make_prior(dim_theta):
    def prior(rng=None):
        rng = np.random.default_rng(rng)
        return {"parameters": rng.uniform(LOW, HIGH, size=dim_theta)}
    return prior


def make_simulator(dim_theta, dim_distractors):
    def simulator(parameters, rng=None):
        rng = np.random.default_rng(rng)
        theta = parameters
        mode = rng.integers(0, 2)
        sign = 1.0 if mode == 0 else -1.0
        y_informative = rng.normal(loc=theta + sign * SHIFT, scale=SIGMA_NOISE, size=dim_theta)
        if dim_distractors > 0:
            y_distractors = rng.uniform(LOW, HIGH, size=dim_distractors)
            return {"obs": np.concatenate([y_informative, y_distractors])}
        return {"obs": y_informative}
    return simulator


def ground_truth_posterior_samples(dim_theta, n_samples, seed=0):
    """GMM posterior at observation=zeros: two modes at ±SHIFT (distractors uninformative)."""
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
BUDGET_LIST = [5_000, 10_000, 20_000, 50_000]   # ascending; capped at 50k
DIST_LIST = [18, 0]            # 18 (core claim) first, then distractor-free
SEEDS = [42, 48930, 1234]
NOF_SAMPLES = 1000
EPOCHS = 120                   # match the reviewer's (Stefan's) setup
BATCH_SIZE = 256

# C2ST success threshold (matches the paper's thr_0.8 frontier). For a given D
# we climb the budget ladder and stop once ALL seeds reach C2ST <= this value:
# once the method works, larger budgets are wasted compute.
SUCCESS_THRESHOLD = 0.8

RESULTS_BASE = os.path.join(os.path.dirname(__file__), "../code/results/mog_benchmark")
METHOD_NAME = "real_bayesflow"


def results_root(dim_distractors):
    sub = "two_modes_distractors" if dim_distractors > 0 else "two_modes"
    return os.path.join(RESULTS_BASE, sub)


def stamp():
    return strftime("%H:%M:%S")


def run_one(dim_theta, budget, seed, dim_distractors):
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    prior_fn = make_prior(dim_theta)
    sim_fn = make_simulator(dim_theta, dim_distractors)
    simulator = bf.make_simulator([prior_fn, sim_fn])

    # observation: zeros for informative dims, random uniform for distractors
    rng = np.random.default_rng(seed)
    obs_informative = np.zeros((1, dim_theta))
    if dim_distractors > 0:
        obs_distractors = rng.uniform(LOW, HIGH, size=(1, dim_distractors))
        obs = np.concatenate([obs_informative, obs_distractors], axis=1)
    else:
        obs = obs_informative

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
    print(f"# systematic real-BayesFlow scan starting at {stamp()}", flush=True)
    print(f"# D={D_LIST} budgets={BUDGET_LIST} distractors={DIST_LIST} seeds={SEEDS}", flush=True)
    print(f"# FlowMatching widths=(128,)*3, no summary net, {EPOCHS} epochs, batch {BATCH_SIZE}", flush=True)
    print(f"# per D: climb budget ladder, stop once all seeds reach C2ST <= {SUCCESS_THRESHOLD}", flush=True)

    for dim_distractors in DIST_LIST:
        root = results_root(dim_distractors)
        for dim in D_LIST:
            d_dir = os.path.join(root, f"D_{dim}")
            os.makedirs(d_dir, exist_ok=True)

            for budget in BUDGET_LIST:                     # ascending
                method_dir = os.path.join(d_dir, f"{METHOD_NAME}_{budget}")
                os.makedirs(method_dir, exist_ok=True)

                budget_scores = []
                for seed in SEEDS:
                    out_dir = os.path.join(method_dir, f"seed_{seed}")
                    c2st_path = os.path.join(out_dir, "c2st.csv")
                    tag = f"dist={dim_distractors} D={dim} budget={budget} seed={seed}"
                    if os.path.exists(c2st_path):
                        score = float(np.loadtxt(c2st_path))
                        print(f"[{stamp()}] [skip] {tag}  (C2ST={score:.4f})", flush=True)
                        budget_scores.append(score)
                        continue

                    print(f"[{stamp()}] running {tag} ...", flush=True)
                    try:
                        score, runtime, samples, samples_gt = run_one(dim, budget, seed, dim_distractors)
                        save(method_dir, samples, samples_gt, score, runtime, seed)
                        print(f"[{stamp()}]   -> C2ST={score:.4f}, runtime={runtime:.1f}s", flush=True)
                        budget_scores.append(score)
                    except Exception as e:
                        print(f"[{stamp()}]   ERROR: {e}", flush=True)

                # Early stop: once all seeds succeed, larger budgets are wasted.
                if len(budget_scores) == len(SEEDS) and all(s <= SUCCESS_THRESHOLD for s in budget_scores):
                    print(f"[{stamp()}] dist={dim_distractors} D={dim}: all seeds <= {SUCCESS_THRESHOLD} "
                          f"at budget={budget}; stopping budget ladder.", flush=True)
                    break

    print(f"# scan finished at {stamp()}", flush=True)
