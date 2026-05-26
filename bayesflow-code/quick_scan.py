import os; os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import bayesflow as bf
from time import perf_counter
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score

LOW, HIGH, SHIFT, SIGMA = -3., 3., 1., 0.2

def make_fns(dim_theta, n_dist):
    def prior(rng=None):
        return {"parameters": np.random.default_rng(rng).uniform(LOW, HIGH, size=dim_theta)}
    def simulator(parameters, rng=None):
        rng = np.random.default_rng(rng)
        sign = 1. if rng.integers(0,2)==0 else -1.
        y = rng.normal(loc=parameters + sign*SHIFT, scale=SIGMA, size=dim_theta)
        if n_dist > 0:
            y = np.concatenate([y, rng.uniform(LOW, HIGH, size=n_dist)])
        return {"obs": y}
    return prior, simulator

def gt_samples(dim_theta, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    modes = rng.integers(0, 2, size=n)
    means = np.where(modes[:,None]==0, -SHIFT, SHIFT) * np.ones((n, dim_theta))
    return rng.normal(loc=means, scale=SIGMA)

def c2st(X, Y, seed=1, n_folds=5):
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

print(f"{'setting':<28} {'budget':>7}  C2ST    time")
print("-" * 56)

for dim_theta, n_dist in [(2,0),(2,18),(5,0),(5,18),(10,0),(10,18),(20,0),(20,18)]:
    prior_fn, sim_fn = make_fns(dim_theta, n_dist)
    sim = bf.make_simulator([prior_fn, sim_fn])
    obs = np.zeros((1, dim_theta + n_dist))
    gt  = gt_samples(dim_theta)
    for budget in [1_000, 5_000, 10_000]:
        train = sim.sample(budget)
        val   = sim.sample(max(200, budget//10))
        wf = bf.BasicWorkflow(
            inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths":(128,)*3}),
            inference_variables=["parameters"],
            inference_conditions=["obs"],
        )
        t0 = perf_counter()
        wf.fit_offline(train, epochs=50, batch_size=256, validation_data=val, verbose=0)
        rt = perf_counter() - t0
        s = wf.sample(conditions={"obs": obs}, num_samples=1000)["parameters"][0]
        score = c2st(s, gt)
        label = f"D={dim_theta}, dist={n_dist}"
        print(f"{label:<28} {budget:>7}  {score:.4f}  {rt:.0f}s", flush=True)
