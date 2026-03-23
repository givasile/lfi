import lfi
import sbibm
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# set seed
np.random.seed(42)

# modeling parameters
dim = 4
prior_mean = np.array([-0.125, -3.0, -0.125, -3.0])
prior_std = np.array([0.5, 0.5, 0.5, 0.5])
dim_y = 20
observation_nof_obs = 1

# inference parameters
budget = 1000
nof_samples = 100

path = "../../results/lotka_volterra/"
if not os.path.exists(path):
    os.makedirs(path)

# Prior
prior = lfi.priors.LogNormal(
    dim=4,
    mean=prior_mean,
    std=prior_std
)

# Run simulator
simulator = lfi.simulators.LotkaVolterra(
    dim=4,
    dim_y=20
)


def sample_prior():
    """Sample one parameter vector from prior."""
    return prior.sample()


def simulate(theta):
    """
    Given theta (shape (dim,)), simulate an observation x.
    Output x shape should match what inference expects as observation.
    """
    # simulator expects input in correct shape (likely (dim,))
    # Output shape depends on lfi.simulators.LotkaVolterra output — assume shape (20, 1)
    # simulator.forward returns simulation for one theta
    x = simulator(theta)
    return x  # shape (dim_y, 1) or similar


def infer_posterior(x_obs):
    """
    Run inference on a single observation x_obs.
    Returns posterior samples array shape (nsamples, dim).
    """
    # Create new inference instance for each run with the same prior and simulator,
    # but new observation x_obs.
    # Use same inference parameters as before.

    inference = lfi.inference.r2omc.R2OMC(
        prior=prior,
        simulator=simulator,
        observation=x_obs,
    )

    inference.fit(
        budget=1000,  # or pass as argument if needed
        fit_kwargs={
            "find_informative_dims": False,
            "inf_dims_nof_th": 30,
            "inf_dims_nof_seeds": 10,
            "inf_dims_threshold": 1e-1,
            "epochs": 8,
            "nof_gd_steps": 10,
            "alpha": 0.001,
            "pcg_to_keep": 0.05,
            "box_algorithm": "blind",
            "dx": 0.1,
            "nof_ls_steps": 10,
            "step_size": 0.02,
        }
    )
    # Return posterior samples
    # th_star shape (nof_samples, dim)
    return inference.th_star_init[:, 0, :], inference.d_star_init[:, 0], inference.th_star[:, 0, :]

# perform inference n
n_rep = 150 # 500
ranks = []
theta_true_list = []
seed_list = []
x_obs_list = []
th_post_list = []
d_post_list = []
th_star_list = []

for ii in range(n_rep):
    print(f"Starting rep {ii + 1} / {n_rep}...")
    tic = time.time()
    theta_true = prior.sample_numpy(1)
    seed = np.random.randint(0, 1e6)
    x_obs = simulator.sample_jax(theta_true[0], seed=seed)
    x_obs = np.expand_dims(x_obs, axis=0)
    th_post, d_post, th_start = infer_posterior(x_obs)

    # # store all
    theta_true_list.append(theta_true)
    seed_list.append(seed)
    x_obs_list.append(x_obs)
    th_post_list.append(th_post)
    th_star_list.append(th_start)
    d_post_list.append(d_post)
    toc = time.time()
    print(f"Completed rep in {toc - tic:.2f} seconds.")


# store results in a np.savez file
np.savez(
    path + "lotka_volterra_sbc_results.npz",
    theta_true_list=theta_true_list,
    seed_list=seed_list,
    x_obs_list=x_obs_list,
    th_post_list=th_post_list,
    d_post_list=d_post_list,
    th_star_list=th_star_list,
)
