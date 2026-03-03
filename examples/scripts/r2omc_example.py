"""
Step-by-step R2OMC running example.
Problem: theta ~ Uniform(-3, 3)^2, y = theta + N(0, 0.1), obs = (0, 0).
Used as a reference script while refining r2omc.py.
"""
import numpy as np
import jax

from lfi.priors import UniformPrior
from lfi.simulators import GaussianNoise
from lfi.inference.r2omc import R2OMC
from lfi.inference.sbi import NPECSingleRound

# ── Setup ─────────────────────────────────────────────────────────────────────

KEY      = jax.random.PRNGKey(21)
prior    = UniformPrior(dim=2, low=-3, high=3)
sim      = GaussianNoise(dim=2, dim_y=2, sigma_noise=0.1)
obs      = np.ones((1, 2), dtype=np.float32) * 1.5
BUDGET   = 200     # small for fast iteration; bump up for quality

method = R2OMC(prior, sim, obs)

# -- All steps together (for reference) ───────────────────────────────────────────
for v in range(3):
    print(f"\n=== verbose={v} ===")
    samples = method.fit_and_sample(budget=BUDGET, nof_samples=100, fit_kwargs={"key": KEY}, sample_kwargs={}, verbose=v)




# # ── Step 1: find informative dims ─────────────────────────────────────────────

# KEY = method.find_informative_dims(KEY)
# print(f"[Step 1] informative_dims: {method.informative_dims}")

# # ── Step 2: sample + optimise objective functions ─────────────────────────────

# KEY = method.sample_objective_functions(KEY, nof_seeds_total=BUDGET, nof_th0=1)
# print(f"[Step 2a] seeds: {method.seeds_init.shape}, "
#       f"d0 mean: {method.d0_init.mean():.4f}")

# for epoch in range(4):
#     method.optimize(nof_gd_steps=50, alpha=0.1)
#     print(f"[Step 2b] epoch {epoch+1}: "
#           f"d_star mean={method.d_star_init.mean():.5f}  "
#           f"min={method.d_star_init.min():.5f}  "
#           f"max={method.d_star_init.max():.5f}")

# # ── Step 3: filter solutions inside prior ─────────────────────────────────────

# method.filter_solutions_inside_prior()
# print(f"[Step 3] inside prior: {method.nof_seeds_inside_prior}/{BUDGET}")

# # ── Step 4: filter solutions by distance ──────────────────────────────────────

# method.filter_solutions(pcg_to_keep=0.8)
# print(f"[Step 4] accepted: {method.nof_seeds_accept}, eps_1={method.eps_1:.5f}")

# # ── Step 5: get directions ────────────────────────────────────────────────────

# method.get_directions(method="hessian")
# print(f"[Step 5] eig_vec shape: {method.eig_vec.shape}")

# # ── Step 6: build bounding boxes ──────────────────────────────────────────────

# eps_2 = method._get_distances(1, dx := 0.1).mean()
# method.get_boxes(eps_2, nof_ls_steps=100, step_size=0.01)
# print(f"[Step 6] limits shape: {method.limits.shape}, "
#       f"eps_2={method.eps_2:.5f}")

# # ── Step 7: sample ────────────────────────────────────────────────────────────

# samples_flat, weights_flat = method.weighted_sampling(
#     sampling_seed=71, samples_per_region=10, eps_3=1.0)
# final = method.importance_resampling(samples_flat, weights_flat, nof_samples=200)
# print(f"[Step 7] samples: {final.shape}, "
#       f"mean={np.mean(final, axis=0).round(3)}, "
#       f"std={np.std(final, axis=0).round(3)}")
