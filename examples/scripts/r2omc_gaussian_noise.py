"""
R2OMC on the Gaussian noise problem.
Problem: theta ~ Uniform(-3, 3)^2, y = theta + N(0, 0.1), obs = (1.5, 1.5).
True posterior: N((1.5, 1.5), 0.1^2 I).
"""
import numpy as np
import jax
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import GaussianNoise
from lfi.inference.r2omc import R2OMC
from lfi.evaluation import c2st

# ── Setup ─────────────────────────────────────────────────────────────────────

KEY        = jax.random.PRNGKey(21)
DIM        = 2
SIGMA      = 0.1
BUDGET     = 2_000
NOF_SAMPLES = 500

prior  = UniformPrior(dim=DIM, low=-3, high=3)
sim    = GaussianNoise(dim=DIM, dim_y=DIM, sigma_noise=SIGMA)
obs    = np.ones((1, DIM), dtype=np.float32) * 1.5

# ground truth: N((1.5, 1.5), sigma^2 I)
rng = np.random.default_rng(42)
gt_samples = rng.normal(loc=1.5, scale=SIGMA, size=(NOF_SAMPLES, DIM)).astype(np.float32)

# ── Fit & sample ──────────────────────────────────────────────────────────────

method = R2OMC(prior, sim, obs)
samples = method.fit_and_sample(
    budget=BUDGET,
    nof_samples=NOF_SAMPLES,
    fit_kwargs={"key": KEY},
    sample_kwargs={},
)

# ── Evaluation ────────────────────────────────────────────────────────────────

score = c2st(gt_samples, samples)
print(f"\nC2ST: {score:.3f}  (0.5 = perfect, 1.0 = completely different)")
print(f"posterior mean : {samples.mean(0).round(3)}  (true: [1.5, 1.5])")
print(f"posterior std  : {samples.std(0).round(3)}   (true: [{SIGMA}, {SIGMA}])")

# ── Plot ──────────────────────────────────────────────────────────────────────

method.plot_posterior_samples(
    samples=samples,
    samples_gt=gt_samples,
    limits=[-3, 3],
    title=f"R2OMC — GaussianNoise  (C2ST={score:.3f})",
    show=False,
)
plt.show(block=False)


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
