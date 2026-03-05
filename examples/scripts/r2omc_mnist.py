"""
R2OMC on MNIST image denoising.

Problem
-------
    theta = clean MNIST digit image, flattened (dim = 784, values in [0, 1])
    y | theta = theta / 10 + 0.5 + N(0, sigma^2)   [ImagePixelWiseTransform]

The observation is a slightly shifted and noisy version of the clean image.
R2OMC automatically detects the ~non-background informative pixels (Step 1)
and recovers the posterior over the clean image.

Prior
-----
    theta ~ Uniform(-5, 5)^784  — wide box containing the [0,1] image range.

MNIST loading
-------------
    Uses torchvision (no tensorflow required).
    The dataset is downloaded to /tmp/mnist on first run (~11 MB).
"""
import numpy as np
import matplotlib.pyplot as plt

from lfi.priors import UniformPrior
from lfi.simulators import ImagePixelWiseTransform
from lfi.inference.r2omc import R2OMC

# ── Setup ─────────────────────────────────────────────────────────────────────

SEED    = 42
IMG_IDX = 0        # which MNIST training image to use as theta_true
BUDGET  = 100
H, W    = 28, 28
DIM     = H * W    # 784

# ── Load one MNIST image (torchvision, no tensorflow) ─────────────────────────

from torchvision.datasets import MNIST
import torchvision.transforms as T

print("Loading MNIST via torchvision (downloads ~11 MB on first run)...")
dataset    = MNIST(root="/tmp/mnist", train=True, download=True, transform=T.ToTensor())
img, label = dataset[IMG_IDX]
theta_true = np.array(img).flatten().astype(np.float32)   # (784,) in [0, 1]
print(f"  digit label : {label}")
print(f"  pixel range : [{theta_true.min():.3f}, {theta_true.max():.3f}]")

# ── Simulator and observation ─────────────────────────────────────────────────

sim = ImagePixelWiseTransform(dim=DIM, dim_y=DIM, H=H, W=W, sigma_noise=0.01)
obs = np.array(sim.sample_jax(theta_true, seed=SEED))[None, :]   # (1, 784)

# ── R2OMC ─────────────────────────────────────────────────────────────────────

prior  = UniformPrior(dim=DIM, low=-5.0, high=5.0)
method = R2OMC(prior, sim, obs)

print("\n── R2OMC ──")
method.fit(
    budget=BUDGET,
    fit_kwargs={
        "find_informative_dims": True,
        "inf_dims_nof_th":       1,
        "inf_dims_nof_seeds":    5,
        "inf_dims_threshold":    1e-5,
        "alpha":                 0.01,
        "epochs":                16,
        "pcg_to_keep":           1.0,
        "box_algorithm":         "blind",
        "dx":                    0.1,
    },
)
print(f"  informative dims : {len(method.informative_dims)} / {DIM}")

NOF_SAMPLES = method.nof_seeds_accept * method.nof_th0
samples = method.sample(
    nof_samples=NOF_SAMPLES,
    sample_kwargs={"return_th_star": True},
)

# ── Plot ──────────────────────────────────────────────────────────────────────

CLIP_THRESHOLD = 0.2   # pixels below this in the posterior mean are set to 0

posterior_mean = samples.mean(axis=0)
posterior_mean_clipped = np.where(posterior_mean < CLIP_THRESHOLD, 0.0, posterior_mean)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panels = [
    ("Clean image (theta_true)", theta_true,             "gray",   0, 1),
    ("Observation (noisy)",      obs[0],                 "gray",   None, None),
    ("Posterior mean",           posterior_mean_clipped, "gray",   0, 1),
]
for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
    ax.imshow(img.reshape(H, W), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

fig.suptitle(f"MNIST digit '{label}' — R2OMC image denoising", fontsize=13)
fig.tight_layout()
plt.show(block=True)
