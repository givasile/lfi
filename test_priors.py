"""
Manual test script for lfi.priors — run line-by-line to verify behaviour.
Not a pytest file: designed for interactive inspection.
"""
import numpy as np
import jax
import jax.numpy as jnp
import lfi

KEY = jax.random.PRNGKey(0)
N = 1000
D = int(1e2)

# add a simple visualtions helper
def plot_samples(samples, title="Samples"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 16), dpi=120)
    plt.title(title)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.show(block=False)


# ==============================================================================
# UniformPrior
# ==============================================================================
print("\n=== UniformPrior ===")
p = lfi.priors.UniformPrior(dim=D, low=-1, high=1)

# sampling — all backends should return (N, dim) arrays
s_np = p.sample_numpy(N)
print("sample_numpy shape:", s_np.shape)               # (5, 2)
print("sample_numpy range:", s_np.min(), s_np.max())   # within [-1, 1]
plot_samples(s_np, title="UniformPrior samples (numpy)")

s_jax = p.sample_jax(KEY, N)
print("sample_jax shape:  ", s_jax.shape)              # (5, 2)
print("sample_jax range:  ", s_jax.min(), s_jax.max()) # within [-1, 1]
plot_samples(np.array(s_jax), title="UniformPrior samples (jax)")

s_pt = p.sample_pytorch(N)
print("sample_pytorch shape:", s_pt.shape)             # torch.Size([5, 2])
print("sample_pytorch range:", s_pt.min(), s_pt.max()) # within [-1, 1]
plot_samples(s_pt.numpy(), title="UniformPrior samples (pytorch)")

# pdf / logpdf
x_in  = np.zeros((D,))       # inside -> pdf = 0.25, logpdf = -log(4) ~ -1.386
x_out = np.ones((D,)) * 2  # outside -> pdf = 0, logpdf = -inf
print("pdf (inside):  ", p.pdf(x_in))
print("pdf (outside): ", p.pdf(x_out))
print("logpdf (inside): ", p.logpdf(x_in))
print("logpdf (outside):", p.logpdf(x_out))

# has_mass
print("has_mass (inside): ", p.has_mass(x_in))   # [True]
print("has_mass (outside):", p.has_mass(x_out))  # [False]

# ==============================================================================
# LogNormal
# ==============================================================================
print("\n=== LogNormal — scalar mean/std (broadcasts) ===")
p = lfi.priors.LogNormal(dim=D, mean=0.0, std=0.5)

s_np = p.sample_numpy(N)
print("sample_numpy shape:   ", s_np.shape)        # (5, 2)
print("sample_numpy all > 0: ", np.all(s_np > 0))  # True (lognormal is positive)

s_jax = p.sample_jax(KEY, N)
print("sample_jax shape:     ", s_jax.shape)            # (5, 2)
print("sample_jax all > 0:   ", bool(jnp.all(s_jax > 0)))  # True

s_pt = p.sample_pytorch(N)
print("sample_pytorch shape: ", s_pt.shape)         # torch.Size([5, 2])
print("sample_pytorch all>0: ", bool((s_pt > 0).all()))  # True
# Rows should be independent
print("rows are independent: ", not np.allclose(s_pt[0].numpy(), s_pt[1].numpy()))  # True

print("\n=== LogNormal — logpdf/pdf ===")
p = lfi.priors.LogNormal(dim=1, mean=0.0, std=1.0)
x_pos = np.array([[1.0]])   # valid input (positive)
x_neg = np.array([[-1.0]])  # invalid (lognormal has no mass at negatives)
print("logpdf at x=1:  ", p.logpdf(x_pos))   # known: -0.5*log(2*pi)
print("logpdf at x=-1: ", p.logpdf(x_neg))   # should be nan (log of negative)

# ==============================================================================
# return_sbi_object
# ==============================================================================
print("\n=== return_sbi_object ===")
import torch
pu = lfi.priors.UniformPrior(dim=D, low=-1, high=1)
pn = lfi.priors.Normal(dim=D, mean=0.0, std=1.0)
pl = lfi.priors.LogNormal(dim=D, mean=0.0, std=0.5)

sbi_u = pu.return_sbi_object()
sbi_n = pn.return_sbi_object()
sbi_l = pl.return_sbi_object()
print("Uniform sbi type:", type(sbi_u))
print("Normal  sbi type:", type(sbi_n))
print("LogNorm sbi type:", type(sbi_l))

s = sbi_n.sample((N,))
print("Normal sbi sample shape:", s.shape)   # torch.Size([5, 2])
s = sbi_l.sample((N,))
print("LogNormal sbi sample shape:", s.shape)  # torch.Size([5, 2])
print("LogNormal sbi sample > 0:", bool((s > 0).all()))  # True

# ==============================================================================
# return_elfi_objects
# ==============================================================================
print("\n=== return_elfi_objects ===")
pu = lfi.priors.UniformPrior(dim=2, low=-1, high=1)
pn = lfi.priors.Normal(dim=2, mean=0.0, std=1.0)
pl = lfi.priors.LogNormal(dim=2, mean=0.0, std=0.5)

elfi_u = pu.return_elfi_objects()
elfi_n = pn.return_elfi_objects()
elfi_l = pl.return_elfi_objects()
print("Uniform elfi objects:", elfi_u)   # list of 2 elfi.Prior nodes
print("Normal  elfi objects:", elfi_n)
print("LogNorm elfi objects:", elfi_l)
print("Uniform len:", len(elfi_u))  # 2
print("Normal  len:", len(elfi_n))  # 2
print("LogNorm len:", len(elfi_l))  # 2

print("\nAll done.")
