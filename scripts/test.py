import sbibm
import lfi
import matplotlib.pyplot as plt
import numpy as np
import jax


task = sbibm.get_task('lotka_volterra')
th_0 = task.get_true_parameters(3)
th_0_np = th_0.numpy().reshape(-1)
x_0 = task.get_observation(3)
th_posterior = task.get_reference_posterior_samples(3)

th_good = np.array([0.5793, 0.0542, 1.072, 0.076])


# check th
for i in range(4):
    plt.figure()
    plt.hist(np.array(th_posterior)[:, i], bins=20, density=True, alpha=0.5)
    plt.axvline(x=th_0[0, i], color='r', linestyle='--')
    plt.title(f'Dimension {i} of observations')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show(block=False)






simulator = lfi.simulators.LotkaVolterra(dim=4, dim_y=20)

sample_jit = jax.jit(simulator.sample_jax)
s = 1595824075
th = np.array([0.54221195, 0.05443878, 1.1391242 , 0.08219181])
x = sample_jit(theta=th, seed=s)


obs = []
seeds = [i for i in range(1000)]

for s in seeds:
    # print every 100 iterations
    if s % 100 == 0:
        print(f"Iteration {s}")
    # x = sample_jit(theta=th_posterior[145, :].numpy(), seed=s)
    x = sample_jit(theta=th_good, seed=s)
    obs.append(x)

# plot i-th dimension of obs as a distribution
# and on top of it, plot with a dot the i-th dimension of x_0

for i in range(20):
    plt.figure()
    plt.hist(np.array(obs)[:, i], bins=20, density=True, alpha=0.5)
    plt.axvline(x=x_0[0, i], color='r', linestyle='--')
    plt.title(f'Dimension {i} of observations')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show(block=False)


