import numpy as np
import simulators
import priors
import jax
import experiments

np.random.seed(21)

exp = experiments.SbiExperiment(
    task_name="gaussian_mixture",
    exp_num=3
)

config = {
    "fit_seed": 21,
    "find_informative_dims": True,
    "inf_dims_nof_th": 100,
    "inf_dims_nof_seeds": 50,
    "nof_th0": 1,
    "nof_gd_steps": 50,
    "alpha": .1,
    "epochs": 8,
    "eps_2": 0.00001,
    "nof_ls_steps": 20,
    "step_size": .01,
    "sample_seed": 71,
    "eps_3": 0.1
}

# exp.run_romc(
#     prior=priors.Uniform(low=-10., high=10., dim=2),
#     sim=simulators.CasesGaussianSBI(dim=2, sigma_1=1, sigma_2=0.1),
#     pcg_of_added_seeds=0.2,
#     config=config
# )

# exp.save()
exp.load()
exp.plot("./paper/figures/")


# import matplotlib.pyplot as plt
# th_romc = exp.romc_method.samples_flat
# th_gt = np.array(exp.samples_gt)[0:1000,:]
#
# plt.figure()
# plt.scatter(th_romc[:,0], th_romc[:,1], c="darkmagenta", label=r"$\mathtt{R2OMC}$")
# plt.legend()
# plt.xlabel(r"$\theta_1$")
# plt.ylabel(r"$\theta_2$")
# plt.xlim(4,11)
# plt.ylim(-6.5,0.5)
# plt.savefig("./paper/figures/exp_sbi_gaussian_mixture_r2omc.pdf", bbox_inches="tight")
# plt.show(block=False)
#
# plt.figure()
# plt.scatter(th_gt[:,0], th_gt[:,1], c="red", label=r"$\mathtt{GT}$")
# plt.legend()
# plt.xlabel(r"$\theta_1$")
# plt.ylabel(r"$\theta_2$")
# plt.xlim(4,11)
# plt.ylim(-6.5,0.5)
# plt.savefig("./paper/figures/exp_sbi_gaussian_mixture_gt.pdf", bbox_inches="tight")
# plt.show(block=False)
