import experiments
import numpy as np
import yaml

np.random.seed(21)

# with open("config_high_dimensional.yaml", "r") as f:
with open("config_high_dimensional_only_plot.yaml", "r") as f:
    config_all = yaml.load(f, Loader=yaml.FullLoader)


for config in config_all:
    print("##" * 40)
    print("Starting experiment: ", config["name"])
    print("##" * 40)
    exp = experiments.HighDimensionalExperiment(
        name=config["name"],
        D_list=config["D_list"],
        nof_repetitions=config["nof_repetitions"],
    )

    if config["load"]:
        exp.load()

    # run experiments
    if "config_romc" in config:
        print("Running ROMC!!!")
        exp.run_romc(config["config_romc"])

    if "config_r2omc" in config:
        print("Running R2OMC!!!")
        exp.run_r2omc(config["config_r2omc"])

    if "config_npe" in config:
        print("Running NPE!!!")
        exp.run_npe(config["config_npe"])

    if "config_snpe" in config:
        print("Running SNPE!!!")
        exp.run_snpe(config["config_snpe"])

    if "config_nle" in config:
        print("Running NLE!!!")
        exp.run_nle(config["config_nle"])

    # save results
    if config["save"]:
        exp.save()

    # plot results
    if config["plot"]:
        exp.plot(config["plot_title"], config["plot_save"])
        exp.plot_runtime(config["plot_title"], config["plot_save"])
