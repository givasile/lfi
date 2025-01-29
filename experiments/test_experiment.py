import lfi
import experiment
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

exp = experiment.SingleRun(
    config=config,
    analyze=False,
    evaluate=True,
    store=True,
    use_mlflow=False,
    experiment_name="test_experiment"
)

exp.run()