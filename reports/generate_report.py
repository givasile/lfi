import yaml
import os
import json

def config_matches(config, target_config):
    """
    Check if the config contains all key-value pairs in target_config.

    Args:
        config (dict): The full configuration dictionary.
        target_config (dict): The partial configuration to match.

    Returns:
        bool: True if all keys and values in target_config are found in config.
    """
    if not isinstance(config, dict) or not isinstance(target_config, dict):
        return False

    for key, value in target_config.items():
        if key not in config:
            return False
        if isinstance(value, dict):
            # Recursively check nested dictionaries
            if not config_matches(config[key], value):
                return False
        else:
            # Check for value match
            if config[key] != value:
                return False

    return True


def find_matching_configs(base_dir, target_config):
    """
    Search for paths whose `config.yml` contains the given target_config.

    Args:
        base_dir (str): The base directory containing the experiment runs.
        target_config (dict): The partial configuration dictionary to match.

    Returns:
        list: A list of paths to matching configurations.
    """
    matching_paths = []

    for root, dirs, files in os.walk(base_dir):
        if 'config.yml' in files:
            config_path = os.path.join(root, 'config.yml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config_matches(config, target_config):
                    matching_paths.append(root)
            except Exception as e:
                print(f"Error reading {config_path}: {e}")

    return matching_paths


def load_metrics_from_paths(paths):
    """
    Load all `metrics.json` files from the provided paths.

    Args:
        paths (list): A list of directories containing `metrics.json` files.

    Returns:
        list: A list of dictionaries, each representing the metrics from a `metrics.json` file.
    """
    metrics_data = []

    for path in paths:
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                metrics_data.append(metrics)
            except Exception as e:
                print(f"Error reading {metrics_path}: {e}")
        else:
            print(f"No metrics.json found in {path}")

    return metrics_data


# Example usage:
if __name__ == "__main__":
    base_directory = "./../experiment_runs"
    target_configuration = {
        "prior": {
            "name": "uniform",
            "params": {
                "dim": 2,
                "low": -10.0,
                "high": 10.0,
            },
        },
        "simulator": {
            "name": "bimodal_gaussian",
            "params": {
                "sigma_noise": 0.1,
            },
        },
    }

    matches = find_matching_configs(base_directory, target_configuration)
    if len(matches) == 0:
        print("No matching paths found.")
    else:
        metrics = load_metrics_from_paths(matches)
        c2st = [metric.get("c2st") for metric in metrics]

    print("Matching paths:")
    for match in matches:
        print(match)
