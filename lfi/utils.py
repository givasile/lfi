import lfi.priors
import lfi.simulators
import lfi.observations
import lfi.inference

PRIOR_TO_CLASS = {
    'uniform': lfi.priors.UniformPrior
}

SIMULATOR_TO_CLASS = {
    'bimodal_gaussian': lfi.simulators.BimodalGaussian
}

OBSERVATION_TO_CLASS = {
    'zeros': lfi.observations.Zeros,
}

INFERENCE_TO_CLASS = {
    'npe_a_single_round': lfi.inference.from_sbi.NPE_A_SingleRound,
    'npe_c_single_round': lfi.inference.from_sbi.NPE_C_SingleRound,
    'mdn': lfi.inference.custom.MixtureDensityNetwork,
}


def flatten_config(config, sep='__'):
    """
    Transforms a nested configuration dictionary into a flattened dictionary.

    Args:
        config (dict): The nested configuration dictionary to flatten.
        sep (str): Separator to use between keys.

    Returns:
        dict: A flattened dictionary where nested keys are merged with the separator.
    """
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten_dict(config)

def unflatten_config(flattened_config, sep='__'):
    """
    Transforms a flattened dictionary back into a nested dictionary.

    Args:
        flattened_config (dict): The flattened dictionary to unflatten.
        sep (str): Separator used in flattened keys.

    Returns:
        dict: A nested dictionary reconstructed from the flattened dictionary.
    """
    nested_config = {}
    for key, value in flattened_config.items():
        keys = key.split(sep)
        d = nested_config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested_config
