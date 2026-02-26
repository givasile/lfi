import lfi.priors
import lfi.simulators
import lfi.observations
import lfi.ground_truth
import lfi.evaluation
import lfi.inference

PRIOR_TO_CLASS = {
    'uniform': lfi.priors.UniformPrior
}

SIMULATOR_TO_CLASS = {
    'bimodal_gaussian': lfi.simulators.BimodalGaussian,
    'gaussian_noise': lfi.simulators.GaussianNoise
}

OBSERVATION_TO_CLASS = {
    'zeros': lfi.observations.Zeros,
}

INFERENCE_TO_CLASS = {
    'sbi_mcabc': lfi.inference.sbi.SBI_MCABC,
    'npe_a_single_round': lfi.inference.sbi.NPEASingleRound,
    'npe_c_single_round': lfi.inference.sbi.NPECSingleRound,
    'fmpe_single_round': lfi.inference.sbi.FMPESingleRound,
    'elfi_rejection_sampling': lfi.inference.elfi.RejectionSampling,
    #'mdn': lfi.inference.custom.MixtureDensityNetwork,
}

GROUND_TRUTH_TO_CLASS = {
    'gaussian': lfi.ground_truth.Gaussian,
    'gaussian_mixture': lfi.ground_truth.GaussianMixture
}

EVALUATION_TO_CLASS = {
    'c2st': lfi.evaluation.c2st
}

def flatten_config(config, sep='__'):
    """
    Transforms a nested configuration dictionary into a flattened dictionary.
    
    Args:
        config (dict): The nested configuration dictionary to flatten.
        sep (str): Separator to use between keys.
        
    Returns:
        dict: A flattened dictionary where nested keys are merged with the separator
    """

    def flatten_dict(d, parent_key=''):
        items=[]
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).item())
            else:
                items.append((new_key, v))
        return dict(items)
    return flatten_dict(config)

def unflatten_config(flattened_config, sep='__'):
    """
    Transforms a flattened dictionary into a nested dictionary
    
    Args:
        flattened_config (dict): The flattened dictionary to unflatten.
        sep (str): Separator used in flattened keys.
        
    Returns:
        dict: A nested dictionary reconstructed from the flattened dictionary
    """
    nested_config={}
    for key, value in flattened_config.items():
        keys = key.split(sep)
        d = nested_config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return nested_config