import torch
from sbi import utils as sbi_utils


def get_uniform(prior_limits, D):
    prior = sbi_utils.BoxUniform(
        low=prior_limits[0] * torch.ones(D),
        high=prior_limits[1] * torch.ones(D),
    )
    return prior
