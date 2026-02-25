import numpy as np


class BaseObservation:
    def __init__(self, name: str, dim_y: int, nof_observations: int):
        self.name = name
        self.dim_y = dim_y
        self.nof_observations = nof_observations

    def sample(self, *args, **kwargs):
        raise NotImplementedError


class Zeros(BaseObservation):
    def __init__(self, dim_y: int, nof_observations: int):
        super().__init__(name="zeros", dim_y=dim_y, nof_observations=nof_observations)

    def sample(self, nof_obs: int = 1, dim_y: int = 1):
        return np.zeros((self.nof_observations, self.dim_y))


class Ones(BaseObservation):
    def __init__(self, dim_y: int, nof_observations: int):
        super().__init__(name="ones", dim_y=dim_y, nof_observations=nof_observations)

    def sample(self, nof_obs: int = 1, dim_y: int = 1):
        return np.ones((self.nof_observations, self.dim_y))



