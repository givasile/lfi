import numpy as np
import sbibm


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


class FromSBIBM(BaseObservation):
    def __init__(self, task_name, exp_num):
        assert task_name in sbibm.get_available_tasks(), f"Task {task_name} is not available in SBIBM."
        assert exp_num >= 1, "Experiment number must be greater than or equal to 1."
        self.task = sbibm.get_task(task_name)
        self.exp_num = exp_num
        super().__init__(name="from_sbi", dim_y=self.task.dim_data, nof_observations=1)

    def sample(self):
        return np.array(self.task.get_observation(self.exp_num))
        # return self.task.get_reference_posterior_samples(self.exp_num)

