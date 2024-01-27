import gym
import numpy as np


class NumpyWrapper(gym.ObservationWrapper):
    """
    A wrapper that turns the observations from torch tensors to numpy arrays
    """

    def __init__(self, env):
        """
        Create a wrapper that turns the observations from torch tensors to numpy arrays
        :param env: the environment to wrap
        """
        super().__init__(env)

    def observation(self, obs):
        """
        Convert each observation from a pytorch tensor to a numpy array
        :param obs: the obs stored in a pytorch tensor
        :return: the output observation (numpy array)
        """
        if isinstance(obs, np.ndarray):
            return obs
        return obs.cpu().numpy()
