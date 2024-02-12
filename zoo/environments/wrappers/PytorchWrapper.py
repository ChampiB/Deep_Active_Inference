import gym
import torch

from zoo.helpers.Device import Device


class PytorchWrapper(gym.ObservationWrapper):
    """
    A wrapper that turns the observations from numpy arrays to torch tensors
    """

    def __init__(self, env):
        """
        Create a wrapper that turns the observations from numpy arrays to torch tensors
        :param env: the environment to wrap
        """
        super().__init__(env)
        self.device = Device.get()

    def observation(self, obs):
        """
        Convert each observation from a numpy array to a pytorch tensor
        :param obs: the obs stored in a numpy array
        :return: the output observation (torch tensor)
        """
        return torch.from_numpy(obs).float().to(self.device)
