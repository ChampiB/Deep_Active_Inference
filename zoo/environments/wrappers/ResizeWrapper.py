import gym
import numpy
import skimage


class ResizeWrapper(gym.ObservationWrapper):
    """
    A wrapper that resizes the observations
    """

    def __init__(self, env, image_shape):
        """
        Create a wrapper that resizes the observations
        :param env: the environment to wrap
        """
        super().__init__(env)
        self.image_shape = image_shape

    def observation(self, obs):
        """
        Resize each observation
        :param obs: the obs to resize
        :return: the resized observation
        """
        obs = skimage.transform.resize(obs, self.image_shape)
        if len(obs.shape) == 4:
            obs = numpy.transpose(obs, (0, 2, 1, 3))
            return obs
        obs = numpy.transpose(obs, (1, 2, 0))
        return obs
