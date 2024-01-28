import abc
import gym


class EnvInterface(gym.Env):
    """
    A class containing the code of the environment interface that all environments must implement.
    """

    def __init__(self, **_):
        """
        Constructor (compatible with OpenAI gym environment)
        """

        # Gym compatibility
        super(EnvInterface, self).__init__()

    @property
    @abc.abstractmethod
    def action_names(self):
        """
        Getter
        :return: the list of action names
        """
        ...
