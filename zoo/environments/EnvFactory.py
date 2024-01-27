import gym
from stable_baselines3.common.monitor import Monitor
from hydra.utils import instantiate


def _get_custom_env(config):
    """
    Create the custom environment according to the configuration
    :param config: the hydra configuration
    :return: the created environment, or None if the environment is not a custom environment
    """

    # Get the environment name.
    env_name = config.environment.name

    # Create the environment, if its name is in the list of custom environments, otherwise return None
    for env in ["d_sprites", "maze"]:
        if env_name.startswith(env):
            return instantiate(config.environment)
    return None


def make(config):
    """
    Create the environment according to the configuration
    :param config: the hydra configuration
    :return: the created environment
    """
    env = _get_custom_env(config)
    if env is None:
        env = gym.make(config.environment.name)
    return Monitor(env)
