from hydra.utils import instantiate
from zoo.agents.save.Checkpoint import Checkpoint


def make(config, env):
    """
    Create the agent according to the configuration
    :param config: the hydra configuration
    :param env: the environment on which the agent will be running
    :return: the created agent
    """

    # Create requested agent.
    archive = Checkpoint(config.agent.tensorboard_dir, config.agent.checkpoint_dir)
    agent = archive.load_agent() if archive.exists() else instantiate(config.agent)

    # Do not proceed if an untrained RL agent is requested for testing or data generation
    if not archive.exists() and config.agent.trainable is True:
        if config.task.name not in ["training", "hyperparameter_tuning"]:
            raise ValueError("This agent must be trained before testing or data generation.")

    # Place environment in agent, and agent in environment
    agent.env = env
    env.agent = agent

    return agent
