from zoo.tasks.TaskInterface import TaskInterface
from zoo.agents import AgentFactory
from zoo.environments import EnvFactory
from zoo.helpers.Seed import Seed


class RunTraining(TaskInterface):
    """
    A class that trains an agent on a specific environment.
    """

    def __init__(self, name, seed, max_n_steps, **_):
        """
        Constructor
        :param name: the task's name
        :param seed: the seed to use for random number generation
        :param max_n_steps: the maximum number of training steps that will be run
        """

        # Call the parent constructor.
        super().__init__(name)

        # Store the task's parameters.
        self.seed = seed
        self.max_n_steps = max_n_steps

    def run(self, hydra_config):
        """
        Train the agent described in the configuration on the requested environment
        :param hydra_config: the configuration where the agent and environment to use are defined
        """

        # Set the seed requested by the user.
        Seed.set(self.seed)

        # Create the environment and agent.
        env = EnvFactory.make(hydra_config)
        agent = AgentFactory.make(hydra_config, env)

        # Train the agent on the environment
        agent.train(env, hydra_config)
