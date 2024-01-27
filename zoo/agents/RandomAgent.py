import numpy as np
from zoo.agents.AgentInterface import AgentInterface


class RandomAgent(AgentInterface):
    """
    Implement an agent acting randomly.
    """

    def __init__(self, name, n_actions, tensorboard_dir, **_):
        """
        Constructor
        :param name: the agent name
        :param n_actions: the number of available actions
        :param tensorboard_dir: the directory for tensorboard runs
        """

        # Call parent constructor
        super().__init__(tensorboard_dir)

        # Store agent name and number of actions
        self.agent_name = name
        self.n_actions = n_actions

    def step(self, obs):
        """
        Select a random action
        :param obs: unused
        :return: the random action
        """
        return np.random.choice(self.n_actions)

    def total_rewards_obtained(self):
        """
        Getter
        :return: the total number of rewards gathered to date
        """
        return 0

    def n_steps_done(self):
        """
        Getter
        :return: the number of training steps performed to date
        """
        return 0

    def name(self):
        """
        Getter
        :return: the agent's name
        """
        return self.agent_name
