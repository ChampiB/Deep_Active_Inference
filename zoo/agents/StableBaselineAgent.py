from datetime import datetime
from os.path import join
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.ppo import MlpPolicy as PPOCnn
from stable_baselines3.dqn import MlpPolicy as DQNCnn
from stable_baselines3.a2c import MlpPolicy as A2CCnn
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.callbacks.StableBaselineCallback import StableBaselineCallback
from zoo.agents.save.Checkpoint import Checkpoint
from zoo.environments.wrappers.NumpyWrapper import NumpyWrapper
import logging
import torch


class StableBaselineAgent(AgentInterface):
    """
    A wrapper around the stable baseline agents, that make them compatible with our project.
    """

    def __init__(
        self, agent_name, tensorboard_dir, checkpoint_dir, total_rewards=0, steps_done=1,
        hp_tuner=None, name=None, trainable=True, **agent_kwargs
    ):
        """
        Create a wrapper around a stable baseline agent
        :param agent_name: the name of the agent that must be wrapped
        :param tensorboard_dir: the directory in which the tensorboard logs must be writen
        :param checkpoint_dir: the path to the checkpoint directory
        :param steps_done: the number of training steps done so far
        :param hp_tuner: this argument is not used by the agent, only used by the hyperparameter tuning script
        :param name: this argument is not used by the agent, only used by the hydra configuration
        :param trainable: whether the agent can be trained
        """

        # Call parent constructor.
        super().__init__(tensorboard_dir, steps_done, need_writer=False)

        # Create the configurations of the agent that can be instantiated,
        # then retrieve the one for the requested agent.
        self.agents_conf = {
            "ppo": (PPO, PPOCnn),
            "dqn": (DQN, DQNCnn),
            "a2c": (A2C, A2CCnn)
        }
        self.agent_class, self.agent_net = self.agents_conf[agent_name]

        # Store the agent keyword arguments.
        self.agent_kwargs = agent_kwargs

        # Store the logging and checkpoint directories.
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = checkpoint_dir

        # Store the information related to the agent and environment.
        self.environment = None
        self.agent = None
        self.initial_name = agent_name
        self.agent_name = f"SB_{agent_name}"

        # Store the total reward obtained and the number of steps done so far.
        self.total_rewards = int(total_rewards)
        self.steps_done = int(steps_done)

    def step(self, obs):
        """
        Select the next action to perform in the environment
        :param obs: the observation available to make the decision
        :return: the next action to perform
        """
        # Check that the agent has been properly initialised
        if self.agent is None:
            logging.error(
                "The step of the StableBaselineAgent could not be performed as "
                "the environment has not been provided using: agent.env = env."
            )
            return None

        # Perform a step using the agent
        return self.agent.predict(obs)[0]

    def name(self):
        """
        Getter
        :return: the agent's name
        """
        return self.agent_name

    def n_steps_done(self):
        """
        Getter
        :return: the number of training steps performed to date
        """
        return self.steps_done

    def total_rewards_obtained(self):
        """
        Getter
        :return: the total number of rewards gathered to date
        """
        return self.total_rewards

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        """

        # Check that the agent has been properly initialised
        if self.agent is None:
            logging.error(
                "The training of the StableBaselineAgent could not be performed as "
                "the environment has not been provided using: agent.env = env."
            )
            return None

        # Train the agent
        logging.info(f"Start the training at {datetime.now()}")
        max_n_steps = config.task.max_n_steps
        self.agent.learn(
            total_timesteps=max_n_steps,
            log_interval=config.tensorboard.log_interval,
            tb_log_name=self.agent_name,
            reset_num_timesteps=False,
            callback=StableBaselineCallback(self, config)
        )

        # Save the final version of the model.
        self.agent.save(join(self.checkpoint_dir, f"stable_baseline_model_{max_n_steps}"))
        self.save(join(self.checkpoint_dir, f"model_{max_n_steps}.pt"))

    def test(self, env, config, reward_name=None, n_steps_done=0, total_rewards=0):
        """
        Test the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :param reward_name: the reward name as displayed in tensorboard
        :param n_steps_done: the number of steps already performed in the environment
        :param total_rewards: the total amount of rewards obtained to date
        """

        # Wrap the environment to convert each observation from a pytorch tensor to a numpy array
        env = NumpyWrapper(env)

        # Call the test function of the parent class
        super().test(env, config, reward_name, n_steps_done, total_rewards)

    @staticmethod
    def load_constructor_parameters(tb_dir, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint
        :param tb_dir: the path of tensorboard directory
        :param checkpoint: the checkpoint from which to load the parameters
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: a dictionary containing the constructor's parameters
        """
        return {
            "agent_name": checkpoint["agent_name"],
            "tensorboard_dir": tb_dir,
            "checkpoint_dir": checkpoint["checkpoint_dir"],
            "steps_done": checkpoint["steps_done"],
            "total_rewards": checkpoint["total_rewards"]
        }

    def save(self, checkpoint_file):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param checkpoint_file: the file in which the model needs to be saved
        """

        # Create directories and files if they do not exist.
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Save the model.
        torch.save({
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
            "agent_name": self.initial_name,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "steps_done": self.steps_done,
            "total_rewards": self.total_rewards,
        }, checkpoint_file)

    @property
    def env(self):
        """
        Getter
        :return: the environment on which the agent is being trained
        """
        return self.environment

    @env.setter
    def env(self, new_env):
        """
        Setter that creates the stable baseline 3 agent
        :param new_env: the environment on which the agent needs to be trained
        """

        # Create the new environment
        self.environment = NumpyWrapper(new_env)

        # Retrieve the latest checkpoint file
        checkpoint_file = Checkpoint.get_latest_checkpoint(self.checkpoint_dir, r'stable_baseline_model_\d+.zip')

        # Load the agent from the latest checkpoint or create a new agent if no checkpoint is available
        if checkpoint_file is None:
            self.agent = self.agent_class(
                self.agent_net, self.environment, tensorboard_log=self.tensorboard_dir, **self.agent_kwargs
            )
        else:
            self.agent_class, self.agent_net = self.agents_conf[self.initial_name]
            self.agent = self.agent_class.load(checkpoint_file, env=self.environment)

        # Prevent the creation of a subdirectory inside the tensorboard directory
        new_logger = configure(self.tensorboard_dir, ["tensorboard"])
        self.agent.set_logger(new_logger)
