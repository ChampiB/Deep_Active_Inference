import ray
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import abc
from abc import ABC
import logging
from stable_baselines3.common.utils import safe_mean
from collections import deque


class AgentInterface(ABC):
    """
    The interface that all agents need to implement.
    """

    def __init__(self, tensorboard_dir, steps_done=1, need_writer=True):
        """
        Construct an agent
        :param tensorboard_dir: the directory in which the tensorboard logs should be written
        :param steps_done: the number of training steps done so far
        :param need_writer: if true create a SummaryWriter
        """

        # Create the queue containing the episode information.
        self.ep_info_buffer = deque(maxlen=100)

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter(tensorboard_dir) if need_writer else None

        # Number of training steps performed to date.
        self.steps_done = steps_done

    @abc.abstractmethod
    def step(self, obs):
        """
        Select the next action to perform in the environment
        :param obs: the observation available to make the decision
        :return: the next action to perform
        """
        ...

    @abc.abstractmethod
    def name(self):
        """
        Getter
        :return: the agent's name
        """
        ...

    @abc.abstractmethod
    def n_steps_done(self):
        """
        Getter
        :return: the number of training steps performed to date
        """
        ...

    @abc.abstractmethod
    def total_rewards_obtained(self):
        """
        Getter
        :return: the total number of rewards gathered to date
        """
        ...

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        """

        # Warn the user that the agent has not implemented the training function.
        logging.warn(f"The agent {self.name()} has not implemented the training function.")

        # Test the agent, as we don't know how to train it.
        self.test(
            env, config,
            n_steps_done=self.n_steps_done(), total_rewards=self.total_rewards_obtained()
        )

    def test(self, env, config, reward_name=None, n_steps_done=0, total_rewards=0):
        """
        Test the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :param reward_name: the reward name as displayed in tensorboard
        :param n_steps_done: the number of steps already performed in the environment
        :param total_rewards: the total amount of rewards obtained to date
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Test the agent.
        task_performed = "training" if reward_name is None else "testing"
        logging.info(f"Start the {task_performed} at {datetime.now()}")
        while n_steps_done < config.task.max_n_steps:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            obs, reward, done, info = env.step(action)

            # Monitor total reward if needed.
            if self.writer is not None:
                total_rewards += reward
                if self.steps_done % config.tensorboard.log_interval == 0:
                    self.writer.add_scalar("total_rewards", total_rewards, self.steps_done)
                    self.log_episode_info(info, config.task.name)

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of iterations performed.
            n_steps_done += 1

        # Close the environment.
        env.close()

    def compute_mean_episodic_reward(self):
        """
        Compute the mean episodic reward
        :return: the mean episodic reward, if it can be computed, None otherwise
        """
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            return safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        return None

    def store_episode_info(self, information):
        """
        Store the episode information into the internal queue
        :param information: the information to store
        """

        # Make sure that the information is stored as a list of dictionary.
        if not isinstance(information, list):
            information = [information]

        # Store episode information in the internal queue.
        for info in information:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.ep_info_buffer.extend([ep_info])

    def log_episode_info(self, information, task_name, steps_done=-1):
        """
        Log episode information in tensorboard
        :param information: the information returned by the environment
        :param task_name: the name of the task being performed
        :param steps_done: the number of steps done so far
        """

        # Make sure that the number of steps done is valid.
        steps_done = steps_done if steps_done >= 0 else self.steps_done

        # Store episode information.
        self.store_episode_info(information)

        # Log mean episodic reward and mean episode length.
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:

            # Log mean episodic reward into tensorboard, and report it to ray tune (if needed).
            ep_rew_mean = self.compute_mean_episodic_reward()
            self.writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, steps_done)
            if task_name == "hyperparameter_tuning":
                ray.train.report({"loss": ep_rew_mean})

            # Log mean episode length.
            ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
            self.writer.add_scalar("rollout/ep_len_mean", ep_len_mean, steps_done)

    def draw_reconstructed_images(self, env, grid_size):
        """
        Get reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        """
        raise Exception("The function 'get_reconstructed_images' is not implemented by this agent.")
