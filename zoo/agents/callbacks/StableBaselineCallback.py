from os.path import join

import ray
from stable_baselines3.common.callbacks import BaseCallback


class StableBaselineCallback(BaseCallback):
    """
    A callback saving the model frequently and reporting the episodic mean reward to ray tune.
    """

    def __init__(self, agent, config, verbose=0):
        """
        Create a callback saving the model frequently
        :param agent: the agent that needs to be saved
        :param config: the hydra configuration containing the saving information
        :param verbose: whether to display debug information
        """
        super(StableBaselineCallback, self).__init__(verbose)
        self.save_interval = config.checkpoint.frequency
        self.save_directory = config.agent.checkpoint_dir
        self.initial_n_steps = agent.steps_done
        self.agent = agent
        self.task_name = config.task.name

    def _on_training_start(self) -> None:
        """
        Save the initial model before training starts.
        """
        if self.agent.steps_done == 0:
            self.model.save(f"{self.save_directory}/stable_baseline_model_{self.agent.steps_done}")
            self.agent.save(f"{self.save_directory}/model_{self.agent.steps_done}.pt")

    def _on_step(self) -> bool:
        """
        Save the model if the current time step is a multiple of the saving interval
        :return: True, to keep the training going
        """

        # Save the agent, if required.
        n_steps = self.initial_n_steps + self.n_calls
        if n_steps % self.save_interval == 0:
            self.agent.steps_done = n_steps
            self.model.save(join(self.save_directory, f"stable_baseline_model_{n_steps}"))
            self.agent.save(join(self.save_directory,f"model_{n_steps}.pt"))

        # Report episodic mean reward to ray tune.
        self.agent.store_episode_info([info for info in self.locals['infos'] if 'episode' in info.keys()])
        ep_rew_mean = self.agent.compute_mean_episodic_reward()
        if self.task_name == "hyperparameter_tuning" and ep_rew_mean is not None:
            ray.train.report({"loss": ep_rew_mean})

        return True
