import glob
import os.path
import re
import logging
import matplotlib.pyplot as plt
from zoo.helpers.FileSystem import FileSystem
from zoo.helpers.TensorBoard import TensorBoard
from zoo.tasks.TaskInterface import TaskInterface
from zoo.helpers.Hydra import Hydra
import pandas as pd
import seaborn as sns
from zoo.helpers.MatPlotLib import MatPlotLib


class DrawPerformanceGraph(TaskInterface):
    """
    A class that computes the mean and standard deviation of the agents performance across several seeds for a specific
    environment, and display them as a graph.
    """

    def __init__(
        self, name, min_n_steps, max_n_steps, max_y_axis, model_dirs, environment_name, seed, monitored_quantity,
        figure_path, df_path_prefix, overwrite, task_directory, jump, **_
    ):
        """
        Constructor
        :param name: the task's name
        :param min_n_steps: the minimum number of training steps after which the plot starts
        :param max_n_steps: the maximum number of training steps that will be run
        :param max_y_axis: the maximum value to display on the y-axis
        :param model_dirs: list of the model directories where the agent have been monitored
        :param environment_name: the environment name
        :param seed: the seed
        :param monitored_quantity: the quantity to that needs to be monitored, i.e., vfe or reward
        :param figure_path: the path where the figure needs to be stored
        :param df_path_prefix: the prefix of the path where the dataframe of tensorboard scores needs to be stored
        :param task_directory: either training or hyperparameter tuning
        :param overwrite: if true, the dataframe are recomputed, otherwise existing dataframes are reused
        :param jump: the number of training steps between two data points displayed in the graph
        """

        # Call the parent constructor.
        super().__init__(name)
        Hydra.register_resolvers()

        # Store the task's parameters.
        self.max_n_steps = max_n_steps
        self.min_n_steps = min_n_steps
        self.jump = jump
        self.figure_path = figure_path
        self.df_path_prefix = df_path_prefix
        self.overwrite = overwrite
        self.task_directory = task_directory
        self.seed = seed
        self.environment_name = environment_name
        self.max_y_axis = max_y_axis
        event_names_map = {
            "reward": ("Rewards", "rollout/ep_rew_mean"),
            "vfe": ("Variational Free Energy", "vfe"),
            "total_reward": ("Total Rewards", "total_rewards"),
        }
        self.values_name, self.scalar_name = event_names_map[monitored_quantity]

        # model_dirs is either a list of specific (environment,agent) to monitor or a starred expression.
        # For example, model_dirs could be ["run/d_sprites/baseline_dqn", "run/d_sprites/baseline_ppo"], or
        # ["run/d_sprites/*/"], where the latest will select all the runs logged for the d_sprites environment.
        self.model_dirs = model_dirs
        if len(model_dirs) == 1:
            self.model_dirs = glob.glob(model_dirs[0])

    @staticmethod
    def try_int(s):
        """
        If the given string contains an integer, convert it to an integer, else return a string
        :param s: the string to process
        :return: the converted integer if s only contained decimal characters, otherwise the initial string
        """
        try:
            return int(s)
        except ValueError:
            return s

    @staticmethod
    def natural_sort(s):
        """
        Turn a string into a list of string and number chunks, e.g., natural_sort("z23a") -> ["z", 23, "a"]
        :param s: the string to process
        :return: a list of string and number chunks
        """
        return [DrawPerformanceGraph.try_int(c) for c in re.split('([0-9]+)', s)]

    @staticmethod
    def format_label(label):
        """
        Format the label passed as input to make pretty in the graph
        :return: the formatted label
        """

        # Format baseline agent names.
        if "baseline_" in label:
            label = label.replace("baseline_", "").upper()

        # Format CHMM agent names.
        if "chmm" in label:
            label = label.replace("chmm", "CHMM")
            if "%" in label:
                label = label.replace("_efe_epsilon_greedy_", "[")
            if "_efe_" in label:
                label = label.replace("_efe_", "[efe,")
            if "_reward_" in label:
                label = label.replace("_reward_", "[reward,")
            label += "]"

        # Format VAE and HMM agent names.
        if label in ["vae", "hmm"]:
            label = label.upper()
        return label

    def run(self, *args, **kwargs):
        """
        Computes the mean and standard deviation of an agent's performance for a specific environment.
        """

        # Create the matplotlib axis, and the labels for the legend.
        ax = None

        # Format the agent names.
        labels = [self.format_label(path.split("/")[-2]) for path in self.model_dirs]

        # Draw the performance of all the models.
        for model_dir in self.model_dirs:

            # Draw the performance of the model described by the path.
            logging.info(f"Plotting curve for {model_dir}")
            ax = self.draw_model_performance(model_dir, ax)

        # Set the legend of the figure, and the axis labels with labels sorted in natural order.
        handles, labels = list(zip(*sorted(zip(*[ax.lines, labels]), key=lambda x: self.natural_sort(x[1]))))
        ax.legend(handles=handles, labels=labels)
        ax.set_xlabel("Training Iterations")
        ax.set_ylabel(self.values_name)

        # Save the full figure, comparing the agents.
        plt.tight_layout()
        plt.savefig(self.figure_path, dpi=300, transparent=True)
        MatPlotLib.close()

    def draw_model_performance(self, model_dir, ax):
        """
        Draw the performance of the model
        :param model_dir: the path to the directory containing the training logs of the model
        :param ax: the axis on which to draw the model performance
        :return: the new axis to use for drawing the next model performance
        """

        # Retrieve the path to the dataframe where the tensorboard scores should be saved.
        agent_name = model_dir.split('/')[-2]
        df_path = f"{self.df_path_prefix}_{agent_name}.tsv"

        # Load all the log data from the file system.
        if os.path.exists(df_path) and self.overwrite is False:
            logging.info(f"{df_path} exists, using already computed logs from tensorboard.")
            rewards = pd.read_csv(df_path, sep="\t")
        else:
            logging.info(f"{df_path} not found, getting logs from tensorboard.")
            directories = self.get_model_directories(model_dir, agent_name)
            rewards = TensorBoard.load_log_directories(directories, df_path, self.values_name, self.scalar_name)

        # Return if the dataframe is empty.
        if rewards.empty:
            return ax

        # Filter only the relevant rewards and group them by training iteration.
        rewards = rewards[rewards[self.values_name] <= self.max_y_axis]
        rewards = rewards[rewards["Steps"] >= self.min_n_steps]
        rewards = rewards[rewards["Steps"] <= self.max_n_steps]
        agg_rewards = rewards.groupby("Steps", as_index=False)

        # Compute the lower and upper bound based on mean and standard deviation.
        mean_rewards = agg_rewards.mean()
        logging.info(mean_rewards)
        mean_rewards = mean_rewards[mean_rewards.index % self.jump == 0]
        logging.info(mean_rewards)

        std_rewards = agg_rewards.std()
        std_rewards = std_rewards[std_rewards.index % self.jump == 0]

        lower_bound = mean_rewards - std_rewards
        upper_bound = mean_rewards + std_rewards

        # Draw the mean reward as a solid curve, and the standard deviation as the shaded area.
        ax = sns.lineplot(mean_rewards, x="Steps", y=self.values_name, ax=ax)
        plt.fill_between(
            mean_rewards.Steps.unique(),
            lower_bound[self.values_name].values,
            upper_bound[self.values_name].values,
            alpha=0.1
        )
        return ax

    def get_model_directories(self, model_dir, agent_name):
        """
        Get the model directories accounting for the task used for training the agent
        :param model_dir: the directory in which all the models are stored
        :param agent_name: the agent name
        :return: the list of directories whose tensorboard event file must be loaded
        """

        # Update the model directory, if the task is hyperparameter tuning.
        if self.task_directory == "hyperparameter_tuning":

            # Get the list of directories present in the model directory.
            sub_directories = FileSystem.sub_directories_of(f"{model_dir}/{self.seed}/")

            # Load the file containing the values of the best hyperparameters.
            filename = f"{model_dir}/{self.seed}/best_parameters_{self.environment_name}_{agent_name}_{self.seed}.tsv"
            best_hp_params = pd.read_csv(filename, sep="\t")

            # Retrieve the parameters names and values.
            best_hp_params = best_hp_params.drop(columns=["loss"])
            hp_param_names = best_hp_params.columns.values.tolist()
            hp_param_values = best_hp_params.values.tolist()[0]

            # Filter only the directory matching the hyperparameter values.
            for name, value in zip(hp_param_names, hp_param_values):
                sub_directories = list(filter(lambda directory: f"{name}={value:.2e}" in directory, sub_directories))

            return [f"{model_dir}/{self.seed}/{sub_directory}" for sub_directory in sub_directories]

        return glob.glob(f"{model_dir}/*/")
