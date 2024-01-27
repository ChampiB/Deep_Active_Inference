from zoo.helpers.MatPlotLib import MatPlotLib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
from zoo.agents.save.Checkpoint import Checkpoint
from zoo.environments import EnvFactory
from zoo.helpers.Data import Data
import logging
from zoo.tasks.TaskInterface import TaskInterface
from zoo.helpers.Hydra import Hydra


class DrawVarianceDistribution(TaskInterface):
    """
    A class that draws the distribution of variance for an agent on an environment
    """

    def __init__(self, name, save_path, n_samples, agent_tensorboard_dir, agent_path, **_):
        """
        Constructor
        :param name: the task's name
        :param save_path: the path where the figure and csv files should be saved
        :param n_samples: the number of samples to use for drawing the figures
        :param agent_tensorboard_dir: the agent's tensorboard directory
        :param agent_path: the path to the agent's checkpoint
        """

        # Call the parent constructor.
        super().__init__(name)
        Hydra.register_resolvers()

        # Store the task's parameters.
        self.save_path = save_path
        self.n_samples = n_samples
        self.agent_tensorboard_dir = agent_tensorboard_dir
        self.agent_path = agent_path

    def create_figure(self, acts, actions, layer):
        """
        Create the figure representing the distribution of variances
        :param acts: the activation of the variance layers
        :param actions: the actions taken
        :param layer: the layer names
        """

        # Set the figures' style.
        sns.set(rc={'figure.figsize': (10, 10)}, font_scale=3)
        sns.set_style("whitegrid", {'axes.grid': False, 'legend.labelspacing': 1.2})

        # Format the dataframe.
        df = pd.DataFrame(acts.tolist()).add_prefix("Latent variable at index ")
        action_map = {0: "Down", 1: "Up", 2: "Left", 3: "Right"}
        df["Action"] = actions.cpu()
        df["Action"] = df["Action"].replace(action_map)
        df = self.drop_outliers(df)

        # Save the dataframe on the filesystem.
        df.to_csv(f"{self.save_path}_{layer}.tsv", sep="\t", index=False)

        # Draw the distribution of variances.
        for i in range(acts.shape[1]):
            plt.figure()
            sns.histplot(
                data=df, x=f"Latent variable at index {i}", hue="Action", multiple="stack",
                hue_order=action_map.values()
            )
            MatPlotLib.save_figure(f"{self.save_path}_{layer}_latent_{i}.pdf")

    @staticmethod
    def drop_outliers(df, z_thresh=1.5):
        """
        Remove the outliers from the dataframe
        :param df: the pandas dataframe
        :param z_thresh: the z-score threshold at which data points are excluded
        :return: the new dataframe without the outliers
        """

        # This a slightly updated version of https://stackoverflow.com/a/56725366
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', np.number]
        constrains = df.select_dtypes(include=numerics).apply(lambda x: np.abs(stats.zscore(x)) < z_thresh).all(axis=1)
        return df.drop(df.index[~constrains])

    def run(self, hydra_conf):
        """
        Draw the distribution of variance for an agent on an environment
        :param hydra_conf: the hydra configuration
        """

        # Create the environment.
        env = EnvFactory.make(hydra_conf)

        # Sample a batch of experiences.
        samples, actions, rewards, done, next_obs = Data.get_batch(batch_size=self.n_samples, env=env)

        model = Checkpoint(self.agent_tensorboard_dir, self.agent_path).load_agent()

        # Retrieve the activation of the variance layers.
        logging.info("Retrieving layer activations...")
        data = (samples, actions)
        acts = Data.get_activations(data, model, log_var_only=True)
        acts = {k: np.exp(v.cpu()) for k, v in acts.items()}

        # Create the figures representing the distribution of variances.
        for layer, act in acts.items():
            logging.debug(f"Activation shape of {layer}: {act.shape}")
            self.create_figure(act, data[1], layer)
