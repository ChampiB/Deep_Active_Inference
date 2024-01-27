import shutil
from zoo.tasks.TaskInterface import TaskInterface
from zoo.helpers.Hydra import Hydra
from zoo.helpers.MatPlotLib import MatPlotLib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class DrawEntropyPriorActions(TaskInterface):
    """
    A class that draw the entropy of the prior over actions.
    """

    def __init__(self, name, max_n_steps, jump, figure_path, task_data_directory, csv_path, **_):
        """
        Constructor
        :param name: the task's name
        :param max_n_steps: the maximum number of training steps ran
        :param jump: the number of training steps between two data points displayed in the graph
        :param figure_path: the path where the figure needs to be stored
        :param csv_path: the path to the file containing the selected actions in CSV format
        :param task_data_directory: the path to the data directory of the task
        """

        # Call the parent constructor.
        super().__init__(name)
        Hydra.register_resolvers()

        # Store the task's parameters.
        self.max_n_steps = max_n_steps
        self.jump = jump
        self.figure_path = figure_path
        self.csv_path = csv_path
        self.task_data_directory = task_data_directory

    def run(self, hydra_config):
        """
        Computes the mean and standard deviation of an agent's performance for a specific environment.
        """

        # Retrieve the data points to be displayed in the graph.
        indices = [i * self.jump for i in range(0, int(self.max_n_steps / self.jump))]
        df = pd.read_csv(self.csv_path)
        df = df.filter(items=indices, axis=0)

        # Set custom color palette
        colors = ["#d65f5f", "#ee854a", "#4878d0", "#6acc64"]
        sns.set_palette(sns.color_palette(colors))
        sns.set_theme(style="whitegrid", palette="muted")

        # Draw a categorical scatter plot to show each observation
        ax = sns.lineplot(data=df, x="Training iterations", y="Entropy")
        ax.set(ylabel="Entropy of prior over actions")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # Save the figure containing the selected actions.
        plt.savefig(self.figure_path)
        MatPlotLib.close()

        # Clean up the filesystem.
        shutil.rmtree(self.task_data_directory, ignore_errors=True)
