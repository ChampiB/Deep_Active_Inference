import shutil
from zoo.tasks.TaskInterface import TaskInterface
from zoo.helpers.Hydra import Hydra
from zoo.helpers.MatPlotLib import MatPlotLib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tkinter as tk


class DrawActionQualities(TaskInterface):
    """
    A class that draw the action qualities as predicted by the critic.
    """

    def __init__(self, name, figure_path, task_data_directory, csv_path, action_names, **_):
        """
        Constructor
        :param name: the task's name
        :param figure_path: the path where the figure needs to be stored
        :param csv_path: the path to the file containing the selected actions in CSV format
        :param task_data_directory: the path to the data directory of the task
        :param action_names: the name of the environment actions
        """

        # Call the parent constructor.
        super().__init__(name)
        Hydra.register_resolvers()

        # Store the task's parameters.
        self.current_index = 0
        self.csv_path = csv_path
        self.qualities = pd.read_csv(self.csv_path)

        # Create the main window.
        self.window = tk.Tk()
        self.window.title("Action Qualities")
        self.window.geometry(self.get_screen_size())

        # Create the buttons to see previous and next action qualities.
        self.prev_btn = tk.Button(self.window, text='Prev', command=self.decrease_index)
        self.prev_btn.grid(row=1, column=0, sticky=tk.NSEW)
        self.next_btn = tk.Button(self.window, text='Next', command=self.increase_index)
        self.next_btn.grid(row=1, column=2, sticky=tk.NSEW)

        # Create the entry displaying the current index.
        self.current_index_var = tk.StringVar(value=str(self.current_index))
        self.current_index_entry = tk.Entry(self.window, textvariable=self.current_index_var)

        # TODO
        self.figure_path = figure_path
        self.task_data_directory = task_data_directory
        self.action_names = action_names

    def run(self, hydra_config):
        """
        Computes the mean and standard deviation of an agent's performance for a specific environment.
        """

        # Set custom color palette
        colors = ["#d65f5f", "#ee854a", "#4878d0", "#6acc64"]
        sns.set_palette(sns.color_palette(colors))
        sns.set_theme(style="whitegrid", palette="muted")

        # Save the figure containing the selected actions.
        plt.savefig(self.figure_path)
        MatPlotLib.close()

        # Clean up the filesystem.
        shutil.rmtree(self.task_data_directory, ignore_errors=True)

    def decrease_index(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = 0
        self.current_index_var.set(value=str(self.current_index))

    def increase_index(self):
        self.current_index += 1
        if self.current_index >= len(self.qualities.index):
            self.current_index = len(self.qualities.index) - 1
        self.current_index_var.set(value=str(self.current_index))

    def get_screen_size(self):
        """
        Getter.
        :return: the screen's size.
        """
        screen_size = str(self.window.winfo_screenwidth() - 85)
        screen_size += "x"
        screen_size += str(self.window.winfo_screenheight() - 75)
        screen_size += "+85+35"
        return screen_size
