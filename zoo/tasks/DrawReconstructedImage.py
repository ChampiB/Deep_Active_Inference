import shutil
from zoo.agents import AgentFactory
from zoo.environments import EnvFactory
from zoo.helpers.Seed import Seed
from zoo.tasks.TaskInterface import TaskInterface
from zoo.helpers.Hydra import Hydra
from zoo.helpers.MatPlotLib import MatPlotLib


class DrawReconstructedImage(TaskInterface):
    """
    A class that computes the mean and standard deviation of the agents performance across several seeds for a specific
    environment, and display them as a graph.
    """

    def __init__(self, name, grid_size, seed, figure_path, task_data_directory, **_):
        """
        Constructor
        :param name: the task's name
        :param seed: the random seed to use
        :param grid_size: the size of the grid to display
        :param figure_path: the path where the figure needs to be stored
        :param task_data_directory: the path to the data directory of the task
        """

        # Call the parent constructor.
        super().__init__(name)
        Hydra.register_resolvers()

        # Store the task's parameters.
        self.seed = seed
        self.grid_size = grid_size
        self.figure_path = figure_path
        self.task_data_directory = task_data_directory

    def run(self, hydra_config):
        """
        Computes the mean and standard deviation of an agent's performance for a specific environment.
        """

        # Update checkpoint directory.
        checkpoint_dir = hydra_config.agent.checkpoint_dir
        hydra_config.agent.checkpoint_dir = checkpoint_dir.replace("draw_reconstructed_images", "training")

        # Set the seed requested by the user.
        Seed.set(self.seed)

        # Create the environment and agent.
        env = EnvFactory.make(hydra_config)
        agent = AgentFactory.make(hydra_config, env)
        agent.writer = None

        # Retrieve the reconstructed images.
        fig = agent.draw_reconstructed_images(env, self.grid_size)

        # Save the figure containing the ground truth and reconstructed images.
        fig.savefig(self.figure_path)
        MatPlotLib.close()

        # Clean up the filesystem.
        shutil.rmtree(self.task_data_directory, ignore_errors=True)
