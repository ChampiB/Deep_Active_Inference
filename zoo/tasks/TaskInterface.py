import abc


class TaskInterface:
    """
    A class that represents a runnable task such as: training, testing, hyperparameter tuning, etc...
    """

    def __init__(self, name):
        """
        Constructor
        :param name: the task's name
        """
        self.name = name

    @abc.abstractmethod
    def run(self, hydra_config):
        """
        Run the task
        :param hydra_config: the hydra configuration of the task
        """
        ...
