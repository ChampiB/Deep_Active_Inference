from zoo.agents.tuning.search_space.SearchSpaceInterface import SearchSpaceInterface
from ray import tune


class LogUniform(SearchSpaceInterface):
    """
    Class sampling hyperparameter values from a log uniform distribution.
    """

    def __init__(self, name, path, lower_bound, upper_bound, **kwargs):
        """
        Constructor
        :param name: the name of the hyperparameter for which the search space is defined
        :param hp_path: the path of the hyperparameter in the hydra configuration
        :param lower_bound: the lower bound of the log uniform from which the hyperparameter needs to be sampled
        :param upper_bound: the upper bound of the log uniform from which the hyperparameter needs to be sampled
        :param kwargs: unused keyword arguments
        """
        super().__init__(name, path, self.sample)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self):
        """
        Sample a hyperparameter value from a log uniform distribution
        :return: the sampled hyperparameter
        """
        return tune.loguniform(self.lower_bound, self.upper_bound)
