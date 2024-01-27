from zoo.agents.tuning.search_space.SearchSpaceInterface import SearchSpaceInterface
from ray import tune


class Choice(SearchSpaceInterface):
    """
    Class sampling hyperparameter values from a (uniform) categorical distribution over a list of values.
    """

    def __init__(self, name, path, values, **kwargs):
        """
        Constructor
        :param name: the name of the hyperparameter for which the search space is defined
        :param path: the path of the hyperparameter in the hydra configuration
        :param values: the list of values from which the hyperparameter needs to be sampled
        :param kwargs: unused keyword arguments
        """
        super().__init__(name, path, self.sample)
        self.values = values

    def sample(self):
        """
        Sample a hyperparameter value from a (uniform) categorical distribution
        :return: the sampled hyperparameter
        """
        return tune.choice(self.values)
