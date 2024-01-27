class SearchSpaceInterface:
    """
    An interface that all (hyperparameter) search spaces must implement.
    """

    def __init__(self, hp_name, hp_path, sampling_function):
        """
        Constructor
        :param hp_name: the name of the hyperparameter for which the search space is defined
        :param hp_path: the path of the hyperparameter in the hydra configuration
        :param sampling_function: the function from which the hyperparameter must be sampled
        """
        self.hp_name = hp_name
        self.hp_path = hp_path
        self.sampling_function = sampling_function

    @property
    def name(self):
        """
        Getter
        :return: the name of the hyperparameter for which the search space is defined
        """
        return self.hp_name

    @property
    def path(self):
        """
        Getter
        :return: the path of the hyperparameter in the hydra configuration
        """
        return self.hp_path

    @property
    def sampling_fc(self):
        """
        Getter
        :return: the function from which the hyperparameter must be sampled
        """
        return self.sampling_function
