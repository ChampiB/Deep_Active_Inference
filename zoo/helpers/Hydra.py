from omegaconf import OmegaConf
from hydra.utils import instantiate


class Hydra:
    """
    Class containing useful functions related to the hydra configuration framework.
    """

    resolvers_have_been_registered = False

    @staticmethod
    def register_resolvers():
        """
        Make hydra able to load tuples, None values, and list of object that needs to be instantiated
        """
        if Hydra.resolvers_have_been_registered is False:
            OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
            OmegaConf.register_new_resolver("instantiate_list", lambda xs: [instantiate(xs[k]) for k in xs.keys()])
            OmegaConf.register_new_resolver("instantiate_model_dirs", Hydra.instantiate_model_dirs)
            Hydra.resolvers_have_been_registered = True

    @staticmethod
    def instantiate_model_dirs(directory, agents):
        """
        Create a list of model directories
        :param directory: the directory in which the models are stored
        :param agents: the agent names
        :return: the list of model directories
        """
        return [directory + "*/"] if len(agents) == 0 else [directory + f"{agent}/" for agent in agents]

    @staticmethod
    def update(config, dictionary):
        """
        Update the hydra configuration using the key-value pairs of the dictionary
        :param config: the configuration that needs to be updated
        :param dictionary: a dictionary containing the key-value pairs to update
        """
        for path, value in dictionary.items():
            OmegaConf.update(config, path, value, merge=True)
