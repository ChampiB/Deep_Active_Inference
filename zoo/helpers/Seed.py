import random
import numpy as np
import torch


class Seed:
    """
    Class containing useful functions related to the seed used for generating random numbers.
    """

    @staticmethod
    def set(seed):
        """
        Set the seed of all the framework used
        :param seed: the seed to use for generating random numbers
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
