from os.path import join
from omegaconf import OmegaConf


class HyperparameterTuner:
    """
    Class used to:
    - retrieve the ray tune configuration from the hydra configuration
    - inject specific hyperparameter values into the hydra configuration
    """

    def __init__(self, hyperparameters, **kwargs):
        """
        Constructor
        :param hyperparameters: a list containing a description of the all hyperparameter search_spaces
        :param kwargs: unused keyword arguments
        """
        self.hyperparameters = hyperparameters

    def get_ray_tune_config(self):
        """
        Getter
        :return: the ray tune configuration
        """
        return {hyperparameter.name: hyperparameter.sampling_fc() for hyperparameter in self.hyperparameters}

    def update_hydra_config(self, rt_config, hydra_config):
        """
        Update the hydra configuration according to the hyperparameter values from the ray tune configuration
        :param rt_config: the ray tune configuration
        :param hydra_config: the hydra configuration
        """

        # Update the hyperparameter values based on ray tune suggestions. We ignore index_trial which is only used to
        # obtain several runs with the same set of parameters and is not part of the agent's parameters.
        for hyperparameter in self.hyperparameters:
            if hyperparameter.name != "trial_index":
                OmegaConf.update(hydra_config, hyperparameter.path, rt_config[hyperparameter.name], merge=True)

        # Create the suffix that must be used for logging in tensorboard and for saving/reloading the model
        suffix = [f"{param_name}={param_value:.2e}" for param_name, param_value in rt_config.items()]
        suffix = "_".join(suffix)

        # Update the tensorboard directory that must be used in tensorboard
        new_tensorboard_dir = join(hydra_config.tensorboard.directory, suffix)
        OmegaConf.update(hydra_config, "agent.tensorboard_dir", new_tensorboard_dir, merge=True)

        # Update the checkpoint directory that must be used for saving/reloading the model
        new_checkpoint_dir = join(hydra_config.checkpoint.directory, suffix)
        OmegaConf.update(hydra_config, "agent.checkpoint_dir", new_checkpoint_dir, merge=True)
