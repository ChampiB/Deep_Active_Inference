import math
from os import listdir, makedirs
from os.path import isfile, isdir, join, exists, dirname
from pathlib import Path
from zoo.helpers.Device import Device
import logging
import importlib
import torch
import re


class Checkpoint:
    """
    Class allowing the loading of model checkpoints.
    """

    def __init__(self, tb_dir, directory):
        """
        Construct the checkpoint from the checkpoint file
        :param tb_dir: the path of tensorboard directory
        :param directory: the checkpoint directory
        """

        # Get the latest checkpoint file
        checkpoint_file = self.get_latest_checkpoint(directory)
        if checkpoint_file is None:
            self.checkpoint = None
            return

        # Load checkpoint from path.
        self.checkpoint = torch.load(checkpoint_file, map_location=Device.get())
        if "checkpoint_dir" in self.checkpoint and self.checkpoint["checkpoint_dir"] != directory:
            self.checkpoint["checkpoint_dir"] = directory
            logging.info("The given checkpoint directory does not match the one found in the file.")
            logging.info("Overwriting checkpoint directory to: " + directory)

        # Store the tensorboard directory
        self.tb_dir = tb_dir
        if "tensorboard_dir" in self.checkpoint and self.checkpoint["tensorboard_dir"] != directory:
            self.checkpoint["tensorboard_dir"] = directory
            logging.info("The given tensorboard directory does not match the one found in the file.")
            logging.info("Overwriting tensorboard directory to: " + directory)

    @staticmethod
    def get_latest_checkpoint(directory, regex=r'model_\d+.pt'):
        """
        Get the latest checkpoint file matching the regex
        :param directory: the checkpoint directory
        :param regex: the regex checking whether a file name is a valid checkpoint file
        :return: None if an error occurred, else the path to the latest checkpoint
        """
        # If the path is not a directory or does not exist, return without trying to load the checkpoint.
        if not exists(directory) or not isdir(directory):
            logging.warning("The following directory was not found: " + directory)
            return None

        # If the directory does not contain any files, return without trying to load the checkpoint.
        files = [file for file in listdir(directory) if isfile(join(directory, file))]
        if len(files) == 0:
            logging.warning("No checkpoint found in directory: " + directory)
            return None

        # Retrieve the file whose name contain the largest number.
        # This number is assumed to be the time step at which the agent was saved.
        max_number = - math.inf
        file = None
        for curr_file in files:
            # Retrieve the number of training steps of the current checkpoint file.
            if len(re.findall(regex, curr_file)) == 0:
                continue
            current_number = max([int(number) for number in re.findall(r'\d+', curr_file)])

            # Remember the checkpoint file with the highest number of training steps.
            if current_number > max_number:
                max_number = current_number
                file = join(directory, curr_file)

        logging.info("Loading agent from the following file: " + file)
        return file

    def exists(self):
        """
        Check whether the checkpoint file exists.
        :return: True if the checkpoint file exists, False otherwise.
        """
        return self.checkpoint is not None

    def load_agent(self, training_mode=True, override=None):
        """
        Load the agent from the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :param override: the key-value pairs that need to be overridden in the checkpoint
        :return: the loaded agent or None if an error occurred.
        """

        # Check if the checkpoint is loadable.
        if not self.exists():
            return None

        # Override key-value pairs in the checkpoint if needed.
        if override is not None:
            for key, value in override.items():
                self.checkpoint[key] = value

        # Load the agent class and module.
        agent_module = importlib.import_module(self.checkpoint["agent_module"])
        agent_class = getattr(agent_module, self.checkpoint["agent_class"])

        # Load the parameters of the constructor from the checkpoint.
        param = agent_class.load_constructor_parameters(self.tb_dir, self.checkpoint, training_mode)

        # Instantiate the agent.
        return agent_class(**param)

    @staticmethod
    def create_dir_and_file(checkpoint_file):
        """
        Create the directory and file of the checkpoint if they do not already exist
        :param checkpoint_file: the checkpoint file
        """
        checkpoint_dir = dirname(checkpoint_file)
        if not exists(checkpoint_dir):
            makedirs(checkpoint_dir)
            file = Path(checkpoint_file)
            file.touch(exist_ok=True)

    @staticmethod
    def set_training_mode(neural_net, training_mode):
        """
        Set the training mode of the neural network sent as parameters
        :param neural_net: the neural network whose training mode needs to be set
        :param training_mode: True if the agent is being loaded for training, False otherwise
        """
        if training_mode:
            neural_net.train()
        else:
            neural_net.eval()

    @staticmethod
    def load_value_network(checkpoint, training_mode, prefix=""):
        """
        Load the value network from the checkpoint passed as parameters
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param prefix: the value network prefix
        :return: the value network
        """

        # Load value network.
        value_net_module = importlib.import_module(checkpoint[f"value_net{prefix}_module"])
        value_net_class = getattr(value_net_module, checkpoint[f"value_net{prefix}_class"])
        value_net = value_net_class(n_actions=checkpoint["n_actions"], n_states=checkpoint["n_states"])
        value_net.load_state_dict(checkpoint[f"value_net{prefix}_state_dict"])

        # Set the training mode of the value network.
        Checkpoint.set_training_mode(value_net, training_mode)
        return value_net

    @staticmethod
    def load_value_networks(checkpoint, training_mode, n_value_networks):
        """
        Load the value networks from the checkpoint passed as parameters
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param n_value_networks: the number of value networks to load
        :return: the value networks
        """
        return [
            Checkpoint.load_value_network(checkpoint, training_mode, prefix=f"_{i}")
            for i in range(n_value_networks)
        ]

    @staticmethod
    def load_decoder(checkpoint, training_mode=True):
        """
        Load the decoder from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the decoder
        """

        # Load number of states and the image shape.
        image_shape = checkpoint["image_shape"]
        n_states = checkpoint["n_states"]

        # Load decoder network.
        decoder_module = importlib.import_module(checkpoint["decoder_net_module"])
        decoder_class = getattr(decoder_module, checkpoint["decoder_net_class"])
        decoder = decoder_class(n_states=n_states, image_shape=image_shape)
        decoder.load_state_dict(checkpoint["decoder_net_state_dict"])

        # Set the training mode of the decoder.
        Checkpoint.set_training_mode(decoder, training_mode)
        return decoder

    @staticmethod
    def load_encoder(checkpoint, training_mode=True):
        """
        Load the encoder from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the encoder
        """

        # Load number of states and the image shape.
        image_shape = checkpoint["image_shape"]
        n_states = checkpoint["n_states"]

        # Load encoder network.
        encoder_module = importlib.import_module(checkpoint["encoder_net_module"])
        encoder_class = getattr(encoder_module, checkpoint["encoder_net_class"])
        encoder = encoder_class(n_states=n_states, image_shape=image_shape)
        encoder.load_state_dict(checkpoint["encoder_net_state_dict"])

        # Set the training mode of the encoder.
        Checkpoint.set_training_mode(encoder, training_mode)
        return encoder

    @staticmethod
    def load_transition(checkpoint, training_mode=True):
        """
        Load the transition from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the transition
        """

        # Load transition network.
        transition_module = importlib.import_module(checkpoint["transition_net_module"])
        transition_class = getattr(transition_module, checkpoint["transition_net_class"])
        transition = transition_class(
            n_states=checkpoint["n_states"], n_actions=checkpoint["n_actions"]
        )
        transition.load_state_dict(checkpoint["transition_net_state_dict"])

        # Set the training mode of the transition.
        Checkpoint.set_training_mode(transition, training_mode)
        return transition

    @staticmethod
    def load_critic(checkpoint, training_mode=True, n_states_key="n_states", network_key="critic_net"):
        """
        Load the critic from the checkpoint
        :param checkpoint: the checkpoint
        :param n_states_key: the key of the dictionary containing the number of states
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param network_key: the prefix of the keys containing the critic's module and class
        :return: the critic.
        """
        # Check validity of inputs
        if network_key + '_module' not in checkpoint.keys() or network_key + '_class' not in checkpoint.keys():
            return None

        # Load critic network.
        critic_module = importlib.import_module(checkpoint[network_key + "_module"])
        critic_class = getattr(critic_module, checkpoint[network_key + "_class"])
        critic = critic_class(
            n_states=checkpoint[n_states_key], n_actions=checkpoint["n_actions"]
        )
        critic.load_state_dict(checkpoint[network_key + "_state_dict"])

        # Set the training mode of the critic.
        Checkpoint.set_training_mode(critic, training_mode)
        return critic

    @staticmethod
    def load_value(checkpoint, training_mode=True, n_states_key="n_states", network_key="value_net"):
        """
        Load the value from the checkpoint
        :param checkpoint: the checkpoint
        :param n_states_key: the key of the dictionary containing the number of states
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param network_key: the prefix of the keys containing the value's module and class
        :return: the value.
        """
        # Check validity of inputs
        if network_key + '_module' not in checkpoint.keys() or network_key + '_class' not in checkpoint.keys():
            return None

        # Load value network.
        value_module = importlib.import_module(checkpoint[network_key + "_module"])
        value_class = getattr(value_module, checkpoint[network_key + "_class"])
        value = value_class(
            n_states=checkpoint[n_states_key], n_actions=checkpoint["n_actions"]
        )
        value.load_state_dict(checkpoint[network_key + "_state_dict"])

        # Set the training mode of the value.
        Checkpoint.set_training_mode(value, training_mode)
        return value

    @staticmethod
    def load_object_from_dictionary(checkpoint, key):
        """
        Load an object from the checkpoint passed as parameters
        :param checkpoint: the checkpoint
        :param key: the key in the dictionary where the object has been serialized
        :return: the loaded object
        """

        # Load the object from the checkpoint.
        obj = checkpoint[key]
        obj_module = importlib.import_module(obj["module"])
        obj_class = getattr(obj_module, obj["class"])
        return obj_class(**obj)
