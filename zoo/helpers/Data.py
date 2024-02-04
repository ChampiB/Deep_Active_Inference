import numpy as np
from zoo.agents.memory.ReplayBuffer import ReplayBuffer, Experience
from torch import nn
from zoo.agents.layers.DiagonalGaussian import DiagonalGaussian
import logging


class Data:

    @staticmethod
    def get_batch(batch_size, env, capacity=50000):
        """
        Collect a batch from the environment.
        :param batch_size: the size of the batch to be generated.
        :param env: the environment from which the samples need to be generated.
        :param capacity: the maximum capacity of the queue.
        :return: the generated batch.
        """

        # Create a replay buffer.
        buffer = ReplayBuffer(capacity=capacity)

        # Generates some experiences.
        for i in range(0, capacity):
            obs = env.reset()
            action = np.random.choice(env.action_space.N)
            next_obs, reward, done, _ = env.step(action)
            buffer.append(Experience(obs, action, reward, done, next_obs))

        # Sample a batch from the replay buffer.
        return buffer.sample(batch_size)

    @staticmethod
    def get_activation(name, activation):
        def hook(model, x, output):
            activation[name] = output.detach()
        return hook

    @staticmethod
    def get_activations(data, model, log_var_only=False):
        """
        Load a model and generate a dictionary of the activations obtained from `data`.
        We assume that the activations of each layer are exposed.
        :param np.array data: A (n_examples, n_features) data matrix
        :param model: The model to use
        :param log_var_only: If True, only return the activations of the log variance layers of model
        :return: A tuple containing the loaded model, list of activations, and list of layer names.
        """
        activations = {}
        hooks = []
        layers_info = Data.select_and_get_layers(model, log_var_only)

        # Register forward hooks to get the activations of all the layers
        for name, layer in layers_info:
            hooks.append(layer.register_forward_hook(Data.get_activation(name, activations)))

        model.predict(data)
        logging.debug(f"Activations obtained after prediction: {activations}")

        # Remove the hooks
        for hook in hooks:
            hook.remove()

        return activations

    @staticmethod
    def select_and_get_layers(model, log_var_only=False):
        layers_info = []
        if hasattr(model, "encoder"):
            layers_info += Data.get_layers(list(model.encoder.modules())[-1], "Encoder", log_var_only)[0]
        if hasattr(model, "transition"):
            layers_info += Data.get_layers(list(model.transition.modules())[1], "Transition", log_var_only)[0]
        if hasattr(model, "critic"):
            layers_info += Data.get_layers(list(model.critic.modules())[1], "Critic", log_var_only)[0]
        logging.debug(f"Found layers {layers_info}")
        return layers_info

    @staticmethod
    def get_layers(model, prefix, log_var_only, i=1):
        # Return the variance layer.
        if log_var_only is True:
            return [("{}_variance".format(prefix), list(model.modules())[-1])], i

        # Get the layers at the current level and annotate them with a generic name
        layers_info = []
        for module in model.modules():
            if not isinstance(module, nn.Sequential) and not isinstance(module, DiagonalGaussian):
                layers_info.append((f"{prefix}_{i}", module))
                i += 1
        return layers_info, i
