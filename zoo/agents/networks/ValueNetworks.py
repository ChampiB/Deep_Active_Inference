from torch import nn, zeros
from math import prod


#
# Class implementing a network modeling the cost of each action given a state.
#
class Convolutional64(nn.Module):

    def __init__(self, image_shape, n_actions, **_):
        """
        Constructor.
        :param image_shape: the shape of the input images.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the convolutional encoder network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, (4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear encoder network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        # Create the full encoder network.
        self.__net = nn.Sequential(
            self.__conv_net,
            self.__linear_net
        )

    def __conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image.
        :return: the shape of the features output by the convolutional encoder.
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = zeros(image_shape)
        return self.__conv_net(input_image).shape

    def forward(self, x):
        """
        Forward pass through the value network.
        :param x: the input.
        :return: the cost of performing each action in that state.
        """
        x = x.permute(0, 3, 1, 2)
        return self.__net(x)


#
# Class implementing a network modeling the cost of each action given a state.
#
class DualLinearRelu4x100(nn.Module):

    def __init__(self, n_states, n_actions, **_):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the value network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2 * n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the value network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)


#
# Class implementing a network modeling the cost of each action given a state.
#
class LinearRelu4x100(nn.Module):

    def __init__(self, n_states, n_actions, **_):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the value network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the value network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)


#
# Class implementing a network modeling the cost of each action given a state.
#
class LinearRelu4x256(nn.Module):

    def __init__(self, n_states, n_actions, **_):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the value network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the value network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)


#
# Class implementing a network modeling the cost of each action given a state.
#
class LinearReluDropout4x100(nn.Module):

    def __init__(self, n_states, n_actions, **_):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the value network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the value network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)


#
# Class implementing a network modeling the cost of each action given a state.
#
class LinearRelu3x128(nn.Module):

    def __init__(self, n_states, n_actions, **_):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the value network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the value network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)
