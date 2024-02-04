from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.actions.SelectRandomAction import SelectRandomAction
from zoo.agents.inference.GaussianMixture import GaussianMixture
from zoo.agents.inference.HierarchicalGM import HierarchicalGM
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.KMeans import KMeans


class AIGM(AgentInterface):
    """
    Implement a Temporal Model with Hierarchical Gaussian Mixture likelihood and Dirichlet prior.
    This model takes action based on the risk over states.
    """

    def __init__(self, n_states, n_observations, n_actions, **_):
        """
        Constructor
        :param n_actions: the number of actions
        :param n_states: the number of latent states, i.e., number of components in the mixture
        :param n_observations: the number of observations
        """

        # Call parent constructor.
        super().__init__("", 0)

        # Store the number of states, observations, and actions.
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions

        # The number of iterations between two learning iterations.
        self.learning_interval = 500

        # Create the class to use for action selection.
        self.action_selection = SelectRandomAction()

        # The dataset used for training.
        self.x = []

        # Create variable that will hold the Gaussian mixture.
        self.gm = None

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()
        self.x.append(obs)

        # Train the agent.
        logging.info("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config.task.max_n_steps:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            obs, reward, done, info = env.step(action)
            self.x.append(obs)

            # Perform one iteration of training (if needed).
            if self.steps_done > 0 and self.steps_done % self.learning_interval == 0:
                self.learn(env)

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()
                self.x.append(obs)

            # Increase the number of steps done.
            self.steps_done += 1

        # Close the environment.
        env.close()

    def learn(self, env, debug=True, verbose=False):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param env: the environment on which the agent is trained
        :param debug: whether to display debug information
        :param verbose: whether to display detailed debug information
        """

        # Initialize the Gaussian mixture using k-means, or update the data of the current Gaussian mixture.
        if self.gm is None:
            self.gm = self.initial_gaussian_mixture(self.x)
        else:
            self.gm.x = self.x

        # Perform variational inference.
        self.gm.learn()
        if debug is True:
            self.gm.show(f"Gaussian Mixture with VFE = {self.gm.vfe}")

        self.gm = HierarchicalGM.split_components(self.gm, init_gm=self.initial_gaussian_mixture)
        if debug is True:
            self.gm.show(f"Hierarchical Gaussian Mixture with VFE = {self.gm.vfe}")

        # Use the posterior as an empirical prior.
        # TODO self.gm.use_posterior_as_empirical_prior()

    def step(self, obs):
        """
        Select a random action
        :param obs: the input observation from which decision should be made
        :return: the random action
        """
        return self.action_selection.select(torch.ones([1, self.n_actions]), self.steps_done)

    def initial_gaussian_mixture(self, x):

        # Initialize the degrees of freedom, and the Dirichlet parameters.
        v = (self.n_observations - 0.99) * torch.ones([self.n_states])
        d = torch.ones([self.n_states])
        β = torch.ones([self.n_states])

        # Perform K-means to initialize the parameter of the posterior over latent variables at time step 1.
        μ, r = KMeans.run(x, self.n_states)

        # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
        precision = KMeans.precision(x, r)
        W = [precision[k] / v[k] for k in range(self.n_states)]

        # Create the initial Gaussian mixture.
        return GaussianMixture(x, W, μ, v, β, d, r)

    @staticmethod
    def load_constructor_parameters(tb_dir, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint.
        :param tb_dir: the path of tensorboard directory.
        :param checkpoint: the checkpoint from which to load the parameters.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: a dictionary containing the constructor's parameters.
        """
        return {
            "name": checkpoint["name"],
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "dataset_size": checkpoint["dataset_size"],
            "tensorboard_dir": tb_dir,
            "checkpoint_dir": checkpoint["checkpoint_dir"],
            "steps_done": checkpoint["steps_done"],
            "n_actions": checkpoint["n_actions"],
            "n_states": checkpoint["n_states"],
            "n_observations": checkpoint["n_observations"],
            "W": checkpoint["W"],
            "v": checkpoint["v"],
            "m": checkpoint["m"],
            "β": checkpoint["β"],
            "d": checkpoint["d"],
            "b": checkpoint["b"],
            "learning_step": checkpoint["learning_step"],
            "min_data_points": checkpoint["min_data_points"],
            "max_planning_steps": checkpoint["max_planning_steps"],
            "exp_const": checkpoint["exp_const"]
        }

    def draw_reconstructed_images(self, env, grid_size):
        """
        Draw the ground truth and reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        :return: the figure containing the images
        """
        raise Exception("Function 'draw_reconstructed_images' not implemented in GM agent.")

    def name(self):
        """
        Getter
        :return: the agent's name
        """
        return "aigm"

    def n_steps_done(self):
        """
        Getter
        :return: the number of training steps performed to date
        """
        return self.steps_done

    def total_rewards_obtained(self):
        """
        Getter
        :return: the total number of rewards gathered to date
        """
        return 0
