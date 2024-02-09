from torch.distributions import Dirichlet

from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.inference.GaussianMixture import GaussianMixture
from zoo.agents.inference.HierarchicalGM import HierarchicalGM
from zoo.agents.planning.MCTS import MCTS
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.KMeans import KMeans
from zoo.helpers.MatPlotLib import MatPlotLib


class AIGM(AgentInterface):
    """
    Implement a Temporal Model with Hierarchical Gaussian Mixture likelihood and Dirichlet prior.
    This model takes action based on the risk over states.
    """

    def __init__(self, n_states, n_observations, n_actions, action_selection, **_):
        """
        Constructor
        :param n_actions: the number of actions
        :param n_states: the number of latent states, i.e., number of components in the mixture
        :param n_observations: the number of observations
        :param action_selection: the action selection to be used
        """

        # Call parent constructor.
        super().__init__("", 0)

        # Store the number of states, observations, and actions.
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions

        # The planning algorithm.
        self.mcts = MCTS(0.5, 100, n_actions, self.predict_next_state, self.efe)

        # Attributes related to the target distribution.
        self.target_temperature = 7
        self.target_state = None

        # The number of iterations between two learning iterations.
        self.learning_interval = 100

        # Create the class to use for action selection.
        self.action_selection = action_selection

        # The dataset used for training.
        self.x = []
        self.a = []
        self.done = []

        # Store the prior and posterior parameters of the transition model.
        self.b = torch.ones([n_actions, n_states, n_states]) * 0.2
        self.b_hat = torch.ones([n_actions, n_states, n_states]) * 0.2

        # Store the mean transition matrix.
        self.B = self.compute_B()

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
        self.done.append(False)

        # Train the agent.
        logging.info("Start the training at {time}".format(time=datetime.now()))
        i = 0
        while self.steps_done < config.task.max_n_steps:

            print(i)

            # Select an action.
            action = self.step(obs)
            self.a.append(action)

            # Execute the action in the environment.
            obs, reward, done, info = env.step(action)

            # Perform one iteration of training (if needed).
            if self.steps_done > 0 and self.steps_done % self.learning_interval == 0:
                self.learn(env)

            # Keep track of the last observation and whether the episode ended.
            self.x.append(obs)
            self.done.append(done)

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()
                self.x.append(obs)
                self.done.append(False)

            # Increase the number of steps done.
            self.steps_done += 1
            i += 1

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

        params = (self.gm.m_hat, self.gm.v_hat, self.gm.W_hat)
        MatPlotLib.draw_gm_graph(params, self.gm.x[-100:], self.gm.r_hat[-100:], title="New Data", clusters=False, ellipses=True)

        self.update_for_b()
        if debug is True:
            self.show(env.action_names, f"Transition Model")

        # Update target.
        self.target_state = self.update_target()

    def update_for_b(self):

        # Initialize the parameter of the prior.
        n_states = self.gm.K
        self.b = torch.ones([self.n_actions, n_states, n_states]) * 0.2

        # Collect the responsibilities and actions.
        r0, _, r1, _, a0 = self.get_training_data()
        a0 = torch.nn.functional.one_hot(torch.tensor(a0), num_classes=self.n_actions)

        # Compute the posterior parameters, and the average B matrices.
        self.b_hat = self.b + torch.einsum("na, nj, nk -> ajk", a0, r1, r0)
        self.B = self.compute_B()

    def get_training_data(self):
        keep = torch.logical_not(torch.tensor(self.done))
        x0 = self.gm.x[keep][:-1]
        r0 = self.gm.r_hat[keep][:-1]

        keep = torch.logical_not(torch.tensor([True] + self.done[:-1]))
        x1 = self.gm.x[keep]
        r1 = self.gm.r_hat[keep]

        a0 = self.a[:-1]
        return r0, x0, r1, x1, a0

    def compute_B(self):
        b = self.b_hat
        b_sum = b.sum(dim=1).unsqueeze(dim=1).repeat((1, b.shape[1], 1))
        b /= b_sum
        return b

    def predict_next_state(self, node, action):

        # Predict the next state.
        next_state = torch.matmul(self.B[action], node.state.squeeze())

        # Collect the responsibilities and actions.
        action = torch.nn.functional.one_hot(torch.tensor(action), num_classes=self.n_actions)

        # Compute the new posterior parameters.
        new_b_hat = node.b + torch.einsum("a, j, k -> ajk", action, next_state, node.state.squeeze())

        return next_state, new_b_hat

    def efe(self, node):
        return 0

        # TODO risk_over_states = node.state * (node.state.log() - self.target_state.log())
        # TODO return risk_over_states.sum()

        # KL divergence between two Dirichlet distributions (sampling solution)
        # TODO n_states = node.b.shape[1]
        # TODO novelty = 0
        # TODO for action in range(node.b.shape[0]):
        # TODO     for state in range(n_states):
        # TODO         d1 = Dirichlet(node.b[action][:][state])
        # TODO         samples = d1.sample()
        # TODO         d2 = Dirichlet(self.b[action][:][state])
        # TODO         novelty += d1.log_prob(samples) - d2.log_prob(samples))

        # KL divergence between two Dirichlet distributions (analytic solution)
        # TODO n_states = node.b.shape[1]
        # TODO novelty = 0
        # TODO for action in range(node.b.shape[0]):
        # TODO     for state in range(n_states):
        # TODO         novelty += torch.lgamma(sum(node.b[action][next_state][state] for next_state in range(n_states)))
        # TODO         novelty -= torch.lgamma(sum(self.b_hat[action][next_state][state] for next_state in range(n_states)))
        # TODO         digamma_sum = torch.digamma(sum(node.b[action][next_state][state] for next_state in range(n_states)))
        # TODO         for next_state in range(n_states):
        # TODO             novelty += torch.lgamma(self.b_hat[action][next_state][state])
        # TODO             novelty -= torch.lgamma(node.b[action][next_state][state])
        # TODO             novelty += (node.b[action][next_state][state] - self.b_hat[action][next_state][state]) * \
        # TODO                        (torch.digamma(node.b[action][next_state][state]) - digamma_sum)
        # TODO return novelty

    def update_target(self):
        return torch.softmax(- self.target_temperature * self.gm.Ns / self.gm.Ns.sum(), dim=0)

    def show(self, action_names, title=""):
        r0, x0, r1, x1, a0 = self.get_training_data()
        r = [r0, r1]
        params = (self.gm.m_hat, self.gm.v_hat, self.gm.W_hat)
        MatPlotLib.draw_dirichlet_tmhgm_graph(action_names, params, params, x0, x1, a0, r, self.b_hat, title=title)

    def step(self, obs):
        """
        Select an action
        :param obs: the input observation from which decision should be made
        :return: the action
        """

        # Perform a random action.
        if self.gm is None:
            return self.action_selection.select(torch.zeros([1, self.n_actions]), self.steps_done)

        # Perform inference.
        state = self.gm.compute_responsibility(obs.unsqueeze(dim=0))

        # Perform planning.
        quality = self.mcts.step(state, self.b)

        # Select an action.
        return self.action_selection.select(quality, self.steps_done)

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
