import math
from os.path import join
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.GaussianMixture import GaussianMixture
import matplotlib.colors as mcolors
from zoo.helpers.MatPlotLib import MatPlotLib


class DirichletGM(AgentInterface):
    """
    Implement a Gaussian Mixture with Dirichlet prior taking random each action.
    """

    def __init__(
        self, name, tensorboard_dir, checkpoint_dir, action_selection, n_states, dataset_size,
        W=None, m=None, v=None, β=None, d=None, n_observations=2, n_actions=4, steps_done=0, verbose=False,
        learning_step=0, **_
    ):
        """
        Constructor
        :param name: the agent name
        :param action_selection: the action selection to be used
        :param n_actions: the number of actions
        :param n_states: the number of latent states, i.e., number of components in the mixture
        :param n_observations: the number of observations
        :param dataset_size: the size of the dataset
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param checkpoint_dir: the directory in which the agent should be saved
        :param steps_done: the number of training iterations performed to date.
        :param verbose: whether to log weights information such as mean, min and max values of layers' weights
        :param W: the scale matrix of the Wishart prior
        :param v: degree of freedom of the Wishart prior
        :param m: the mean of the prior over μ
        :param β: the scaling coefficient of the precision matrix of the prior over μ
        :param d: the parameters of the Dirichlet prior over the random vector D
        :param learning_step: the number of learning steps performed so far
        """

        # Call parent constructor.
        super().__init__(tensorboard_dir, steps_done)

        # Miscellaneous.
        self.agent_name = name
        self.total_rewards = 0.0
        self.dataset_size = dataset_size
        self.steps_done = steps_done
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = checkpoint_dir
        self.action_selection = action_selection
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_observations = n_observations
        self.verbose = verbose
        self.colors = ['red', 'green', 'blue', 'purple', 'gray', 'pink', 'turquoise', 'orange', 'brown', 'cyan']
        if n_states > 10:
            self.colors = list(mcolors.CSS4_COLORS.keys())

        # The number of learning steps performed so far.
        self.learning_step = learning_step

        # The dataset used for training.
        self.x = []

        # Gaussian mixture prior parameters.
        self.W = [torch.ones([n_observations, n_observations]) if W is None else W[k].cpu() for k in range(n_states)]
        self.m = [torch.zeros([n_observations]) if m is None else m[k].cpu() for k in range(n_states)]
        self.v = (n_observations - 0.99) * torch.ones([n_states]) if v is None else v.cpu()
        self.β = torch.ones([n_states]) if β is None else β.cpu()
        self.d = torch.ones([n_states]) if d is None else d.cpu()

        # Gaussian mixture posterior parameters.
        self.W_hat = [torch.ones([n_observations, n_observations]) for _ in range(n_states)]
        self.m_hat = [torch.ones([n_observations]) for _ in range(n_states)]
        self.v_hat = (n_observations - 0.99) * torch.ones([n_states])
        self.β_hat = torch.ones([n_states])
        self.r_hat = torch.softmax(torch.rand([dataset_size, n_states]), dim=1)
        self.d_hat = torch.ones([n_states])

    def name(self):
        """
        Getter
        :return: the agent's name
        """
        return self.agent_name

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
        return self.total_rewards

    def step(self, obs):
        """
        Select a random action
        :param obs: the input observation from which decision should be made
        :return: the random action
        """

        # Select a random action.
        return self.action_selection.select(torch.zeros([1, self.n_actions]), self.steps_done)

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Train the agent.
        logging.info("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config.task.max_n_steps:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            obs, reward, done, info = env.step(action)

            # Add the new observation to the dataset.
            self.x.append(obs)

            # Perform one iteration of training (if needed).
            if len(self.x) >= self.dataset_size:
                self.learn()

            # Save the agent (if needed).
            if self.steps_done % config.checkpoint.frequency == 0:
                self.save(config)

            # Log the reward (if needed).
            if self.writer is not None:
                self.total_rewards += reward
                if self.steps_done % config.tensorboard.log_interval == 0:
                    self.writer.add_scalar("total_rewards", self.total_rewards, self.steps_done)
                    self.log_episode_info(info, config.task.name)

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of steps done.
            self.steps_done += 1

        # Save the final version of the model.
        self.save(config, final_model=True)

        # Close the environment.
        env.close()

    def learn(self, debug=True, verbose=False):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param debug: whether to display debug information
        :param verbose: whether to display detailed debug information
        """

        if self.learning_step == 0:

            # Initialize the parameter of the Gaussian Mixture using the K-means algorithm.
            self.m, self.m_hat, self.r_hat, self.W_hat, self.W = \
                GaussianMixture.k_means_init([self.x], self.v, self.v_hat)

            # Display the result of the k-means algorithm, if needed.
            if verbose is True:
                MatPlotLib.draw_gm_graph(
                    title="After K-means Optimization", params=(self.m_hat, self.v_hat, self.W_hat),
                    data=self.x, r=self.r_hat, ellipses=False, clusters=True
                )

        # Display the model's beliefs, if needed.
        if debug is True or verbose is True:
            self.draw_beliefs_graphs("Before Optimization")

        # Perform inference.
        for i in range(20):  # TODO implement a better stopping condition

            # Perform the update for the latent variable D, and display the model's beliefs (if needed).
            self.update_for_d()
            if verbose is True:
                self.draw_beliefs_graphs(f"[{i}] After D update")

            # Perform the update for the latent variable Z, and display the model's beliefs (if needed).
            self.update_for_z()
            if verbose is True:
                self.draw_beliefs_graphs(f"[{i}] After Z update")

            # Perform the update for the latent variables μ and Λ, and display the model's beliefs (if needed).
            self.update_for_μ_and_Λ()
            if verbose is True:
                self.draw_beliefs_graphs(f"[{i}] After μ and Λ update")

        # Display the model's beliefs, if needed.
        if debug is True or verbose is True:
            self.draw_beliefs_graphs("After Optimization")

        # Perform empirical Bayes.
        self.W = self.W_hat
        self.m = self.m_hat
        self.v = self.v_hat
        self.β = self.β_hat

        # Clear the dataset and increase learning step.
        self.x.clear()
        self.learning_step += 1

    def draw_beliefs_graphs(self, title):
        params = (self.m_hat, self.v_hat, self.W_hat)
        MatPlotLib.draw_gm_graph(title=title, params=params, data=self.x, r=self.r_hat)
        MatPlotLib.draw_gm_graph(title=title, params=params, data=self.x, r=self.r_hat, ellipses=False)

    def update_for_d(self):
        self.d_hat = self.d + self.r_hat.sum(dim=0)

    def update_for_z(self):

        # Compute the non-normalized state probabilities.
        log_D = GaussianMixture.expected_log_D(self.d, self.dataset_size)
        log_det = GaussianMixture.expected_log_det_Λ(self.v_hat, self.W_hat, self.dataset_size)
        quadratic_form = GaussianMixture.expected_quadratic_form(self.x, self.m_hat, self.β_hat, self.v_hat, self.W_hat)
        log_ρ = torch.zeros([self.dataset_size, self.n_states])
        log_ρ += log_D - self.n_states / 2 * math.log(2 * math.pi) + 0.5 * log_det - 0.5 * quadratic_form

        # Normalize the state probabilities.
        self.r_hat = torch.softmax(log_ρ, dim=1)

    def update_for_μ_and_Λ(self):
        N = torch.sum(self.r_hat, dim=0) + 0.0001
        x_bar = GaussianMixture.gm_x_bar(self.r_hat, self.x, N)

        self.v_hat = self.v + N
        self.β_hat = self.β + N
        self.m_hat = [(self.β[k] * self.m[k] + N[k] * x_bar[k]) / self.β_hat[k] for k in range(self.n_states)]
        self.W_hat = [self.compute_W_hat(N, k, x_bar) for k in range(self.n_states)]

    def compute_W_hat(self, N, k, x_bar):
        W_hat = torch.inverse(self.W[k])
        for n in range(self.dataset_size):
            x = self.x[n] - x_bar[k]
            W_hat += self.r_hat[n][k] * torch.outer(x, x)
        x = x_bar[k] - self.m[k]
        W_hat += (self.β[k] * N[k] / self.β_hat[k]) * torch.outer(x, x)
        return torch.inverse(W_hat)

    def predict(self, data):
        """
        Do one forward pass using the given observations and actions.
        :param data: a tuple containing the observations and actions at time t
        :return: the outputs of the encoder, transition, and critic model
        """
        raise Exception("Function 'predict' not implemented in GM agent.")

    def save(self, config, final_model=False):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param config: the hydra configuration
        :param final_model: True if the model being saved is the final version, False otherwise
        """

        # Create directories and files if they do not exist.
        model_id = config.task.max_n_steps if final_model is True else self.steps_done
        checkpoint_file = join(self.checkpoint_dir, f"model_{model_id}.pt")
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Save the model.
        torch.save({
            "name": self.agent_name,
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
            "n_states": self.n_states,
            "n_observations": self.n_observations,
            "dataset_size": self.dataset_size,
            "n_actions": self.n_actions,
            "steps_done": self.steps_done,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "action_selection": dict(self.action_selection),
            "W": self.W,
            "v": self.v,
            "m": self.m,
            "β": self.β,
            "d": self.d,
            "learning_step": self.learning_step
        }, checkpoint_file)

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
            "learning_step": checkpoint["learning_step"]
        }

    def draw_reconstructed_images(self, env, grid_size):
        """
        Draw the ground truth and reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        :return: the figure containing the images
        """
        raise Exception("Function 'draw_reconstructed_images' not implemented in GM agent.")
