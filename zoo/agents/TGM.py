import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from os.path import join
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.KMeans import KMeans


class TGM(AgentInterface):
    """
    Implement a Temporal Gaussian Mixture taking random each action.
    """

    def __init__(
        self, name, discount_factor, tensorboard_dir, checkpoint_dir, action_selection, n_states, dataset_size,
        W=None, m=None, v=None, β=None, D=None, n_observations=2, n_actions=4, steps_done=0, verbose=False,
        learning_step=0, **_
    ):
        """
        Constructor
        :param name: the agent name
        :param action_selection: the action selection to be used
        :param n_actions: the number of actions
        :param n_states: the number of latent states, i.e., number of components in the mixture
        :param n_observations: the number of observations
        :param discount_factor: the factor by which the future EFE is discounted
        :param dataset_size: the size of the dataset
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param checkpoint_dir: the directory in which the agent should be saved
        :param steps_done: the number of training iterations performed to date.
        :param verbose: whether to log weights information such as mean, min and max values of layers' weights
        :param W: the scale matrix of the Wishart prior
        :param v: degree of freedom of the Wishart prior
        :param m: the mean of the prior over μ
        :param β: the scaling coefficient of the precision matrix of the prior over μ
        :param D: the prior probability of the Gaussian components
        :param learning_step: the number of learning steps performed so far
        """

        # Call parent constructor.
        super().__init__(tensorboard_dir, steps_done)

        # Miscellaneous.
        self.agent_name = name
        self.total_rewards = 0.0
        self.discount_factor = discount_factor
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

        # The number of learning steps performed so far.
        self.learning_step = learning_step

        # The dataset used for training.
        self.x = []

        # Gaussian mixture prior parameters.
        self.W = [self.random_psd_matrix([n_observations, n_observations]) for _ in range(n_states)] if W is None else W
        self.m = [torch.zeros([n_observations]) for _ in range(n_states)] if m is None else m
        self.v = (n_observations - 0.99) * torch.ones([n_states]) if v is None else v
        self.β = torch.ones([n_states]) if β is None else β
        self.D = torch.ones([n_states]) / n_states if D is None else D

        # Gaussian mixture posterior parameters.
        self.W_hat = [self.random_psd_matrix([n_observations, n_observations]) for _ in range(n_states)]
        self.m_hat = [torch.ones([n_observations]) for _ in range(n_states)]
        self.v_hat = (n_observations - 0.99) * torch.ones([n_states])
        self.β_hat = torch.abs(torch.rand([n_states]))
        self.r_hat = torch.softmax(torch.rand([dataset_size, n_states]), dim=1)

    @staticmethod
    def random_psd_matrix(shape):
        """
        Generate a random positive semi-definite matrix
        :param shape: the matrix shape
        :return: the matrix
        """
        a = torch.rand(shape)
        a = torch.abs(a + a.t())
        return torch.matmul(a, a.t())

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
                self.learn(config)

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

    def learn(self, config):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param config: the hydra configuration
        """

        if self.learning_step == 0:

            # Perform K-means to initialize means of prior and posterior distributions.
            μ, r = KMeans.run(self.x, self.n_states)
            # TODO self.draw_graph(title="After K-means Optimization", ellipses=False, r=r, μ=μ)
            self.m = μ
            self.m_hat = μ
            self.r_hat = r

            # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
            precision = KMeans.precision(self.x, r)
            self.W_hat = [precision[k] / self.v_hat[k] for k in range(self.n_states)]
            self.W = [precision[k] / self.v[k] for k in range(self.n_states)]

        # Perform inference.
        self.draw_graph(title="Before Optimization")
        self.draw_graph(title="Before Optimization", ellipses=False)
        for i in range(200):  # TODO implement a better stopping condition
            self.update_for_z()
            # TODO self.draw_graph(title=f"[{i}] After Z update")
            # TODO self.draw_graph(title=f"[{i}] After Z update", ellipses=False)
            self.update_for_μ_and_Λ()
            # TODO self.draw_graph(title=f"[{i}] After μ and Λ update")
            # TODO self.draw_graph(title=f"[{i}] After μ and Λ update", ellipses=False)
        self.draw_graph(title="After Optimization")
        self.draw_graph(title="After Optimization", ellipses=False)

        # Perform empirical Bayes.
        self.W = self.W_hat
        self.m = self.m_hat
        self.v = self.v_hat
        self.β = self.β_hat

        # Clear the dataset and increase learning step.
        self.x.clear()
        self.learning_step += 1

    def draw_graph(self, title="", data=True, ellipses=True, r=None, μ=None):

        # Draw the data points.
        if data is True:
            if r is None:
                r = self.r_hat
            x = [x_tensor[0] for x_tensor in self.x]
            y = [x_tensor[1] for x_tensor in self.x]

            c = [tuple(r_hat) for r_hat in r] if r.shape[1] == 3 else [self.colors[torch.argmax(r_hat)] for r_hat in r]
            plt.scatter(x=x, y=y, c=c)

        # Draw the ellipses corresponding to the current model believes.
        if ellipses is True:
            self.make_ellipses()

        if μ is not None:
            x = [μ_k[0] for μ_k in μ]
            y = [μ_k[1] for μ_k in μ]
            plt.scatter(x=x, y=y, marker="X")

        plt.title(title)
        plt.show()

    def make_ellipses(self):
        for k in range(self.n_states):
            color = self.colors[k]
            covariances = torch.inverse(self.v_hat[k] * self.W_hat[k])
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 3. * np.sqrt(2.) * np.sqrt(v)
            mean = self.m_hat[k]
            mean = mean.reshape(2, 1)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(plt.gca().bbox)
            ell.set_alpha(0.5)
            plt.gca().add_artist(ell)
            plt.gca().set_aspect('equal', 'datalim')

    def update_for_z(self):
        # Compute the non-normalized state probabilities.
        log_D = self.repeat_across_data_points(torch.log(self.D))
        log_det = self.repeat_across_data_points(self.expected_log_determinant())
        quadratic_form = self.expected_quadratic_form()
        log_ρ = torch.zeros([self.dataset_size, self.n_states])
        log_ρ += log_D - self.n_states / 2 * math.log(2 * math.pi) + 0.5 * log_det - 0.5 * quadratic_form

        # Normalize the state probabilities.
        self.r_hat = torch.softmax(log_ρ, dim=1)

    def expected_log_determinant(self):
        log_det = []
        for k in range(self.n_states):
            digamma_sum = sum([torch.digamma((self.v_hat[k] - i) / 2) for i in range(self.n_states)])
            log_det.append(self.n_states * math.log(2) + torch.logdet(self.W_hat[k]) + digamma_sum)
        return torch.tensor(log_det)

    def expected_quadratic_form(self):
        quadratic_form = torch.zeros([self.dataset_size, self.n_states])
        for n in range(self.dataset_size):
            for k in range(self.n_states):
                x = self.x[n] - self.m_hat[k]
                quadratic_form[n][k] = self.n_states / self.β_hat[k]
                quadratic_form[n][k] += self.v_hat[k] * torch.matmul(torch.matmul(x.t(), self.W_hat[k]), x)
        return quadratic_form

    def update_for_μ_and_Λ(self):
        N = torch.sum(self.r_hat, dim=0) + 0.0001
        x_bar = self.compute_x_bar(N)

        self.v_hat = self.v + N
        self.β_hat = self.β + N
        self.m_hat = [(self.β[k] * self.m[k] + N[k] * x_bar[k]) / self.β_hat[k] for k in range(self.n_states)]
        self.W_hat = [self.compute_W_hat(N, k, x_bar) for k in range(self.n_states)]

    def compute_x_bar(self, N):
        return \
            [sum([self.r_hat[n][k] * self.x[n] for n in range(self.dataset_size)]) / N[k] for k in range(self.n_states)]

    def compute_W_hat(self, N, k, x_bar):
        W_hat = torch.inverse(self.W[k])
        for n in range(self.dataset_size):
            x = self.x[n] - x_bar[k]
            W_hat += self.r_hat[n][k] * torch.outer(x, x)
        x = x_bar[k] - self.m[k]
        W_hat += (self.β[k] * N[k] / self.β_hat[k]) * torch.outer(x, x)
        return torch.inverse(W_hat)

    def repeat_across_data_points(self, x):
        return torch.unsqueeze(x, dim=0).repeat(self.dataset_size, 1)

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
            "discount_factor": self.discount_factor,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "action_selection": dict(self.action_selection),
            "W": self.W,
            "v": self.v,
            "m": self.m,
            "β": self.β,
            "D": self.D,
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
            "discount_factor": checkpoint["discount_factor"],
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
            "D": checkpoint["D"],
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
