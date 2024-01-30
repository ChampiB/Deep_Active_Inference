import math
from os.path import join
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.GaussianMixture import GaussianMixture
from zoo.helpers.MatPlotLib import MatPlotLib


class DirichletTMGM(AgentInterface):
    """
    Implement a Temporal Model with Gaussian Mixture likelihood and Dirichlet prior taking random each action.
    """

    def __init__(
        self, name, tensorboard_dir, checkpoint_dir, action_selection, n_states, dataset_size,
        W=None, m=None, v=None, β=None, d=None, b=None, n_observations=2, n_actions=4, steps_done=0,
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
        :param W: the scale matrix of the Wishart prior
        :param v: degree of freedom of the Wishart prior
        :param m: the mean of the prior over μ
        :param β: the scaling coefficient of the precision matrix of the prior over μ
        :param D: the prior probability of the Gaussian components at time step 0
        :param B: the prior probability of the Gaussian components at time step 1
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
        self.colors = ['red', 'green', 'blue', 'purple', 'gray', 'pink', 'turquoise', 'orange', 'brown', 'cyan']

        # The number of learning steps performed so far.
        self.learning_step = learning_step

        # The dataset used for training.
        self.x0 = []
        self.x1 = []
        self.a0 = []

        # Prior parameters.
        self.W = [torch.ones([n_observations, n_observations]) if W is None else W[k].cpu() for k in range(n_states)]
        self.m = [torch.zeros([n_observations]) if m is None else m[k].cpu() for k in range(n_states)]
        self.v = (n_observations - 0.99) * torch.ones([n_states]) if v is None else v.cpu()
        self.β = torch.ones([n_states]) if β is None else β.cpu()
        self.d = torch.ones([n_states]) if d is None else d.cpu()
        self.b = torch.ones([n_actions, n_states, n_states]) * 0.2 if b is None else b.cpu()

        # Posterior parameters.
        self.W_hat = [torch.ones([n_observations, n_observations]) for _ in range(n_states)]
        self.m_hat = [torch.ones([n_observations]) for _ in range(n_states)]
        self.v_hat = (n_observations - 0.99) * torch.ones([n_states])
        self.β_hat = torch.ones([n_states])
        self.r_hat = [torch.softmax(torch.rand([dataset_size, n_states]), dim=1) for _ in range(2)]
        self.d_hat = torch.ones([n_states])
        self.b_hat = torch.ones([n_actions, n_states, n_states]) * 0.2

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
            old_obs = obs
            obs, reward, done, info = env.step(action)

            # Add the new observation to the dataset.
            self.x0.append(old_obs.clone())
            self.x1.append(obs.clone())
            self.a0.append(action)

            # Perform one iteration of training (if needed).
            if len(self.x0) >= self.dataset_size:
                self.learn(env)

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

    def learn(self, env, debug=True, verbose=False):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param env: the environment on which the agent is trained
        :param debug: whether to display debug information
        :param verbose: whether to display detailed debug information
        """

        if self.learning_step == 0:

            # Initialize the parameter of the Gaussian Mixture using the K-means algorithm.
            self.m, self.m_hat, self.r_hat[1], self.W_hat, self.W, self.r_hat[0] = \
                GaussianMixture.k_means_init([self.x0, self.x1], self.v, self.v_hat)

            # Display the result of the k-means algorithm, if needed.
            if verbose is True:
                MatPlotLib.draw_tgm_graph(
                    action_names=env.action_names, title="After K-means Optimization",
                    params=(self.m_hat, self.v_hat, self.W_hat), x0=self.x0, x1=self.x1, a0=self.a0, r=self.r_hat,
                    ellipses=False, clusters=True
                )

        # Display the model's beliefs, if needed.
        if debug is True or verbose is True:
            self.draw_beliefs_graphs(env.action_names, "Before Optimization")

        # Perform inference.
        for i in range(20):  # TODO implement a better stopping condition

            # Perform the update for the latent variable D, and display the model's beliefs (if needed).
            self.update_for_d()
            if verbose is True:
                self.draw_beliefs_graphs(env.action_names, f"[{i}] After D update")

            # Perform the update for the latent variable B, and display the model's beliefs (if needed).
            self.update_for_b()
            if verbose is True:
                self.draw_beliefs_graphs(env.action_names, f"[{i}] After B update")

            # Perform the update for the latent variable Z0, and display the model's beliefs (if needed).
            self.update_for_z0()
            if verbose is True:
                self.draw_beliefs_graphs(env.action_names, f"[{i}] After Z0 update")

            self.update_for_z1()
            if verbose is True:
                self.draw_beliefs_graphs(env.action_names, f"[{i}] After Z1 update")

            self.update_for_μ_and_Λ()
            if verbose is True:
                self.draw_beliefs_graphs(env.action_names, f"[{i}] After μ and Λ update")

        # Display the model's beliefs, if needed.
        if debug is True or verbose is True:
            self.draw_beliefs_graphs(env.action_names, "After Optimization")

        # Perform empirical Bayes.
        self.W = self.W_hat
        self.m = self.m_hat
        self.v = self.v_hat
        self.β = self.β_hat
        self.d = self.d_hat
        self.b = self.b_hat

        # Clear the dataset and increase learning step.
        self.x0.clear()
        self.x1.clear()
        self.a0.clear()
        self.learning_step += 1

    def draw_beliefs_graphs(self, action_names, title):
        params = (self.m_hat, self.v_hat, self.W_hat)
        MatPlotLib.draw_tgm_graph(
            action_names=action_names, title=title, params=params, x0=self.x0, x1=self.x1, a0=self.a0, r=self.r_hat
        )
        MatPlotLib.draw_tgm_graph(
            action_names=action_names, title=title, params=params, x0=self.x0, x1=self.x1, a0=self.a0, r=self.r_hat,
            ellipses=False
        )

    def update_for_d(self):
        self.d_hat = self.d + self.r_hat[0].sum(dim=0) + self.r_hat[1].sum(dim=0)

    def update_for_b(self):
        a = torch.nn.functional.one_hot(torch.tensor(self.a0))
        self.b_hat = self.b + torch.einsum("na, nj, nk -> ajk", a, self.r_hat[1], self.r_hat[0])

    def update_for_z0(self):

        # Compute the non-normalized state probabilities.
        log_D = GaussianMixture.expected_log_D(self.d, self.dataset_size)
        log_det = GaussianMixture.expected_log_det_Λ(self.v_hat, self.W_hat, self.dataset_size)
        quadratic_form = GaussianMixture.expected_quadratic_form(self.x0, self.m_hat, self.β_hat, self.v_hat, self.W_hat)
        log_ρ = torch.zeros([self.dataset_size, self.n_states])
        log_ρ += log_D - self.n_states / 2 * math.log(2 * math.pi) + 0.5 * log_det - 0.5 * quadratic_form

        # Normalize the state probabilities.
        self.r_hat[0] = torch.softmax(log_ρ, dim=1)

    def update_for_z1(self):

        # Compute the non-normalized state probabilities.
        log_D = GaussianMixture.expected_log_D(self.d, self.dataset_size)
        log_det = GaussianMixture.expected_log_det_Λ(self.v_hat, self.W_hat, self.dataset_size)
        quadratic_form = GaussianMixture.expected_quadratic_form(self.x1, self.m_hat, self.β_hat, self.v_hat, self.W_hat)
        log_ρ = torch.zeros([self.dataset_size, self.n_states])
        log_ρ += log_D - self.n_states / 2 * math.log(2 * math.pi) + 0.5 * log_det - 0.5 * quadratic_form

        # Normalize the state probabilities.
        self.r_hat[1] = torch.softmax(log_ρ, dim=1)

    def expected_log_Bj(self, log_B_hat):
        log_B = torch.zeros([self.dataset_size, self.n_states])
        for n in range(self.dataset_size):
            for j in range(self.n_states):
                log_B[n][j] = sum(
                    [self.r_hat[0][n][j] * log_B_hat[self.a0[n]][j][k] for k in range(self.n_states)]
                )
        return log_B

    def update_for_μ_and_Λ(self):
        N = torch.sum(self.r_hat[0], dim=0) + torch.sum(self.r_hat[1], dim=0) + 0.0001
        x_bar = GaussianMixture.tgm_x_bar(self.r_hat, self.x0, self.x1, N)

        self.v_hat = self.v + N
        self.β_hat = self.β + N
        self.m_hat = [(self.β[k] * self.m[k] + N[k] * x_bar[k]) / self.β_hat[k] for k in range(self.n_states)]
        self.W_hat = [self.compute_W_hat(N, k, x_bar) for k in range(self.n_states)]

    def compute_W_hat(self, N, k, x_bar):
        W_hat = torch.inverse(self.W[k])
        for n in range(self.dataset_size):
            x = self.x0[n] - x_bar[k]
            W_hat += self.r_hat[0][n][k] * torch.outer(x, x)
            x = self.x1[n] - x_bar[k]
            W_hat += self.r_hat[1][n][k] * torch.outer(x, x)
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
            "b": self.b,
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
            "b": checkpoint["b"],
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
