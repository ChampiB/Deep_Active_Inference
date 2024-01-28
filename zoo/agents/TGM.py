from matplotlib import colors
import matplotlib.image as mpimg
from io import BytesIO
import matplotlib as mpl
import math
from os.path import join
import pydot
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.KMeans import KMeans
import matplotlib.pyplot as plt
import numpy as np


class TGM(AgentInterface):
    """
    Implement a Temporal Gaussian Mixture taking random each action.
    """

    def __init__(
        self, name, tensorboard_dir, checkpoint_dir, action_selection, n_states, dataset_size,
        W=None, m=None, v=None, β=None, D=None, B=None, n_observations=2, n_actions=4, steps_done=0, verbose=False,
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
        self.verbose = verbose
        self.colors = ['red', 'green', 'blue', 'purple', 'gray', 'pink', 'turquoise', 'orange', 'brown', 'cyan']

        # The number of learning steps performed so far.
        self.learning_step = learning_step

        # The dataset used for training.
        self.x0 = []
        self.x1 = []
        self.a0 = []

        # Gaussian mixture prior parameters.
        self.W = [self.random_psd_matrix([n_observations, n_observations]) if W is None else W[k].cpu() for k in range(n_states)]
        self.m = [torch.zeros([n_observations]) if m is None else m[k].cpu() for k in range(n_states)]
        self.v = (n_observations - 0.99) * torch.ones([n_states]) if v is None else v.cpu()
        self.β = torch.ones([n_states]) if β is None else β.cpu()
        self.D = torch.ones([n_states]) / n_states if D is None else D.cpu()
        self.B = torch.ones([n_actions, n_states, n_states]) / n_states if B is None else B.cpu()

        # Gaussian mixture posterior parameters.
        self.W_hat = [self.random_psd_matrix([n_observations, n_observations]) for _ in range(n_states)]
        self.m_hat = [torch.ones([n_observations]) for _ in range(n_states)]
        self.v_hat = (n_observations - 0.99) * torch.ones([n_states])
        self.β_hat = torch.ones([n_states])
        self.r_hat = [torch.softmax(torch.rand([dataset_size, n_states]), dim=1) for _ in range(2)]

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
            old_obs = obs
            obs, reward, done, info = env.step(action)

            # Add the new observation to the dataset.
            self.x0.append(old_obs.clone())
            self.x1.append(obs.clone())
            self.a0.append(action)

            # Perform one iteration of training (if needed).
            if len(self.x0) >= self.dataset_size:
                self.learn(config)

            # Save the agent (if needed).
            # TODO if self.steps_done % config.checkpoint.frequency == 0:
            # TODO     self.save(config)

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

            # Perform K-means to initialize the parameter of the posterior over latent variables at time step 1.
            μ, r = KMeans.run(self.x1, self.n_states)
            # TODO self.draw_graph(title="After K-means Optimization", ellipses=False, r=r, μ=μ)
            self.m = μ
            self.m_hat = μ
            self.r_hat[1] = r

            # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
            precision = KMeans.precision(self.x1, r)
            self.W_hat = [precision[k] / self.v_hat[k] for k in range(self.n_states)]
            self.W = [precision[k] / self.v[k] for k in range(self.n_states)]

            # Perform K-means to initialize means of prior and posterior distributions at time step 0.
            r = KMeans.update_responsibilities(self.x0, μ)
            # TODO self.draw_graph(title="After K-means Optimization", ellipses=False, r=r, μ=μ)
            self.r_hat[0] = r

        # Perform inference.
        self.draw_graph(title="Before Optimization")
        self.draw_graph(title="Before Optimization", ellipses=False)
        for i in range(20):  # TODO implement a better stopping condition
            for j in range(10):
                self.update_for_z0()
                self.draw_graph(title=f"[{i}] After Z0 update")
                self.draw_graph(title=f"[{i}] After Z0 update", ellipses=False)
                self.update_for_z1()
                self.draw_graph(title=f"[{i}] After Z1 update")
                self.draw_graph(title=f"[{i}] After Z1 update", ellipses=False)
            self.update_for_μ_and_Λ()
            self.draw_graph(title=f"[{i}] After μ and Λ update")
            self.draw_graph(title=f"[{i}] After μ and Λ update", ellipses=False)
        self.draw_graph(title="After Optimization")
        self.draw_graph(title="After Optimization", ellipses=False)

        # Perform empirical Bayes.
        self.W = self.W_hat
        self.m = self.m_hat
        self.v = self.v_hat
        self.β = self.β_hat

        # Clear the dataset and increase learning step.
        self.x0.clear()
        self.x1.clear()
        self.a0.clear()
        self.learning_step += 1

    def draw_graph(self, title="", data=True, ellipses=True, r0=None, r1=None, μ0=None, μ1=None):

        # Create the subplots.
        f, axes = plt.subplots(nrows=1 + math.ceil(self.n_actions / 2.0), ncols=2)
        axes[0][0].set_title("Observation at t = 0")
        axes[0][1].set_title("Observation at t = 1")
        for action in range(self.n_actions):
            axes[1 + int(action / 2)][action % 2].set_title(f"Transition for action = {action}")

        # Draw the data points.
        if data is True:

            # Draw the data points of t = 0.
            if r0 is None:
                r0 = self.r_hat[0]
            x = [x_tensor[0] for x_tensor in self.x0]
            y = [x_tensor[1] for x_tensor in self.x0]

            c = [tuple(r) for r in r0] if r0.shape[1] == 3 else [self.colors[torch.argmax(r)] for r in r0]
            axes[0][0].scatter(x=x, y=y, c=c)

            # Draw the data points of t = 0.
            if r1 is None:
                r1 = self.r_hat[1]
            x = [x_tensor[0] for x_tensor in self.x1]
            y = [x_tensor[1] for x_tensor in self.x1]

            c = [tuple(r) for r in r1] if r1.shape[1] == 3 else [self.colors[torch.argmax(r)] for r in r1]
            axes[0][1].scatter(x=x, y=y, c=c)

        # Draw the ellipses corresponding to the current model believes.
        if ellipses is True:
            self.make_ellipses(axes[0][0])
            self.make_ellipses(axes[0][1])

        # Draw the cluster center.
        if μ0 is not None:
            x = [μ_k[0] for μ_k in μ0]
            y = [μ_k[1] for μ_k in μ0]
            axes[0][0].scatter(x=x, y=y, marker="X")
        if μ1 is not None:
            x = [μ_k[0] for μ_k in μ1]
            y = [μ_k[1] for μ_k in μ1]
            axes[0][1].scatter(x=x, y=y, marker="X")

        # Display the graph corresponding to each action.
        for action in range(self.n_actions):
            axis = axes[1 + int(action / 2)][action % 2]
            self.draw_transition_graph(axis, action)

        f.suptitle(title)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

    def draw_transition_graph(self, axis, action):

        # Create the graph.
        graph = pydot.Dot()
        for state in range(self.n_states):
            color = colors.to_rgba(self.colors[state])
            color = [hex(int(c * 255)).replace("0x", "") for c in list(color)[0:3]]
            color = [c if len(c) == 2 else c + c for c in list(color)[0:3]]
            color = f"#{''.join(color)}88"
            graph.add_node(pydot.Node(state, label=str(state), style="filled", color=color))

        # Create the adjacency matrix.
        states = [
            [torch.argmax(self.r_hat[0][n]), torch.argmax(self.r_hat[1][n])]
            for n in range(self.dataset_size) if self.a0[n] == action
        ]
        adjacency_matrix = torch.zeros([self.n_states, self.n_states])
        for z0, z1 in states:
            adjacency_matrix[z0][z1] += 1

        # Add the edges to the graph and create the graph label.
        sum_columns = adjacency_matrix.sum(dim=1)
        for z0 in range(self.n_states):
            for z1 in range(self.n_states):
                if adjacency_matrix[z0][z1] != 0:
                    label = round(float(adjacency_matrix[z0][z1] / sum_columns[z0]), 2)
                    graph.add_edge(pydot.Edge(z0, z1, label=label))

        # Draw the graph.
        png_img = graph.create_png()
        sio = BytesIO()
        sio.write(png_img)
        sio.seek(0)
        img = mpimg.imread(sio)
        axis.imshow(img)

    def make_ellipses(self, axis):
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
            ell.set_clip_box(axis.bbox)
            ell.set_alpha(0.5)
            axis.add_artist(ell)
            axis.set_aspect('equal', 'datalim')

    def update_for_z0(self):
        # Compute the non-normalized state probabilities.
        log_D = self.repeat_across_data_points(torch.log(self.D))
        log_B = self.expected_log_Bk()
        log_det = self.repeat_across_data_points(self.expected_log_determinant())
        quadratic_form = self.expected_quadratic_form(t=0)
        log_ρ = torch.zeros([self.dataset_size, self.n_states])
        log_ρ += log_D + log_B - self.n_states / 2 * math.log(2 * math.pi) + 0.5 * log_det - 0.5 * quadratic_form

        # Normalize the state probabilities.
        self.r_hat[0] = torch.softmax(log_ρ, dim=1)

    def expected_log_Bk(self):
        log_B = torch.zeros([self.dataset_size, self.n_states])
        for n in range(self.dataset_size):
            for k in range(self.n_states):
                log_B[n][k] = sum(
                    [self.r_hat[1][n][j] * torch.log(self.B[self.a0[n]])[j][k] for j in range(self.n_states)]
                )
        return log_B

    def expected_log_determinant(self):
        log_det = []
        for k in range(self.n_states):
            digamma_sum = sum([torch.digamma((self.v_hat[k] - i) / 2) for i in range(self.n_states)])
            log_det.append(self.n_states * math.log(2) + torch.logdet(self.W_hat[k]) + digamma_sum)
        return torch.tensor(log_det)

    def expected_quadratic_form(self, t):
        quadratic_form = torch.zeros([self.dataset_size, self.n_states])
        for n in range(self.dataset_size):
            for k in range(self.n_states):
                x = (self.x0[n] - self.m_hat[k]) if t == 0 else (self.x1[n] - self.m_hat[k])
                quadratic_form[n][k] = self.n_states / self.β_hat[k]
                quadratic_form[n][k] += self.v_hat[k] * torch.matmul(torch.matmul(x.t(), self.W_hat[k]), x)
        return quadratic_form

    def update_for_z1(self):
        # Compute the non-normalized state probabilities.
        log_B = self.expected_log_Bj()
        log_det = self.repeat_across_data_points(self.expected_log_determinant())
        quadratic_form = self.expected_quadratic_form(t=1)
        log_ρ = torch.zeros([self.dataset_size, self.n_states])
        log_ρ += log_B - self.n_states / 2 * math.log(2 * math.pi) + 0.5 * log_det - 0.5 * quadratic_form

        # Normalize the state probabilities.
        self.r_hat[1] = torch.softmax(log_ρ, dim=1)

    def expected_log_Bj(self):
        log_B = torch.zeros([self.dataset_size, self.n_states])
        for n in range(self.dataset_size):
            for j in range(self.n_states):
                log_B[n][j] = sum(
                    [self.r_hat[0][n][j] * torch.log(self.B[self.a0[n]])[j][k] for k in range(self.n_states)]
                )
        return log_B

    def update_for_μ_and_Λ(self):
        N = torch.sum(self.r_hat[0], dim=0) + torch.sum(self.r_hat[1], dim=0) + 0.0001
        x_bar = self.compute_x_bar(N)

        self.v_hat = self.v + N
        self.β_hat = self.β + N
        self.m_hat = [(self.β[k] * self.m[k] + N[k] * x_bar[k]) / self.β_hat[k] for k in range(self.n_states)]
        self.W_hat = [self.compute_W_hat(N, k, x_bar) for k in range(self.n_states)]

    def compute_x_bar(self, N):
        return [
            sum([self.r_hat[0][n][k] * self.x0[n] + self.r_hat[1][n][k] * self.x1[n] for n in range(self.dataset_size)]) / N[k]
            for k in range(self.n_states)
        ]

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
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "action_selection": dict(self.action_selection),
            "W": self.W,
            "v": self.v,
            "m": self.m,
            "β": self.β,
            "D": self.D,
            "B": self.B,
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
            "D": checkpoint["D"],
            "B": checkpoint["B"],
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
