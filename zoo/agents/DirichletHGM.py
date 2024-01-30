from os.path import join
from bigtree import Node
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.DirichletGM import DirichletGM
from zoo.agents.GM import GM
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.MatPlotLib import MatPlotLib


class DirichletHGM(AgentInterface):
    """
    Implement a Dirichlet Hierarchical Gaussian Mixture taking random each action.
    """

    def __init__(
        self, name, tensorboard_dir, checkpoint_dir, action_selection, n_states, dataset_size,
        W=None, m=None, v=None, β=None, d=None, n_observations=2, n_actions=4, steps_done=0,
        learning_step=0, min_data_points=10, **_
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
        :param W: the scale matrix of the Wishart prior
        :param v: degree of freedom of the Wishart prior
        :param m: the mean of the prior over μ
        :param β: the scaling coefficient of the precision matrix of the prior over μ
        :param D: the prior probability of the Gaussian components
        :param learning_step: the number of learning steps performed so far
        :param min_data_points: the minimum number of data points required for a new hierarchical to start
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
        self.min_data_points = min_data_points
        self.colors = ['red', 'green', 'blue', 'purple', 'gray', 'pink', 'turquoise', 'orange', 'brown', 'cyan']

        # The number of learning steps performed so far.
        self.learning_step = learning_step

        # The dataset used for training.
        self.x = []

        # The Gaussian Mixture containing all the components of the Hierarchical Gaussian Mixture.
        self.gm = DirichletGM(
            n_states=n_states, dataset_size=dataset_size, n_observations=n_observations, n_actions=n_actions,
            W=W, m=m, v=v, β=β, d=d, learning_step=learning_step
        )

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

        # Learns the root Gaussian Mixture.
        self.gm.x = self.x
        self.gm.learn(clear=False, debug=verbose, verbose=verbose)

        # Recursively learns all sub Gaussian Mixture.
        root = Node("gm", gm=self.gm)
        self.learn_sub_gm(root, debug=verbose, verbose=verbose)

        # Combine all the Gaussian Mixtures in the tree to create the overall Gaussian Mixture.
        self.gm = self.combine(root)
        self.gm.x = self.x
        self.gm.learn(clear=False, debug=debug, verbose=verbose)

        # Clear the dataset and increase learning step.
        self.x.clear()
        self.learning_step += 1

    def combine(self, root):

        # Retrieve the components of the combined Gaussian Mixture.
        components = self.find_terminal_components(root)

        # Unpack the parameters of the components.
        params = list(zip(*components))
        W_hat, m_hat, v_hat, β_hat = [list(param) for param in params]
        v_hat = torch.stack(v_hat)
        β_hat = torch.stack(β_hat)

        # Create the combined Gaussian Mixture.
        gm = DirichletGM(
            n_states=len(components), dataset_size=self.dataset_size, n_observations=self.n_observations,
            n_actions=self.n_actions, learning_step=1, W=W_hat, m=m_hat, v=v_hat, β=β_hat
        )
        gm.W_hat = W_hat
        gm.m_hat = m_hat
        gm.v_hat = v_hat
        gm.β_hat = β_hat
        return gm

    def find_terminal_components(self, parent):

        # Initialize the list of terminal components.
        components = []

        # Retrieve expanded nodes that are terminal.
        if len(parent.children) == 0:
            active_ks = parent.gm.active_components
            for k in active_ks:
                components.append((parent.gm.W_hat[k], parent.gm.m_hat[k], parent.gm.v_hat[k], parent.gm.β_hat[k]))
            return components

        # Keep track of active components, and call the function recursively for each child.
        active_ks = parent.gm.active_components
        for child in parent.children:
            active_ks = active_ks.difference({int(child.name)})
            components.extend(self.find_terminal_components(child))

        # Retrieve non-expanded nodes that are terminal.
        for k in active_ks:
            components.append((parent.gm.W_hat[k], parent.gm.m_hat[k], parent.gm.v_hat[k], parent.gm.β_hat[k]))

        return components

    def learn_sub_gm(self, node, debug=True, verbose=False):

        # Check whether to stop the recursion.
        if node.parent is not None and node.gm.n_active_components() == 1 and node.parent.gm.n_active_components() == 1:
            return

        for state in range(self.n_states):

            # Retrieving data for the Gaussian Mixture corresponding to the current state.
            x = node.gm.data_of_component(state)
            if len(x) < self.min_data_points:
                continue

            # Learns the Gaussian Mixture corresponding to the current state.
            sub_gm = DirichletGM(n_states=self.n_states, dataset_size=len(x), n_observations=self.n_observations)
            sub_gm.x = x
            sub_gm.learn(clear=False, debug=debug, verbose=verbose)
            child = Node(str(state), gm=sub_gm, parent=node)
            self.learn_sub_gm(child, debug=debug, verbose=verbose)

    def draw_beliefs_graphs(self, title):
        params = (self.gm.m_hat, self.gm.v_hat, self.gm.W_hat)
        MatPlotLib.draw_gm_graph(title=title, params=params, data=self.x, r=self.gm.r_hat)
        MatPlotLib.draw_gm_graph(title=title, params=params, data=self.x, r=self.gm.r_hat, ellipses=False)

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
            "W": self.gm.W,
            "v": self.gm.v,
            "m": self.gm.m,
            "β": self.gm.β,
            "d": self.gm.d,
            "learning_step": self.learning_step,
            "min_data_points": self.min_data_points
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
            "learning_step": checkpoint["learning_step"],
            "min_data_points": checkpoint["min_data_points"]
        }

    def draw_reconstructed_images(self, env, grid_size):
        """
        Draw the ground truth and reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        :return: the figure containing the images
        """
        raise Exception("Function 'draw_reconstructed_images' not implemented in GM agent.")
