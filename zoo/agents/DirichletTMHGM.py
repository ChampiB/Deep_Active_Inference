import copy
from os.path import join
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.DirichletGM import DirichletGM
from zoo.agents.DirichletHGM import DirichletHGM
from zoo.agents.save.Checkpoint import Checkpoint
from datetime import datetime
import torch
import logging
from zoo.helpers.MatPlotLib import MatPlotLib


class DirichletTMHGM(AgentInterface):
    """
    Implement a Temporal Model with Hierarchical Gaussian Mixture likelihood and Dirichlet prior.
    This model takes random each action.
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

        # The Dirichlet Hierarchical Gaussian Mixture to use for the perception model at time step 1.
        self.gm1 = DirichletHGM(n_states=n_states, dataset_size=dataset_size, W=W, m=m, v=v, β=β, d=d)

        # The Dirichlet Gaussian Mixture to use for the perception model at time step 1.
        self.gm0 = DirichletGM(n_states=n_states, dataset_size=dataset_size, W=W, m=m, v=v, β=β, d=d)

        # Prior parameters.
        self.b = torch.ones([n_actions, n_states, n_states]) * 0.2 if b is None else b.cpu()

        # Posterior parameters.
        self.r_hat = [torch.softmax(torch.rand([dataset_size, n_states]), dim=1) for _ in range(2)]
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

        # Fit the perception models.
        self.gm1.x = self.x1
        self.gm1.learn(clear=False, debug=True)  # TODO verbose)
        self.gm0 = self.copy_gm(self.gm1.gm)
        self.gm0.x = self.x0
        self.gm0.learn(clear=False, debug=True)  # TODO verbose)

        # Retrieve the responsibilities.
        self.r_hat[0] = self.gm0.r_hat
        self.r_hat[1] = self.gm1.r_hat

        # Display the model's beliefs, if needed.
        if debug is True or verbose is True:
            self.draw_beliefs_graphs(env.action_names, "Before Optimization")

        # Perform inference on the prediction model.
        for i in range(20):  # TODO implement a better stopping condition

            # Perform the update for the latent variable B, and display the model's beliefs (if needed).
            self.update_for_b()
            if verbose is True:
                self.draw_beliefs_graphs(env.action_names, f"[{i}] After B update")

        # Display the model's beliefs, if needed.
        if debug is True or verbose is True:
            self.draw_beliefs_graphs(env.action_names, "After Optimization")

        # Perform empirical Bayes.
        self.b = self.b_hat

        # Clear the dataset and increase learning step.
        self.x0.clear()
        self.x1.clear()
        self.a0.clear()
        self.learning_step += 1

    def copy_gm(self, gm):
        new_gm = DirichletGM(n_states=gm.n_states, dataset_size=gm.dataset_size)
        new_gm.W = copy.deepcopy(gm.W)
        new_gm.m = copy.deepcopy(gm.m)
        new_gm.v = copy.deepcopy(gm.v)
        new_gm.β = copy.deepcopy(gm.β)
        new_gm.d = copy.deepcopy(gm.d)
        new_gm.W_hat = copy.deepcopy(gm.W_hat)
        new_gm.m_hat = copy.deepcopy(gm.m_hat)
        new_gm.v_hat = copy.deepcopy(gm.v_hat)
        new_gm.β_hat = copy.deepcopy(gm.β_hat)
        new_gm.d_hat = copy.deepcopy(gm.d_hat)
        return new_gm

    def draw_beliefs_graphs(self, action_names, title):
        MatPlotLib.draw_dirichlet_tmhgm_graph(
            action_names=action_names, title=title,
            params0=(self.gm1.m_hat, self.gm1.v_hat, self.gm1.W_hat),
            params1=(self.gm1.m_hat, self.gm1.v_hat, self.gm1.W_hat),
            x0=self.x0, x1=self.x1, a0=self.a0, r=self.r_hat
        )
        MatPlotLib.draw_dirichlet_tmhgm_graph(
            action_names=action_names, title=title,
            params0=(self.gm1.m_hat, self.gm1.v_hat, self.gm1.W_hat),
            params1=(self.gm1.m_hat, self.gm1.v_hat, self.gm1.W_hat),
            x0=self.x0, x1=self.x1, a0=self.a0, r=self.r_hat,
            ellipses=False
        )

    def update_for_b(self):
        a = torch.nn.functional.one_hot(torch.tensor(self.a0))
        self.b_hat = self.b + torch.einsum("na, nj, nk -> ajk", a, self.r_hat[1], self.r_hat[0])

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
            "W": self.gm1.W_hat,
            "v": self.gm1.v_hat,
            "m": self.gm1.m_hat,
            "β": self.gm1.β_hat,
            "d": self.gm1.d_hat,
            "b": self.b_hat,
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