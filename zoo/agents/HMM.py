import matplotlib.pyplot as plt
from zoo.helpers.MatPlotLib import MatPlotLib
from os.path import join
from zoo.agents.AgentInterface import AgentInterface
from zoo.agents.save.Checkpoint import Checkpoint
import numpy as np
from datetime import datetime
from zoo.agents.memory.ReplayBuffer import ReplayBuffer, Experience
import zoo.agents.math_fc.functions as mathfc
from zoo.helpers.Device import Device
from zoo.agents.learning import Optimizers
import torch
import logging


class HMM(AgentInterface):
    """
    Implement a Hidden Markov Model agent acting randomly.
    """

    def __init__(
        self, name, encoder, decoder, transition, n_steps_beta_reset, beta, lr, beta_starting_step, beta_rate,
        queue_capacity, action_selection, n_actions, image_shape, n_states, tensorboard_dir, checkpoint_dir,
        steps_done=0, verbose=False, **_
    ):
        """
        Constructor
        :param name: the agent name
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param n_steps_beta_reset: the number of steps after with beta is reset
        :param beta_starting_step: the number of steps after which beta start increasing
        :param beta: the initial value for beta
        :param beta_rate: the rate at which the beta parameter is increased
        :param lr: the learning rate
        :param n_actions: the number of possible actions
        :param n_states: the number of latent states
        :param image_shape: the shape of the input image
        :param queue_capacity: the maximum capacity of the queue
        :param action_selection: the action selection to be used
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param checkpoint_dir: the directory in which the agent should be saved
        :param steps_done: the number of training iterations performed to date
        :param verbose: whether to log weights information such as mean, min and max values of layers' weights
        """

        # Call parent constructor.
        super().__init__(tensorboard_dir, steps_done)

        # Neural networks.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition

        # Ensure models are on the right device.
        Device.send([self.encoder, self.decoder, self.transition])

        # Optimizer.
        self.optimizer = Optimizers.get_adam([encoder, decoder, transition], lr)

        # Beta scheduling.
        self.n_steps_beta_reset = n_steps_beta_reset
        self.beta_starting_step = beta_starting_step
        self.beta = beta
        self.beta_rate = beta_rate

        # Miscellaneous.
        self.agent_name = name
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.lr = lr
        self.queue_capacity = queue_capacity
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = checkpoint_dir
        self.action_selection = action_selection
        self.n_actions = n_actions
        self.n_states = n_states
        self.image_shape = image_shape
        self.total_rewards = 0
        self.verbose = verbose

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
        :param obs: unused
        :return: the action to be performed
        """
        quality = torch.zeros([1, self.n_actions]).to(Device.get())
        return self.action_selection.select(quality, self.steps_done)

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :return: nothing
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Train the agent.
        logging.info(f"Start the training at {datetime.now()}")
        while self.steps_done < config.task.max_n_steps:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, info = env.step(action)

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training (if needed).
            if len(self.buffer) >= config.task.learning_starts:
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
        :return: nothing
        """

        # Sample the replay buffer.
        obs, action, _, _, next_obs = self.buffer.sample(config.task.batch_size)

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(config, obs, action, next_obs)
        if vfe_loss is None:
            return

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        vfe_loss.backward()
        self.optimizer.step()

        # Implement the cyclical scheduling for beta.
        if self.steps_done >= self.beta_starting_step:
            self.beta = np.clip(self.beta + self.beta_rate, 0, 1)
        if self.steps_done % self.n_steps_beta_reset == 0:
            self.beta = 0

    def compute_vfe(self, config, obs, actions, next_obs):
        """
        Compute the variational free energy
        :param config: the hydra configuration
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = mathfc.reparameterize(mean_hat, log_var_hat)
        alpha = self.decoder(states)
        kl_div_hs_t0 = mathfc.kl_div_gaussian(mean_hat, log_var_hat)
        log_likelihood_t0 = mathfc.log_bernoulli_with_logits(obs, alpha)

        mean_hat, log_var_hat = self.encoder(next_obs)
        next_states = mathfc.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        next_alpha = self.decoder(next_states)

        # Compute the variational free energy.
        kl_div_hs_t1 = mathfc.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood_t1 = mathfc.log_bernoulli_with_logits(next_obs, next_alpha)
        vfe_loss = self.beta * kl_div_hs_t1 - log_likelihood_t1 + self.beta * kl_div_hs_t0 - log_likelihood_t0
        if torch.isnan(vfe_loss) or torch.isinf(vfe_loss) or vfe_loss > 1e5:
            return None

        # Display debug information, if needed.
        if self.writer is not None and self.steps_done % min(config.tensorboard.log_interval, 50) == 0:

            # Log the mean, min and max values of weights, if requested by user.
            if self.verbose and self.steps_done % min(config.tensorboard.log_interval * 10, 500) == 0:

                for neural_network in [self.encoder, self.decoder, self.transition]:
                    for name, param in neural_network.named_parameters():
                        self.writer.add_scalar(f"{name}.mean", param.mean(), self.steps_done)
                        self.writer.add_scalar(f"{name}.min", param.min(), self.steps_done)
                        self.writer.add_scalar(f"{name}.max", param.min(), self.steps_done)

            # Log the KL-divergence, the negative log likelihood, beta and the variational free energy.
            self.writer.add_scalar("kl_div_hs_t", kl_div_hs_t0, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood_t", - log_likelihood_t0, self.steps_done)
            self.writer.add_scalar("kl_div_hs_t+1", kl_div_hs_t1, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood_t+1", - log_likelihood_t1, self.steps_done)
            self.writer.add_scalar("beta", self.beta, self.steps_done)
            self.writer.add_scalar("vfe", vfe_loss, self.steps_done)

        return vfe_loss

    def predict(self, data):
        """
        Do one forward pass using the given observations and actions.
        :param data: a tuple containing the observations and actions at time t
        :return: the outputs of the encoder and transition model
        """
        obs, actions = data
        mean_hat_t, log_var_hat_t = self.encoder(obs)
        # To be coherent with CHMM and DAI, we do not reparametrise before the transition.
        transition_pred = self.transition(mean_hat_t, actions)
        return (mean_hat_t, log_var_hat_t), transition_pred

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
            "image_shape": self.image_shape,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "decoder_net_state_dict": self.decoder.state_dict(),
            "decoder_net_module": str(self.decoder.__module__),
            "decoder_net_class": str(self.decoder.__class__.__name__),
            "encoder_net_state_dict": self.encoder.state_dict(),
            "encoder_net_module": str(self.encoder.__module__),
            "encoder_net_class": str(self.encoder.__class__.__name__),
            "transition_net_state_dict": self.transition.state_dict(),
            "transition_net_module": str(self.transition.__module__),
            "transition_net_class": str(self.transition.__class__.__name__),
            "action_selection": dict(self.action_selection),
            "lr": self.lr,
            "beta": self.beta,
            "n_steps_beta_reset": self.n_steps_beta_reset,
            "beta_starting_step": self.beta_starting_step,
            "beta_rate": self.beta_rate,
            "steps_done": self.steps_done,
            "queue_capacity": self.queue_capacity,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
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
            "encoder": Checkpoint.load_encoder(checkpoint, training_mode),
            "decoder": Checkpoint.load_decoder(checkpoint, training_mode),
            "transition": Checkpoint.load_transition(checkpoint, training_mode),
            "lr": checkpoint["lr"],
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "beta": checkpoint["beta"],
            "n_actions": checkpoint["n_actions"],
            "n_states": checkpoint["n_states"],
            "image_shape": checkpoint["image_shape"],
            "n_steps_beta_reset": checkpoint["n_steps_beta_reset"],
            "beta_starting_step": checkpoint["beta_starting_step"],
            "beta_rate": checkpoint["beta_rate"],
            "steps_done": checkpoint["steps_done"],
            "queue_capacity": checkpoint["queue_capacity"],
            "tensorboard_dir": tb_dir,
            "checkpoint_dir": checkpoint["checkpoint_dir"],
        }

    def draw_reconstructed_images(self, env, grid_size):
        """
        Draw the ground truth and reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        :return: the figure containing the images
        """

        # Create the figure and the grid specification.
        height, width = grid_size
        n_cols = 2
        fig = plt.figure(figsize=(width + n_cols, height * 2))
        gs = fig.add_gridspec(height * 2, width + n_cols)

        # Iterate over the grid's rows.
        h = 0
        while h < height:

            # Draw the ground truth label for each row.
            fig.add_subplot(gs[2 * h, 0:3])
            plt.text(0.08, 0.45, "Ground Truth Image:", fontsize=10)
            plt.axis('off')

            # Draw the reconstructed image label for each row.
            fig.add_subplot(gs[2 * h + 1, 0:3])
            plt.text(0.08, 0.45, "Reconstructed Image:", fontsize=10)
            plt.axis('off')

            # Retrieve the initial ground truth and reconstructed images.
            obs = torch.unsqueeze(env.reset(), dim=0)
            state, _ = self.encoder(obs)
            reconstructed_obs = self.decoder(state)

            # Iterate over the grid's columns.
            for w in range(width):

                # Draw the ground truth image.
                fig.add_subplot(gs[2 * h, w + n_cols])
                plt.imshow(MatPlotLib.format_image(obs))
                plt.axis('off')

                # Draw the reconstructed image.
                fig.add_subplot(gs[2 * h + 1, w + n_cols])
                plt.imshow(MatPlotLib.format_image(reconstructed_obs))
                plt.axis('off')

                # Execute the agent's action in the environment to obtain the next ground truth observation.
                action = self.step(obs)
                obs, _, done, _ = env.step(action)
                obs = torch.unsqueeze(obs, dim=0)
                action = torch.tensor([action]).to(Device.get())

                # Simulate the agent's action to obtain the next reconstructed observation.
                state, _ = self.transition(state, action)
                reconstructed_obs = self.decoder(state)

                # Increase row index.
                if done:
                    h -= 1
                    break

            # Increase row index.
            h += 1

        # Set spacing between subplots.
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.1)
        return fig
