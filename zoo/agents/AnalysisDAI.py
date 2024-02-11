import imageio
from os.path import join
from zoo.agents.AgentInterface import AgentInterface
from zoo.helpers.MatPlotLib import MatPlotLib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from zoo.agents.learning import Optimizers
from zoo.agents.save.Checkpoint import Checkpoint
from torch.utils.tensorboard import SummaryWriter
import copy
from datetime import datetime
import logging
from zoo.agents.memory.ReplayBuffer import ReplayBuffer, Experience
import zoo.agents.math_fc.functions as math_fc
from zoo.helpers.Device import Device
import pandas as pd
from torch import nn, unsqueeze
import torch
from scipy.stats import entropy


class AnalysisDAI(AgentInterface):
    """
    Implement a Deep Active Inference agent able to evaluate the qualities of each action (from observation).
    """

    def __init__(
        self, name, encoder, decoder, transition, critic, discount_factor, n_steps_info_gain_incr,
        info_gain_percentage, vfe_lr, queue_capacity, n_steps_between_synchro, tensorboard_dir, checkpoint_dir,
        g_value, image_shape, n_states, action_selection, task_dir, reward_coefficient, n_actions=4, steps_done=0,
        inhibition_of_return=False, verbose=False, **_
    ):
        """
        Constructor
        :param name: the agent name
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param critic: the critic network
        :param action_selection: the action selection to be used
        :param n_actions: the number of actions
        :param n_states: the number of latent states
        :param discount_factor: the factor by which the future EFE is discounted
        :param n_steps_info_gain_incr: the number of steps after with the information gain is increased
        :param info_gain_percentage: the percentage of information gain to be added to the EFE
        :param image_shape: the shape of the input image
        :param vfe_lr: the learning rate of the other networks
        :param queue_capacity: the maximum capacity of the queue
        :param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param task_dir: the task directory within the data directory
        :param checkpoint_dir: the directory in which the agent should be saved
        :param g_value: the type of value to be used, i.e. "reward" or "efe"
        :param reward_coefficient: the coefficient by which the reward is multiplied
        :param steps_done: the number of training iterations performed to date
        :param inhibition_of_return: the last
        :param verbose: whether to log weights information such as mean, min and max values of layers' weights
        """

        # Call parent constructor.
        super().__init__(tensorboard_dir, steps_done)

        # Neural networks.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.critic = critic
        self.target = copy.deepcopy(self.critic)
        self.target.eval()

        # Ensure models are on the right device.
        Device.send([self.encoder, self.decoder, self.transition, self.critic, self.target])

        # Optimizers.
        self.vfe_optimizer = Optimizers.get_adam([encoder, decoder, transition, critic], vfe_lr)

        # Information gain scheduling.
        self.n_steps_info_gain_incr = n_steps_info_gain_incr
        self.info_gain_percentage = info_gain_percentage

        # Miscellaneous.
        self.agent_name = name
        self.total_rewards = 0.0
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.g_value = g_value
        self.vfe_lr = vfe_lr
        self.image_shape = image_shape
        self.reward_coefficient = reward_coefficient
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = checkpoint_dir
        self.queue_capacity = queue_capacity
        self.action_selection = action_selection
        self.n_actions = n_actions
        self.n_states = n_states
        self.actions_picked = pd.DataFrame(columns=["Training iterations", "Actions"])
        self.entropy = pd.DataFrame(columns=["Training iterations", "Entropy"])
        self.verbose = verbose
        self.task_dir = task_dir
        self.inhibition_of_return = inhibition_of_return
        self.last_action = None

        # Create summary writer for monitoring
        self.writer = SummaryWriter(tensorboard_dir)

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
        Select a random action based on the critic output
        :param obs: the input observation from which decision should be made
        :return: the random action
        """

        # Select an action.
        obs = torch.unsqueeze(obs, dim=0)
        quality = self.critic(obs)
        action = self.action_selection.select(quality, self.steps_done)

        # Select another action, if this action was.
        while self.inhibition_of_return is True and self.agent_want_to_go_back(action):
            action = self.action_selection.select(quality, self.steps_done)
            quality[0][action] = quality.min() - 1
        self.last_action = action

        # Save action taken.
        action_name = ["Down", "Up", "Left", "Right"][action]
        new_row = pd.DataFrame({"Training iterations": [self.steps_done], "Actions": [action_name]})
        self.actions_picked = pd.concat([self.actions_picked, new_row], ignore_index=True, axis=0)

        # Compute entropy of prior over actions.
        sm = nn.Softmax(dim=1)(self.critic(obs))
        e = entropy(sm[0].detach().cpu())
        new_row = pd.DataFrame({"Training iterations": [self.steps_done], "Entropy": [e]})
        self.entropy = pd.concat([self.entropy, new_row], ignore_index=True, axis=0)

        return action

    def agent_want_to_go_back(self, action):
        if self.last_action is None:
            return False
        if action == 0 and self.last_action == 1:
            return True
        if action == 1 and self.last_action == 0:
            return True
        if action == 2 and self.last_action == 3:
            return True
        if action == 3 and self.last_action == 2:
            return True
        return False

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
        logging.info("Start the training at {time}".format(time=datetime.now()))
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
                self.save(env, config)

            # Render the environment and monitor total rewards (if needed).
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
        self.save(env, config, final_model=True)

        # Close the environment.
        env.close()

        # Display graph.
        self.save_actions_picked()
        self.save_actions_prior_entropy()

    def save_actions_prior_entropy(self):
        """
        Save the entropy of the prior over actions during training
        :return: nothing
        """

        # Create the directory in which the dataframe should be saved.
        directory = self.task_dir.replace("[[DIRECTORY]]", "entropy_prior_actions")
        os.makedirs(directory, exist_ok=True)

        # Save dataframe to CSV.
        filepath = Path(f"{directory}/entropy_prior_actions.csv")
        self.entropy.to_csv(filepath)

    def save_actions_picked(self):
        """
        Save the action picked by the agent during training
        :return: nothing
        """

        # Create the directory in which the dataframe should be saved.
        directory = self.task_dir.replace("[[DIRECTORY]]", "selected_actions")
        os.makedirs(directory, exist_ok=True)

        # Save dataframe to CSV.
        filepath = Path(f"{directory}/selected_actions.csv")
        self.actions_picked.to_csv(filepath)

    def learn(self, config):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param config: the hydra configuration
        :return: nothing
        """

        # Synchronize the target with the critic (if needed).
        if self.steps_done % self.n_steps_between_synchro == 0:
            self.synchronize_target()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(config.task.batch_size)

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(config, obs, actions, next_obs, done, rewards)

        # Perform one step of gradient descent on the other networks.
        if vfe_loss is not None:
            self.vfe_optimizer.zero_grad()
            vfe_loss.backward()
            self.vfe_optimizer.step()

    def compute_efe(self, config, next_obs, done, rewards, mean, log_var, mean_hat, log_var_hat):
        """
        Compute the expected free energy
        :param config: the hydra configuration
        :param next_obs: the observations at time t + 1
        :param done: did the simulation ended at time t + 1 after performing the actions at time t
        :param rewards: the rewards at time t + 1
        :param mean: the mean from the transition network
        :param log_var: the log variance from the transition network
        :param mean_hat: the mean from the encoder network
        :param log_var_hat: the log variance from the encoder network
        :return: expected free energy loss
        """

        # For each batch entry where the simulation did not stop, compute the value of the next states.
        future_efe = torch.zeros(config.task.batch_size, device=Device.get())
        future_efe[torch.logical_not(done)] = self.target(next_obs[torch.logical_not(done)]).max(1)[0]

        # Compute the information gain (if needed).
        percentage = self.info_gain_percentage / 100.0 if self.n_steps_info_gain_incr <= self.steps_done else 0.0
        info_gain = math_fc.compute_info_gain(self.g_value, mean, log_var, mean_hat, log_var_hat)

        # Compute the immediate G-value.
        immediate_efe = self.reward_coefficient * rewards + percentage * info_gain
        immediate_efe = immediate_efe.to(torch.float32)

        # Display debug information, if needed.
        if self.writer is not None and self.steps_done % min(config.tensorboard.log_interval, 50) == 0:
            self.writer.add_scalar("efe_mean_reward", rewards.mean(), self.steps_done)
            self.writer.add_scalar("efe_mean_info_gain", info_gain.mean(), self.steps_done)
            self.writer.add_scalar("info_gain_percentage", percentage * 100, self.steps_done)

        # Compute the discounted G values.
        efe = immediate_efe + self.discount_factor * future_efe
        return efe.detach()

    def compute_vfe(self, config, obs, actions, next_obs, done, rewards):
        """
        Compute the variational free energy
        :param config: the hydra configuration
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :param done: did the simulation ended at time t + 1 after performing the actions at time t
        :param rewards: the rewards at time t + 1
        :return: the variational free energy
        """

        # Compute required vectors.
        mean_hat_t, log_var_hat_t = self.encoder(obs)
        states = math_fc.reparameterize(mean_hat_t, log_var_hat_t)
        alpha = self.decoder(states)

        mean_hat, log_var_hat = self.encoder(next_obs)
        next_states = math_fc.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        next_alpha = self.decoder(next_states)

        # Compute the EFE of each action given the current observation (as predicted by the target).
        efe = self.compute_efe(config, next_obs, done, rewards, mean, log_var, mean_hat, log_var_hat)
        efe = nn.functional.softmax(efe, dim=1)

        # Compute the EFE of each action given the current observation (as predicted by the critic).
        critic_prediction = self.critic(obs).gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))
        critic_prediction = nn.functional.softmax(critic_prediction, dim=1)

        # Compute the variational free energy.
        kl_div_hs_t0 = math_fc.kl_div_gaussian(mean_hat_t, log_var_hat_t)
        log_likelihood_t0 = math_fc.log_bernoulli_with_logits(obs, alpha)

        kl_div_hs_t1 = math_fc.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood_t1 = math_fc.log_bernoulli_with_logits(next_obs, next_alpha)

        kl_div_a = math_fc.kl_div_categorical(critic_prediction, efe)

        vfe_loss = kl_div_hs_t1 - log_likelihood_t1 + kl_div_hs_t0 - log_likelihood_t0 + kl_div_a
        if torch.isnan(vfe_loss) or torch.isinf(vfe_loss) or vfe_loss > 1e5:
            return None

        # Display debug information, if needed.
        if self.writer is not None and self.steps_done % min(config.tensorboard.log_interval, 50) == 0:

            # Log the mean, min and max values of weights, if requested by user.
            if self.verbose and self.steps_done % min(config.tensorboard.log_interval * 10, 500) == 0:

                for neural_network in [self.encoder, self.decoder, self.transition, self.critic, self.target]:
                    for name, param in neural_network.named_parameters():
                        self.writer.add_scalar(f"{name}.mean", param.mean(), self.steps_done)
                        self.writer.add_scalar(f"{name}.min", param.min(), self.steps_done)
                        self.writer.add_scalar(f"{name}.max", param.min(), self.steps_done)

            # Log the KL-divergence, the negative log likelihood, beta and the variational free energy.
            self.writer.add_scalar("kl_div_a", kl_div_a, self.steps_done)
            self.writer.add_scalar("kl_div_hs_t", kl_div_hs_t0, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood_t", - log_likelihood_t0, self.steps_done)
            self.writer.add_scalar("kl_div_hs_t+1", kl_div_hs_t1, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood_t+1", - log_likelihood_t1, self.steps_done)
            self.writer.add_scalar("vfe", vfe_loss, self.steps_done)

        return vfe_loss

    def predict(self, data):
        """
        Do one forward pass using the given observations and actions.
        :param data: a tuple containing the observations and actions at time t
        :return: the outputs of the encoder, transition, and critic model
        """
        obs, actions = data
        mean_hat_t, log_var_hat_t = self.encoder(obs)
        transition_prediction = self.transition(mean_hat_t, actions)
        critic_prediction = self.critic(obs)
        return (mean_hat_t, log_var_hat_t), transition_prediction, critic_prediction

    def synchronize_target(self):
        """
        Synchronize the target with the critic.
        :return: nothing.
        """
        self.target = copy.deepcopy(self.critic)
        self.target.eval()

    def save(self, env, config, final_model=False):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param config: the hydra configuration
        :param final_model: True if the model being saved is the final version, False otherwise
        :return: nothing
        """

        # Create directories and files if they do not exist.
        model_id = config.task.max_n_steps if final_model is True else self.steps_done
        checkpoint_file = join(self.checkpoint_dir, f"model_{model_id}.pt")
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Create a GIF showing the actions taken by the agent.
        gif_file = join(self.checkpoint_dir, f"actions_taken_by_model_{model_id}.gif")
        self.create_actions_gif(env, gif_file)

        # Save the model.
        torch.save({
            "name": self.agent_name,
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
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
            "critic_net_state_dict": self.critic.state_dict(),
            "critic_net_module": str(self.critic.__module__),
            "critic_net_class": str(self.critic.__class__.__name__),
            "n_steps_info_gain_incr": self.n_steps_info_gain_incr,
            "info_gain_percentage": self.info_gain_percentage,
            "steps_done": self.steps_done,
            "g_value": self.g_value,
            "vfe_lr": self.vfe_lr,
            "task_dir": self.task_dir,
            "image_shape": self.image_shape,
            "reward_coefficient": self.reward_coefficient,
            "discount_factor": self.discount_factor,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "queue_capacity": self.queue_capacity,
            "n_steps_between_synchro": self.n_steps_between_synchro,
            "action_selection": dict(self.action_selection),
            "inhibition_of_return": self.inhibition_of_return
        }, checkpoint_file)

    def create_actions_gif(self, env, gif_file, n_steps=500):

        # Retrieve the initial observation from the environment.
        obs = env.reset()
        images = [obs.cpu().numpy()]

        # Train the agent.
        steps_done = 0
        while steps_done < n_steps:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            obs, reward, done, info = env.step(action)
            images.append(obs.cpu().numpy())

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()
                images.append(obs.cpu().numpy())

            # Increase the number of steps done so far.
            steps_done += 1

        # Save the GIF on the filesystem.
        imageio.mimsave(gif_file, images)

    @staticmethod
    def load_constructor_parameters(tb_dir, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint
        :param tb_dir: the path of tensorboard directory
        :param checkpoint: the checkpoint from which to load the parameters
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: a dictionary containing the constructor's parameters
        """
        return {
            "name": checkpoint["name"],
            "encoder": Checkpoint.load_encoder(checkpoint, training_mode),
            "decoder": Checkpoint.load_decoder(checkpoint, training_mode),
            "transition": Checkpoint.load_transition(checkpoint, training_mode),
            "critic": Checkpoint.load_critic(checkpoint, training_mode),
            "image_shape": checkpoint["image_shape"],
            "vfe_lr": checkpoint["vfe_lr"],
            "reward_coefficient": checkpoint["reward_coefficient"],
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "n_steps_info_gain_incr": checkpoint["n_steps_info_gain_incr"],
            "info_gain_percentage": checkpoint["info_gain_percentage"],
            "discount_factor": checkpoint["discount_factor"],
            "tensorboard_dir": tb_dir,
            "checkpoint_dir": checkpoint["checkpoint_dir"],
            "g_value": checkpoint["g_value"],
            "queue_capacity": checkpoint["queue_capacity"],
            "n_steps_between_synchro": checkpoint["n_steps_between_synchro"],
            "task_dir": checkpoint["task_dir"],
            "steps_done": checkpoint["steps_done"],
            "n_actions": checkpoint["n_actions"],
            "n_states": checkpoint["n_states"],
            "inhibition_of_return": checkpoint["inhibition_of_return"]
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
            obs = env.reset()
            state, _ = self.encoder(torch.unsqueeze(obs, dim=0))
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
