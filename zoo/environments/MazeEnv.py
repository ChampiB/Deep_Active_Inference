import copy
import numpy as np
import gym
from gym import spaces
import torch

from zoo.environments.EnvInterface import EnvInterface


class MazeEnv(EnvInterface):
    """
    A class containing the code of the maze environment.
    """

    def __init__(self, maze_path, max_episode_length, **_):
        """
        Constructor (compatible with OpenAI gym environment)
        :param maze_path: the path to the file describing the maze that must be used
        :param max_episode_length: the maximum length of an episode
        """

        # Gym compatibility
        super(MazeEnv, self).__init__()

        self.np_precision = np.float64
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=self.np_precision)

        self.end_state = None
        self.initial_state = None
        self.state = None
        self.frame_id = 0
        self.last_r = 0
        self.max_episode_length = max_episode_length
        self.maze = self.load_maze(maze_path)
        self.reset()

    def load_maze(self, maze_path):
        """
        Load the maze from the file
        :param maze_path: the path to file containing the maze description
        """

        # Load the maze from the file.
        with open(maze_path, "r") as f:

            # Load the maze data, number of rows and columns.
            maze_data = f.readlines()
            n_rows, n_columns = maze_data[0].split(" ")
            n_rows = int(n_rows)
            n_columns = int(n_columns)

            # Turn the maze data into a list.
            maze = []
            for row in range(n_rows):
                maze.append([])
                for column in range(n_columns):
                    maze[row].append(1 if maze_data[row + 1][column] == "X" else 0)
                    if maze_data[row + 1][column] == "S":
                        self.initial_state = [float(row), float(column)]
                        self.state = [float(row), float(column)]
                    if maze_data[row + 1][column] == "E":
                        self.end_state = [float(row), float(column)]

            # Check that the file was valid.
            if self.state is None:
                raise Exception("The maze file does not specify the agent starting position.")
            if self.end_state is None:
                raise Exception("The maze file does not specify the target position that the agent must reach.")
            return maze

    def get_state(self):
        """
        Getter on the current state of the system.
        :return: the current state.
        """
        return torch.normal(mean=torch.tensor([self.state[1], self.state[0]]), std=torch.ones([2]) * 0.15)

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        :return: the first observation.
        """
        self.frame_id = 0
        self.last_r = 0
        self.state = copy.deepcopy(self.initial_state)
        return self.get_state()

    @property
    def action_names(self):
        """
        Getter
        :return: the list of action names
        """
        return ["Down", "Up", "Left", "Right"]

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: the action to perform.
        :return: next observation, reward, is the trial done?, information
        """

        # Increase the frame index, that count the number of frames since the beginning of the episode.
        self.frame_id += 1

        # Simulate the action requested by the user.
        actions_fn = [self.down, self.up, self.left, self.right]
        if not isinstance(action, int):
            action = action.item()
        if action < 0 or action > 3:
            exit('Invalid action.')
        done = actions_fn[action]()
        self.last_r = self.compute_reward()
        if done:
            return self.get_state(), self.last_r, True, {}

        # Make sure the environment is reset if the maximum number of steps in
        # the episode has been reached.
        if self.frame_id >= self.max_episode_length:
            return self.get_state(), -1.0, True, {}
        else:
            return self.get_state(), self.last_r, False, {}

    #
    # Actions
    #

    def down(self):
        """
        Execute the action "down" in the environment.
        :return: true of the state is terminal and false otherwise.
        """

        # Increase y coordinate, if needed.
        if self.maze[self.y_pos + 1][self.x_pos] == 0:
            self.y_pos += 1.0

        # Return true of the state is terminal and false otherwise.
        return self.is_terminal_state()

    def up(self):
        """
        Execute the action "up" in the environment.
        :return: true of the state is terminal and false otherwise.
        """
        # Increase y coordinate, if needed.
        if self.maze[self.y_pos - 1][self.x_pos] == 0:
            self.y_pos -= 1.0

        # Return true of the state is terminal and false otherwise.
        return self.is_terminal_state()

    def right(self):
        """
        Execute the action "right" in the environment.
        :return: true of the state is terminal and false otherwise.
        """
        # Increase y coordinate, if needed.
        if self.maze[self.y_pos][self.x_pos + 1] == 0:
            self.x_pos += 1.0

        # Return true of the state is terminal and false otherwise.
        return self.is_terminal_state()

    def left(self):
        """
        Execute the action "left" in the environment.
        :return: true of the state is terminal and false otherwise.
        """
        # Increase y coordinate, if needed.
        if self.maze[self.y_pos][self.x_pos - 1] == 0:
            self.x_pos -= 1.0

        # Return true of the state is terminal and false otherwise.
        return self.is_terminal_state()

    #
    # Reward computation
    #
    def is_terminal_state(self):
        """
        Getter
        :return: True if the agent reached the target state, False otherwise
        """
        return True if self.state[0] == self.end_state[0] and self.state[1] == self.end_state[1] else False

    def compute_reward(self):
        """
        Compute the reward obtained by the agent
        :return: the reward
        """
        return 1 if self.is_terminal_state() else 0

    #
    # Getter and setter.
    #

    @property
    def y_pos(self):
        """
        Getter.
        :return: the current position of the object on the y-axis.
        """
        return int(self.state[0])

    @y_pos.setter
    def y_pos(self, new_value):
        """
        Setter.
        :param new_value: the new position of the object on the y-axis.
        :return: nothing.
        """
        self.state[0] = float(new_value)

    @property
    def x_pos(self):
        """
        Getter.
        :return: the current position of the object on the x-axis.
        """
        return int(self.state[1])

    @x_pos.setter
    def x_pos(self, new_value):
        """
        Setter.
        :param new_value: the new position of the object on the x-axis.
        :return: nothing.
        """
        self.state[1] = float(new_value)
