from zoo.tasks.TaskInterface import TaskInterface
from zoo.helpers.Hydra import Hydra
from zoo.agents.save.Checkpoint import Checkpoint
from zoo.environments import EnvFactory
import logging
from zoo.helpers.CKA import CKA
from zoo.helpers.Data import Data
from zoo.helpers.MatPlotLib import MatPlotLib
import pandas as pd
import seaborn as sns


class DrawCKAScores(TaskInterface):
    """
    A class that computes the CKA scores between the layers of two agents.
    """

    def __init__(
        self, name, save_path, n_samples,
        a1_name, a1_tensorboard_dir, a1_path, a2_name, a2_tensorboard_dir, a2_path, **_
    ):
        """
        Constructor
        :param name: the task's name
        :param save_path: the path where the csv and figure should be saved
        :param n_samples: the number of samples used to estimate the CKA scores
        :param a1_name: the name of the first agent
        :param a1_tensorboard_dir: the tensorboard directory of the first agent
        :param a1_path: the path to the checkpoint of the first agent
        :param a2_name: the name of the second agent
        :param a2_tensorboard_dir: the tensorboard directory of the first agent
        :param a2_path: the path to the checkpoint of the second agent
        """

        # Call the parent constructor.
        super().__init__(name)
        Hydra.register_resolvers()

        # Store the task's parameters.
        self.save_path = save_path
        self.n_samples = n_samples
        self.a1_name = a1_name
        self.a1_tensorboard_dir = a1_tensorboard_dir
        self.a1_path = a1_path
        self.a2_name = a2_name
        self.a2_tensorboard_dir = a2_tensorboard_dir
        self.a2_path = a2_path

    def save(self, scores):
        """
        Save the CKA score as a csv file, and create a figure displaying the scores
        :param scores: the score to be saved
        """

        # When we have FC/conv + activation function, we only keep the activation function.
        # We also drop activations from dropout and reshape layers as they are not very informative.
        logging.debug(f"Dataframe before pre-processing: {scores}")
        cols_to_keep = {
            "Encoder_2": "Encoder_1",
            "Encoder_4": "Encoder_2",
            "Encoder_6": "Encoder_3",
            "Encoder_8": "Encoder_4",
            "Encoder_11": "Encoder_5",
            "Encoder_12": "Encoder_mean",
            "Encoder_13": "Encoder_variance",
            "Transition_2": "Transition_1",
            "Transition_4": "Transition_2",
            "Transition_5": "Transition_mean",
            "Transition_6": "Transition_variance",
            "Critic_2": "Critic_1",
            "Critic_4": "Critic_2",
            "Critic_6": "Critic_3",
            "Critic_7": "Critic_4",
        }
        rows_to_keep = cols_to_keep
        df = scores.loc[scores.index.isin(rows_to_keep.keys()), scores.columns.isin(cols_to_keep.keys())]
        df.rename(columns=cols_to_keep, index=rows_to_keep, inplace=True)

        logging.debug(f"Dataframe after pre-processing: {df}")
        ax = sns.heatmap(df, vmin=0, vmax=1, annot_kws={"fontsize": 13})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        MatPlotLib.save_figure(f"{self.save_path}.pdf")

    @staticmethod
    def get_display_name(name):
        """
        Getter
        :param name: the agent name
        :return: the name to display in the CKA figure
        """
        display_name = name.upper()
        if name == "chmm_efe_epsilon_greedy_0%":
            display_name = "Model = CHMM, Action = \u03B5-greedy,\n Gain = Reward"
        if name == "chmm_reward_epsilon_greedy":
            display_name = "Model = CHMM, Action = \u03B5-greedy,\n Gain = Reward"
        if name == "chmm_efe_epsilon_greedy_100%":
            display_name = "Model = CHMM, Action = \u03B5-greedy,\n Gain = EFE"
        if name == "chmm_efe_epsilon_greedy":
            display_name = "Model = CHMM, Action = \u03B5-greedy,\n Gain = EFE"
        return display_name

    def compute_similarity_metric(self, model1, model2, samples):
        """
        Compute the CKA score between the layers of the two models passed as parameters
        :param model1: the first model
        :param model2: the second model
        :param samples: the number of sample to use for estimating CKA scores
        """

        # Create the CKA algorithm and the layer activations.
        logging.info("Instantiating CKA...")
        metric = CKA()
        acts1 = Data.get_activations(samples, model1)
        acts2 = Data.get_activations(samples, model2)

        # Prepare the layer activations.
        logging.info("Preparing layer activations...")
        f = lambda x: metric.center(CKA.prepare_activations(x))
        acts1 = {k: f(v) for k, v in acts1.items()}
        acts2 = {k: f(v) for k, v in acts2.items()}
        scores = {}
        for l1, act1 in acts1.items():
            scores[l1] = {}
            for l2, act2 in acts2.items():
                logging.info(f"Computing similarity of {l1} and {l2}")
                scores[l1][l2] = float(metric(act1, act2))
        scores = pd.DataFrame(scores).T

        # Save csv with m1 layers as header, m2 layers as indexes
        a1_name = self.get_display_name(self.a1_name)
        a2_name = self.get_display_name(self.a2_name)
        scores = scores.rename_axis(a2_name, axis="columns")
        scores = scores.rename_axis(a1_name)
        scores.to_csv(f"{self.save_path}.tsv", sep="\t")
        self.save(scores)

    def run(self, hydra_config):
        """
        Computes the CKA scores between the layers of two agents.
        """

        # Create the environment.
        env = EnvFactory.make(hydra_config)

        # Sample a batch of experiences.
        samples, actions, rewards, done, next_obs = Data.get_batch(batch_size=self.n_samples, env=env)

        # Load the agents.
        m1 = Checkpoint(self.a1_tensorboard_dir, self.a1_path).load_agent()
        m2 = Checkpoint(self.a2_tensorboard_dir, self.a2_path).load_agent()

        # Compute the CKA scores.
        self.compute_similarity_metric(m1, m2, (samples, actions))
