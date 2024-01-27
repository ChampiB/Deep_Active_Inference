from math import floor
import ray
from ray import tune
import logging
from ray.tune.search import BasicVariantGenerator
from zoo.helpers.RayMemory import auto_garbage_collect
from zoo.tasks.TaskInterface import TaskInterface
from zoo.agents import AgentFactory
from zoo.environments import EnvFactory
from zoo.helpers.Hydra import Hydra
from hydra.utils import instantiate
from functools import partial
from zoo.helpers.Seed import Seed


class RunHyperparameterTuning(TaskInterface):
    """
    A class that tunes the hyperparameters of an agent for a specific environment.
    """

    def __init__(
        self, name, seed, grace_period, max_n_steps, n_cpu, n_gpu, memory, n_hp_samples, local_directory, save_file,
        tensorboard_dir, **_
    ):
        """
        Constructor
        :param name: the task's name
        :param seed: the seed to use for random number generation
        :param grace_period: the number of training steps for which no trials can be stopped
        :param max_n_steps: the maximum number of training steps that will be run
        :param n_cpu: the number of CPU to use for each trial
        :param n_gpu: the number of GPU to use for each trial
        :param memory: the total memory available in Gb
        :param n_hp_samples: the number of trials to run (each trail has its own hyperparameter values)
        :param local_directory: the local directory that ray tune must use for logging
        :param save_file: the file used to save the best hyperparameters
        :param tensorboard_dir: the tensorboard directory
        """

        # Call the parent constructor.
        super().__init__(name)

        # Store the task's parameters.
        self.seed = seed
        self.grace_period = grace_period
        self.max_n_steps = max_n_steps
        self.n_cpu = n_cpu
        self.n_gpu = n_gpu
        self.memory = memory
        self.n_hp_samples = n_hp_samples
        self.local_directory = local_directory
        self.save_file = save_file
        self.tensorboard_dir = tensorboard_dir

    def run(self, hydra_config):
        """
        Search the agent's hyperparameter space
        :param hydra_config: the configuration where the agent, environment, and tuning strategy to use are defined
        """

        # Set the seed requested by the user.
        Seed.set(self.seed)

        # Register all the hydra resolvers.
        Hydra.register_resolvers()

        # Retrieve the hyperparameter configuration that must be provided to ray tune.
        hp_tuner = instantiate(hydra_config.agent.hp_tuner)
        rt_config = hp_tuner.get_ray_tune_config()

        # Init ray instance to use only num_cpus and num_gpus.
        # When using slurm, if num_cpus is set to 4, one should ensure to use --ntasks=1 --cpus-per-task=4 in sbatch
        # For gpus, if num_gpus is set to 4, one should ensure to use --gres=gpu:4 in sbatch
        ray.init(ignore_reinit_error=True, num_cpus=self.n_cpu, num_gpus=self.n_gpu)
        logging.info("Resources allocated to ray instance: {}".format(ray.cluster_resources()))

        # Allocate the resources for each trial. For example, if self.n_cpu=2 and self.n_gpu=1,
        # this will create 4 concurrent workers using each 0.5 cpu and 0.25 gpu
        # When using slurm, if mem is set to 15, one should ensure to use --mem=15G in sbatch
        n_gpus = self.n_gpu / self.n_cpu
        n_gpus = floor(n_gpus * 100) / 100
        memory = self.memory // self.n_cpu
        resources_per_trial = {"cpu": 1, "gpu": n_gpus, "memory": memory * 1024 ** 3}
        logging.info("Resources allocated to each of the {} concurrent trials: {}".format(
            self.n_cpu, resources_per_trial
        ))

        # No scheduling is performed as this may lead to inconsistent results when doing cross validation
        # (see https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.Repeater.html). The repeater unfortunately
        # does not support cross validation ATM, so we instead rely on using a trial_index parameter which is
        # grid searched as suggested in https://github.com/ray-project/ray/issues/7744#issuecomment-888148735
        searcher = BasicVariantGenerator(constant_grid_search=True)

        # TODO: tune.run will become deprecated in the next major version. We may want to migrate to Tuner.fit() before
        # TODO: any upgrade of ray tune.

        # Run the hyperparameter tuning
        result = tune.run(
            partial(self.train_agent, hydra_config=hydra_config, hp_tuner=hp_tuner),
            resources_per_trial=resources_per_trial,
            config=rt_config,
            num_samples=self.n_hp_samples,
            search_alg=searcher,
            verbose=0,
            local_dir=self.local_directory
        )

        # Save the results of all the trials in case we need it later on
        df = result.results_df
        to_drop = df.columns.difference([col for col in df if col.startswith('config/') or col == "loss"])
        df.drop(columns=to_drop, inplace=True)
        df.columns = df.columns.str.replace('config/', '')
        df.to_csv(f"{self.tensorboard_dir}/full_{self.save_file}", sep="\t", index=False)

        # Display and save the configuration with the better average metric value over n trial indexes
        group_cols = df.columns.difference(["loss", "trial_index"])
        best_query = "loss == loss.max()"
        best_trial = df.groupby(list(group_cols)).mean().reset_index().drop(columns="trial_index").query(best_query)
        logging.info(f"Best configuration of the hyperparameters: {best_trial}")
        best_trial.to_csv(f"{self.tensorboard_dir}/best_{self.save_file}", sep="\t", index=False)

    @staticmethod
    def train_agent(rt_config, hydra_config, hp_tuner):
        """
        Update the hydra configuration to fit the current hyperparameter configuration, and train the agent
        :param rt_config: the current hyperparameter configuration picked by ray tune
        :param hydra_config: the hydra configuration
        :param hp_tuner: an instance of the HyperparameterTuner class
        """

        # Register all the hydra resolvers again because this is not propagated to the subprocess training the agent.
        Hydra.register_resolvers()

        # Update the hydra configuration to fit the hyperparameters selected by ray tune.
        hp_tuner.update_hydra_config(rt_config, hydra_config)

        # Create the environment and agent.
        env = EnvFactory.make(hydra_config)
        agent = AgentFactory.make(hydra_config, env)

        # Train the agent on the environment
        agent.train(env, hydra_config)

        # Force ray to free any unused memory to avoid memory buildup
        auto_garbage_collect()
