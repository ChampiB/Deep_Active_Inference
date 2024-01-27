import dataclasses
import pandas as pd
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent
from dataclasses import dataclass
import logging


@dataclass
class ScalarEventList:
    """
    Stores scalar events as a list using dataclass to ease unpacking
    """
    scalar_event_list: list[ScalarEvent]


class TensorBoard:
    """
    Class containing useful functions related to the TensorBoard monitoring framework.
    """

    @staticmethod
    def load_log_file(file, scalar_name):
        """
        Convert single tensorflow log file to pandas DataFrame
        :param file: path to tensorflow log file
        :param scalar_name: the name of the scalar entries in the tensorboard event file
        :return: a dataframe containing the logg file information
        """

        try:
            # Load all the data present in the log file.
            event_acc = EventAccumulator(file, size_guidance={
                "compressedHistograms": 1,
                "images": 1,
                "scalars": 0,
                "histograms": 1,
            })
            event_acc.Reload()

            event_list = dataclasses.astuple(ScalarEventList(event_acc.Scalars(scalar_name)))[0]
            _, steps, values = zip(*event_list)
            return steps, values

        except Exception as e:
            # Tell the user that a file could not be loaded.
            logging.error("Cannot process {}: {}".format(file, e))
            # traceback.print_exc()
            return [], []

    @staticmethod
    def load_log_directory(directory, df_path, values_name, scalar_name, return_df=True):
        """
        Load all the event file present in the directory
        :param directory: the target directory
        :param df_path: the path where the dataframe should be saved
        :param values_name: the name to give to the values extracted from the tensorboard event file
        :param scalar_name: the name of the scalar entries in the tensorboard event file
        :param return_df: whether to return a dataframe or not
        :return: a dataframe containing the scores of all the event files in the directory
        """

        # Walk through the directory's sub-folders recursively.
        all_steps, all_values = [], []
        for dir_path, _, files in os.walk(directory):
            for file in files:
                path = f"{dir_path}/{file}"
                logging.info(f"Processing file {path}")
                steps, values = TensorBoard.load_log_file(path, scalar_name)
                all_steps += steps
                all_values += values

        # Return a tuple containing the steps and associated values.
        if not return_df:
            return all_steps, all_values

        # Return a dataframe containing the steps and associated values.
        df = pd.DataFrame({"Steps": all_steps, values_name: all_values})
        if len(df.index) != 0:
            df.to_csv(df_path, sep="\t", index=False)
        return df

    @staticmethod
    def load_log_directories(directories, df_path, values_name, scalar_name):
        """
        Load all the event file present in the directory
        :param directories: the target directories
        :param df_path: the path where the dataframe should be saved
        :param values_name: the name to give to the values extracted from the tensorboard event file
        :param scalar_name: the name of the scalar entries in the tensorboard event file
        :return: a dataframe containing the scores of all the event files in the directory
        """

        # Walk through the directory's sub-folders recursively.
        all_steps, all_values = [], []
        for directory in directories:
            steps, values = TensorBoard.load_log_directory(directory, df_path, values_name, scalar_name, return_df=False)
            all_steps.append(steps)
            all_values.append(values)

        # Identify the largest list of steps.
        largest_steps = []
        for steps in all_steps:
            if len(steps) > len(largest_steps):
                largest_steps = steps

        # Unify the steps of the values.
        final_steps = []
        final_values = []
        for values in all_values:
            final_steps += largest_steps[:len(values)]
            final_values += values

        # Return a dataframe containing the steps and associated values.
        df = pd.DataFrame({"Steps": final_steps, values_name: final_values})
        if len(df.index) != 0:
            df.to_csv(df_path, sep="\t", index=False)
        return df

