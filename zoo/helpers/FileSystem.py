from os.path import isdir, isfile, join, exists
import os


class FileSystem:
    """
    A helper class providing useful functions to interact with the filesystem.
    """

    @staticmethod
    def files_in(directory):
        """
        Retrieve the files within the directory passed as parameters
        :param directory: the directory whose files must be returned
        :return: the files
        """

        # Create the list of files.
        files = []

        # Iterate over all directory entries.
        for entry in os.listdir(directory):

            # Add the current entry, if it is a file.
            if isfile(join(directory, entry)):
                files.append(entry)

        return files

    @staticmethod
    def sub_directories_of(directory):
        """
        Retrieve the subdirectories within the directory passed as parameters
        :param directory: the directory whose subdirectories must be returned
        :return: the subdirectories
        """

        # Create the list of subdirectories.
        subdirectories = []

        # Iterate over all directory entries.
        for entry in os.listdir(directory):

            # Add the current entry, if it is a directory.
            if isdir(join(directory, entry)):
                subdirectories.append(entry)

        return subdirectories

    @staticmethod
    def remove_empty_sub_directories_of(directory):
        """
        # Remove all the empty subdirectories of in the directory passed as input.
        :param directory: the directory whose empty subdirectories must be removed
        """

        # Iterate over all subdirectories of the directory passed as input (bottom-up).
        for root, dirs, files in os.walk(directory, topdown=False):

            # Exclude the files and directories that got deleted in the previous loop iterations.
            dirs = [directory for directory in dirs if exists(join(root, directory))]
            files = [file for file in files if exists(join(root, file))]

            # If the subdirectory is empty, remove it.
            if len(files) == 0 and len(dirs) == 0:
                os.rmdir(root)
