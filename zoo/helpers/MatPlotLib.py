import gc
import matplotlib.pyplot as plt
import torch


class MatPlotLib:
    """
    A helper class providing useful functions for interacting with matplotlib.
    """

    @staticmethod
    def close(fig=None):
        """
        Close the figure passed as parameter or the current figure
        :param fig: the figure to close
        """

        # Clear the current axes.
        plt.cla()

        # Clear the current figure.
        plt.clf()

        # Closes all the figure windows.
        plt.close('all')

        # Closes the matplotlib figure
        plt.close(plt.gcf() if fig is None else fig)

        # Forces the garbage collection
        gc.collect()

    @staticmethod
    def format_image(img):
        """
        Turn a 4d pytorch tensor into a 3d numpy array
        :param img: the 4d tensor
        :return: the 3d array
        """
        return torch.squeeze(img).detach().cpu().numpy()

    @staticmethod
    def save_figure(out_f_name, dpi=300, tight=True):
        """
        Save a matplotlib figure in an `out_f_name` file.
        :param str out_f_name: Name of the file used to save the figure.
        :param int dpi: Number of dpi, Default 300.
        :param bool tight: If True, use plt.tight_layout() before saving. Default True.
        """
        if tight is True:
            plt.tight_layout()
        plt.savefig(out_f_name, dpi=dpi, transparent=True)
        MatPlotLib.close()
