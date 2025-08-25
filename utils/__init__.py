import numpy as np
import matplotlib.pyplot as plt

def significance_connector(x1, x2, y, h, text, ax=None):
    """
    Draws a significance connector between two points on a plot.

    Args:
        x1 (float): The x-coordinate of the first point.
        x2 (float): The x-coordinate of the second point.
        y (float): The y-coordinate where the connector starts.
        h (float): The height of the connector.
        text (str): The text to display above the connector.
        ax (matplotlib.axes.Axes, optional): The axes to draw on. Defaults to None.
    """

    if ax is None:
        ax = plt.gca()
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='k')
def smooth(y, box_pts):
    """
    Smooths a 1D array using a moving average filter.

    Args:
        y (np.ndarray): The input 1D array to be smoothed.
        box_pts (int): The size of the moving average window.

    Returns:
        np.ndarray: The smoothed array.
    """
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

def loadmat(filename):
    """
    Load .mat file using the appropriate function based on the file version.

    Args:
        filename (str): The path to the .mat file.

    Returns:
        dict: The loaded data.
    """
    from scipy.io import loadmat as loadmat_scipy
    from mat73 import loadmat as loadmat_mat73

    try:
        return loadmat_scipy(filename)
    except Exception as e:
        return loadmat_mat73(filename)


