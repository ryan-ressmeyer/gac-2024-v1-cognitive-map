import numpy as np

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


