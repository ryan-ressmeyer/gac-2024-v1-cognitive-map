import numpy as np
import matplotlib.pyplot as plt

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

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
    ax.text((x1+x2)*.5, y+h*1.1, text, ha='center', va='bottom', color='k')
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


def pvalue_to_stars(pval):
    """
    Convert p-value to star notation for significance display.

    Args:
        pval: p-value from statistical test

    Returns:
        str: Star notation (*** for p<0.001, ** for p<0.01, * for p<0.05, n.s. otherwise)
    """
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'n.s.'


def add_significance_window(ax, window, pval, y_pos=0.6, bar_height=0.02):
    """
    Add a horizontal bar showing the counting window with significance stars.

    Args:
        ax: matplotlib axis
        window: tuple of (start_time, end_time) for the counting window
        pval: p-value to display
        y_pos: y-position for the bar (default 0.6)
        bar_height: height of the bar in data coordinates
    """
    # Draw horizontal bar for the window
    ax.plot([window[0], window[1]], [y_pos, y_pos], 'k-', linewidth=3, solid_capstyle='butt')

    # Add vertical caps at the ends
    ax.plot([window[0], window[0]], [y_pos - bar_height/2, y_pos + bar_height/2], 'k-', linewidth=2)
    ax.plot([window[1], window[1]], [y_pos - bar_height/2, y_pos + bar_height/2], 'k-', linewidth=2)

    # Add p-value text with stars above the bar
    stars = pvalue_to_stars(pval)
    window_center = (window[0] + window[1]) / 2
    if stars == 'n.s.':
        text = f'{stars}\n(p={pval:.3f})'
    else:
        text = f'{stars}\n(p={pval:.1e})'
    ax.text(window_center, y_pos + 0.05, text,
            horizontalalignment='center', verticalalignment='bottom',
            fontsize=14)


