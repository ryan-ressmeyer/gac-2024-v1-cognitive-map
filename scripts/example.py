#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    except NotImplementedError:
        return loadmat_mat73(filename)

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

def plot_mua_gac_dataset(datadir_gen, monkeys, tasks, snr_th):
    """
    Generates and displays plots for the MUA-GAC dataset.

    Args:
        datadir_gen (str): The base directory for the data.
        monkeys (list): A list of monkey identifiers.
        tasks (list): A list of task names.
        snr_th (float): The SNR threshold for channel selection.
    """
    # Create a Path object for the base data directory
    base_data_dir = Path(datadir_gen)

    # Loop through each monkey
    for monkey in monkeys:
        # Define channel ranges based on the monkey
        if monkey == 'monkeyF':
            array_chns = np.arange(129, 193)
        else:
            array_chns = np.arange(193, 257)
        
        # Create a boolean mask for the channels of the current monkey
        all_chns = np.zeros(512, dtype=bool)
        all_chns[array_chns - 1] = True # Subtract 1 for 0-based indexing

        # Loop through each task
        for task in tasks:
            # Construct file paths using pathlib
            mat_filename = base_data_dir / monkey / f"ObjAtt_GAC2_{task}_MUA_trials.mat"
            norm_mua_filename = base_data_dir / monkey / f"ObjAtt_GAC2_{task}_normMUA.mat"

            # Load the .mat files
            allmat_data = loadmat(mat_filename)
            ALLMAT = allmat_data['ALLMAT']
            tb = allmat_data['tb'].flatten()

            norm_mua_data = loadmat(norm_mua_filename)
            normMUA = norm_mua_data['normMUA']
            SNR = norm_mua_data['SNR'].flatten()

            # Set up task-specific parameters
            if task == 'lums':
                control_var_idx = 7  # 8 in MATLAB (1-based) -> 7 in Python (0-based)
                control_combs = np.array([[1, 3], [2, 5], [3, 1], [4, 6], [5, 2], [6, 4]])
            elif task == 'sacc':
                control_var_idx = 8  # 9 in MATLAB (1-based) -> 8 in Python (0-based)
                control_combs = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

            # Get unique control values
            controls = np.unique(ALLMAT[:, control_var_idx])
            
            # Define a boolean mask for channels that meet the SNR threshold
            snr_mask = (SNR > snr_th) & all_chns

            # --- Plot 1: Average activity for attended vs. unattended ---
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            
            # Condition for unattended trials (ALLMAT[:, 4] == 2)
            unattended_mask = ALLMAT[:, 3] == 2 # Index 3 for 4th column
            unattended_activity = np.nanmean(normMUA[np.ix_(snr_mask, unattended_mask)], axis=(0, 1))
            
            # Condition for attended trials (ALLMAT[:, 4] == 1)
            attended_mask = ALLMAT[:, 3] == 1 # Index 3 for 4th column
            attended_activity = np.nanmean(normMUA[np.ix_(snr_mask, attended_mask)], axis=(0, 1))

            ax1.plot(smooth(unattended_activity, 20), color=[.7, .6, .8], linewidth=2, label='Unattended')
            ax1.plot(smooth(attended_activity, 20), color=[.3, .7, .2], linewidth=2, label='Attended')

            # --- Formatting for Plot 1 ---
            ax1.set_xlim(0, 700)
            ax1.set_ylim(-0.3, 1.1)
            ax1.axvline(x=200, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=1)
            
            tick_indices = np.arange(0, len(tb), 100)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels(np.round(tb[tick_indices], 2))
            ax1.set_yticks([0, 1])

            ax1.set_ylabel('Normalized activity')
            ax1.set_xlabel('Time (ms)')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.legend()
            ax1.set_title(f'{monkey.capitalize()} - {task.capitalize()} Task: Overall Activity')
            fig1.tight_layout()

            # --- Plot 2: Activity split by control conditions ---
            fig2, axes2 = plt.subplots(1, len(controls), figsize=(5 * len(controls), 5), sharey=True)
            if len(controls) == 1: # Ensure axes2 is always a list
                axes2 = [axes2]

            for i, (ax, c_val) in enumerate(zip(axes2, controls)):
                control_comb = control_combs[i]

                # Unattended trials for the specific control condition
                unattended_mask = (ALLMAT[:, 3] == 2) & (ALLMAT[:, control_var_idx] == control_comb[1])
                unattended_activity = np.nanmean(normMUA[np.ix_(snr_mask, unattended_mask)], axis=(0, 1))

                # Attended trials for the specific control condition
                attended_mask = (ALLMAT[:, 3] == 1) & (ALLMAT[:, control_var_idx] == control_comb[0])
                attended_activity = np.nanmean(normMUA[np.ix_(snr_mask, attended_mask)], axis=(0, 1))
                
                ax.plot(smooth(unattended_activity, 20), color=[.7, .6, .8], linewidth=2)
                ax.plot(smooth(attended_activity, 20), color=[.3, .7, .2], linewidth=2)

                # --- Formatting for Subplots ---
                ax.set_ylim(-0.3, 1.4)
                ax.set_xlim(0, 700)
                ax.axvline(x=200, color=[0.5, 0.5, 0.5], linestyle='--', linewidth=1)

                ax.set_xticks(tick_indices)
                ax.set_xticklabels(np.round(tb[tick_indices], 2))
                ax.set_yticks([0, 1])

                if i == 0:
                    ax.set_ylabel('Normalized activity')
                ax.set_xlabel('Time (ms)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title(f'Control Condition {i+1}')

            fig2.suptitle(f'{monkey.capitalize()} - {task.capitalize()} Task: Activity by Control Condition', fontsize=16)
            fig2.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

if __name__ == '__main__':
    # --- Constants and Script Execution ---
    # Define the base directory for your data
    # IMPORTANT: Update this path to your actual data directory
    DATADIR_GEN = '/home/ryanress/code/Data_for_paper/'
    
    # Define the monkeys and tasks to be analyzed
    MONKEYS = ['monkeyF', 'monkeyN']
    TASKS = ['lums', 'sacc']
    
    # Set the Signal-to-Noise Ratio (SNR) threshold
    SNR_THRESHOLD = 1.0
    
    # Run the main plotting function
    plot_mua_gac_dataset(DATADIR_GEN, MONKEYS, TASKS, SNR_THRESHOLD)
# %%
