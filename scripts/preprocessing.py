"""
Preprocessing Script for V1 Cognitive Map Study
===============================================

This script loads and preprocesses neural and eye tracking data for both monkeys,
applying quality filters and signal processing, then saves to HDF5 for fast analysis.

PROCESSING STEPS:
-----------------
1. Load neural data (MUA) and behavioral metadata (ALLMAT)
2. Load and calibrate eye tracking data
3. Filter trials based on:
   - Eye position bounds (monkey-specific)
   - Eye tracker dropout (skipped frames)
4. Preprocess neural data:
   - Select high-SNR channels
   - Average across channels
   - Apply baseline correction
5. Preprocess eye tracking data:
   - Decimate from 30kHz to 1kHz
   - Center and filter (Savitzky-Golay)
   - Calculate velocity and speed
6. Save to HDF5 file for analysis scripts

USAGE:
------
    uv run python scripts/preprocessing.py

OUTPUT:
-------
    data/preprocessed_data.h5 - Contains all preprocessed data for both monkeys

Author: Ryan Ressmeyer
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import decimate, savgol_filter, savgol_coeffs, freqz
from tqdm import tqdm
from utils import loadmat, is_notebook
import h5py

if is_notebook():
    matplotlib.use('inline')
else:
    matplotlib.use('Qt5Agg')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# **IMPORTANT**: Update this path to your actual data directory
DATAGEN_DIR = '/home/ryanress/code/gac-2024-v1-cognitive-map/data/'

# Analysis parameters
MONKEYS = ['monkeyF', 'monkeyN']
TASKS = ['lums']  # Focus on luminance attention task
CURRENT_TASK = TASKS[0]

# Output file
OUTPUT_FILE = Path(DATAGEN_DIR) / 'preprocessed_data.h5'

# Signal processing parameters
SNR_THRESHOLD = 1.0  # Minimum signal-to-noise ratio for channel inclusion

EYE = 'right'  # Use right eye data
EYE_SAMPLING_RATE = 30000  # Original eye tracking sampling rate (Hz)
MUA_SAMPLING_RATE = 1000  # MUA sampling rate (Hz)
DECIMATION_FACTOR = int(EYE_SAMPLING_RATE / MUA_SAMPLING_RATE)  # Decimation factor

# Eye position unit conversion
PIXELS_PER_DEGREE = 26.9  # Conversion factor from screen pixels to degrees of visual angle

SAVGOL_WINDOW = 35  # Savitzky-Golay filter window (must be odd)
SAVGOL_ORDER = 3   # Savitzky-Golay polynomial order

# Trial filtering parameters
MAX_SKIPPED_FRAMES = 5  # Maximum allowed dropped frames
DPI_SAMPLING_RATE = 1/500  # Original DPI sampling rate before upsampling

# Figure output
SAVE_FIGS = True
if SAVE_FIGS:
    FIGURE_DIR = Path('./figures/preprocessing/')
    FIGURE_DIR.mkdir(exist_ok=True, parents=True)

# Create Path object for data directory
BASE_DATA_DIR = Path(DATAGEN_DIR)

#%%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_monkey_channels(monkey_name):
    """
    Get the channel range for a specific monkey.

    Parameters:
    -----------
    monkey_name : str
        Either 'monkeyF' or 'monkeyN'

    Returns:
    --------
    np.ndarray : Array channel indices for the specified monkey
    """
    if monkey_name == 'monkeyF':
        return np.arange(129, 193)  # Channels 129-192 (64 channels)
    else:  # monkeyN
        return np.arange(193, 257)  # Channels 193-256 (64 channels)

def load_monkey_data(base_dir, monkey_name, task_name):
    """
    Load all required data files for a specific monkey and task.

    This function consolidates the loading of neural data, eye tracking data,
    and calibration files for a given experimental subject.

    Parameters:
    -----------
    base_dir : Path
        Base data directory
    monkey_name : str
        Monkey identifier ('monkeyF' or 'monkeyN')
    task_name : str
        Task identifier (e.g., 'lums')

    Returns:
    --------
    dict : Dictionary containing all loaded data arrays and metadata
    """
    monkey_dir = base_dir / monkey_name

    # Load main trial data and neural activity
    mat_filename = monkey_dir / f"ObjAtt_GAC2_{task_name}_MUA_trials.mat"
    norm_mua_filename = monkey_dir / f"ObjAtt_GAC2_{task_name}_normMUA.mat"

    print(f"Loading data for {monkey_name}...")
    allmat_data = loadmat(mat_filename)
    norm_mua_data = loadmat(norm_mua_filename)

    # Load calibration data
    if monkey_name == 'monkeyN':
        cals_filename = monkey_dir / 'cals_monkeyN_20250218_B1.mat'
    else:
        cals_filename = monkey_dir / 'cals_monkeyF_20250228_B1.mat'

    cals_data = loadmat(cals_filename)

    return {
        'ALLMAT': allmat_data['ALLMAT'],
        'time_base_mua': allmat_data['tb'].flatten() / 1000,  # Convert ms to seconds
        'normMUA': norm_mua_data['normMUA'],
        'SNR': norm_mua_data['SNR'],
        'calibration': cals_data,
        'monkey_dir': monkey_dir
    }

def load_eye_tracking_data(monkey_dir, task_name, calibration_data, t_rel_stim):
    """
    Load and calibrate eye tracking data from all session files.

    Eye tracking data is stored in separate files for each recording session.
    This function loads all sessions and applies calibration transforms.

    Parameters:
    -----------
    monkey_dir : Path
        Directory containing monkey data files
    task_name : str
        Task name for file pattern matching
    calibration_data : dict
        Calibration matrices and offsets
    t_rel_stim : np.ndarray
        Time vector relative to stimulus onset (from MUA data)

    Returns:
    --------
    tuple : (dpi_left_trials, dpi_right_trials) - calibrated eye position data in degrees of visual angle
    """
    # Find all session files
    obj_att_files = list(monkey_dir.glob(f'ObjAtt_GAC2_{task_name}_*_B*.mat'))
    obj_att_files.sort(reverse=False)

    # Calculate sample counts to match MUA time base after decimation
    # Find stimulus onset (time = 0) in time base
    stimulus_onset_idx = np.argmin(np.abs(t_rel_stim))
    n_pre = stimulus_onset_idx * DECIMATION_FACTOR
    n_post = (len(t_rel_stim) - stimulus_onset_idx) * DECIMATION_FACTOR

    dpi_left_trials = []
    dpi_right_trials = []

    for session_file in obj_att_files:
        print(f'Processing session: {session_file.name}')

        # Load session data
        session_data = loadmat(session_file)

        # Load corresponding eye tracking files
        dpi_filename = session_file.parent / f'TEMP_DPI_{session_file.name[-15:]}'
        eye_filename = session_file.parent / f'TEMP_EYE_{session_file.name[-15:]}'

        dpi_data = loadmat(dpi_filename)
        eye_data = loadmat(eye_filename)
        eye_trial_indices = eye_data['Trials_corrected'].flatten()

        # Apply calibration transforms
        dpi_left_cal = (dpi_data['dpi_l'][:, :2] @ calibration_data['left_gains'] +
                       calibration_data['left_offset'])
        dpi_right_cal = (dpi_data['dpi_r'][:, :2] @ calibration_data['right_gains'] +
                        calibration_data['right_offset'])

        # Convert from pixels to degrees of visual angle
        dpi_left_cal = dpi_left_cal / PIXELS_PER_DEGREE
        dpi_right_cal = dpi_right_cal / PIXELS_PER_DEGREE

        # Extract trials with valid behavioral responses
        for trial_idx, eye_index in enumerate(eye_trial_indices):
            eye_index = int(eye_index)
            if session_data['MAT'][trial_idx, -1] >= 1:  # Valid trial criterion
                dpi_left_trials.append(dpi_left_cal[eye_index-n_pre:eye_index+n_post])
                dpi_right_trials.append(dpi_right_cal[eye_index-n_pre:eye_index+n_post])

    return np.stack(dpi_left_trials, axis=0), np.stack(dpi_right_trials, axis=0)

def filter_trials_by_skipped_frames(dpi_trials, max_skipped_frames, dpi_sampling_rate, eye_sampling_rate):
    """
    Identify trials with excessive skipped frames based on inter-frame intervals.

    This function uses an acceleration-based method to detect frames where the eye tracker
    likely dropped samples, leading to interpolated data. Trials with inter-frame intervals
    exceeding a threshold are flagged for exclusion.

    Parameters:
    -----------
    dpi_trials : np.ndarray
        Eye position data [trials x time x coordinates]
    max_skipped_frames : int
        Maximum allowable skipped frames before exclusion
    dpi_sampling_rate : float
        DPI sampling rate for inter-frame interval calculations
    eye_sampling_rate : int
        Eye tracking sampling rate

    Returns:
    --------
    np.ndarray : Boolean mask of valid trials
    """
    n_trials = dpi_trials.shape[0]
    valid_mask = np.ones(n_trials, dtype=bool)

    # Calculate maximum allowable inter-frame interval
    max_ifi = max_skipped_frames * dpi_sampling_rate

    for trial_idx in range(n_trials):
        trial_data = dpi_trials[trial_idx]
        trial_acceleration = np.linalg.norm(
            np.diff(np.diff(trial_data, axis=0, prepend=0), axis=0, prepend=0), axis=1
        )

        # Detect frames where acceleration changes (indicating real vs. interpolated data)
        acceleration_changes = np.diff(trial_acceleration, prepend=-1, append=1) > 1e-6
        frame_indices = np.where(acceleration_changes)[0]
        inter_frame_intervals = np.diff(frame_indices) / eye_sampling_rate

        if np.any(inter_frame_intervals > max_ifi):
            valid_mask[trial_idx] = False

    return valid_mask

def filter_trials_by_eye_position(dpi_trials, trial_indices_dict, position_limits):
    """
    Filter trials based on eye position bounds. This is helpful for removing trials
    with poor eye tracking quality that could confound analyses.

    Parameters:
    -----------
    dpi_trials : np.ndarray
        Eye position data in degrees of visual angle [trials x time x coordinates]
    trial_indices_dict : dict
        Dictionary with 'attended' and 'unattended' trial indices
    position_limits : dict
        Dictionary with position limits (in degrees) per attention condition

    Returns:
    --------
    np.ndarray : Boolean mask of valid trials
    """
    n_trials = dpi_trials.shape[0]
    valid_mask = np.ones(n_trials, dtype=bool)

    for trial_idx in range(n_trials):
        # Check position bounds based on attention condition
        if trial_idx in trial_indices_dict['attended']:
            # Check Y position bounds
            if np.any((dpi_trials[trial_idx, :, 1] < position_limits['attended_y_lim'][0]) |
                     (dpi_trials[trial_idx, :, 1] > position_limits['attended_y_lim'][1])):
                valid_mask[trial_idx] = False

            # Check X position bounds
            if np.any((dpi_trials[trial_idx, :, 0] < position_limits['attended_x_lim'][0]) |
                     (dpi_trials[trial_idx, :, 0] > position_limits['attended_x_lim'][1])):
                valid_mask[trial_idx] = False

        elif trial_idx in trial_indices_dict['unattended']:
            # Check Y position bounds
            if np.any((dpi_trials[trial_idx, :, 1] < position_limits['unattended_y_lim'][0]) |
                     (dpi_trials[trial_idx, :, 1] > position_limits['unattended_y_lim'][1])):
                valid_mask[trial_idx] = False

            # Check X position bounds
            if np.any((dpi_trials[trial_idx, :, 0] < position_limits['unattended_x_lim'][0]) |
                     (dpi_trials[trial_idx, :, 0] > position_limits['unattended_x_lim'][1])):
                valid_mask[trial_idx] = False

    return valid_mask

def preprocess_eye_data(dpi_trials, t_rel_stim, sampling_rate=EYE_SAMPLING_RATE,
                       decimation_factor=DECIMATION_FACTOR):
    """
    Preprocess eye position data: decimate, center, and calculate velocity.

    Raw eye data is collected at high sampling rates but contains noise.
    This function applies standard preprocessing steps for eye movement analysis.

    Scientific Rationale:
    - Decimation reduces computational load while preserving relevant eye movement dynamics
    - De-medianing centers the data around zero to account for session-to-session offsets
    - Savitzky-Golay filtering smooths data while preserving peak structure for microsaccade detection
    - Velocity calculation enables microsaccade detection via speed thresholding

    Parameters:
    -----------
    dpi_trials : np.ndarray
        Raw eye position data in degrees of visual angle [trials x time x coordinates]
    t_rel_stim : np.ndarray
        Time vector relative to stimulus onset (matches MUA time base)
    sampling_rate : int
        Original sampling rate (Hz)
    decimation_factor : int
        Factor by which to reduce sampling rate

    Returns:
    --------
    dict : Dictionary containing processed eye movement data (all in degrees or degrees/s for velocity)
    """
    # Decimate to reduce sampling rate (30 kHz -> 1 kHz)
    dpi_decimated = decimate(dpi_trials, decimation_factor, axis=1)

    # Center data by removing median across all trials and timepoints
    # This accounts for session-to-session baseline shifts in eye position
    dpi_centered = dpi_decimated - np.median(dpi_decimated, axis=(0, 1), keepdims=True)

    # Apply Savitzky-Golay filter for smoothing while preserving peak structure
    # This filter is optimal for derivative calculations needed for velocity
    dpi_filtered = savgol_filter(dpi_centered, SAVGOL_WINDOW, SAVGOL_ORDER, axis=1)

    dt = t_rel_stim[1] - t_rel_stim[0]  # Time step after decimation
    # Calculate velocity (first derivative) and speed (magnitude)
    dpi_velocity = np.diff(dpi_filtered, axis=1, prepend=dpi_filtered[:, :1, :]) / dt
    dpi_speed = np.linalg.norm(dpi_velocity, axis=2)

    # Use provided time vector (aligned with MUA data)
    new_sampling_rate = sampling_rate / decimation_factor

    return {
        'position_raw': dpi_centered,
        'position_filtered': dpi_filtered,
        'velocity': dpi_velocity,
        'speed': dpi_speed,
        'time_vector': t_rel_stim,
        'sampling_rate_new': new_sampling_rate
    }

#%%
# =============================================================================
# MAIN PREPROCESSING LOOP
# =============================================================================

print("="*70)
print("PREPROCESSING V1 COGNITIVE MAP DATA")
print("="*70)
print(f"Task: {CURRENT_TASK}")
print(f"Data directory: {BASE_DATA_DIR}")
print(f"Output file: {OUTPUT_FILE}")
print("="*70)

# Create HDF5 file for preprocessed data
with h5py.File(OUTPUT_FILE, 'w') as hf:

    # Loop over both monkeys
    for monkey_name in MONKEYS:
        print(f"\n{'='*70}")
        print(f"PREPROCESSING {monkey_name.upper()}")
        print('='*70)

        # =====================================================================
        # STEP 1: LOAD NEURAL AND BEHAVIORAL DATA
        # =====================================================================
        print(f"\n1. Loading neural and behavioral data for {monkey_name}...")
        monkey_data = load_monkey_data(BASE_DATA_DIR, monkey_name, CURRENT_TASK)

        # Extract loaded data
        ALLMAT = monkey_data['ALLMAT']
        t_rel_stim = monkey_data['time_base_mua']
        normalized_mua = monkey_data['normMUA']
        channel_snr = monkey_data['SNR']

        print(f"   - Loaded {ALLMAT.shape[0]} trials")
        print(f"   - Time vector: {len(t_rel_stim)} samples ({t_rel_stim[0]:.3f} to {t_rel_stim[-1]:.3f} s)")
        print(f"   - Neural data shape: {normalized_mua.shape}")

        # Validate and filter trials based on attention condition
        # ALLMAT column 4 (Python index 3): 2 = attended, 1 = unattended
        attention_condition = ALLMAT[:, 3]
        valid_attention_mask = (attention_condition == 1) | (attention_condition == 2)

        n_invalid_trials = np.sum(~valid_attention_mask)
        if n_invalid_trials > 0:
            print(f"   - WARNING: Found {n_invalid_trials} trials with invalid attention condition (not 1 or 2)")
            print(f"   - Filtering out invalid trials...")
            ALLMAT = ALLMAT[valid_attention_mask]
            normalized_mua = normalized_mua[:, valid_attention_mask]
            attention_condition = attention_condition[valid_attention_mask]
            print(f"   - Remaining trials after attention filtering: {ALLMAT.shape[0]}")

        # Create trial_attended vector: 1 for attended, 0 for unattended
        trial_attended = (attention_condition == 2).astype(np.int32)

        # Define attention conditions for later use
        attended_trials_mask = trial_attended == 1
        unattended_trials_mask = trial_attended == 0
        attended_trial_indices = np.where(attended_trials_mask)[0]
        unattended_trial_indices = np.where(unattended_trials_mask)[0]

        print(f"   - Attended trials: {attended_trial_indices.shape[0]}")
        print(f"   - Unattended trials: {unattended_trial_indices.shape[0]}")

        # =====================================================================
        # STEP 2: LOAD EYE TRACKING DATA
        # =====================================================================
        print(f"\n2. Loading and calibrating eye tracking data for {monkey_name}...")
        dpi_left_trials, dpi_right_trials = load_eye_tracking_data(
            monkey_data['monkey_dir'], CURRENT_TASK, monkey_data['calibration'], t_rel_stim
        )

        total_trials = len(dpi_right_trials)
        print(f"   - Total eye tracking trials: {total_trials}")
        print(f"   - Eye data shape: {dpi_right_trials.shape}")

        # Verify data alignment
        if total_trials != ALLMAT.shape[0]:
            raise ValueError(f"Mismatch between behavioral ({ALLMAT.shape[0]}) and eye tracking ({total_trials}) trials")

        # =====================================================================
        # STEP 3: FILTER TRIALS BY QUALITY
        # =====================================================================
        print(f"\n3. Filtering trials based on eye position and data quality for {monkey_name}...")

        # Get monkey-specific eye position limits (in degrees of visual angle)
        if monkey_name == 'monkeyN':
            eye_position_limits = {
                'attended_y_lim': [0, 14.87],        # Converted from [0, 400] pixels
                'unattended_y_lim': [9.29, 16.73],   # Converted from [250, 450] pixels
                'attended_x_lim': [-7.43, 7.43],     # Converted from [-200, 200] pixels
                'unattended_x_lim': [-7.43, 0]       # Converted from [-200, 0] pixels
            }
        else:  # monkeyF
            eye_position_limits = {
                'attended_y_lim': [-3.72, 11.15],    # Converted from [-100, 300] pixels
                'unattended_y_lim': [0, 14.87],      # Converted from [0, 400] pixels
                'attended_x_lim': [-7.43, 3.72],     # Converted from [-200, 100] pixels
                'unattended_x_lim': [-7.43, 3.72]    # Converted from [-200, 100] pixels
            }

        trial_indices_for_filtering = {
            'attended': attended_trial_indices,
            'unattended': unattended_trial_indices
        }

        # Filter trials with excessive skipped frames
        skipped_mask = filter_trials_by_skipped_frames(
            dpi_right_trials, MAX_SKIPPED_FRAMES, DPI_SAMPLING_RATE, EYE_SAMPLING_RATE)

        # Filter trials based on eye position limits
        position_mask = filter_trials_by_eye_position(
            dpi_right_trials, trial_indices_for_filtering, eye_position_limits)

        valid_trials_mask = skipped_mask & position_mask

        # Visualize raw eye position data after filtering
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0,0].plot(dpi_right_trials[attended_trials_mask & skipped_mask & position_mask,:,0].T, alpha=.1, c='k')
        axs[0,0].plot(dpi_right_trials[attended_trials_mask & skipped_mask & ~position_mask,:,0].T, alpha=.1, c='r')
        axs[0,0].axhline(y=eye_position_limits['attended_x_lim'][0], color='r', linestyle='--')
        axs[0,0].axhline(y=eye_position_limits['attended_x_lim'][1], color='r', linestyle='--')
        axs[0,0].set_title('Attended X Position')
        axs[0,0].set_ylabel('Position (degrees)')
        axs[0,1].plot(dpi_right_trials[attended_trials_mask & skipped_mask & position_mask,:,1].T, alpha=.1, c='k')
        axs[0,1].plot(dpi_right_trials[attended_trials_mask & skipped_mask & ~position_mask,:,1].T, alpha=.1, c='r')
        axs[0,1].axhline(y=eye_position_limits['attended_y_lim'][0], color='r', linestyle='--')
        axs[0,1].axhline(y=eye_position_limits['attended_y_lim'][1], color='r', linestyle='--')
        axs[0,1].set_title('Attended Y Position')
        axs[1,0].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & position_mask,:,0].T, alpha=.1, c='k')
        axs[1,0].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & ~position_mask,:,0].T, alpha=.1, c='r')
        axs[1,0].axhline(y=eye_position_limits['unattended_x_lim'][0], color='r', linestyle='--')
        axs[1,0].axhline(y=eye_position_limits['unattended_x_lim'][1], color='r', linestyle='--')
        axs[1,0].set_title('Unattended X Position')
        axs[1,0].set_ylabel('Position (degrees)')
        axs[1,1].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & position_mask,:,1].T, alpha=.1, c='k')
        axs[1,1].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & ~position_mask,:,1].T, alpha=.1, c='r')
        axs[1,1].axhline(y=eye_position_limits['unattended_y_lim'][0], color='r', linestyle='--')
        axs[1,1].axhline(y=eye_position_limits['unattended_y_lim'][1], color='r', linestyle='--')
        axs[1,1].set_title('Unattended Y Position')
        fig.suptitle(f'{monkey_name} - Raw Eye Position in Degrees (All Trials Overlayed)')
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig(FIGURE_DIR / f'{monkey_name}_raw_eye_position_post_filtering.png', dpi=300)
            fig.savefig(FIGURE_DIR / f'{monkey_name}_raw_eye_position_post_filtering.svg')
        plt.close(fig)

        # Apply filtering to all data arrays
        dpi_right_trials_filtered = dpi_right_trials[valid_trials_mask]
        dpi_left_trials_filtered = dpi_left_trials[valid_trials_mask]
        ALLMAT_filtered = ALLMAT[valid_trials_mask]
        normalized_mua_filtered = normalized_mua[:, valid_trials_mask]
        trial_attended_filtered = trial_attended[valid_trials_mask]

        # Update trial counts
        total_trials_filtered = np.sum(valid_trials_mask)
        print(f"   - Excluded {total_trials - total_trials_filtered} trials ({(total_trials - total_trials_filtered)/total_trials*100:.1f}%)")
        print(f"   - Remaining trials: {total_trials_filtered}")

        # Update variables to use filtered data
        ALLMAT = ALLMAT_filtered
        normalized_mua = normalized_mua_filtered
        dpi_right_trials = dpi_right_trials_filtered
        dpi_left_trials = dpi_left_trials_filtered
        trial_attended = trial_attended_filtered

        attended_trial_indices = np.where(trial_attended == 1)[0]
        unattended_trial_indices = np.where(trial_attended == 0)[0]

        # =====================================================================
        # STEP 4: PREPROCESS NEURAL DATA
        # =====================================================================
        print(f"\n4. Preprocessing neural data and selecting high-quality channels for {monkey_name}...")

        # Get monkey-specific channels
        array_channels = get_monkey_channels(monkey_name)

        # Create channel selection mask
        all_channels_mask = np.zeros(512, dtype=bool)
        all_channels_mask[array_channels - 1] = True

        # Select high SNR channels
        high_snr_channels_mask = (channel_snr.flatten() > SNR_THRESHOLD) & all_channels_mask
        n_selected_channels = np.sum(high_snr_channels_mask)

        print(f"   - Total available channels: {len(all_channels_mask)}")
        print(f"   - Monkey {monkey_name} channels: {np.sum(all_channels_mask)}")
        print(f"   - High SNR channels selected: {n_selected_channels}")
        print(f"   - SNR threshold: {SNR_THRESHOLD}")

        # Average across selected channels
        population_mua_raw = normalized_mua[high_snr_channels_mask, :, :].mean(axis=0)

        # Apply baseline correction
        pre_stimulus_samples = 200
        population_mua = (population_mua_raw -
                         population_mua_raw[:, :pre_stimulus_samples].mean(axis=1)[:, None])

        print(f"   - Population MUA shape: {population_mua.shape}")
        print(f"   - Applied baseline correction using first {pre_stimulus_samples} samples")

        # Visualization: Compare raw vs baseline-corrected data
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(population_mua_raw, vmin=-2, vmax=2, aspect='auto', cmap='coolwarm')
        plt.title('Raw Population MUA')
        plt.xlabel('Time (samples)')
        plt.ylabel('Trials')
        plt.colorbar(label='Normalized Activity')

        plt.subplot(1, 2, 2)
        plt.imshow(population_mua, vmin=-2, vmax=2, aspect='auto', cmap='coolwarm')
        plt.title('Baseline-Corrected Population MUA')
        plt.xlabel('Time (samples)')
        plt.ylabel('Trials')
        plt.colorbar(label='Normalized Activity')

        plt.suptitle(f'{monkey_name.capitalize()} - Neural Data Preprocessing')
        plt.tight_layout()
        if SAVE_FIGS:
            plt.savefig(FIGURE_DIR / f'{monkey_name}_neural_data_preprocessing.png', dpi=300)
            plt.savefig(FIGURE_DIR / f'{monkey_name}_neural_data_preprocessing.svg')
        plt.close(fig)

        # =====================================================================
        # STEP 5: PREPROCESS EYE MOVEMENT DATA
        # =====================================================================
        print(f"\n5. Preprocessing eye movement data for {monkey_name}...")

        # Display filter characteristics
        fig = plt.figure(figsize=(8, 5))
        filter_coefficients = savgol_coeffs(SAVGOL_WINDOW, SAVGOL_ORDER)
        frequencies, response = freqz(filter_coefficients)
        new_sampling_rate = EYE_SAMPLING_RATE / DECIMATION_FACTOR

        plt.plot(0.5 * new_sampling_rate * frequencies / np.pi, np.abs(response), 'b', linewidth=2)
        plt.title(f'Savitzky-Golay Filter Response\n(Order={SAVGOL_ORDER}, Window={SAVGOL_WINDOW}, Fs={new_sampling_rate} Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=50, color='r', linestyle='--', alpha=0.7, label='~50 Hz cutoff')
        plt.legend()
        plt.tight_layout()
        if SAVE_FIGS:
            plt.savefig(FIGURE_DIR / f'{monkey_name}_savgol_filter_response.png', dpi=300)
            plt.savefig(FIGURE_DIR / f'{monkey_name}_savgol_filter_response.svg')
        plt.close(fig)

        print(f"   - Filter preserves frequencies below ~50 Hz while attenuating noise")
        print(f"   - Processing {EYE} eye data with decimation factor {DECIMATION_FACTOR}...")

        # Apply preprocessing pipeline
        assert EYE in ['right', 'left'], "EYE must be either 'right' or 'left'"
        eye_data = preprocess_eye_data(
            dpi_right_trials if EYE == 'right' else dpi_left_trials,
            t_rel_stim=t_rel_stim
        )

        # Extract processed data
        eye_position_raw = eye_data['position_raw']
        eye_position_filtered = eye_data['position_filtered']
        eye_velocity = eye_data['velocity']
        eye_speed = eye_data['speed']
        eye_sampling_rate_new = eye_data['sampling_rate_new']

        print(f"   - Original sampling rate: {EYE_SAMPLING_RATE} Hz")
        print(f"   - New sampling rate: {eye_sampling_rate_new} Hz")
        print(f"   - Eye data shape: {eye_position_filtered.shape}")
        print(f"   - Time vector aligned with MUA: {t_rel_stim[0]:.3f} to {t_rel_stim[-1]:.3f} seconds")

        # =====================================================================
        # STEP 6: SAVE TO HDF5
        # =====================================================================
        print(f"\n6. Saving preprocessed data to HDF5 for {monkey_name}...")

        # Create group for this monkey
        monkey_group = hf.create_group(monkey_name)

        # Save preprocessed arrays
        monkey_group.create_dataset('population_mua', data=population_mua, compression='gzip')
        monkey_group.create_dataset('eye_position_raw', data=eye_position_raw, compression='gzip')
        monkey_group.create_dataset('eye_position_filtered', data=eye_position_filtered, compression='gzip')
        monkey_group.create_dataset('eye_velocity', data=eye_velocity, compression='gzip')
        monkey_group.create_dataset('eye_speed', data=eye_speed, compression='gzip')
        monkey_group.create_dataset('t_rel_stim', data=t_rel_stim, compression='gzip')
        monkey_group.create_dataset('trial_attended', data=trial_attended, compression='gzip')

        # Save metadata
        meta = monkey_group.create_group('metadata')
        meta.create_dataset('ALLMAT', data=ALLMAT, compression='gzip')
        meta.create_dataset('channel_snr', data=channel_snr, compression='gzip')
        meta.create_dataset('high_snr_channels_mask', data=high_snr_channels_mask, compression='gzip')
        meta.attrs['n_selected_channels'] = n_selected_channels
        meta.attrs['eye_sampling_rate_new'] = eye_sampling_rate_new
        meta.attrs['snr_threshold'] = SNR_THRESHOLD
        meta.attrs['decimation_factor'] = DECIMATION_FACTOR
        meta.attrs['savgol_window'] = SAVGOL_WINDOW
        meta.attrs['savgol_order'] = SAVGOL_ORDER
        meta.attrs['pre_stimulus_samples'] = pre_stimulus_samples
        meta.attrs['pixels_per_degree'] = PIXELS_PER_DEGREE  # Conversion factor from pixels to degrees
        meta.attrs['eye_position_units'] = 'degrees'  # Eye position data is in degrees of visual angle

        print(f"   - Saved {monkey_name} data to HDF5")
        print(f"   - Arrays saved: population_mua, eye_position_*, eye_velocity, eye_speed, t_rel_stim, trial_attended")
        print(f"   - Metadata saved: ALLMAT, channel info, processing parameters")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
print(f"Output file: {OUTPUT_FILE}")
print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")
print("\nData ready for analysis!")
print("="*70)
