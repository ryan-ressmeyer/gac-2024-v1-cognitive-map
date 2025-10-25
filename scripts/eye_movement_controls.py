"""
Eye Movement Control Analysis for V1 Cognitive Map Study
=======================================================

This script analyzes the relationship between multi-unit activity (MUA) and eye movements 
during an object attention task to control for potential confounds in neural attention effects
due to movements of the eyes. The primary goal is to demonstrate that attentional modulation of 
V1 multi-unit activity is not driven by systematic differences in eye movements between attended 
and unattended conditions. This is critical for interpreting V1 as a cognitive map rather than simply 
reflecting oculomotor behavior.

ANALYSES:
-------------
1. Eye Position Control: Tests whether differences in mean gaze position between attention 
   conditions could explain neural differences
2. Microsaccade Control: Identifies and excludes trials with microsaccades during stimulus 
   presentation to eliminate feedforward saccadic effects
3. Drift Control (Path Length): Controls for overall ocular drift by analyzing neural 
   activity within quartiles of eye movement path length

DEPENDENCIES:
-------------
- numpy, matplotlib, scipy (signal processing)
- pathlib (file handling)
- tqdm (progress bars)
- Custom utils module (loadmat, smooth, significance_connector)

Install dependencies using uv:
    uv sync
    uv run python scripts/eye_movement_controls.py

DATA REQUIREMENTS:
------------------
The script expects the following directory structure:
    data/
    ├── monkeyN/
    │   ├── ObjAtt_GAC2_lums_MUA_trials.mat      # Trial metadata and time base
    │   ├── ObjAtt_GAC2_lums_normMUA.mat         # Normalized MUA data with SNR
    │   ├── cals_monkeyF_20250228_B1.mat         # Eye tracking calibration data
    │   ├── ObjAtt_GAC2_lums_*_B*.mat           # Session-specific trial data
    │   ├── TEMP_DPI_*_B*.mat                   # Eye position data (DPI format)
    │   └── TEMP_EYE_*_B*.mat                   # Eye tracking trial indices
    └── monkeyF/ (similar structure)

USAGE:
------
1. **CRITICAL**: Update the DATAGEN_DIR variable below to point to your local data directory
2. Configure analysis parameters in the Configuration section
3. Run the script: python scripts/eye_movement_controls.py

OUTPUTS:
--------
- Statistical comparisons of eye movements between attention conditions
- Microsaccade detection visualization (optional PDF export)
- Neural activity plots stratified by eye movement controls
- Demonstrates persistence of attention effects after controlling for eye movements

Author: Ryan Ressmeyer
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import decimate, savgol_filter, savgol_coeffs, freqz, find_peaks
from scipy.stats import ttest_ind
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from utils import loadmat, smooth, significance_connector, is_notebook

if is_notebook():
    matplotlib.use('inline')  # Use inline backend for scripts
else:
    matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plots
# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# **IMPORTANT**: Update this path to your actual data directory
DATAGEN_DIR = '/home/ryanress/code/gac-2024-v1-cognitive-map/data/'

# Analysis parameters
MONKEYS = ['monkeyF', 'monkeyN']
TASKS = ['lums']  # Focus on luminance attention task
CURRENT_MONKEY = 'monkeyF'  # Switch between 'monkeyF' and 'monkeyN'
CURRENT_TASK = TASKS[0]

# Signal processing parameters
SNR_THRESHOLD = 1.0  # Minimum signal-to-noise ratio for channel inclusion

EYE_SAMPLING_RATE = 30000  # Original eye tracking sampling rate (Hz)
EYE = 'right'  # Use right eye data (left not fully implemented yet)
DECIMATION_FACTOR = 30  # Downsample to 1 kHz for analysis
SAVGOL_WINDOW = 35  # Savitzky-Golay filter window (must be odd)
SAVGOL_ORDER = 3   # Savitzky-Golay polynomial order


# Microsaccade detection parameters
MICROSACCADE_THRESHOLD = 0.35  # Velocity threshold (deg/s)
MIN_SACCADE_INTERVAL = 100  # Minimum time between saccades (ms at 1kHz = samples)
EXPORT_SACCADE_PDF = True  # Export PDF with microsaccade detection for inspection
SAVE_FIGS = True  # Save figures to disk
if SAVE_FIGS:
    FIGURE_DIR = Path('./figures')
    FIGURE_DIR.mkdir(exist_ok=True, parents=True)

# Analysis time windows
PRE_STIMULUS_DURATION = 0.2  # seconds
POST_STIMULUS_DURATION = 0.5  # seconds
ATTENTION_WINDOW = [0.15, 0.5]
MICROSACCADE_WINDOW = [0, 0.5]  # Time window for microsaccade exclusion (s)
DRIFT_ANALYSIS_WINDOW = [0, 0.3]  # Time window for path length analysis (s)
EYE_POSITION_WINDOW = [0.3, 0.4]  # Time window for mean position analysis (s)

# Trial filtering parameters (monkey-specific eye position limits)
MAX_SKIPPED_FRAMES = 5  # Maximum allowed dropped frames
DPI_SAMPLING_RATE = 1/500  # Original DPI sampling rate before upsampling

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

def load_eye_tracking_data(monkey_dir, task_name, calibration_data):
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
        
    Returns:
    --------
    tuple : (dpi_left_trials, dpi_right_trials) - calibrated eye position data
    """
    # Find all session files
    obj_att_files = list(monkey_dir.glob(f'ObjAtt_GAC2_{task_name}_*_B*.mat'))
    obj_att_files.sort(reverse=False)
    
    # Time window for eye data extraction  
    n_pre = int(PRE_STIMULUS_DURATION * EYE_SAMPLING_RATE)
    n_post = int(POST_STIMULUS_DURATION * EYE_SAMPLING_RATE)
    
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
        
        # Extract trials with valid behavioral responses
        for trial_idx, eye_index in enumerate(eye_trial_indices):
            eye_index = int(eye_index)
            if session_data['MAT'][trial_idx, -1] >= 1:  # Valid trial criterion
                dpi_left_trials.append(dpi_left_cal[eye_index-n_pre:eye_index+n_post])
                dpi_right_trials.append(dpi_right_cal[eye_index-n_pre:eye_index+n_post])
    
    return np.stack(dpi_left_trials, axis=0), np.stack(dpi_right_trials, axis=0)

print("="*70)
print("EYE MOVEMENT CONTROL ANALYSIS")
print("="*70)
print(f"Analyzing monkey: {CURRENT_MONKEY}")
print(f"Task: {CURRENT_TASK}")
print(f"Data directory: {BASE_DATA_DIR}")
print("="*70)

print("\n1. Loading neural and behavioral data...")
monkey_data = load_monkey_data(BASE_DATA_DIR, CURRENT_MONKEY, CURRENT_TASK)

# Extract loaded data
ALLMAT = monkey_data['ALLMAT']  
time_base_mua = monkey_data['time_base_mua']
normalized_mua = monkey_data['normMUA']
channel_snr = monkey_data['SNR']

print(f"   - Loaded {ALLMAT.shape[0]} trials")
print(f"   - MUA time base: {len(time_base_mua)} samples")
print(f"   - Neural data shape: {normalized_mua.shape}")

# Define attention conditions based on ALLMAT column 4 (Python index 3)
# 1 = attended, 2 = unattended
attended_trials_mask = ALLMAT[:, 3] == 1
unattended_trials_mask = ALLMAT[:, 3] == 2
attended_trial_indices = np.where(attended_trials_mask)[0]
unattended_trial_indices = np.where(unattended_trials_mask)[0]

print(f"   - Attended trials: {attended_trial_indices.shape[0]}")
print(f"   - Unattended trials: {unattended_trial_indices.shape[0]}")

print("\n2. Loading and calibrating eye tracking data...")
dpi_left_trials, dpi_right_trials = load_eye_tracking_data(
    monkey_data['monkey_dir'], CURRENT_TASK, monkey_data['calibration']
)

total_trials = len(dpi_right_trials)
print(f"   - Total eye tracking trials: {total_trials}")
print(f"   - Eye data shape: {dpi_right_trials.shape}")

# Verify data alignment
if total_trials != ALLMAT.shape[0]:
    print(f"ERROR: Mismatch between behavioral ({ALLMAT.shape[0]}) and eye tracking ({total_trials}) trials")
    print("Truncating to the minimum number of trials...(this could be wrong)")
    min_trials = min(total_trials, ALLMAT.shape[0])
    ALLMAT = ALLMAT[:min_trials]
    normalized_mua = normalized_mua[:, :min_trials]
    dpi_left_trials = dpi_left_trials[:min_trials]
    dpi_right_trials = dpi_right_trials[:min_trials] 
    total_trials = min_trials
    
    # Recompute trial indices after truncation
    attended_trials_mask = ALLMAT[:, 3] == 1
    unattended_trials_mask = ALLMAT[:, 3] == 2
    attended_trial_indices = np.where(attended_trials_mask)[0]
    unattended_trial_indices = np.where(unattended_trials_mask)[0]
    print(f"   - Aligned to {total_trials} trials")

#%%
print("""
# =============================================================================  
# TRIAL FILTERING AND QUALITY CONTROL
# =============================================================================

This section filters out trials with poor eye tracking quality that could confound analyses.

FILTERING CRITERIA:
1. Eye Position Bounds: Removes trials where gaze deviated outside acceptable regions
   - Different limits for attended vs unattended conditions based on task requirements
   - MonkeyN has stricter limits based on experimental setup geometry
   
2. Eye Tracker Dropout Detection: Identifies trials with excessive missing eye tracking samples
   - Uses acceleration-based method to detect interpolated vs. real data points  
   - Excludes trials with inter-frame intervals exceeding physiological limits
""")


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
    sampling_rate : int
        Sampling rate for inter-frame interval calculations
        
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
            max_interval_ms = np.max(inter_frame_intervals) * 1000
            #print(f'Trial {trial_idx} excluded: max inter-frame interval ({max_ifi:.1f} ms) = {max_interval_ms:.3f} ms')
    
    return valid_mask

def filter_trials_by_eye_position(dpi_trials, trial_indices_dict, position_limits):
    """
    Filter trials based on eye position bounds. This is helpful for removing trials
    with poor eye tracking quality that could confound analyses.
       
    Parameters:
    -----------
    dpi_trials : np.ndarray
        Eye position data [trials x time x coordinates]
    trial_indices_dict : dict
        Dictionary with 'attended' and 'unattended' trial indices  
    position_limits : dict
        Dictionary with position limits per attention condition
    sampling_rate : int
        Sampling rate for inter-frame interval calculations
        
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
                #print(f'Trial {trial_idx} excluded: y position out of bounds')
                
            # Check X position bounds  
            if np.any((dpi_trials[trial_idx, :, 0] < position_limits['attended_x_lim'][0]) |
                     (dpi_trials[trial_idx, :, 0] > position_limits['attended_x_lim'][1])):
                valid_mask[trial_idx] = False
                #print(f'Trial {trial_idx} excluded: x position out of bounds')
                
        elif trial_idx in trial_indices_dict['unattended']:
            # Check Y position bounds
            if np.any((dpi_trials[trial_idx, :, 1] < position_limits['unattended_y_lim'][0]) |
                     (dpi_trials[trial_idx, :, 1] > position_limits['unattended_y_lim'][1])):
                valid_mask[trial_idx] = False
                #print(f'Trial {trial_idx} excluded: y position out of bounds')
                
            # Check X position bounds
            if np.any((dpi_trials[trial_idx, :, 0] < position_limits['unattended_x_lim'][0]) |
                     (dpi_trials[trial_idx, :, 0] > position_limits['unattended_x_lim'][1])):
                valid_mask[trial_idx] = False
                #print(f'Trial {trial_idx} excluded: x position out of bounds')

    return valid_mask

print("\n3. Filtering trials based on eye position and data quality...")

# Apply trial filtering using the refactored function
trial_indices_for_filtering = {
    'attended': attended_trial_indices,
    'unattended': unattended_trial_indices  
}

# Get monkey-specific eye position limits
if CURRENT_MONKEY == 'monkeyN':
    eye_position_limits = {
        'attended_y_lim': [0, 400],
        'unattended_y_lim': [250, 450], 
        'attended_x_lim': [-200, 200],
        'unattended_x_lim': [-200, 0]
    }
else:  # monkeyF - haven't set specific limits yet
    eye_position_limits = {
        'attended_y_lim': [-100, 300],
        'unattended_y_lim': [0, 400],
        'attended_x_lim': [-200, 100],
        'unattended_x_lim': [-200, 100]
    }

# Step 1: Filter trials with excessive skipped frames
skipped_mask = filter_trials_by_skipped_frames(
    dpi_right_trials, MAX_SKIPPED_FRAMES, DPI_SAMPLING_RATE, EYE_SAMPLING_RATE)

# Step 2: Filter trials based on eye position limits
position_mask = filter_trials_by_eye_position(
    dpi_right_trials, trial_indices_for_filtering, eye_position_limits)

valid_trials_mask = skipped_mask & position_mask

# Visualize raw eye position data after filtering to confirm quality
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0,0].plot(dpi_right_trials[attended_trials_mask & skipped_mask & position_mask,:,0].T, alpha = .1, c='k')
axs[0,0].plot(dpi_right_trials[attended_trials_mask & skipped_mask & ~position_mask,:,0].T, alpha = .1, c='r')
axs[0,0].axhline(y=eye_position_limits['attended_x_lim'][0], color='r', linestyle='--')
axs[0,0].axhline(y=eye_position_limits['attended_x_lim'][1], color='r', linestyle='--')
axs[0,0].set_title('Attended X Position')
axs[0,1].plot(dpi_right_trials[attended_trials_mask & skipped_mask & position_mask,:,1].T, alpha = .1, c='k')
axs[0,1].plot(dpi_right_trials[attended_trials_mask & skipped_mask & ~position_mask,:,1].T, alpha = .1, c='r')
axs[0,1].axhline(y=eye_position_limits['attended_y_lim'][0], color='r', linestyle='--')
axs[0,1].axhline(y=eye_position_limits['attended_y_lim'][1], color='r', linestyle='--')
axs[0,1].set_title('Attended Y Position')
axs[1,0].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & position_mask,:,0].T, alpha = .1, c='k')
axs[1,0].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & ~position_mask,:,0].T, alpha = .1, c='r')
axs[1,0].axhline(y=eye_position_limits['unattended_x_lim'][0], color='r', linestyle='--')
axs[1,0].axhline(y=eye_position_limits['unattended_x_lim'][1], color='r', linestyle='--')
axs[1,0].set_title('Unattended X Position') 
axs[1,1].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & position_mask,:,1].T, alpha = .1, c='k')
axs[1,1].plot(dpi_right_trials[unattended_trials_mask & skipped_mask & ~position_mask,:,1].T, alpha = .1, c='r')
axs[1,1].axhline(y=eye_position_limits['unattended_y_lim'][0], color='r', linestyle='--')
axs[1,1].axhline(y=eye_position_limits['unattended_y_lim'][1], color='r', linestyle='--')
axs[1,1].set_title('Unattended Y Position')
fig.suptitle('Raw Eye Position (All Trials Overlayed)')
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_raw_eye_position_post_filtering.png', dpi=300)
    fig.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_raw_eye_position_post_filtering.svg')
plt.show()

# Apply filtering to all data arrays
dpi_right_trials_filtered = dpi_right_trials[valid_trials_mask]
dpi_left_trials_filtered = dpi_left_trials[valid_trials_mask]
ALLMAT_filtered = ALLMAT[valid_trials_mask]
normalized_mua_filtered = normalized_mua[:, valid_trials_mask]

# Update trial counts and indices after filtering
total_trials_filtered = np.sum(valid_trials_mask)
attended_trials_mask_filtered = ALLMAT_filtered[:, 3] == 2
unattended_trials_mask_filtered = ALLMAT_filtered[:, 3] == 1
attended_trial_indices_filtered = np.where(attended_trials_mask_filtered)[0]
unattended_trial_indices_filtered = np.where(unattended_trials_mask_filtered)[0]

print(f"   - Excluded {total_trials - total_trials_filtered} trials ({(total_trials - total_trials_filtered)/total_trials*100:.1f}%)")
print(f"   - Remaining trials: {total_trials_filtered}")
print(f"   - Attended: {len(attended_trial_indices_filtered)}")
print(f"   - Unattended: {len(unattended_trial_indices_filtered)}")

# Update variables to use filtered data
total_trials = total_trials_filtered
ALLMAT = ALLMAT_filtered
normalized_mua = normalized_mua_filtered
dpi_right_trials = dpi_right_trials_filtered
dpi_left_trials = dpi_left_trials_filtered
attended_trial_indices = attended_trial_indices_filtered
unattended_trial_indices = unattended_trial_indices_filtered
attended_trials_mask = attended_trials_mask_filtered
unattended_trials_mask = unattended_trials_mask_filtered

#%%
print("""
# =============================================================================
# NEURAL DATA PREPROCESSING AND CHANNEL SELECTION  
# =============================================================================

This section prepares the neural data for analysis by:

1. CHANNEL SELECTION: Selects channels with sufficient SNR
   
2. BASELINE CORRECTION: Removes trial-to-trial DC offset variations
   - Subtracts pre-stimulus mean to eliminate baseline shifts
   - Necessary for preventing spurious correlations with eye movements because
     there is an early block of higher baseline activity that could correlate with drift
""")

print("\n4. Preprocessing neural data and selecting high-quality channels...")

# Get monkey-specific parameters
array_channels = get_monkey_channels(CURRENT_MONKEY)

# Create a boolean mask for the channels of the current monkey
all_channels_mask = np.zeros(512, dtype=bool)
all_channels_mask[array_channels - 1] = True  # Subtract 1 for 0-based indexing

# Create channel selection mask: high SNR channels from current monkey's array
high_snr_channels_mask = (channel_snr.flatten() > SNR_THRESHOLD) & all_channels_mask
n_selected_channels = np.sum(high_snr_channels_mask)

print(f"   - Total available channels: {len(all_channels_mask)}")
print(f"   - Monkey {CURRENT_MONKEY} channels: {np.sum(all_channels_mask)}")
print(f"   - High SNR channels selected: {n_selected_channels}")
print(f"   - SNR threshold: {SNR_THRESHOLD}")

# Average across selected channels to get population MUA response
# Shape: [trials x time] 
population_mua_raw = normalized_mua[high_snr_channels_mask, :, :].mean(axis=0)

# CRITICAL: Remove pre-stimulus baseline to eliminate DC offset variations
# This prevents eye movement artifacts from creating spurious neural correlations
pre_stimulus_samples = 200  # First 200 samples are pre-stimulus period
population_mua = (population_mua_raw - 
                         population_mua_raw[:, :pre_stimulus_samples].mean(axis=1)[:, None])

print(f"   - Population MUA shape: {population_mua.shape}")
print(f"   - Applied baseline correction using first {pre_stimulus_samples} samples")

# Visualization: Compare raw vs baseline-corrected data
plt.figure(figsize=(12, 5))
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

plt.suptitle(f'{CURRENT_MONKEY.capitalize()} - Neural Data Preprocessing')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_neural_data_preprocessing.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_neural_data_preprocessing.svg')
plt.show()

#%%
# =============================================================================
# BASELINE NEURAL ATTENTION EFFECT
# =============================================================================
print("""
This section establishes the core attention effect that we will control for.

ANALYSIS: Compare average neural activity between attended and unattended trials
- This is the primary effect we want to show persists after eye movement controls
- Provides the baseline against which we will compare controlled analyses
""")

print("\n5. Analyzing baseline attention effect on neural activity...")

# Calculate attention effect: attended vs unattended average responses

all_population_mua = population_mua.mean(axis=0)
attended_population_mua = population_mua[attended_trial_indices].mean(axis=0)
unattended_population_mua = population_mua[unattended_trial_indices].mean(axis=0)

print(f"   - Attended trials: {len(attended_trial_indices)} trials")
print(f"   - Unattended trials: {len(unattended_trial_indices)} trials")

# Create the baseline attention effect plot using our helper function  
plt.figure()
plt.plot(time_base_mua, all_population_mua, color='gray', label='All Trials', alpha=0.7)
plt.plot(time_base_mua, unattended_population_mua, color='b', linewidth=2, label='Unattended')
plt.plot(time_base_mua, attended_population_mua, color='r', linewidth=2, label='Attended')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
plt.grid()
plt.xlabel('Time (ms)')
plt.ylabel('Normalized MUA')
plt.legend()
plt.title(f'{CURRENT_MONKEY.capitalize()} - Baseline Attention Effect')
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_baseline_attention_effect.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_baseline_attention_effect.svg')
plt.show()

# Calculate and report attention effect magnitude
attention_window_mask = (time_base_mua >= ATTENTION_WINDOW[0]) & (time_base_mua <= ATTENTION_WINDOW[1])
attended_population_mua_window = population_mua[attended_trial_indices][:,attention_window_mask].mean(axis=1)
unattended_population_mua_window = population_mua[unattended_trial_indices][:,attention_window_mask].mean(axis=1)
attention_pvalue = ttest_ind(attended_population_mua_window, unattended_population_mua_window).pvalue
print(f'   - Mean attended activity: {attended_population_mua_window.mean():.3f}')
print(f'   - Mean unattended activity: {unattended_population_mua_window.mean():.3f}')
print(f'   - Attention effect magnitude (mean difference): {attended_population_mua_window.mean() - unattended_population_mua_window.mean():.3f}')
print(f'   - Mean attention effect p-value (t-test): {attention_pvalue:.3e}')

#%%
# =============================================================================
# EYE MOVEMENT DATA PREPROCESSING
# =============================================================================
print("""
This section processes the raw eye tracking data for subsequent analysis.

PROCESSING STEPS:
1. DECIMATION: Reduce sampling rate from 30 kHz to 1 kHz  
   - Original eye data was 500 Hz, upsampled to 30 kHz for alignment with neural data
   - 1 kHz provides sufficient temporal resolution for eye movement analysis
   
2. CENTERING: Remove median offset across all trials and timepoints
   - Accounts for session-to-session calibration differences
   - Centers eye position distributions around zero
   
3. SAVITZKY-GOLAY FILTERING: Smooth data while preserving microsaccade peaks
   - Optimal for calculating derivatives (velocity) needed for microsaccade detection that amplitfy noise
   - Common in eye tracking literature because it preserves signal shape better than simple moving averages
""")

def preprocess_eye_data(dpi_trials, sampling_rate=EYE_SAMPLING_RATE, 
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
        Raw eye position data [trials x time x coordinates] 
    sampling_rate : int
        Original sampling rate (Hz)
    decimation_factor : int
        Factor by which to reduce sampling rate
        
    Returns:
    --------
    dict : Dictionary containing processed eye movement data
    """
    # Decimate to reduce sampling rate (30 kHz -> 1 kHz)
    dpi_decimated = decimate(dpi_trials, decimation_factor, axis=1)
    
    # Center data by removing median across all trials and timepoints
    # This accounts for session-to-session baseline shifts in eye position
    dpi_centered = dpi_decimated - np.median(dpi_decimated, axis=(0, 1), keepdims=True)
    
    # Apply Savitzky-Golay filter for smoothing while preserving peak structure
    # This filter is optimal for derivative calculations needed for velocity
    dpi_filtered = savgol_filter(dpi_centered, SAVGOL_WINDOW, SAVGOL_ORDER, axis=1)
    
    # Calculate velocity (first derivative) and speed (magnitude)
    dpi_velocity = np.diff(dpi_filtered, axis=1, prepend=0)
    dpi_speed = np.linalg.norm(dpi_velocity, axis=2)
    
    # Create time vector for eye data
    new_sampling_rate = sampling_rate / decimation_factor
    n_pre_samples = int(PRE_STIMULUS_DURATION * sampling_rate)
    time_vector = (np.arange(dpi_filtered.shape[1]) / new_sampling_rate - 
                  n_pre_samples / sampling_rate)
    
    return {
        'position_raw': dpi_centered,
        'position_filtered': dpi_filtered, 
        'velocity': dpi_velocity,
        'speed': dpi_speed,
        'time_vector': time_vector,
        'sampling_rate_new': new_sampling_rate
    }

print("\n6. Preprocessing eye movement data...")

# Display filter characteristics for transparency
plt.figure(figsize=(8, 5))
filter_coefficients = savgol_coeffs(SAVGOL_WINDOW, SAVGOL_ORDER)
frequencies, response = freqz(filter_coefficients)
new_sampling_rate = EYE_SAMPLING_RATE / DECIMATION_FACTOR

plt.plot(0.5 * new_sampling_rate * frequencies / np.pi, np.abs(response), 'b', linewidth=2)
plt.title(f'Savitzky-Golay Filter Response\\n(Order={SAVGOL_ORDER}, Window={SAVGOL_WINDOW}, Fs={new_sampling_rate} Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True, alpha=0.3)
plt.axvline(x=50, color='r', linestyle='--', alpha=0.7, label='~50 Hz cutoff')
plt.legend()
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_savgol_filter_response.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_savgol_filter_response.svg')
plt.show()

print(f"   - Filter preserves frequencies below ~50 Hz while attenuating noise")
print(f"   - Processing {EYE} eye data with decimation factor {DECIMATION_FACTOR}...")

# Apply preprocessing pipeline using our helper function
assert EYE in ['right', 'left'], "EYE must be either 'right' or 'left'"
eye_data = preprocess_eye_data(dpi_right_trials if EYE == 'right' else dpi_left_trials)

# Extract processed data for easier access
eye_position_raw = eye_data['position_raw']
eye_position_filtered = eye_data['position_filtered'] 
eye_velocity = eye_data['velocity']
eye_speed = eye_data['speed']
time_vector_eye = eye_data['time_vector']
eye_sampling_rate_new = eye_data['sampling_rate_new']

print(f"   - Original sampling rate: {EYE_SAMPLING_RATE} Hz")
print(f"   - New sampling rate: {eye_sampling_rate_new} Hz")
print(f"   - Eye data shape: {eye_position_filtered.shape}")
print(f"   - Time vector range: {time_vector_eye[0]:.3f} to {time_vector_eye[-1]:.3f} seconds")
#%%

print("\nVisualizing sample eye position trials...")
print("Two things to look for:")
print("1. Presence of microsaccades")
print("2. Drift over time, especially in the y direction")


np.random.seed(104)

n_trials_to_plot = 5
attn_plot = np.random.permutation(attended_trial_indices)[:n_trials_to_plot]
unattn_plot = np.random.permutation(unattended_trial_indices)[:n_trials_to_plot]
t_mask = (0 < time_vector_eye) & (time_vector_eye < 500)
bins = np.linspace(-30, 30, 100)
ep_unattn = eye_position_filtered[unattn_plot]
ep_unattn_all = eye_position_filtered[unattended_trial_indices]
ep_attn = eye_position_filtered[attn_plot]
ep_attn_all = eye_position_filtered[attended_trial_indices]



fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0,0].plot(time_vector_eye, ep_unattn[:,:,0].T, alpha = .6, c='k')
axs[0,0].plot(time_vector_eye, ep_unattn[:,:,0].T, c='r', linewidth=2)
axs[0,0].axvline(x=0, color='k', linestyle='--')
axs[0,0].set_title('Unattended X')
axs[0,0].set_ylim(-20, 20)
axs[0,1].plot(time_vector_eye, ep_unattn[:,:,1].T, alpha = .6, c='k')
axs[0,1].plot(time_vector_eye, ep_unattn[:,:,1].T, c='r', linewidth=2)
axs[0,1].axvline(x=0, color='k', linestyle='--')
axs[0,1].set_title('Unattended Y')
axs[0,1].set_ylim(-20, 20)
h1 = axs[0,2].hist2d(ep_unattn_all[:,t_mask,0].flatten(), ep_unattn_all[:,t_mask,1].flatten(), bins=[bins, bins], cmap='inferno')
axs[0,2].set_title('Unattended 2D Histogram (0-500 ms)')
axs[1,0].plot(time_vector_eye, ep_attn[:,:,0].T, alpha = .6, c='k')
axs[1,0].plot(time_vector_eye, ep_attn[:,:,0].T, c='b', linewidth=2)
axs[1,0].axvline(x=0, color='k', linestyle='--')
axs[1,0].set_title('Attended X')
axs[1,0].set_ylim(-20, 20)
axs[1,1].plot(time_vector_eye, ep_attn[:,:,1].T, alpha = .6, c='k')
axs[1,1].plot(time_vector_eye, ep_attn[:,:,1].T, c='b', linewidth=2)
axs[1,1].axvline(x=0, color='k', linestyle='--')
axs[1,1].set_title('Attended Y')
axs[1,1].set_ylim(-20, 20)
h2 = axs[1,2].hist2d(ep_attn_all[:,t_mask,0].flatten(), ep_attn_all[:,t_mask,1].flatten(), bins=[bins, bins], cmap='inferno')
axs[1,2].set_title('Attended 2D Histogram (0-500 ms)')
fig.tight_layout()
fig.colorbar(h1[3], ax=axs[0,2], label='Counts')
fig.colorbar(h2[3], ax=axs[1,2], label='Counts')
if SAVE_FIGS:
    fig.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_sample_eye_position_trials.png', dpi=300)
    fig.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_sample_eye_position_trials.svg')
plt.show()

#%%

print("""
# =============================================================================
# CONTROL 1: OVERALL EYE POSITION ANALYSIS
# =============================================================================

Test whether systematic differences in mean gaze position between attention conditions
could account for observed neural differences.

HYPOTHESIS: If attention effects are attributable to differences in eye position, then:
1. Attended and unattended trials should have different mean eye positions
2. Neural activity should vary systematically with eye position  
3. Attention effects should disappear when controlling for eye position

METHOD:
- Compare mean eye position between attention conditions during stimulus period
- Divide trials into quartiles based on eye position
- Test for attention effects within each position quartile
""")

print("\n7. CONTROL 1: Testing for differences in eye position between attention conditions...")

# Calculate mean and 95% confidence intervals for eye position in each condition
# Note: Using filtered, centered, and smoothed eye position data for this analysis
mean_dpi_attended = np.mean(eye_position_filtered[attended_trials_mask], axis=0)
ste_dpi_attended = np.std(eye_position_filtered[attended_trials_mask], axis=0) / np.sqrt(len(attended_trial_indices))
mean_dpi_unattended = np.mean(eye_position_filtered[unattended_trials_mask], axis=0)
ste_dpi_unattended = np.std(eye_position_filtered[unattended_trials_mask], axis=0) / np.sqrt(len(unattended_trial_indices))


# Plot the mean eye position over time for both attention conditions
plt.figure(figsize=(10, 10))

# X-Position Plot
plt.subplot(2, 1, 1)
# Plot 95% confidence interval for unattended trials
plt.fill_between(time_vector_eye, 
                 mean_dpi_unattended[:,0] - ste_dpi_unattended[:,0]*1.96, 
                 mean_dpi_unattended[:,0] + ste_dpi_unattended[:,0]*1.96,
                 color='b', alpha=0.2, label='Unattended - 95% CI')
plt.plot(time_vector_eye, mean_dpi_unattended[:,0], c='b', label='Unattended - Mean')
# Plot 95% confidence interval for attended trials
plt.fill_between(time_vector_eye, 
                 mean_dpi_attended[:,0] - ste_dpi_attended[:,0]*1.96, 
                 mean_dpi_attended[:,0] + ste_dpi_attended[:,0]*1.96,
                 color='r', alpha=0.2, label='Attended - 95% CI')
plt.plot(time_vector_eye, mean_dpi_attended[:,0], c='r', label='Attended - Mean')
plt.legend()
plt.title(f'{CURRENT_MONKEY.capitalize()} - {CURRENT_TASK.capitalize()} Task: Eye Position X')
plt.axvline(x=0, color='k', linestyle='--')
plt.ylabel('Horizontal Position (pixels)')
plt.ylim(-15, 15)
plt.grid(True, alpha=0.3)


# Y-Position Plot
plt.subplot(2, 1, 2)
plt.fill_between(time_vector_eye, 
                 mean_dpi_unattended[:,1] - ste_dpi_unattended[:,1]*1.96, 
                 mean_dpi_unattended[:,1] + ste_dpi_unattended[:,1]*1.96,
                 color='b', alpha=0.2, label='Unattended - 95% CI')
plt.plot(time_vector_eye, mean_dpi_unattended[:,1], c='b', label='Unattended - Mean')
plt.fill_between(time_vector_eye, 
                 mean_dpi_attended[:,1] - ste_dpi_attended[:,1]*1.96, 
                 mean_dpi_attended[:,1] + ste_dpi_attended[:,1]*1.96,
                 color='r', alpha=0.2, label='Attended - 95% CI')
plt.plot(time_vector_eye, mean_dpi_attended[:,1], c='r', label='Attended - Mean')
# Highlight the analysis window for mean position
plt.fill_betweenx([-15, 15], EYE_POSITION_WINDOW[0], EYE_POSITION_WINDOW[1], color='g', alpha=0.1, label='Position Window', zorder=0)
plt.axvline(x=0, color='k', linestyle='--')
plt.ylim(-15, 15)
plt.legend()
plt.title('Y-Position')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Position (pixels)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_eye_position_by_attention_condition.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_eye_position_by_attention_condition.svg')
plt.show()  

print("   - Green shaded region indicates time window for position analysis.")
print("   - X position shows no significant difference between attention conditions.")
print("   - Y position shows a small but consistent difference between conditions.")
print("   - Next, we will test if this small difference can explain the attention effect.")

#%%
# Stratify by eye position to control for the observed difference.
# We will split the distribution of Y eye positions into quartiles and check if the 
# attention effect is present within each quartile.

print("\n   Analyzing MUA within eye position quartiles...")
# Define the time mask for calculating mean eye position per trial
pos_mask = (time_vector_eye >= EYE_POSITION_WINDOW[0]) & (time_vector_eye <= EYE_POSITION_WINDOW[1])

# Calculate the mean Y position for each trial within the defined window
trial_pos_y = eye_position_filtered[:, pos_mask, 1].mean(axis=1)

# Compare mean positions between conditions
attn_pos_y = trial_pos_y[attended_trials_mask].mean()
unattn_pos_y = trial_pos_y[unattended_trials_mask].mean()
pval = ttest_ind(trial_pos_y[attended_trials_mask], trial_pos_y[unattended_trials_mask]).pvalue
print(f'   - Mean attended Y position: {attn_pos_y:.2f} pixels')
print(f'   - Mean unattended Y position: {unattn_pos_y:.2f} pixels')
print(f'   - T-test for difference in mean Y position: p = {pval:.3f}')

# Define quartile bins based on the overall distribution of eye positions
n_bins = 4
edges = np.percentile(trial_pos_y, np.linspace(0, 100, n_bins + 1))
print(f'   - Y position bin edges (quartiles): {np.round(edges, 2)}')

# Plot the distribution of eye positions for visualization
hist_bins = np.linspace(edges[0], edges[-1], 30)
plt.figure(figsize=(8, 6))
plt.hist(trial_pos_y, bins=hist_bins, label='All trials', color='gray', alpha=0.6)
plt.hist(trial_pos_y[attended_trials_mask], bins=hist_bins, alpha=0.7, label='Attended', color='r')
plt.hist(trial_pos_y[unattended_trials_mask], bins=hist_bins, alpha=0.7, label='Unattended', color='b')

# Add mean indicators and significance connector
plt.scatter([attn_pos_y], [60], color='r', marker='v', s=100, label='Attended Mean', zorder=5)
plt.scatter([unattn_pos_y], [60], color='b', marker='v', s=100, label='Unattended Mean', zorder=5)
significance_connector(attn_pos_y, unattn_pos_y, 65, 5, f'p={pval:.3f}' if pval < 0.05 else 'n.s.')
for e in edges:
    plt.axvline(x=e, color='k', linestyle='--', label='Quartile edges' if e == edges[0] else None, alpha=0.5)
plt.legend()
plt.title('Distribution of Mean Y Eye Position (0.3-0.4s)')
plt.xlabel('Mean Vertical Eye Position (pixels)')
plt.ylabel('Number of Trials')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_eye_position_distribution.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_eye_position_distribution.svg')
plt.show()

print("   - The difference in mean eye position is small compared to the overall distribution.")
print("   - Now, let's see if the MUA attention effect persists within each position quartile.")

#%%
# Plot MUA for attended vs. unattended trials within each eye position quartile
fig, axs = plt.subplots(n_bins+1, 1, figsize=(8, 12), sharex=True, sharey=True)
fig.suptitle(f'{CURRENT_MONKEY.capitalize()} - MUA Conditioned on Eye Position', y=1.02, fontsize=16)


for i in range(n_bins):
    e0, e1 = edges[i], edges[i+1]
    # Create a mask for trials within the current position quartile
    quartile_mask = (trial_pos_y >= e0) & (trial_pos_y <= e1)
    
    # Calculate MUA for all, attended, and unattended trials in this quartile
    all_activity = population_mua[quartile_mask].mean(axis=0)
    attn_activity = population_mua[quartile_mask & attended_trials_mask]
    unattn_activity = population_mua[quartile_mask & unattended_trials_mask]
    pval = ttest_ind(attn_activity[:, attention_window_mask].mean(axis=1), 
                     unattn_activity[:, attention_window_mask].mean(axis=1)).pvalue
    
    # Plotting
    axs[0].plot(time_base_mua, all_activity, label=f'Quartile {i+1}: [{e0:.1f}, {e1:.1f}] pixels')
    axs[i+1].plot(time_base_mua, all_activity, c='gray', alpha=0.5, label='All trials')
    axs[i+1].plot(time_base_mua, attn_activity.mean(axis=0), label='Attended', color='r')
    axs[i+1].plot(time_base_mua, unattn_activity.mean(axis=0), label='Unattended', color='b')
    axs[i+1].axvline(x=0, color='k', linestyle='--')
    axs[i+1].set_ylim(-.3, 1.3)
    axs[i+1].axhline(y=0, color='k', linestyle='--', alpha=0.3, zorder=0)
    axs[i+1].set_title(f'Quartile {i+1}: Y Position [{e0:.1f}, {e1:.1f}] pixels - ' + (f'p={pval:.1e}' if pval < 0.05 else 'n.s.'))
    axs[i+1].grid(True, alpha=0.2)
    if i == 0:
        axs[i+1].legend()

axs[0].set_ylabel('Normalized MUA')
axs[0].set_title('All Trials by Eye Position Quartile')
axs[0].legend()
axs[-1].set_xlabel('Time (s)')
fig.text(0.04, 0.5, 'Normalized MUA', va='center', rotation='vertical')
plt.tight_layout(rect=[0.05, 0, 1, 1])
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_mua_by_eye_position_quartile.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_mua_by_eye_position_quartile.svg')
plt.show()

print("\nCONCLUSION for Control 1:")
print("   - A robust attention effect (Attended > Unattended) is present within each eye position quartile.")
print("   - Therefore, the V1 attention effect is NOT driven by the small systematic difference in mean eye position.")

#%%
print("""
# =============================================================================
# CONTROL 2: MICROSACCADE ANALYSIS
# =============================================================================

Test whether microsaccades during stimulus presentation could confound attention effects.

BACKGROUND:
Microsaccades are small, rapid eye movements (0.1-2°) that occur even during fixation.
They can influence neural activity through both feedforward effects on the retinal image and 
corollary discharge signals. Previous studies have shown that microsaccade rate and direction
can be modulated by attention, potentially confounding neural attention effects.

HYPOTHESIS: If attention effects are due to microsaccade differences, then including only trials 
without microsaccades during stimulus presentation should eliminate the attention effect.

METHOD:
- Detect microsaccades using velocity threshold method
- Identify trials with microsaccades during stimulus period (0-400ms)
- Compare attention effects in trials with vs without microsaccades
""")

print("\n8. CONTROL 2: Microsaccade detection and analysis...")  
print("   Using velocity thresholding method for microsaccade detection")
print(f"   - Threshold: {MICROSACCADE_THRESHOLD} pixels/ms")
print(f"   - Minimum intersaccade distance: {MIN_SACCADE_INTERVAL} ms")
print(f"   - Analysis window: {MICROSACCADE_WINDOW[0]}-{MICROSACCADE_WINDOW[1]} s")

# Initialize lists to store detected saccade properties
saccade_trials = []
saccade_samples = []
saccade_times = []

# Optionally create a PDF for manual inspection of detection quality
if EXPORT_SACCADE_PDF:
    pdf_filename = f'{CURRENT_MONKEY}_{CURRENT_TASK}_microsaccades_inspection.pdf'
    print(f"   - Exporting saccade detection plots to: {pdf_filename}")
    pdf = PdfPages(pdf_filename)

try:
    # Loop through each trial to detect microsaccades
    for iT in tqdm(range(total_trials), desc="Detecting microsaccades"):
        # Use scipy's find_peaks on the eye speed signal
        peaks, _ = find_peaks(eye_speed[iT, :], height=MICROSACCADE_THRESHOLD, distance=MIN_SACCADE_INTERVAL)
        
        # Store results if any peaks are found
        if len(peaks) > 0:
            saccade_samples.append(peaks)
            saccade_trials.append(np.ones(len(peaks)) * iT)
            saccade_times.append(time_vector_eye[peaks])

        # If exporting, create a plot for the current trial and save to PDF
        if EXPORT_SACCADE_PDF:
            fig = plt.figure(figsize=(10, 6))
            # Plot X and Y position
            plt.subplot(2, 1, 1)
            plt.plot(time_vector_eye, eye_position_filtered[iT, :, 0], label='X-pos')
            plt.plot(time_vector_eye, eye_position_filtered[iT, :, 1], label='Y-pos', color='gray')
            if len(peaks) > 0:
                for p_time in time_vector_eye[peaks]:
                    plt.axvline(x=p_time, color='r', linestyle='--', alpha=0.7)
            plt.title(f'Trial {iT} - Eye Position')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            # Plot eye speed and detected peaks
            plt.subplot(2, 1, 2)
            plt.plot(time_vector_eye, eye_speed[iT, :], label='Speed')
            if len(peaks) > 0:
                plt.plot(time_vector_eye[peaks], eye_speed[iT, peaks], 'rx', label='Detected Saccade')
            plt.axhline(y=MICROSACCADE_THRESHOLD, color='r', linestyle='--', label='Threshold')
            plt.ylim(0, 1)
            plt.title('Eye Speed')
            plt.xlabel('Time (s)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
finally:
    if EXPORT_SACCADE_PDF:
        pdf.close()

# Concatenate results from all trials into single numpy arrays
saccade_trials = np.concatenate(saccade_trials).astype(int)
saccade_times = np.concatenate(saccade_times)

print(f'   - Detected {len(saccade_times)} microsaccades across {total_trials} trials')

#%%
# Visualize microsaccade timing across all trials
fig, axs = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[2, 1], sharex=True)
fig.suptitle("Microsaccade Timing Across All Trials", fontsize=16)

# Raster plot of microsaccade times for each trial
axs[0].eventplot([saccade_times[saccade_trials==iT] for iT in range(total_trials)], linelengths=5, linewidths=2, colors='k')
axs[0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
axs[0].set_xlim(time_vector_eye[0], time_vector_eye[-1])
axs[0].set_ylim(0, total_trials)
axs[0].set_ylabel('Trial')
axs[0].set_title('Microsaccade Raster Plot')
axs[0].grid(True, axis='x', alpha=0.3)

# Histogram of microsaccade times
axs[1].hist(saccade_times, bins=np.linspace(time_vector_eye[0], time_vector_eye[-1], 70), color='k')
axs[1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Stimulus Onset')
axs[1].annotate('Pre-stimulus\nmicrosaccades', xy=(-.1, 20), xytext=(-.05, 40),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='center')
axs[1].annotate('Post-stimulus\nsuppression', xy=(0.1, 5), xytext=(0.15, 30),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='center')
axs[1].annotate('Choice saccades', xy=(.45, 9), xytext=(.35, 40),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='center')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Number of Saccades')
axs[1].set_title('Microsaccade Rate Histogram')
axs[1].legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_microsaccade_timing.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_microsaccade_timing.svg')
plt.show()

print("   - Note the characteristic suppression of microsaccades shortly after stimulus onset.")
print("   - Since feedforward effects are most critical, we will exclude trials with any microsaccades in the early stimulus window.")

#%%
# Compare attention effect in trials with vs. without microsaccades in the critical window.

# Identify trials that have at least one microsaccade within the specified window
msacc_window_mask = (saccade_times > MICROSACCADE_WINDOW[0]) & (saccade_times < MICROSACCADE_WINDOW[1])
trials_with_msacc = np.unique(saccade_trials[msacc_window_mask])

# Create a boolean mask for all trials
msacc_trials_mask = np.zeros(total_trials, dtype=bool)
msacc_trials_mask[trials_with_msacc] = True
no_msacc_trials_mask = ~msacc_trials_mask

print(f'   - {np.sum(msacc_trials_mask)} trials WITH microsaccades between {MICROSACCADE_WINDOW[0]}-{MICROSACCADE_WINDOW[1]} s')
print(f'   - {np.sum(no_msacc_trials_mask)} trials WITHOUT microsaccades in this window.')

# Calculate MUA for trials without microsaccades, separated by attention
no_saccade_attended_mua = population_mua[no_msacc_trials_mask & attended_trials_mask].mean(axis=0)
no_saccade_unattended_mua = population_mua[no_msacc_trials_mask & unattended_trials_mask].mean(axis=0)

pval = ttest_ind(
    population_mua[no_msacc_trials_mask & attended_trials_mask][:, attention_window_mask].mean(axis=1),
    population_mua[no_msacc_trials_mask & unattended_trials_mask][:, attention_window_mask].mean(axis=1)
).pvalue

print(f'   - Attention effect in no-microsaccade trials: p = {pval:.3e}')

# Calculate MUA for saccade vs. no-saccade trials (collapsed across attention)
saccade_mua = population_mua[msacc_trials_mask].mean(axis=0)
no_saccade_mua = population_mua[no_msacc_trials_mask].mean(axis=0)

# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Controlling for Microsaccade Effects", fontsize=16)

# Plot 1: Attention effect in trials WITHOUT microsaccades
axs[0].plot(time_base_mua, no_saccade_unattended_mua, label='Unattended', color='b')
axs[0].plot(time_base_mua, no_saccade_attended_mua, label='Attended', color='r')
axs[0].axvline(x=0, color='k', linestyle='--')
axs[0].set_title(f'Attention Effect ({np.sum(no_msacc_trials_mask)} Trials WITHOUT Microsaccades)')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Normalized MUA')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Plot 2: MUA difference between saccade and no-saccade trials
axs[1].plot(time_base_mua, saccade_mua, label='Saccade Trials', color='g')
axs[1].plot(time_base_mua, no_saccade_mua, label='No Saccade Trials', color='k')
axs[1].axvline(x=0, color='k', linestyle='--')
axs[1].set_title('MUA: Saccade vs. No Saccade Trials')
axs[1].set_xlabel('Time (s)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_mua_control_microsaccades.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_mua_control_microsaccades.svg')
plt.show()

print("\nCONCLUSION for Control 2:")
print(f"   - A strong attention effect persists in trials completely free of microsaccades during the stimulus period (p={pval:.1e}).")
print("   - Therefore, the V1 attention effect is NOT driven by microsaccadic eye movements.")

# %%

print("""
# =============================================================================
# CONTROL 3: OCULAR DRIFT (PATH LENGTH) ANALYSIS  
# =============================================================================

Test whether differences in ocular drift between attention conditions could account
for observed neural differences.

BACKGROUND:
Even during attempted fixation, the eyes exhibit slow drift movements. This drift can:
1. Change the retinal image location continuously
2. Activate different populations of neurons with different receptive fields
3. Create apparent "attention" effects if drift patterns differ between conditions

DRIFT QUANTIFICATION:
Path length = cumulative eye speed during stimulus period. A sensitive measure of ocular instability.

HYPOTHESIS: If attention effects are due to drift differences, then:
1. Attended and unattended trials should have different path lengths
2. Neural activity should vary systematically with path length
3. Attention effects should disappear when controlling for path length
""")

print("\n9. CONTROL 3: Ocular drift (path length) analysis...")
print(f"   - Analysis window: {DRIFT_ANALYSIS_WINDOW[0]}-{DRIFT_ANALYSIS_WINDOW[1]} s")

# Define time mask for drift analysis
drift_mask = (time_vector_eye >= DRIFT_ANALYSIS_WINDOW[0]) & (time_vector_eye <= DRIFT_ANALYSIS_WINDOW[1])

# Calculate path length for each trial by summing speed within the window
trial_path_length = np.sum(eye_speed[:, drift_mask], axis=1)

# Separate path lengths by attention condition
attended_path_length = trial_path_length[attended_trials_mask]
unattended_path_length = trial_path_length[unattended_trials_mask]

# Compare mean path length between conditions
mean_attended_path_length = np.mean(attended_path_length)
mean_unattended_path_length = np.mean(unattended_path_length)
pval = ttest_ind(attended_path_length, unattended_path_length).pvalue
print(f'   - Mean attended path length: {mean_attended_path_length:.2f}')
print(f'   - Mean unattended path length: {mean_unattended_path_length:.2f}')
print(f'   - T-test for difference in path length: p = {pval:.3f}')

# Define quartile bins based on the overall distribution of path lengths
n_bins = 4
edges = np.percentile(trial_path_length, np.linspace(0, 100, n_bins + 1))
print(f'   - Path length bin edges (quartiles): {np.round(edges, 2)}')

# Plot the distribution of path lengths
bins = np.linspace(trial_path_length.min(), trial_path_length.max(), 30)
plt.figure(figsize=(8, 6))
plt.hist(trial_path_length, bins=bins, color='gray', label='All Trials', alpha=0.6)
plt.hist(attended_path_length, bins=bins, alpha=0.7, label='Attended', color='r')
plt.hist(unattended_path_length, bins=bins, alpha=0.7, label='Unattended', color='b')

# Add mean indicators and significance
plt.scatter([mean_attended_path_length], [60], color='r', marker='v', s=100, label='Attended Mean', zorder=5)
plt.scatter([mean_unattended_path_length], [60], color='b', marker='v', s=100, label='Unattended Mean', zorder=5)
significance_connector(mean_attended_path_length, mean_unattended_path_length, 65, 5, 'n.s.' if pval > 0.05 else f'p={pval:.3f}')
for e in edges:
    plt.axvline(x=e, color='k', linestyle='--', label='Quartile edges' if e == edges[0] else None, alpha=0.5)
plt.legend()
plt.title(f'Distribution of Path Lengths ({DRIFT_ANALYSIS_WINDOW[0]}-{DRIFT_ANALYSIS_WINDOW[1]}s)')
plt.xlabel('Path Length (arbitrary units)')
plt.ylabel('Number of Trials')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_path_length_distribution.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_path_length_distribution.svg')
plt.show()

print("   - There is no significant difference in path length between attended and unattended trials.")
print("   - Nevertheless, we will check if the attention effect persists within each path length quartile as a stringent control.")

#%%
# Plot MUA for attended vs. unattended trials within each path length quartile
fig, axs = plt.subplots(n_bins+1, 1, figsize=(8, 12), sharex=True, sharey=True)
fig.suptitle(f'{CURRENT_MONKEY.capitalize()} - MUA Conditioned on Path Length', y=1.02, fontsize=16)


for i in range(n_bins):
    e0, e1 = edges[i], edges[i+1]
    # Create a mask for trials within the current path length quartile
    quartile_mask = (trial_path_length >= e0) & (trial_path_length <= e1)

    # Calculate MUA for all, attended, and unattended trials in this quartile
    all_mua = population_mua[quartile_mask].mean(axis=0)
    attended_mua = population_mua[quartile_mask & attended_trials_mask].mean(axis=0)
    unattended_mua = population_mua[quartile_mask & unattended_trials_mask].mean(axis=0)

    pval = ttest_ind(
        population_mua[quartile_mask & attended_trials_mask][:, attention_window_mask].mean(axis=1),
        population_mua[quartile_mask & unattended_trials_mask][:, attention_window_mask].mean(axis=1)
    ).pvalue

    # Plotting
    axs[0].plot(time_base_mua, all_mua, label='Quartile ' + str(i+1))
    axs[i+1].plot(time_base_mua, all_mua, c='gray', alpha=0.5, label='All trials')
    axs[i+1].plot(time_base_mua, attended_mua, label='Attended', color='r')
    axs[i+1].plot(time_base_mua, unattended_mua, label='Unattended', color='b')
    axs[i+1].set_title(f'Quartile {i+1}: Path Length [{edges[i]:.2f}, {edges[i+1]:.2f}] - ' + (f'p={pval:.1e}' if pval < 0.05 else 'n.s.'))
    axs[i+1].axvline(x=0, color='k', linestyle='--')
    axs[i+1].grid(True, alpha=0.2)
    axs[i+1].axhline(y=0, color='k', linestyle='--', alpha=0.3, zorder=0)
    if i == 0:
        axs[i+1].legend()

axs[0].set_title('All Trials by Path Length Quartile')
axs[0].legend()
axs[-1].set_xlabel('Time (s)')
fig.text(0.04, 0.5, 'Normalized MUA', va='center', rotation='vertical')
plt.tight_layout(rect=[0.05, 0, 1, 1])
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_mua_by_path_length_quartile.png', dpi=300)
    plt.savefig(FIGURE_DIR / f'{CURRENT_MONKEY}_mua_by_path_length_quartile.svg')
plt.show()

print("""
KEY FINDING: Attention effects persist across all path length quartiles.

INTERPRETATION:
- Although there is a slight modulation of MUA by path length (higher drift = slightly higher MUA),
  the attention effect is robustly present in all quartiles.
- This demonstrates that neural attention effects are not mere artifacts of ocular drift.

CONCLUSION for Control 3: The attentional modulation of V1 multi-unit activity is not driven by 
differences in ocular drift between attended and unattended conditions.
""")

print("\n" + "="*70)
print("SUMMARY OF EYE MOVEMENT CONTROLS")  
print("="*70)
print("The three comprehensive eye movement controls demonstrate that:")
print("1. EYE POSITION CONTROL: Attention effects persist within each eye position quartile.")
print("2. MICROSACCADE CONTROL: Attention effects remain in trials without microsaccades.") 
print("3. DRIFT CONTROL: Attention effects are present across all ocular drift levels.")
print()
print("CONCLUSION:")
print("These controls were unable to explain away the V1 attentional affects as being")
print("due to eye movements during fixation.")
print("="*70)
