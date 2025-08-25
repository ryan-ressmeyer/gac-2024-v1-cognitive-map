#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

#%%
# --- Constants and Script Execution ---
# Define the base directory for your data
# IMPORTANT: Update this path to your actual data directory
datagen_dir = '/home/ryanress/code/gac-2024-v1-cognitive-map/data/'

# Define the monkeys and tasks to be analyzed
monkeys = ['monkeyF', 'monkeyN']
#tasks = ['lums', 'sacc']
tasks = ['lums']
task = tasks[0]


# Set the Signal-to-Noise Ratio (SNR) threshold
snr_threshold = 1.0

# Create a Path object for the base data directory
base_data_dir = Path(datagen_dir)


#%%

monkey = 'monkeyF'
monkey = 'monkeyN'
# Define channel ranges based on the monkey
if monkey == 'monkeyF':
    array_chns = np.arange(129, 193)
else:
    array_chns = np.arange(193, 257)

# Create a boolean mask for the channels of the current monkey
all_chns = np.zeros(512, dtype=bool)
all_chns[array_chns - 1] = True # Subtract 1 for 0-based indexing

# Construct file paths using pathlib
mat_filename = base_data_dir / monkey / f"ObjAtt_GAC2_lums_MUA_trials.mat"
norm_mua_filename = base_data_dir / monkey / f"ObjAtt_GAC2_lums_normMUA.mat"

# Load the .mat files
allmat_data = loadmat(mat_filename)
ALLMAT = allmat_data['ALLMAT']
tb = allmat_data['tb'].flatten()

norm_mua_data = loadmat(norm_mua_filename)
normMUA = norm_mua_data['normMUA']

cals_file = base_data_dir / monkey 
if monkey == 'monkeyN':
    cals_file /= 'cals_monkeyN_20250218_B1.mat'
else:
    cals_file /= 'cals_monkeyF_20250228_B1.mat'

cals = loadmat(cals_file)

obj_att_files = list((base_data_dir / monkey).glob('ObjAtt_GAC2_lums_*_B*.mat'))
obj_att_files.sort(reverse=False)

n_pre = int(.2 * 30000)
n_post = int(.5 * 30000)

dpi_l_trials = []
dpi_r_trials = []
for f in obj_att_files:
    data = loadmat(f)
    print(f'File: {f.name}')

    dpi_file = f.parent/ (f'TEMP_DPI_{f.name[-15:]}')
    print(f'DPI File: {dpi_file.name}')
    dpi = loadmat(dpi_file)

    eye_file = f.parent/ (f'TEMP_EYE_{f.name[-15:]}')
    print(f'Eye File: {eye_file.name}')
    eye = loadmat(eye_file)
    eye_trials = eye['Trials_corrected'].flatten()
    print(eye_trials.shape)

    dpi_l = dpi['dpi_l'][:,:2] @ cals['left_gains'] + cals['left_offset']
    dpi_r = dpi['dpi_r'][:,:2] @ cals['right_gains'] + cals['right_offset']

    for iT, ind in enumerate(eye_trials):
        ind = int(ind)
        if data['MAT'][iT,-1] < 1:
            continue
        dpi_l_trials.append(dpi_l[ind-n_pre:ind+n_post])
        dpi_r_trials.append(dpi_r[ind-n_pre:ind+n_post])

dpi_l_trials = np.stack(dpi_l_trials, axis=0)
dpi_r_trials = np.stack(dpi_r_trials, axis=0)
n_trials = len(dpi_l_trials)

ALLMAT = allmat_data['ALLMAT']

unattended_trials = np.where(ALLMAT[:, 3] == 2)[0]
attended_trials = np.where(ALLMAT[:, 3] == 1)[0]
print(dpi_l_trials.shape)
print(ALLMAT.shape)

#%%
# filter bad trials

if monkey == 'monkeyN':
    attended_y_lim = [0, 400]
    unattended_y_lim = [250, 450]
    attended_x_lim = [-200, 200]
    unattended_x_lim = [-200, 0]
else:
    attended_y_lim = [-np.inf, np.inf]
    unattended_y_lim = [-np.inf, np.inf]
    attended_x_lim = [-np.inf, np.inf]
    unattended_x_lim = [-np.inf, np.inf]

max_skipped_frames = 5
dpi_sampling_rate = 1/500
max_ifi = max_skipped_frames * dpi_sampling_rate

valid_mask = np.ones(dpi_r_trials.shape[0], dtype=bool)
for iT in range(n_trials):
    if iT in attended_trials:
        if np.any((dpi_r_trials[iT,:,1] < attended_y_lim[0]) | (dpi_r_trials[iT,:,1] > attended_y_lim[1])):
            valid_mask[iT] = False
        if np.any((dpi_r_trials[iT,:,0] < attended_x_lim[0]) | (dpi_r_trials[iT,:,0] > attended_x_lim[1])):
            valid_mask[iT] = False
    elif iT in unattended_trials:
        if np.any((dpi_r_trials[iT,:,1] < unattended_y_lim[0]) | (dpi_r_trials[iT,:,1] > unattended_y_lim[1])):
            valid_mask[iT] = False
        if np.any((dpi_r_trials[iT,:,0] < unattended_x_lim[0]) | (dpi_r_trials[iT,:,0] > unattended_x_lim[1])):
            valid_mask[iT] = False
    else:
        print('Error: trial not attended or unattended')

    trial = dpi_r_trials[iT]
    trial_acc = np.linalg.norm(np.diff(np.diff(trial, axis=0, prepend=0), axis=0, prepend=0), axis=1)
    acc_changing = np.diff(trial_acc, prepend=-1, append=1) > 1e-6
    frame_inds = np.where(acc_changing)[0]
    ifi = np.diff(frame_inds) / 30000

    if np.any(ifi > max_ifi):
        valid_mask[iT] = False
        print(f'Trial {iT} has maximum interframe interval of {np.max(ifi)*1000:.3f} ms')


dpi_r_trials = dpi_r_trials[valid_mask]
dpi_l_trials = dpi_l_trials[valid_mask]
ALLMAT = ALLMAT[valid_mask]
n_trials = len(dpi_r_trials)
unattended_trials = np.where(ALLMAT[:, 3] == 2)[0]
attended_trials = np.where(ALLMAT[:, 3] == 1)[0]
print(n_trials)

normMUA = normMUA[:,valid_mask]


# Set up task-specific parameters
control_var_idx = 7  # 8 in MATLAB (1-based) -> 7 in Python (0-based)

# Get unique control values
controls = np.unique(ALLMAT[:, control_var_idx])

# Define a boolean mask for channels that meet the SNR threshold
snr_mask = (norm_mua_data['SNR'].flatten() > snr_threshold) & all_chns

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

#%%

t_trial = np.arange(-n_pre, n_post)/30

n_trials = 100
plt.figure()
plt.subplot(221)
plt.plot(t_trial, dpi_r_trials[unattended_trials[:n_trials],:,0].T, alpha = .6, c='k')
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Unattended X')
plt.subplot(223)
plt.plot(t_trial, dpi_r_trials[attended_trials[:n_trials],:,0].T, alpha = .6, c='r')
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Attended X')
plt.subplot(222)
plt.plot(t_trial, dpi_r_trials[unattended_trials[:n_trials],:,1].T, alpha = .6, c='k')
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Unattended Y')
plt.subplot(224)
plt.plot(t_trial, dpi_r_trials[attended_trials[:n_trials],:,1].T, alpha = .6, c='r')
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Attended Y')
plt.show()

#%%

trial_mua = normMUA[snr_mask,:,:].mean(axis=0)
norm_trial_mua = trial_mua - trial_mua[:,:200].mean(axis=1)[:,None]

attended_mua = norm_trial_mua[attended_trials].mean(axis=0)
unattended_mua = norm_trial_mua[unattended_trials].mean(axis=0)

plt.figure()
plt.subplot(121)
plt.title('raw')
plt.imshow(trial_mua, vmin=-2, vmax=2, aspect='auto', cmap='coolwarm')
plt.subplot(122)
plt.title('normalized')
plt.imshow(norm_trial_mua, vmin=-2, vmax=2, aspect='auto', cmap='coolwarm')
plt.show()

plt.figure()
plt.plot(attended_mua, label='Attended')
plt.plot(unattended_mua, label='Unattended')
plt.plot()

#%%
from scipy.signal import decimate 

dpi_r_trials_dec = decimate(dpi_r_trials, 30, axis=1)

from scipy.signal import savgol_filter

#%%
dpi_r_trials_sg = savgol_filter(dpi_r_trials_dec, 35, 3, axis=1)

plt.figure()
plt.plot(dpi_r_trials_dec[7,:,0])
plt.plot(dpi_r_trials_sg[7,:,0])
plt.show()

#%%
dpi_r_trials_vel = np.diff(dpi_r_trials_sg, axis=1)
dpi_r_trials_spd = np.linalg.norm(dpi_r_trials_vel, axis=2)

#%%

from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
th = .35

#from tqdm import tqdm
# n_trials = len(dpi_r_trials_spd)
# with PdfPages(f'{monkey}_{task}_microsaccades.pdf') as pdf:
#     for iT in range(n_trials):
#         peaks, _ = find_peaks(dpi_r_trials_spd[iT,:], height=th, distance=100)
#         plt.figure()
#         plt.subplot(211)
#         plt.plot(dpi_r_trials_sg[iT,:,0])
#         if len(peaks) > 0:
#             for p in peaks:
#                 plt.axvline(x=p, color='r', linestyle='--')
#         plt.title(f'Trial {iT}')
#         ax2 = plt.twinx()
#         ax2.plot(dpi_r_trials_sg[iT,:,1], 'k')
#         plt.subplot(212)
#         plt.plot(dpi_r_trials_spd[iT,:])
#         if len(peaks) > 0:
#             plt.plot(peaks, dpi_r_trials_spd[iT,peaks], 'rx')
#         plt.axhline(y=th, color='r', linestyle='--')
#         plt.ylim(0, 1)
#         pdf.savefig()
#         plt.close()

#%%


saccade_trials = []
saccade_samples = []
saccade_directions = []
saccade_speeds = []

n_trials = len(dpi_r_trials_spd)
for iT in range(n_trials):
    peaks, _ = find_peaks(dpi_r_trials_spd[iT,:], height=th, distance=100)
    saccade_samples.append(peaks)
    saccade_trials.append(np.ones(len(peaks)) * iT)
    saccade_directions.append(np.atan2(dpi_r_trials_vel[iT,peaks,1], dpi_r_trials_vel[iT,peaks,0]))
    saccade_speeds.append(dpi_r_trials_spd[iT,peaks])

saccade_trials = np.concatenate(saccade_trials)
saccade_samples = np.concatenate(saccade_samples)
saccade_directions = np.concatenate(saccade_directions)
saccade_speeds = np.concatenate(saccade_speeds)

#%%
plt.figure()
plt.eventplot([saccade_samples[saccade_trials==iT] for iT in range(n_trials)], linelengths=2.5, linewidths=2.5)
plt.show()

#%%
# Stimulus on during these times
i0 = 200
i1 = 500 
msacc_trials = saccade_trials[(saccade_samples > i0) & (saccade_samples < i1)].astype(int)
msacc_mask = np.zeros(n_trials, dtype=bool)
msacc_mask[msacc_trials] = True

print(msacc_mask.sum())


saccade_mua = norm_trial_mua[msacc_mask].mean(axis=0)
no_saccade_mua = norm_trial_mua[~msacc_mask].mean(axis=0)
plt.figure()
plt.plot(tb, saccade_mua, label='Saccade')
plt.plot(tb, no_saccade_mua, label='No Saccade')
plt.legend()
plt.show()

#%%
no_saccade_attended = norm_trial_mua[~msacc_mask & attended_mask].mean(axis=0)
no_saccade_unattended = norm_trial_mua[~msacc_mask & unattended_mask].mean(axis=0)

plt.figure()
plt.plot(tb, no_saccade_attended, label='No Saccade - Attended')
plt.plot(tb, no_saccade_unattended, label='No Saccade - Unattended')
plt.legend()
plt.show()

#%%

attended_samples = saccade_samples[np.isin(saccade_trials, attended_trials)]
unattended_samples = saccade_samples[np.isin(saccade_trials, unattended_trials)]

plt.figure()
plt.hist(attended_samples, color='r', bins=100)
plt.axvline(x=200, color='r', linestyle='--')
plt.show()
plt.figure()
plt.hist(unattended_samples, color='k', bins=100)
plt.axvline(x=200, color='k', linestyle='--')
plt.show()

#%%

s0 = 500
s1 = 700

attended_directions = []
unattended_directions = []

for iS in range(len(saccade_samples)):
    sample = saccade_samples[iS]
    trial = saccade_trials[iS]
    if sample > s0 and sample < s1:
        if trial in attended_trials:
            attended_directions.append(saccade_directions[iS])
        elif trial in unattended_trials:
            unattended_directions.append(saccade_directions[iS])
attended_directions = np.array(attended_directions)
unattended_directions = np.array(unattended_directions)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(attended_directions, bins=8, alpha=.5, label='Attended')
ax.hist(unattended_directions, bins=8, alpha=.5, label='Unattended')
ax.legend()
ax.set_title('Microsaccade Directions')
plt.show()

# %%

i0 = 200
i1 = 500

path_length = np.sum(np.linalg.norm(np.diff(dpi_r_trials_sg[:,i0:i1,:], axis=1), axis=2), axis=1)

plt.figure()
plt.hist(path_length, bins=50)
plt.legend()
plt.show()

#%%
# look at 10 longest path lengths
n_plot = 10
longest_paths = np.argsort(path_length)[::1][:n_plot]
for i in longest_paths:
    y = dpi_r_trials_sg[i,i0:i1,1]
    y -= y[0]
    x = dpi_r_trials_sg[i,i0:i1,0]
    x -= x[0]
    plt.figure()
    plt.plot(dpi_r_trials_sg[i,i0:i1,0])
    plt.plot(dpi_r_trials_sg[i,i0:i1,1])
    plt.show()

#%%
# compare mua for longest and shortest paths

n_divs = 4
divs = np.percentile(path_length, np.linspace(0, 100, n_divs + 1))
plt.figure()
for i in range(n_divs):
    mask = (divs[i] < path_length) & (path_length < divs[i+1])
    mean_trial_num = np.mean(np.arange(n_trials)[mask])
    print(f'Mean trial number: {mean_trial_num}')
    mua = norm_trial_mua[mask].mean(axis=0)
    plt.plot(tb, mua, label=f'{divs[i]:.2f} - {divs[i+1]:.2f}')
plt.legend()
plt.show()
#%%
# coimpare mua over trials

divs = np.percentile(np.arange(n_trials), np.linspace(0, 100, n_divs + 1))
fig, axs = plt.subplots(n_divs, 1, figsize=(10, 10))
for i in range(n_divs):
    mask = (divs[i] < np.arange(n_trials)) & (np.arange(n_trials) < divs[i+1])
    div_attended_mask = mask & attended_mask
    div_unattended_mask = mask & unattended_mask
    attended_mua = norm_trial_mua[div_attended_mask].mean(axis=0)
    unattended_mua = norm_trial_mua[div_unattended_mask].mean(axis=0)
    axs[i].plot(tb, attended_mua, label='Attended')
    axs[i].plot(tb, unattended_mua, label='Unattended')
    axs[i].set_title(f'Trials {divs[i]} - {divs[i+1]}')
plt.show()

#%%
# compare mua for each division

fig, axs = plt.subplots(n_divs, 1, figsize=(10, 10))
for i in range(n_divs):
    mask = (divs[i] < path_length) & (path_length < divs[i+1])
    div_attended_mask = mask & attended_mask
    div_unattended_mask = mask & unattended_mask
    attended_mua = norm_trial_mua[div_attended_mask].mean(axis=0)
    unattended_mua = norm_trial_mua[div_unattended_mask].mean(axis=0)
    axs[i].plot(tb, attended_mua, label='Attended')
    axs[i].plot(tb, unattended_mua, label='Unattended')
    axs[i].set_title(f'{divs[i]:.2f} - {divs[i+1]:.2f}')
    axs[i].legend()
plt.show()
#%%
# compare distribution of path length for different types of trials

attended_path_length = path_length[attended_mask]
unattended_path_length = path_length[unattended_mask]

plt.figure()
plt.hist(attended_path_length, bins=50, alpha=0.5, label='Attended')
plt.hist(unattended_path_length, bins=50, alpha=0.5, label='Unattended')
plt.legend()
plt.show()

#%%
# more path lengths in the beginning
plt.figure()
plt.plot(path_length)
plt.show()

#%%
# exclude the top div and look at attended and unattended separately

divs = np.percentile(path_length, np.linspace(0, 100, n_divs + 1))
for i in range(n_divs-1):
    mask = (divs[i] < path_length) & (path_length < divs[i+1])
    attended_mask = mask & attended_mask
    unattended_mask = mask & unattended_mask
    attended_mua = norm_trial_mua[attended_mask].mean(axis=0)
    unattended_mua = norm_trial_mua[unattended_mask].mean(axis=0)
    plt.figure()
    plt.plot(tb, attended_mua, label='Attended')
    plt.plot(tb, unattended_mua, label='Unattended')
    plt.legend()
    plt.show()

#%%
# look at motion in various directions

i0 = 200
i1 = 500
#%%

dpi_r_vel_dir = np.arctan2(dpi_r_trials_vel[:,i0:i1,1], dpi_r_trials_vel[:,i0:i1,0])
dpi_r_vel_amp = np.linalg.norm(dpi_r_trials_vel[:,i0:i1,:], axis=2)

plt.figure()
plt.hist(dpi_r_vel_dir[0], bins=50)
plt.show()
plt.figure()
plt.hist(dpi_r_vel_amp[0], bins=50)
plt.show()

#%%
r = .2
n_bins = 50
bins = np.linspace(-r, r, n_bins)
plt.figure()
plt.hist2d(dpi_r_trials_vel[attended_mask,i0:i1,0].flatten(), dpi_r_trials_vel[attended_mask,i0:i1,1].flatten(), bins=(bins, bins))
plt.show()
plt.figure()
plt.hist2d(dpi_r_trials_vel[unattended_mask,i0:i1,0].flatten(), dpi_r_trials_vel[unattended_mask,i0:i1,1].flatten(), bins=(bins, bins))
plt.show()

#%%

i0 = 200
i1 = 500

r = 20
n_bins = 50
bins = np.linspace(-r, r, n_bins)

y_pos = dpi_r_trials_sg[:,i0:i1,0].copy()
y_pos -= np.median(y_pos)
x_pos = dpi_r_trials_sg[:,i0:i1,1].copy()
x_pos -= np.median(x_pos)

y_pos_attn = y_pos[attended_mask]
x_pos_attn = x_pos[attended_mask]
y_pos_unattn = y_pos[unattended_mask]
x_pos_unattn = x_pos[unattended_mask]

plt.figure()
plt.hist2d(x_pos_attn.flatten(), y_pos_attn.flatten(), bins=(bins, bins), cmap='coolwarm')
plt.show()
plt.figure()
plt.hist2d(x_pos_unattn.flatten(), y_pos_unattn.flatten(), bins=(bins, bins), cmap='coolwarm')
plt.show()

#%%

plt.figure()
for i in range(0,60):

    iT = attended_trials[i]
    plt.plot(dpi_r_trials_sg[iT,:,0], dpi_r_trials_sg[iT,:,1], 'r')
    iT = unattended_trials[i]
    plt.plot(dpi_r_trials_sg[iT,:,0], dpi_r_trials_sg[iT,:,1], 'k')
plt.show()
