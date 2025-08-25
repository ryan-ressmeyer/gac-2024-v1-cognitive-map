#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
if 'trial_ind' not in locals():
    trial_ind = 0
else:
    trial_ind += 1

plt.figure(figsize=(12, 10))

# Time domain plots
plt.subplot(321)
plt.plot(t_trial, dpi_r_trials[trial_ind,:,0], c='k')
plt.title(f'Trial {trial_ind} - {"Attended" if (trial_ind in attended_trials) else "Unattended"} X')
plt.xlabel('Time (s)')
plt.ylabel('Position')

plt.subplot(323)
plt.plot(t_trial, dpi_r_trials[trial_ind,:,1], c='k')
plt.title('Y')
plt.xlabel('Time (s)')
plt.ylabel('Position')

plt.subplot(325)
plt.plot(np.arange(-200, 500), np.nanmean(normMUA[snr_mask, trial_ind, :], axis=0))
plt.title('MUA')
plt.xlabel('Time (ms)')
plt.ylabel('Activity')

# FFT and power spectrum plots
# Compute FFT for X and Y signals
x_signal = dpi_r_trials[trial_ind,:,0]
y_signal = dpi_r_trials[trial_ind,:,1]

# Compute FFT
x_fft = np.fft.fft(x_signal)
y_fft = np.fft.fft(y_signal)

# Compute power spectrum (magnitude squared)
x_power = np.abs(x_fft)**2
y_power = np.abs(y_fft)**2

# Create frequency arrays
# Assuming sampling rate can be inferred from t_trial
fs = 30000  # sampling frequency
dt = 1 / fs  # time step

freqs_x = np.fft.fftfreq(len(x_signal), dt)
freqs_y = np.fft.fftfreq(len(y_signal), dt)

# Plot power spectra (only positive frequencies)
plt.subplot(322)
pos_freqs_x = freqs_x[:len(freqs_x)//2]
plt.semilogy(pos_freqs_x[1:], x_power[1:len(x_power)//2], c='k')
plt.title('X Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0, 250)  # Show up to quarter of sampling frequency

plt.subplot(324)
pos_freqs_y = freqs_y[:len(freqs_y)//2]
plt.semilogy(pos_freqs_y[1:], y_power[1:len(y_power)//2], c='k')
plt.title('Y Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0, 250)  # Show up to quarter of sampling frequency

plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(6, 12))
plt.subplot(211)
for i in range(40):
    iT = attended_trials[i]
    plt.plot(t_trial, dpi_r_trials[iT,:,1]+i*20, alpha = 1)
plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
plt.legend()
plt.title(f'40 Attended Trials - Vertical Eye Position')
plt.subplot(212)
for i in range(40):
    iT = unattended_trials[i]
    plt.plot(t_trial, dpi_r_trials[iT,:,1]+i*20, alpha = 1)
plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
plt.legend()
plt.title(f'40 Unattended Trials - Vertical Eye Position')
plt.show()


#%%

mean_attended_eyepos = np.mean(dpi_r_trials[attended_trials,:], axis=0)
ste_attended_eyepos = np.std(dpi_r_trials[attended_trials,:], axis=0) / np.sqrt(dpi_r_trials[attended_trials,:].shape[0])
mean_unattended_eyepos = np.mean(dpi_r_trials[unattended_trials,:], axis=0)
ste_unattended_eyepos = np.std(dpi_r_trials[unattended_trials,:], axis=0) / np.sqrt(dpi_r_trials[unattended_trials,:].shape[0])

plt.figure(figsize=(8, 12))
plt.subplot(211)
plt.fill_between(t_trial, 
                 mean_unattended_eyepos[:,0] - ste_unattended_eyepos[:,0]*1.96, 
                 mean_unattended_eyepos[:,0] + ste_unattended_eyepos[:,0]*1.96,
                 color='k', alpha=0.2, label='Unattended - 95% CI')
plt.plot(t_trial, mean_unattended_eyepos[:,0], c='k', label='Unattended - Mean')
plt.fill_between(t_trial, 
                 mean_attended_eyepos[:,0] - ste_attended_eyepos[:,0]*1.96, 
                 mean_attended_eyepos[:,0] + ste_attended_eyepos[:,0]*1.96,
                 color='r', alpha=0.2, label='Attended - 95% CI')
plt.plot(t_trial, mean_attended_eyepos[:,0], c='r', label='Attended - Mean')
plt.legend()
plt.title(f'{monkey.capitalize()} - {task.capitalize()} Task: Eye Position\nX')
plt.axvline(x=0, color='k', linestyle='--')
plt.subplot(212)
plt.fill_between(t_trial, 
                 mean_unattended_eyepos[:,1] - ste_unattended_eyepos[:,1]*1.96, 
                 mean_unattended_eyepos[:,1] + ste_unattended_eyepos[:,1]*1.96,
                 color='k', alpha=0.2, label='Unattended - 95% CI')
plt.fill_between(t_trial, 
                 mean_attended_eyepos[:,1] - ste_attended_eyepos[:,1]*1.96, 
                 mean_attended_eyepos[:,1] + ste_attended_eyepos[:,1]*1.96,
                 color='r', alpha=0.2, label='Attended - 95% CI')
plt.plot(t_trial, mean_unattended_eyepos[:,1], c='k', label='Unattended - Mean')
plt.plot(t_trial, mean_attended_eyepos[:,1], c='r', label='Attended - Mean')
plt.axvline(x=0, color='k', linestyle='--')
plt.legend()
plt.title('Y')
plt.tight_layout()
plt.show()

#%%

attended_speed = np.linalg.norm(np.diff(dpi_r_trials[attended_trials,:], axis=1), axis=2)
unattended_speed = np.linalg.norm(np.diff(dpi_r_trials[unattended_trials,:], axis=1), axis=2)

plt.figure()
plt.plot(t_trial[:-1], attended_speed[:n_trials,:].T, c='r', alpha=.5)
plt.plot(t_trial[:-1], unattended_speed[:n_trials,:].T, c='k', alpha=.5)
plt.show()
#%%
med_attended_speed = np.mean(attended_speed, axis=0)
med_unattended_speed = np.mean(unattended_speed, axis=0)

plt.figure()
plt.plot(t_trial[:-1], med_unattended_speed, c='k')
plt.plot(t_trial[:-1], med_attended_speed, c='r')
plt.show()


# %%
# marginalize on position

window = [200, 400]
n_bins = 4
mask = (window[0] < t_trial) & (t_trial < window[1])
dpi_r_y_win = np.mean(dpi_r_trials[:,mask,1], axis=1)
edges = np.percentile(dpi_r_y_win, np.linspace(0, 100, n_bins + 1))

plt.figure(figsize=(8, 12))
plt.subplot(211)
plt.plot(t_trial, mean_attended_eyepos[:,1], c='r')
plt.plot(t_trial, mean_unattended_eyepos[:,1], c='k')
plt.axvline(x=window[1], color='b', linestyle='--', label='Window')
plt.axvline(x=window[0], color='b', linestyle='--')
plt.title(f'Mean position window - {window[0]} - {window[1]} ms')
plt.legend()
plt.subplot(212)
plt.hist(dpi_r_y_win, bins=50, alpha=0.5, label='All')
plt.hist(dpi_r_y_win[attended_trials], bins=50, alpha=0.5, label='Attended')
plt.hist(dpi_r_y_win[unattended_trials], bins=50, alpha=0.5, label='Unattended')
for e in edges:
    plt.axvline(x=e, color='k', linestyle='--', label='Bin edges' if e == edges[0] else None)
plt.legend()
plt.title('Position histogram - binned')
plt.tight_layout()
plt.show()

#%%

plt.figure()
for i in range(n_bins):
    e0, e1 = edges[i], edges[i+1]
    mask = (e0 < dpi_r_y_win) & (dpi_r_y_win < e1)
    activity = norm_trial_mua[mask].mean(axis=0)
    plt.plot(activity, label=f'{e0:.1f} - {e1:.1f}')
plt.legend()
plt.title('MUA conditioned on position')
plt.show()

#%%
eyepos_sorted = np.argsort(dpi_r_y_win)

plt.figure()
plt.imshow(norm_trial_mua[eyepos_sorted], vmin=-2, vmax=2, cmap='coolwarm')
plt.colorbar()
plt.title('MUA sorted by vertical eye position')
plt.show()



# %%


plt.figure(figsize=(8, 12))
for i in range(n_bins):
    e0, e1 = edges[i], edges[i+1]
    mask = (e0 < dpi_r_y_win) & (dpi_r_y_win < e1)
    attn_activity_ctrl = norm_trial_mua[mask & attended_mask].mean(axis=0)
    unattn_activity_ctrl = norm_trial_mua[mask & unattended_mask].mean(axis=0)
    plt.subplot(n_bins, 1, i+1)
    plt.plot(tb, attended_activity, c='k')
    plt.plot(tb, unattended_activity, c='k')
    plt.plot(tb, attn_activity_ctrl, label='Attended')
    plt.plot(tb, unattn_activity_ctrl, label='Unattended')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.ylim(-.3, 1.3)
    plt.axhline(y=0, color='k', linestyle='--', zorder=0)
    plt.title(f'{e0:.1f} - {e1:.1f}')
    plt.legend()
plt.tight_layout()
plt.show()

#%%
# CONTROL -> Check to see if the affect changes over time and if 
# the eye position also varies over time

from scipy.stats import linregress
n_blocks = 4
n_trials = norm_trial_mua.shape[0]
plt.figure()
plt.subplot(211)
for i in range(n_blocks):
    i0, i1 = i*n_trials//n_blocks, (i+1)*n_trials//n_blocks
    signal = norm_trial_mua[i0:i1].mean(axis=0)
    plt.plot(tb, signal, label=f'Block {i}: {i0} - {i1}')
plt.legend()
plt.axvline(x=0, color='k', linestyle='--')
plt.subplot(212)
plt.plot(dpi_r_y_win)
slope, intercept, _, p, _ = linregress(np.arange(n_trials), dpi_r_y_win)
plt.plot(intercept + slope*np.arange(n_trials), 'r--', label=f'slope = {slope:.2f}, p = {p:.2e}')
plt.legend()
plt.show()

#%%


