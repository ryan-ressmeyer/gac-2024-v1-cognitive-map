#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import loadmat, smooth, significance_connector

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
#monkey = 'monkeyN'
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
# Filter bad trials where either the eye position is out of bounds
# or there are large gaps in the data due to dropped frames

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

# Apply the valid mask to filter trials
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

#%%

# One thing I noticed is that the MUA has DC offset that varies across trials
# I therefore removed the mean of the pre-stimulus period from each trial to avoid
# any spurious correlations with eye position

trial_mua = normMUA[snr_mask,:,:].mean(axis=0)
norm_trial_mua = trial_mua - trial_mua[:,:200].mean(axis=1)[:,None]

plt.figure()
plt.subplot(121)
plt.title('raw')
plt.imshow(trial_mua, vmin=-2, vmax=2, aspect='auto', cmap='coolwarm')
plt.subplot(122)
plt.title('normalized')
plt.imshow(norm_trial_mua, vmin=-2, vmax=2, aspect='auto', cmap='coolwarm')
plt.show()

#%%
# --- Plot 1: Average activity for attended vs. unattended ---
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Condition for unattended trials (ALLMAT[:, 4] == 2)
unattended_mask = ALLMAT[:, 3] == 2 # Index 3 for 4th column
unattended_mua = norm_trial_mua[unattended_trials].mean(axis=0)

# Condition for attended trials (ALLMAT[:, 4] == 1)
attended_mask = ALLMAT[:, 3] == 1 # Index 3 for 4th column
attended_mua = norm_trial_mua[attended_trials].mean(axis=0)

ax1.plot(unattended_mua, color=[.7, .6, .8], linewidth=2, label='Unattended')
ax1.plot(attended_mua, color=[.3, .7, .2], linewidth=2, label='Attended')

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

# The eye position data was collected at 500 Hz, but upsampled to 30 kHz
# We will decimate the data back down to 1 kHz for analysis
# and then smooth it with a Savitzky-Golay filter to reduce high frequency noise
# We are also only going to use the right eye data for analyses.

from scipy.signal import decimate 
from scipy.signal import savgol_filter, savgol_coeffs, freqz

fs = 30000
decim_factor = 30
fs_new = fs / decim_factor
savgol_order = 3
savgol_window = 35  # must be odd

# plot the frequency response of the filter
plt.figure()
plt.title(f'Savitzky-Golay Filter (order={savgol_order}, window={savgol_window}, fs={fs_new} Hz)')
b = savgol_coeffs(savgol_window, savgol_order)
w, h = freqz(b)
plt.plot(0.5*fs_new*w/np.pi, np.abs(h), 'b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid()
plt.show()
#%%

eye = 'right'

dpi_trials_raw = dpi_r_trials if eye == 'right' else dpi_l_trials
dpi_trials = decimate(dpi_r_trials, 30, axis=1)
dpi_trials -= np.median(dpi_trials, axis=(0, 1), keepdims=True) # demedian to center the data
dpi_trials_filt = savgol_filter(dpi_trials, savgol_window, savgol_order, axis=1)
dpi_trials_vel = np.diff(dpi_trials_filt, axis=1, prepend=0)
dpi_trials_spd = np.linalg.norm(dpi_trials_vel, axis=2)
t_dpi = np.arange(dpi_trials.shape[1]) / fs_new - n_pre/fs
#%%

# Now let's visualize a few eye position trials

np.random.seed(104)

n_trials_to_plot = 5
attn_plot = np.random.permutation(attended_trials)[:n_trials_to_plot]
unattn_plot = np.random.permutation(unattended_trials)[:n_trials_to_plot]

plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.plot(t_dpi, dpi_trials[unattn_plot,:,0].T, alpha = .6, c='k')
plt.plot(t_dpi, dpi_trials_filt[unattn_plot,:,0].T, c='r', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Unattended X')
plt.ylim(-20, 20)
plt.subplot(223)
plt.plot(t_dpi, dpi_trials[attn_plot,:,0].T, alpha = .6, c='k')
plt.plot(t_dpi, dpi_trials_filt[attn_plot,:,0].T, c='b', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Attended X')
plt.ylim(-20, 20)
plt.subplot(222)
plt.plot(t_dpi, dpi_trials[unattn_plot,:,1].T, alpha = .6, c='k')
plt.plot(t_dpi, dpi_trials_filt[unattn_plot,:,1].T, c='r', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Unattended Y')
plt.ylim(-20, 20)
plt.subplot(224)
plt.plot(t_dpi, dpi_trials[attn_plot,:,1].T, alpha = .6, c='k')
plt.plot(t_dpi, dpi_trials_filt[attn_plot,:,1].T, c='b', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Attended Y')
plt.ylim(-20, 20)
plt.tight_layout()
plt.show()

# Let's also look at the distribution of eye position across all trials during the stimulus period
mask = (0 < t_dpi) & (t_dpi < 500)
bins = np.linspace(-30, 30, 100)

plt.figure(figsize=(6, 5))
plt.hist2d(dpi_trials_filt[:,mask,0].flatten(), dpi_trials_filt[:,mask,1].flatten(), bins=[bins, bins], cmap='inferno')
plt.colorbar(label='Counts')
plt.title('2D Histogram of Eye Position (0-500 ms)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()


# Two things I noticed:
# 1. There are microsaccades in the data
# 2. There's quite a bit of drift, especially in the y direction, that is higher amplitude than some of the microsaccades

# We're now going to analyse three different aspects of the eye movement data to control for their effects on MUA:
# 1. Overall eye position (x and y)
# 2. Microsaccades
# 3. Amount of drift (path length)

#%%
# Control 1: Overall eye position
# Let's see if there are any obvious differences in eye position between attended and unattended trials


mean_dpi_attended = np.mean(dpi_trials_filt[attended_trials], axis=0)
ste_dpi_attended = np.std(dpi_trials_filt[attended_trials], axis=0) / np.sqrt(dpi_trials_filt[attended_trials].shape[0])
mean_dpi_unattended = np.mean(dpi_trials_filt[unattended_trials], axis=0)
ste_dpi_unattended = np.std(dpi_trials_filt[unattended_trials], axis=0) / np.sqrt(dpi_trials_filt[unattended_trials].shape[0])

pos_window = [.3, .4]

plt.figure(figsize=(8, 12))
plt.subplot(211)
plt.fill_between(t_dpi, 
                 mean_dpi_unattended[:,0] - ste_dpi_unattended[:,0]*1.96, 
                 mean_dpi_unattended[:,0] + ste_dpi_unattended[:,0]*1.96,
                 color='k', alpha=0.2, label='Unattended - 95% CI')
plt.plot(t_dpi, mean_dpi_unattended[:,0], c='k', label='Unattended - Mean')
plt.fill_between(t_dpi, 
                 mean_dpi_attended[:,0] - ste_dpi_attended[:,0]*1.96, 
                 mean_dpi_attended[:,0] + ste_dpi_attended[:,0]*1.96,
                 color='r', alpha=0.2, label='Attended - 95% CI')
plt.plot(t_dpi, mean_dpi_attended[:,0], c='r', label='Attended - Mean')
plt.legend()
plt.title(f'{monkey.capitalize()} - {task.capitalize()} Task: Eye Position\nX')
plt.axvline(x=0, color='k', linestyle='--')
plt.ylim(-10, 10)
plt.subplot(212)
plt.fill_between(t_dpi, 
                 mean_dpi_unattended[:,1] - ste_dpi_unattended[:,1]*1.96, 
                 mean_dpi_unattended[:,1] + ste_dpi_unattended[:,1]*1.96,
                 color='k', alpha=0.2, label='Unattended - 95% CI')
plt.fill_between(t_dpi, 
                 mean_dpi_attended[:,1] - ste_dpi_attended[:,1]*1.96, 
                 mean_dpi_attended[:,1] + ste_dpi_attended[:,1]*1.96,
                 color='r', alpha=0.2, label='Attended - 95% CI')
plt.plot(t_dpi, mean_dpi_unattended[:,1], c='k', label='Unattended - Mean')
plt.plot(t_dpi, mean_dpi_attended[:,1], c='r', label='Attended - Mean')
plt.fill_betweenx([-10, 10], pos_window[0], pos_window[1], color='g', alpha=0.1, label='Position Window', zorder=0)
plt.axvline(x=0, color='k', linestyle='--')
plt.ylim(-10, 10)
plt.legend()
plt.title('Y')
plt.tight_layout()
plt.show()  
# There does seem to be a small difference in vertical eye position between attended and unattended trials
# Let's see if the MUA varies with eye position in the window we highlighted

#%%

# We are going to split the distribution of eye positions into quartiles and see if the MUA varies with eye position
# We will also see if the affect of attention is still present in each quartiles


from scipy.stats import ttest_ind
pos_mask = (pos_window[0] < t_dpi) & (t_dpi < pos_window[1])
trial_pos = dpi_trials_filt[:,pos_mask,1].mean(axis=1)
attn_pos = trial_pos[attended_trials].mean()
unattn_pos = trial_pos[unattended_trials].mean()
mean_attn_pos = np.mean(trial_pos[attended_trials])
mean_unattn_pos = np.mean(trial_pos[unattended_trials])
pval = ttest_ind(trial_pos[attended_trials], trial_pos[unattended_trials]).pvalue
print(f'Mean attended position: {mean_attn_pos:.2f}')
print(f'Mean unattended position: {mean_unattn_pos:.2f}')
print(f'T-test p-value: {pval:.3f}')

n_bins = 4
edges = np.percentile(trial_pos, np.linspace(0, 100, n_bins+1))
print('Bin edges:', edges)

min_pos = edges[0]
max_pos = edges[-1]
hist_bins = np.linspace(min_pos, max_pos, 30)

plt.figure(figsize=(6, 5))
plt.hist(trial_pos, bins=hist_bins, label='All trials', color='gray', zorder=-1)
plt.hist(trial_pos[attended_trials], bins=hist_bins, alpha=0.5, label='Attended', color='r')
plt.hist(trial_pos[unattended_trials], bins=hist_bins, alpha=0.5, label='Unattended', color='b')
significance_connector(mean_attn_pos, mean_unattn_pos, 65, 5, f'**' if pval < 0.05 else 'n.s.')
# draw arrow to indicate the mean
plt.scatter([attn_pos], [60], color='r', marker='v', s=100, label='Attended Mean')
plt.scatter([unattn_pos], [60], color='b', marker='v', s=100, label='Unattended Mean')
for e in edges:
    plt.axvline(x=e, color='k', linestyle='--', label='Bin edges' if e == edges[0] else None, alpha=0.5, zorder=0)
plt.legend()
plt.title('Eye position distribution')
plt.tight_layout()
plt.show()

# The difference in mean eye position between attended and unattended trials is quite small compared to the overall distribution
# and there is substantial overlap between the two conditions
# Let's see if the MUA varies with eye position
#%%


plt.figure()
for i in range(n_bins):
    e0, e1 = edges[i], edges[i+1]
    mask = (e0 < trial_pos) & (trial_pos < e1)
    activity = norm_trial_mua[mask].mean(axis=0)
    plt.plot(activity, label=f'Quartile {i+1}: {e0:.1f} - {e1:.1f}')
plt.legend()
plt.title('MUA conditioned on position')
plt.show()
# So there is a small effect of eye position on MUA
# Let's see if the effect of attention is still present within each quartile

#%%
fig, axs = plt.subplots(n_bins, 1, figsize=(8, 12), sharex=True, sharey=True)
for i in range(n_bins):
    e0, e1 = edges[i], edges[i+1]
    mask = (e0 < trial_pos) & (trial_pos < e1)
    all_activity = norm_trial_mua[mask].mean(axis=0)
    attn_activity = norm_trial_mua[mask & attended_mask].mean(axis=0)
    unattn_activity = norm_trial_mua[mask & unattended_mask].mean(axis=0)
    axs[i].plot(tb, all_activity, c='gray', alpha=0.5, label='All trials')
    axs[i].plot(tb, attn_activity, label='Attended', color='r')
    axs[i].plot(tb, unattn_activity, label='Unattended', color='b')
    axs[i].axvline(x=0, color='k', linestyle='--')
    axs[i].set_ylim(-.3, 1.3)
    axs[i].axhline(y=0, color='k', linestyle='--', zorder=0)
    axs[i].set_title(f'Quartile {i+1}: {e0:.1f} - {e1:.1f}')
    if i == 0:
        axs[i].legend()
axs[-1].set_xlabel('Time (ms)')
fig.suptitle(f'{monkey.capitalize()} - {task.capitalize()} Task: MUA conditioned on Eye Position', y=1.02)
plt.tight_layout()
plt.show()

# Indeed, the effect of attention is still present within each quartile of eye position
# So we conclude that the effect of attention on MUA is not driven by differences in eye position

#%%

# Control 2: Microsaccades

# We are going to start by detecting microsaccades using a simple velocity thresholding method
# This is not perfect, but should be sufficient for a first pass analysis.

from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
th = .35 # Threshold for microsaccade detection
distance = 100 # Minimum distance between microsaccades in samples (100 samples at 1 kHz = 100 ms)
export_pdf = True

# Export a PDF with all trials and detected microsaccades for manual inspection

saccade_trials = []
saccade_samples = []
saccade_times = []
saccade_directions = []
saccade_speeds = []

try:
    if export_pdf:
        pdf = PdfPages(f'{monkey}_{task}_microsaccades.pdf')
    for iT in tqdm(range(n_trials)):
        peaks, _ = find_peaks(dpi_trials_spd[iT,:], height=th, distance=distance)
        saccade_samples.append(peaks)
        saccade_trials.append(np.ones(len(peaks)) * iT)
        saccade_times.append(t_dpi[peaks])
        saccade_directions.append(np.arctan2(dpi_trials_vel[iT,peaks,1], dpi_trials_vel[iT,peaks,0]))
        saccade_speeds.append(dpi_trials_spd[iT,peaks])

        if export_pdf:
            plt.figure()
            plt.subplot(211)
            plt.plot(dpi_trials_filt[iT,:,0])
            if len(peaks) > 0:
             for p in peaks:
                 plt.axvline(x=p, color='r', linestyle='--')
            plt.title(f'Trial {iT}')
            ax2 = plt.twinx()
            ax2.plot(dpi_trials_filt[iT,:,1], 'k')
            plt.subplot(212)
            plt.plot(dpi_trials_spd[iT,:])
            if len(peaks) > 0:
             plt.plot(peaks, dpi_trials_spd[iT,peaks], 'rx')
            plt.axhline(y=th, color='r', linestyle='--')
            plt.ylim(0, 1)
            pdf.savefig()
            plt.close()
except Exception as e:
    raise e
finally:
    if export_pdf:
        pdf.close()

saccade_trials = np.concatenate(saccade_trials)
saccade_samples = np.concatenate(saccade_samples)
saccade_times = np.concatenate(saccade_times)
saccade_directions = np.concatenate(saccade_directions)
saccade_speeds = np.concatenate(saccade_speeds)

print(f'Detected {len(saccade_samples)} microsaccades across {n_trials} trials')

#%%
fig, axs = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[2, 1])
axs[0].eventplot([saccade_times[saccade_trials==iT] for iT in range(n_trials)], linelengths=7.5, linewidths=2.5, colors='k')
axs[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axs[0].set_xlim(t_dpi[0], t_dpi[-1])
axs[0].set_ylim(0, n_trials)
axs[0].set_ylabel('Trial')

axs[1].hist(saccade_times, bins=np.linspace(t_dpi[0], t_dpi[-1], 50), color='k')
axs[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axs[1].annotate('Pre-stimulus\nmicrosaccades', xy=(-.1, 20), xytext=(-.05, 40),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='center')
axs[1].annotate('Post-stimulus\nsuppression of\nmicrosaccades', xy=(.1, 5), xytext=(.15, 30),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='center')
axs[1].annotate('Choice saccades', xy=(.45, 9), xytext=(.35, 40),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='center')
axs[1].set_xlim(t_dpi[0], t_dpi[-1])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Number of Saccades')

plt.show()

# What we see is that there aren't many microsaccades when the stimulus is actually on the screen
# There are more microsaccades in the pre-stimulus period

#%%

# Only microsaccades that occur when the stimulus is on the screen
# can affect spiking activity in a feedforward manner. Since there aren't 
# many microsaccades during this period, it's easy to control for by 
# excluding trials with microsaccades during this period
# We'll also look at the difference between trials with and without microsaccades

msacc_window = [0, 0.4] # seconds
msacc_trials = saccade_trials[(saccade_times > msacc_window[0]) & (saccade_times < msacc_window[1])].astype(int)
msacc_mask = np.zeros(n_trials, dtype=bool)
msacc_mask[msacc_trials] = True

print(f'{msacc_mask.sum()} trials with microsaccades between {msacc_window[0]} and {msacc_window[1]} seconds')
print(f'{n_trials - msacc_mask.sum()} trials without microsaccades.')

saccade_mua = norm_trial_mua[msacc_mask].mean(axis=0)
no_saccade_mua = norm_trial_mua[~msacc_mask].mean(axis=0)
no_saccade_attended = norm_trial_mua[~msacc_mask & attended_mask].mean(axis=0)
no_saccade_unattended = norm_trial_mua[~msacc_mask & unattended_mask].mean(axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(tb, no_saccade_mua, label='No Saccade', color='k', alpha=0.7)
axs[0].plot(tb, no_saccade_attended, label='No Saccade - Attended', color='r')
axs[0].plot(tb, no_saccade_unattended, label='No Saccade - Unattended', color='b')
axs[0].axvline(x=0, color='k', linestyle='--')
axs[0].set_title('Trials without Microsaccades')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Normalized MUA')
axs[0].legend()
axs[1].plot(tb, saccade_mua, label='Saccade', color='g')
axs[1].plot(tb, no_saccade_mua, label='No Saccade', color='k')
axs[1].axvline(x=0, color='k', linestyle='--')
axs[1].set_title('Microsaccade Trials vs No Microsaccade Trials')
axs[1].set_xlabel('Time (ms)')
plt.legend()
plt.show()

# Since there is still an effect of attention on trials without microsaccades
# we conclude that the effect of attention on MUA is not driven by microsaccades

# %%

# Control 3: Path Length
# There is quite a bit of drift in the eye position data. We will therefore test to see if the amount of drift
# affects MUA and if the effect of attention is still present when controlling for drift, similar to what we did for overall eye position.

# We will quantify drift by calculating the path length of the eye position trace during the stimulus period
drift_window = [0, 0.3] # seconds
drift_mask = (drift_window[0] < t_dpi) & (t_dpi < drift_window[1])

trial_path_length = np.sum(dpi_trials_spd[:,drift_mask], axis=1)
attended_path_length = trial_path_length[attended_mask]
unattended_path_length = trial_path_length[unattended_mask]

mean_attended_path_length = np.mean(attended_path_length)
mean_unattended_path_length = np.mean(unattended_path_length)

pval = ttest_ind(attended_path_length, unattended_path_length).pvalue
print(f'Mean attended path length: {mean_attended_path_length:.2f}')
print(f'Mean unattended path length: {mean_unattended_path_length:.2f}')
print(f'T-test p-value: {pval:.3f}')

n_divs = 4
edges = np.percentile(trial_path_length, np.linspace(0, 100, n_divs+1))
bins = np.linspace(trial_path_length.min(), trial_path_length.max(), 30)

plt.figure()
plt.hist(trial_path_length, bins=bins, color='gray', zorder=-1)
plt.hist(attended_path_length, bins=bins, alpha=0.5, label='Attended', color='r')
plt.hist(unattended_path_length, bins=bins, alpha=0.5, label='Unattended', color='b')
plt.scatter([mean_attended_path_length], [60], color='r', marker='v', s=100, label='Attended Mean')
plt.scatter([mean_unattended_path_length], [60], color='b', marker='v', s=100, label='Unattended Mean')
significance_connector(mean_attended_path_length, mean_unattended_path_length, 65, 5, f'n.s.')
for e in edges:
    plt.axvline(x=e, color='k', linestyle='--', label='Quartile edges' if e == edges[0] else None, alpha=0.5, zorder=0)
plt.legend()
plt.title('Distribution of Path Lengths')
plt.xlabel('Path Length (degrees)')
plt.ylabel('Number of Trials')
plt.show()

# So there isn't a significant difference in path length between attended and unattended trials
# Let's still see if MUA varies with path length and if the effect of attention is still present within each quartile of path length

#%%
plt.figure()
for i in range(n_divs):
    mask = (edges[i] < trial_path_length) & (trial_path_length < edges[i+1])
    div_mua = norm_trial_mua[mask].mean(axis=0)
    plt.plot(div_mua, label=f'Quartile {i+1}: {edges[i]:.2f} - {edges[i+1]:.2f}')
plt.legend()
plt.title('MUA conditioned on Path Length')
plt.show()

# So there does seem to be a small correlation between path length and MUA

#%%
# Let's now compare attended and unattended trials within each quartile of path length
# compare mua for each division
fig, axs = plt.subplots(n_divs, 1, figsize=(10, 10))
for i in range(n_divs):
    mask = (edges[i] < trial_path_length) & (trial_path_length < edges[i+1])
    div_attended_mask = mask & attended_mask
    div_unattended_mask = mask & unattended_mask
    div_mua = norm_trial_mua[mask].mean(axis=0)
    attended_mua = norm_trial_mua[div_attended_mask].mean(axis=0)
    unattended_mua = norm_trial_mua[div_unattended_mask].mean(axis=0)
    axs[i].plot(tb, div_mua, c='gray', alpha=0.5, label='All trials')
    axs[i].plot(tb, attended_mua, label='Attended')
    axs[i].plot(tb, unattended_mua, label='Unattended')
    axs[i].set_title(f'Quartile {i+1}: {edges[i]:.2f} - {edges[i+1]:.2f}')
    axs[i].axvline(x=0, color='k', linestyle='--')
    axs[i].legend()
plt.xlabel('Time (ms)')
plt.suptitle(f'{monkey.capitalize()} - {task.capitalize()} Task: MUA conditioned on Path Length', y=1.02)
plt.tight_layout()
plt.show()

# Interestingly, the effect of attention seems to be largest on trials with the smallest path length.
# Still, the attentional effect is present in all quartiles
# So we conclude that the effect of attention on MUA is not driven by differences in path length.


