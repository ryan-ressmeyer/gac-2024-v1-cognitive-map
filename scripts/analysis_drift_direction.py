"""
Drift Direction Analysis for V1 Cognitive Map Study
====================================================

This script analyzes the angular direction of eye drift movements to test whether
attention conditions differ in their directional patterns.

BACKGROUND:
-----------
While X and Y movement magnitudes may be similar, the preferred directions of drift
could still differ between attention conditions. This analysis examines the angular
distribution of eye velocity vectors.

HYPOTHESIS:
-----------
If attention effects are not due to differential eye movements, then:
1. Angular distributions should be similar between attended and unattended conditions
2. Any preferred directions should be consistent across conditions

METHOD:
-------
- For each time point in each trial, compute drift direction: θ = arctan2(vy, vx)
- Pool all angles across time and trials for each attention condition
- Use circular statistics to compare distributions
- Visualize with rose plots (circular histograms)

CIRCULAR STATISTICS:
--------------------
- Angles are circular data (0° = 360°), requiring special statistical methods
- Mean direction and concentration (inverse of circular variance)
- Watson-Williams test for comparing mean directions between groups

Author: Ryan Ressmeyer
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import circmean, circstd
from utils import is_notebook
import h5py

np.random.seed(1001)  # For reproducibility

if is_notebook():
    matplotlib.use('inline')
else:
    matplotlib.use('Qt5Agg')

#%%
# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Data paths
DATAGEN_DIR = '/home/ryanress/code/gac-2024-v1-cognitive-map/data/'
PREPROCESSED_DATA_FILE = Path(DATAGEN_DIR) / 'preprocessed_data.h5'

# Analysis parameters
MONKEYS = ['monkeyF', 'monkeyN']
CURRENT_TASK = 'lums'

# Analysis time window
MOVEMENT_ANALYSIS_WINDOW = [0, 0.35]  # Time window for direction analysis (s)

# Velocity threshold for angle computation (to avoid noisy angles from small movements)
VELOCITY_THRESHOLD = 0.00  # deg/s - only compute angles for speeds above this

# Circular histogram parameters
N_ANGLE_BINS = 36  # 10° bins

# Figure output
SAVE_FIGS = True
if SAVE_FIGS:
    FIGURE_DIR = Path('./figures/drift_direction')
    FIGURE_DIR.mkdir(exist_ok=True, parents=True)

#%%
# =============================================================================
# CIRCULAR STATISTICS HELPER FUNCTIONS
# =============================================================================

def circular_mean_deg(angles_deg):
    """Compute circular mean of angles in degrees."""
    angles_rad = np.deg2rad(angles_deg)
    mean_rad = circmean(angles_rad, high=2*np.pi, low=0)
    return np.rad2deg(mean_rad)

def circular_std_deg(angles_deg):
    """Compute circular standard deviation of angles in degrees."""
    angles_rad = np.deg2rad(angles_deg)
    std_rad = circstd(angles_rad, high=2*np.pi, low=0)
    return np.rad2deg(std_rad)

def rayleigh_test(angles_deg):
    """
    Rayleigh test for non-uniformity of circular data.
    Tests whether the population is uniformly distributed around the circle.

    Returns:
        R: mean resultant length (0 = uniform, 1 = all same direction)
        p: p-value (significant = non-uniform distribution)
    """
    angles_rad = np.deg2rad(angles_deg)
    n = len(angles_rad)

    # Compute mean resultant vector
    C = np.sum(np.cos(angles_rad))
    S = np.sum(np.sin(angles_rad))
    R = np.sqrt(C**2 + S**2) / n

    # Rayleigh test statistic
    Z = n * R**2

    # Approximate p-value
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))

    return R, p

def watson_williams_test(angles1_deg, angles2_deg):
    """
    Watson-Williams test for difference in mean direction between two groups.
    This is a circular analog of the two-sample t-test.

    Returns:
        F: test statistic
        p: p-value (significant = different mean directions)
    """
    # Convert to radians
    angles1_rad = np.deg2rad(angles1_deg)
    angles2_rad = np.deg2rad(angles2_deg)

    n1 = len(angles1_rad)
    n2 = len(angles2_rad)

    # Compute resultant vectors for each group
    C1 = np.sum(np.cos(angles1_rad))
    S1 = np.sum(np.sin(angles1_rad))
    R1 = np.sqrt(C1**2 + S1**2)

    C2 = np.sum(np.cos(angles2_rad))
    S2 = np.sum(np.sin(angles2_rad))
    R2 = np.sqrt(C2**2 + S2**2)

    # Combined group
    all_angles = np.concatenate([angles1_rad, angles2_rad])
    C_all = np.sum(np.cos(all_angles))
    S_all = np.sum(np.sin(all_angles))
    R_all = np.sqrt(C_all**2 + S_all**2)

    N = n1 + n2

    # Watson-Williams F-statistic
    numerator = (N - 2) * (R1 + R2 - R_all)
    denominator = (N - R1 - R2)

    if denominator == 0:
        return np.nan, np.nan

    F = numerator / denominator

    # Approximate p-value using F-distribution with df1=1, df2=N-2
    from scipy.stats import f as f_dist
    p = 1 - f_dist.cdf(F, 1, N - 2)

    return F, p

#%%
# =============================================================================
# LOAD PREPROCESSED DATA FROM HDF5
# =============================================================================

print("="*70)
print("DRIFT DIRECTION ANALYSIS")
print("="*70)
print(f"Loading preprocessed data from: {PREPROCESSED_DATA_FILE}")
print(f"Analyzing monkeys: {', '.join(MONKEYS)}")
print(f"Task: {CURRENT_TASK}")
print("="*70)

# Load data for both monkeys from HDF5
monkey_data = {}

with h5py.File(PREPROCESSED_DATA_FILE, 'r') as f:
    for monkey in MONKEYS:
        monkey_group = f[monkey]

        # Load arrays for this monkey
        monkey_data[monkey] = {
            'eye_velocity': monkey_group['eye_velocity'][:],  # [trials x time x 2]
            't_rel_stim': monkey_group['t_rel_stim'][:],
            'trial_attended': monkey_group['trial_attended'][:]
        }

# Print summary statistics
print("\nData Summary:")
for monkey in MONKEYS:
    data = monkey_data[monkey]
    total_trials = data['eye_velocity'].shape[0]
    n_attended = np.sum(data['trial_attended'] == 1)
    n_unattended = np.sum(data['trial_attended'] == 0)

    print(f"\n{monkey}:")
    print(f"   - Total trials: {total_trials}")
    print(f"   - Attended trials: {n_attended}")
    print(f"   - Unattended trials: {n_unattended}")

#%%
# =============================================================================
# COMPUTE DRIFT DIRECTIONS
# =============================================================================

print(f"\n--- Computing drift directions ---")
print(f"Analysis window: {MOVEMENT_ANALYSIS_WINDOW[0]}-{MOVEMENT_ANALYSIS_WINDOW[1]} s")
print(f"Velocity threshold: {VELOCITY_THRESHOLD} deg/s")

direction_data = {}

for monkey in MONKEYS:
    data = monkey_data[monkey]
    t_rel_stim = data['t_rel_stim']
    eye_velocity = data['eye_velocity']  # [trials x time x 2]
    trial_attended = data['trial_attended']

    # Get time mask for analysis window
    time_mask = (t_rel_stim >= MOVEMENT_ANALYSIS_WINDOW[0]) & (t_rel_stim <= MOVEMENT_ANALYSIS_WINDOW[1])

    # Extract velocities in analysis window
    vel_x = eye_velocity[:, time_mask, 0]  # [trials x time_samples]
    vel_y = eye_velocity[:, time_mask, 1]  # [trials x time_samples]

    # Compute speed for thresholding
    speed = np.sqrt(vel_x**2 + vel_y**2)

    # Compute angles in degrees (-180 to 180)
    angles_all = np.rad2deg(np.arctan2(vel_y, vel_x))

    # Apply velocity threshold - only keep angles where speed > threshold
    valid_mask = speed > VELOCITY_THRESHOLD
    angles_valid = angles_all[valid_mask]

    # Split by attention condition
    # Need to expand trial_attended to match the time dimension
    trial_attended_expanded = np.repeat(trial_attended[:, np.newaxis], time_mask.sum(), axis=1)
    attended_mask = (trial_attended_expanded == 1) & valid_mask
    unattended_mask = (trial_attended_expanded == 0) & valid_mask

    angles_attended = angles_all[attended_mask]
    angles_unattended = angles_all[unattended_mask]

    # Convert to 0-360 range for consistency
    angles_attended = (angles_attended + 360) % 360
    angles_unattended = (angles_unattended + 360) % 360
    angles_valid = (angles_valid + 360) % 360

    # Compute circular statistics
    mean_attended = circular_mean_deg(angles_attended)
    mean_unattended = circular_mean_deg(angles_unattended)
    std_attended = circular_std_deg(angles_attended)
    std_unattended = circular_std_deg(angles_unattended)

    # Rayleigh tests (test for non-uniformity)
    R_attended, p_rayleigh_attended = rayleigh_test(angles_attended)
    R_unattended, p_rayleigh_unattended = rayleigh_test(angles_unattended)

    # Watson-Williams test (test for difference in mean direction)
    F_ww, p_ww = watson_williams_test(angles_attended, angles_unattended)

    direction_data[monkey] = {
        'angles_attended': angles_attended,
        'angles_unattended': angles_unattended,
        'mean_attended': mean_attended,
        'mean_unattended': mean_unattended,
        'std_attended': std_attended,
        'std_unattended': std_unattended,
        'R_attended': R_attended,
        'R_unattended': R_unattended,
        'p_rayleigh_attended': p_rayleigh_attended,
        'p_rayleigh_unattended': p_rayleigh_unattended,
        'F_watson_williams': F_ww,
        'p_watson_williams': p_ww,
        'n_samples_attended': len(angles_attended),
        'n_samples_unattended': len(angles_unattended)
    }

    print(f'\n{monkey}:')
    print(f'   Attended:')
    print(f'      - Mean direction: {mean_attended:.1f}°')
    print(f'      - Circular SD: {std_attended:.1f}°')
    print(f'      - Mean resultant length (R): {R_attended:.3f}')
    print(f'      - Rayleigh test p-value: {p_rayleigh_attended:.3f}')
    print(f'      - N samples: {len(angles_attended)}')
    print(f'   Unattended:')
    print(f'      - Mean direction: {mean_unattended:.1f}°')
    print(f'      - Circular SD: {std_unattended:.1f}°')
    print(f'      - Mean resultant length (R): {R_unattended:.3f}')
    print(f'      - Rayleigh test p-value: {p_rayleigh_unattended:.3f}')
    print(f'      - N samples: {len(angles_unattended)}')
    print(f'   Watson-Williams test (difference in mean direction):')
    print(f'      - F-statistic: {F_ww:.2f}')
    print(f'      - p-value: {p_ww:.3f}')

#%%
# =============================================================================
# VISUALIZATION: ROSE PLOTS (2x2 GRID)
# =============================================================================

print("\n--- Creating rose plots (circular histograms) ---")

fig = plt.figure(figsize=(14, 12))
fig.suptitle('Drift Direction Distributions by Attention Condition',
             fontsize=16, y=0.98)

# Create angle bins
angle_bins = np.linspace(0, 360, N_ANGLE_BINS + 1)
angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
angle_bin_width = 2 * np.pi / N_ANGLE_BINS  # Width in radians for rose plot

# Layout: rows = monkeys, columns = attended vs unattended
for row_idx, monkey in enumerate(['monkeyN', 'monkeyF']):
    data = direction_data[monkey]

    # Attended (left column)
    ax_att = fig.add_subplot(2, 2, row_idx*2 + 1, projection='polar')
    angles_att = data['angles_attended']

    # Create histogram
    counts_att, _ = np.histogram(angles_att, bins=angle_bins)
    counts_att = counts_att / counts_att.sum()  # Normalize to proportions

    # Plot rose
    theta = np.deg2rad(angle_bin_centers)
    bars = ax_att.bar(theta, counts_att, width=angle_bin_width, bottom=0.0,
                       color='red', alpha=0.7, edgecolor='darkred', linewidth=1)

    # Add mean direction arrow
    mean_theta = np.deg2rad(data['mean_attended'])
    max_radius = counts_att.max() * 1.2
    ax_att.arrow(mean_theta, 0, 0, max_radius * 0.8,
                 head_width=0.2, head_length=max_radius*0.15,
                 fc='darkred', ec='darkred', linewidth=2, zorder=5)

    ax_att.set_theta_zero_location('E')
    ax_att.set_theta_direction(1)
    ax_att.set_title(f'Monkey {monkey[-1].upper()} - Attended\n' +
                     f'μ={data["mean_attended"]:.1f}°, R={data["R_attended"]:.2f}\n' +
                     f'Rayleigh p={data["p_rayleigh_attended"]:.3f}',
                     pad=20, fontsize=11)
    ax_att.set_ylim(0, max_radius)

    # Unattended (right column)
    ax_unatt = fig.add_subplot(2, 2, row_idx*2 + 2, projection='polar')
    angles_unatt = data['angles_unattended']

    # Create histogram
    counts_unatt, _ = np.histogram(angles_unatt, bins=angle_bins)
    counts_unatt = counts_unatt / counts_unatt.sum()  # Normalize to proportions

    # Plot rose
    bars = ax_unatt.bar(theta, counts_unatt, width=angle_bin_width, bottom=0.0,
                         color='blue', alpha=0.7, edgecolor='darkblue', linewidth=1)

    # Add mean direction arrow
    mean_theta = np.deg2rad(data['mean_unattended'])
    max_radius = counts_unatt.max() * 1.2
    ax_unatt.arrow(mean_theta, 0, 0, max_radius * 0.8,
                   head_width=0.2, head_length=max_radius*0.15,
                   fc='darkblue', ec='darkblue', linewidth=2, zorder=5)

    ax_unatt.set_theta_zero_location('E')
    ax_unatt.set_theta_direction(1)
    ax_unatt.set_title(f'Monkey {monkey[-1].upper()} - Unattended\n' +
                       f'μ={data["mean_unattended"]:.1f}°, R={data["R_unattended"]:.2f}\n' +
                       f'Rayleigh p={data["p_rayleigh_unattended"]:.3f}',
                       pad=20, fontsize=11)
    ax_unatt.set_ylim(0, max_radius)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / 'drift_direction_rose_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'drift_direction_rose_plots.svg', bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# OVERLAY COMPARISON PLOTS
# =============================================================================

print("\n--- Creating overlay comparison plots ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
fig.suptitle('Drift Direction: Attended vs Unattended (Overlaid)',
             fontsize=16, y=1.02)

for col_idx, monkey in enumerate(['monkeyN', 'monkeyF']):
    ax = axes[col_idx]
    data = direction_data[monkey]

    angles_att = data['angles_attended']
    angles_unatt = data['angles_unattended']

    # Create histograms
    counts_att, _ = np.histogram(angles_att, bins=angle_bins)
    counts_unatt, _ = np.histogram(angles_unatt, bins=angle_bins)

    # Normalize
    counts_att = counts_att / counts_att.sum()
    counts_unatt = counts_unatt / counts_unatt.sum()

    theta = np.deg2rad(angle_bin_centers)

    # Plot both with transparency
    ax.bar(theta, counts_att, width=angle_bin_width, bottom=0.0,
           color='red', alpha=0.5, edgecolor='darkred', linewidth=1, label='Attended')
    ax.bar(theta, counts_unatt, width=angle_bin_width, bottom=0.0,
           color='blue', alpha=0.5, edgecolor='darkblue', linewidth=1, label='Unattended')

    # Add mean direction arrows
    mean_theta_att = np.deg2rad(data['mean_attended'])
    mean_theta_unatt = np.deg2rad(data['mean_unattended'])
    max_radius = max(counts_att.max(), counts_unatt.max()) * 1.2

    ax.arrow(mean_theta_att, 0, 0, max_radius * 0.7,
             head_width=0.2, head_length=max_radius*0.15,
             fc='darkred', ec='darkred', linewidth=2, zorder=5)
    ax.arrow(mean_theta_unatt, 0, 0, max_radius * 0.7,
             head_width=0.2, head_length=max_radius*0.15,
             fc='darkblue', ec='darkblue', linewidth=2, zorder=5)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # Title with Watson-Williams test result
    p_ww = data['p_watson_williams']
    sig_text = '***' if p_ww < 0.001 else ('**' if p_ww < 0.01 else ('*' if p_ww < 0.05 else 'n.s.'))

    ax.set_title(f'Monkey {monkey[-1].upper()}\n' +
                 f'Attended μ={data["mean_attended"]:.1f}° vs Unattended μ={data["mean_unattended"]:.1f}°\n' +
                 f'Watson-Williams: F={data["F_watson_williams"]:.2f}, p={p_ww:.3f} {sig_text}',
                 pad=20, fontsize=11)
    ax.set_ylim(0, max_radius)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / 'drift_direction_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'drift_direction_comparison.svg', bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: DRIFT DIRECTION ANALYSIS")
print("="*70)
print("\nThis analysis examined whether attended and unattended trials differ in")
print("the angular direction of eye drift movements using circular statistics.")
print("\nRESULTS:")

for monkey in MONKEYS:
    data = direction_data[monkey]
    print(f"\n{monkey}:")
    print(f"   Mean direction difference: {abs(data['mean_attended'] - data['mean_unattended']):.1f}°")
    print(f"   Watson-Williams test: F={data['F_watson_williams']:.2f}, p={data['p_watson_williams']:.3f}")

    if data['p_watson_williams'] < 0.05:
        print(f"   → SIGNIFICANT difference in mean drift direction")
    else:
        print(f"   → No significant difference in mean drift direction")

print("\nINTERPRETATION:")
print("The Watson-Williams test is a circular analog of the t-test that compares")
print("mean directions between groups. Non-significant results indicate that")
print("attended and unattended conditions have similar drift direction patterns,")
print("supporting that attention effects are not due to directional eye movement biases.")
print("\nThe Rayleigh test indicates whether directions are uniformly distributed.")
print("Low R values and non-significant Rayleigh tests indicate relatively uniform")
print("drift directions (no strong directional bias).")
print("="*70)
