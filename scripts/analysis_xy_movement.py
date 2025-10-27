"""
X and Y Eye Movement Analysis for V1 Cognitive Map Study
=========================================================

This script analyzes horizontal (X) and vertical (Y) eye movements separately
to test whether attention conditions differ in their directional eye movement patterns.

BACKGROUND:
-----------
While path length provides an overall measure of eye movement, examining X and Y
components separately can reveal whether:
1. Attention conditions differ in horizontal vs vertical movement patterns
2. Movement is isotropic (equal in all directions) or anisotropic
3. Any attention effects are direction-specific

HYPOTHESIS:
-----------
If attention effects are not due to differential eye movements, then:
1. X and Y movement distributions should be similar between attention conditions
2. Any differences should be small and non-systematic

METHOD:
-------
- For each trial, compute integrated absolute velocity in X and Y separately
- Compare distributions between attended and unattended conditions
- Visualize as 2x2 grid: (monkeyN vs monkeyF) × (X vs Y)

Author: Ryan Ressmeyer
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind
from utils import pvalue_to_stars, significance_connector, is_notebook
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

# Analysis time window (matches path length analysis)
MOVEMENT_ANALYSIS_WINDOW = [0, 0.35]  # Time window for movement analysis (s)

# Figure output
SAVE_FIGS = True
if SAVE_FIGS:
    FIGURE_DIR = Path('./figures/xy_movement')
    FIGURE_DIR.mkdir(exist_ok=True, parents=True)

#%%
# =============================================================================
# LOAD PREPROCESSED DATA FROM HDF5
# =============================================================================

print("="*70)
print("X AND Y EYE MOVEMENT ANALYSIS")
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
            'eye_velocity': monkey_group['eye_velocity'][:],  # Should be [trials x time x 2]
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
    print(f"   - Eye velocity shape: {data['eye_velocity'].shape}")

#%%
# =============================================================================
# COMPUTE X AND Y MOVEMENTS
# =============================================================================

print(f"\n--- Computing X and Y movements ---")
print(f"Analysis window: {MOVEMENT_ANALYSIS_WINDOW[0]}-{MOVEMENT_ANALYSIS_WINDOW[1]} s")

movement_data = {}

for monkey in MONKEYS:
    data = monkey_data[monkey]
    t_rel_stim = data['t_rel_stim']
    eye_velocity = data['eye_velocity']  # [trials x time x 2]
    trial_attended = data['trial_attended']

    # Calculate dt from the time vector
    dt = t_rel_stim[1] - t_rel_stim[0]

    # Get time mask for analysis window
    time_mask = (t_rel_stim >= MOVEMENT_ANALYSIS_WINDOW[0]) & (t_rel_stim <= MOVEMENT_ANALYSIS_WINDOW[1])

    # Extract X and Y velocities
    eye_velocity_x = eye_velocity[:, :, 0]  # [trials x time]
    eye_velocity_y = eye_velocity[:, :, 1]  # [trials x time]

    # Compute integrated absolute velocity for each trial
    x_movement = np.sum(np.abs(eye_velocity_x[:, time_mask]), axis=1) * dt  # degrees
    y_movement = np.sum(np.abs(eye_velocity_y[:, time_mask]), axis=1) * dt  # degrees

    # Split by attention condition
    x_attended = x_movement[trial_attended == 1]
    x_unattended = x_movement[trial_attended == 0]
    y_attended = y_movement[trial_attended == 1]
    y_unattended = y_movement[trial_attended == 0]

    # Compute statistics
    x_mean_att = x_attended.mean()
    x_mean_unatt = x_unattended.mean()
    y_mean_att = y_attended.mean()
    y_mean_unatt = y_unattended.mean()

    x_pval = ttest_ind(x_attended, x_unattended).pvalue
    y_pval = ttest_ind(y_attended, y_unattended).pvalue

    movement_data[monkey] = {
        'x_movement': x_movement,
        'y_movement': y_movement,
        'x_attended': x_attended,
        'x_unattended': x_unattended,
        'y_attended': y_attended,
        'y_unattended': y_unattended,
        'x_mean_att': x_mean_att,
        'x_mean_unatt': x_mean_unatt,
        'y_mean_att': y_mean_att,
        'y_mean_unatt': y_mean_unatt,
        'x_pval': x_pval,
        'y_pval': y_pval
    }

    print(f'\n{monkey}:')
    print(f'   X movement:')
    print(f'      - Mean attended: {x_mean_att:.2f} deg')
    print(f'      - Mean unattended: {x_mean_unatt:.2f} deg')
    print(f'      - T-test p-value: {x_pval:.3f}')
    print(f'   Y movement:')
    print(f'      - Mean attended: {y_mean_att:.2f} deg')
    print(f'      - Mean unattended: {y_mean_unatt:.2f} deg')
    print(f'      - T-test p-value: {y_pval:.3f}')

#%%
# =============================================================================
# VISUALIZATION: 2x2 GRID (MONKEY × DIRECTION)
# =============================================================================

print("\n--- Creating 2x2 visualization ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('X and Y Eye Movement Distributions by Attention Condition',
             fontsize=16, y=0.995)

# Layout: rows = monkeys, columns = directions
for row_idx, monkey in enumerate(['monkeyN', 'monkeyF']):
    data = movement_data[monkey]
    trial_attended = monkey_data[monkey]['trial_attended']

    # X movement (left column)
    ax_x = axes[row_idx, 0]
    x_movement = data['x_movement']
    x_bins = np.linspace(x_movement.min(), x_movement.max(), 30)

    ax_x.hist(x_movement, bins=x_bins, color='gray', label='All Trials', alpha=0.6)
    ax_x.hist(data['x_attended'], bins=x_bins, alpha=0.7, label='Attended', color='r')
    ax_x.hist(data['x_unattended'], bins=x_bins, alpha=0.7, label='Unattended', color='b')

    # Add mean indicators
    y_pos = ax_x.get_ylim()[1] * 0.6
    ax_x.scatter([data['x_mean_att']], [y_pos], color='r', marker='v', s=100,
                 label='Attended Mean', zorder=5)
    ax_x.scatter([data['x_mean_unatt']], [y_pos], color='b', marker='v', s=100,
                 label='Unattended Mean', zorder=5)

    # Add significance connector
    pval = data['x_pval']
    pval_text = f'{pvalue_to_stars(pval)}\n' + ('p<0.001' if pval < 0.001 else f'p={pval:.3f}')
    significance_connector(data['x_mean_att'], data['x_mean_unatt'],
                          y_pos * 1.1, y_pos * 0.08, pval_text, ax=ax_x)

    ax_x.set_title(f'Monkey {monkey[-1].upper()} - X (Horizontal) Movement')
    ax_x.set_xlabel('X Movement (degrees)')
    ax_x.set_ylabel('Number of Trials')
    if row_idx == 0:
        ax_x.legend(loc='upper right')
    ax_x.grid(True, alpha=0.2)

    # Y movement (right column)
    ax_y = axes[row_idx, 1]
    y_movement = data['y_movement']
    y_bins = np.linspace(y_movement.min(), y_movement.max(), 30)

    ax_y.hist(y_movement, bins=y_bins, color='gray', label='All Trials', alpha=0.6)
    ax_y.hist(data['y_attended'], bins=y_bins, alpha=0.7, label='Attended', color='r')
    ax_y.hist(data['y_unattended'], bins=y_bins, alpha=0.7, label='Unattended', color='b')

    # Add mean indicators
    y_pos = ax_y.get_ylim()[1] * 0.6
    ax_y.scatter([data['y_mean_att']], [y_pos], color='r', marker='v', s=100,
                 label='Attended Mean', zorder=5)
    ax_y.scatter([data['y_mean_unatt']], [y_pos], color='b', marker='v', s=100,
                 label='Unattended Mean', zorder=5)

    # Add significance connector
    pval = data['y_pval']
    pval_text = f'{pvalue_to_stars(pval)}\n' + ('p<0.001' if pval < 0.001 else f'p={pval:.3f}')
    significance_connector(data['y_mean_att'], data['y_mean_unatt'],
                          y_pos * 1.1, y_pos * 0.08, pval_text, ax=ax_y)

    ax_y.set_title(f'Monkey {monkey[-1].upper()} - Y (Vertical) Movement')
    ax_y.set_xlabel('Y Movement (degrees)')
    ax_y.set_ylabel('Number of Trials')
    if row_idx == 0:
        ax_y.legend(loc='upper right')
    ax_y.grid(True, alpha=0.2)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(FIGURE_DIR / 'xy_movement_distributions.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'xy_movement_distributions.svg')
plt.show()

#%%
# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: X AND Y EYE MOVEMENT ANALYSIS")
print("="*70)
print("\nThis analysis examined whether attended and unattended trials differ in")
print("their horizontal (X) and vertical (Y) eye movement patterns.")
print("\nRESULTS:")
for monkey in MONKEYS:
    data = movement_data[monkey]
    print(f"\n{monkey}:")
    print(f"   X movement: {pvalue_to_stars(data['x_pval'])} (p={data['x_pval']:.3f})")
    print(f"   Y movement: {pvalue_to_stars(data['y_pval'])} (p={data['y_pval']:.3f})")

print("\nINTERPRETATION:")
print("If there are no significant differences in X or Y movement between attention")
print("conditions, this supports that attention effects are not due to systematic")
print("differences in eye movement patterns.")
print("="*70)
