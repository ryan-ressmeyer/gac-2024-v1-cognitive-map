# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroscience research project for the 2024 Generative Adversarial Collaboration (GAC) investigating whether V1 (primary visual cortex) functions as a cognitive map. The project analyzes multi-unit activity (MUA) data from two monkeys (monkeyF and monkeyN) performing attention tasks with eye tracking controls.

## Development Environment

This project uses Python with `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate

# Run Python scripts
python scripts/example.py
python scripts/eye_movement_controls.py
```

## Project Structure

### Data Organization
- `data/monkeyF/` and `data/monkeyN/` - Neural and eye tracking data for each subject
- Data files follow naming convention: `ObjAtt_GAC2_{task}_{datatype}_{session}.mat`
- Tasks: `lums` (luminance attention) and `sacc` (saccade attention)
- Key data types: `MUA_trials`, `normMUA`, calibration files (`CALIB_`, `cals_`)

### Code Structure
- `scripts/` - Analysis scripts (Python and MATLAB)
- `utils/` - Shared utility functions for data loading and visualization
- `results/` - Output directory for analysis results

## Key Components

### Data Loading (`utils/__init__.py`)
- `loadmat()` - Robust .mat file loader that handles both scipy and mat73 formats
- `smooth()` - Moving average filter for signal smoothing
- `significance_connector()` - Statistical significance visualization helper

### Analysis Pipeline
1. **Neural Data Processing**: Load MUA trials and normalized data, apply SNR thresholds
2. **Eye Movement Controls**: Process eye tracking data, detect microsaccades, analyze drift
3. **Attention Analysis**: Compare attended vs unattended conditions across control variables

### Channel Organization
- MonkeyF: channels 129-192 (64 channels)
- MonkeyN: channels 193-256 (64 channels) 
- Total recording array: 512 channels
- SNR threshold typically set to 1.0 for channel selection

### Experimental Design
- **Luminance task**: Control variable index 7 (MATLAB: 8), 6 control combinations
- **Saccade task**: Control variable index 8 (MATLAB: 9), 4 control combinations
- Attention conditions: 1=attended, 2=unattended (ALLMAT column 4, Python index 3)

## Data Analysis Patterns

### Time Series Analysis
- Time base (`tb`) typically spans -200ms to +500ms relative to stimulus onset
- Stimulus onset at 200ms mark in time series
- Standard smoothing window: 20 samples for visualization

### Eye Movement Processing
- Original sampling: 30kHz (upsampled from 500Hz)
- Analysis sampling: 1kHz after decimation
- Microsaccade detection: velocity threshold ~0.35 deg/s
- Eye position calibration applied using gain/offset matrices

### Statistical Controls
The analysis implements three key eye movement controls:
1. **Eye Position**: Quartile-based analysis of mean gaze position
2. **Microsaccades**: Exclusion of trials with microsaccades during stimulus period
3. **Drift (Path Length)**: Control for overall eye movement during analysis window

## File Dependencies

Scripts expect data directory structure with:
- `.mat` files containing `ALLMAT` (trial metadata) and `tb` (time base)
- Normalized MUA data with `normMUA` and `SNR` arrays
- Eye tracking data in `TEMP_DPI_*`, `TEMP_EYE_*` files
- Calibration data in `cals_*` files with gain/offset matrices

## Common Workflows

### Running Basic Analysis
```python
from utils import loadmat, smooth
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze MUA data
data_dir = "data/monkeyF/"
allmat_file = data_dir + "ObjAtt_GAC2_lums_MUA_trials.mat"
mua_file = data_dir + "ObjAtt_GAC2_lums_normMUA.mat"

allmat_data = loadmat(allmat_file)
mua_data = loadmat(mua_file)
```

### Eye Movement Analysis
The eye movement controls script provides comprehensive microsaccade detection and drift analysis. Key parameters:
- Microsaccade threshold: 0.35 deg/s
- Minimum inter-saccade interval: 100ms
- Analysis windows: typically 0-400ms for stimulus period

### Output Generation
- PDF exports for microsaccade detection validation
- Statistical comparisons between attention conditions
- Visualization of neural activity time courses with confidence intervals

## Data Paths

Update data paths in scripts to match your local setup:
- Example script: Update `DATADIR_GEN` variable
- Eye movement controls: Update `datagen_dir` variable