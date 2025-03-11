#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For questions, please reach out to Patrick Mayerhofer at pmayerho@sfu.ca

Configuration settings for the running classification project.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RAW_DIR = os.path.join(DATA_DIR, 'Raw')
PREPARED_DIR = os.path.join(DATA_DIR, 'Prepared')
RESULTS_DIR = os.path.join(DATA_DIR, 'Results')

# Create required directories if they don't exist
for directory in [DATA_DIR, RAW_DIR, PREPARED_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data processing settings
DATA_CONFIG = {
    'subjects': [1,2,3,8,9,20,21,24,28,32,33,35,36,42,45,49,55,79,108,110,111,122,131,133],
    'speeds': [2.5, 3.0, 3.5],
    'trials': [3],
    'data_sampling_fq': 500,
    'time_steps': 10000,
    'step': 5000
}

# Spectrogram settings
SPECTROGRAM_CONFIG = {
    'nperseg': 1000,  # for 251, 251 use 500; for 223, 223 use 1120
    'noverlap': 800,  # for 251, 251 use 462; for 223, 223 use 1080
    'num_samples': 12  # each window is a sample
}

# Model settings
MODEL_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.001
    },
    'dropout_rate': 0.5
}

# Feature settings
FEATURES = {
    'accelerometer': ['ax', 'ay', 'az'],
    'gyroscope': ['gx', 'gy', 'gz']
}

# Processing flags
FLAGS = {
    'treadmill': False,
    'overground': True,
    'plot_treadmill': True,
    'plot_overground': True,
    'save': False,
    'save_overview': False,
    'debugging': False
} 