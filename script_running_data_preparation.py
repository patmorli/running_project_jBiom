#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For questions, please reach out to Patrick Mayerhofer at pmayerho@sfu.ca

Running Data Preparation Script
=============================

This script processes raw running sensor data and prepares it for classification.

Key Functions:
-------------
1. Data Loading:
   - Imports raw sensor data from treadmill and/or overground running
   - Supports multiple subjects, trials, and speeds
   - Handles data from the Raw/SENSOR{xxx} directories

2. Data Processing:
   - Cleans treadmill data by removing acceleration/deceleration periods
   - Processes overground data for consistent speed sections
   - Labels usable and non-usable data segments
   - Visualizes data for quality control

3. Data Export:
   - Saves processed data to CSV files
   - Organizes output by treadmill/overground categories
   - Updates overview file with processing status

Directory Configuration:
----------------------
Before running this script, update the following paths:
1. dir_root: Root directory of your project
2. dir_data_raw: Location of raw sensor data
3. dir_overview_file: Path to the overview CSV file

Example Directory Structure:
    project_root/
    ├── Code/
    │   └── functions_running_data_preparation.py
    └── Data/
        ├── Raw/
        │   └── SENSOR{xxx}/
        └── Prepared/
            └── csv/

Required Dependencies:
--------------------
- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib: Data visualization
- bottleneck: Fast numerical operations
- functions_running_data_preparation: Custom utility functions

Author: Patrick Mayerhofer
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
import os
import functions_running_data_preparation as frg
import pickle

# Configuration Variables
# ----------------------

# Subject and trial settings
subjects = [1,2,3,8,9,20,21,24,28,32,33,35,36,42,45,49,55,79,108,110,111,122,131,133]
speeds = [2.5, 3.0, 3.5]
trials = [3]

# Processing flags
treadmill_flag = 0
overground_flag = 1
plot_flag_treadmill = 1
plot_flag_overground = 1
save_flag = 0
save_overview_flag = 0
x_debugging = 0  # for files with _x suffix

# Directory Configuration
# ---------------------
# TODO: Replace these paths with your project paths
dir_root = '/path/to/your/project/'  # Set this to your project root directory
dir_data_raw = os.path.join(dir_root, 'Data', 'Raw')
dir_overview_file = os.path.join(dir_root, 'Data', 'my_overview_file.csv')

# Import overview file
overview_file = pd.read_csv(dir_overview_file)


"""import raw data and add subject id, trial id, and speeds to each dataset"""
"""mark unusable data from treadmill running and/or overground running with 0 and rest with 1""" 
"""save data"""
if treadmill_flag:
    data_treadmill_raw_key, data_treadmill_raw, overview_file = frg.import_treadmill_data(dir_data_raw, subjects, speeds, trials, overview_file, x_debugging)
    data_treadmill_raw = frg.clean_treadmill_data(data_treadmill_raw, data_treadmill_raw_key, plot_flag_treadmill)
    
    
    """save each subjects's data with the key as its name"""
    if save_flag:
        for i in range(1,len(data_treadmill_raw_key)):
            for z in range(0,len(data_treadmill_raw_key[i])):
                for a in range(0,len(data_treadmill_raw_key[i][z])):
                    if data_treadmill_raw_key[i][z][a] != 'not':
                        filepath = dir_root + 'Data/Prepared/csv/Treadmill/' 
                        if os.path.isdir(filepath) != True:
                            os.makedirs(filepath)
                        file = data_treadmill_raw[i][z][a]
                        file.to_csv(filepath + data_treadmill_raw_key[i][z][a] + '.csv', index = False)
        
    
    

 
    
    
if overground_flag:
    data_overground_raw_key, data_overground_raw, overview_file = frg.import_overground_data(dir_data_raw, subjects, trials, overview_file, x_debugging)
    data_overground_raw = frg.clean_overground_data(data_overground_raw, data_overground_raw_key, plot_flag_treadmill)
    
    
    """save each subjects's data with the key as its name"""
    if save_flag:
        for i in range(1,len(data_overground_raw_key)):
            for z in range(0,len(data_overground_raw_key[i])):
                    if data_overground_raw_key[i][z] != 'not':
                        filepath = dir_root + 'Data/Prepared/csv/Overground/'
                        if os.path.isdir(filepath) != True:
                            os.makedirs(filepath)
                        file = data_overground_raw[i][z]
                        file.to_csv(filepath + data_overground_raw_key[i][z] + '.csv', index = False)

# also save overview file
if save_overview_flag:
    overview_file.to_csv(dir_overview_file, index = False)    
       


