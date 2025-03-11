#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For questions, please reach out to Patrick Mayerhofer at pmayerho@sfu.ca

TFRecord Creation Script for Running Data
=======================================

This script converts processed running data into TFRecord format with spectrograms
for efficient deep learning model training.

Key Functions:
-------------
1. Data Loading:
   - Reads processed CSV files from treadmill/overground data
   - Supports multiple subjects, trials, and speeds
   - Handles windowed data segments

2. Spectrogram Generation:
   - Creates spectrograms for each sensor channel
   - Configurable window sizes and overlap
   - Supports visualization for quality control

3. TFRecord Creation:
   - Converts data into TensorFlow's efficient TFRecord format
   - Includes metadata (subject ID, trial info, etc.)
   - Organizes output by data type (treadmill/overground)

Directory Configuration:
----------------------
Before running this script, update these paths in the configuration section:
1. dir_root: Root directory containing the Data folder
2. dir_prepared: Location of prepared data
3. Other derived paths will be created automatically

Example Directory Structure:
    project_root/
    └── Data/
        ├── Prepared/
        │   ├── csv/
        │   │   ├── Treadmill/
        │   │   └── Overground/
        │   ├── tfrecords/
        │   └── parquet/
        └── Results/

Configuration Parameters:
----------------------
- data_sampling_fq: Sampling frequency of the sensor data
- time_steps: Window size for spectrogram generation
- step: Step size between windows
- nperseg: Length of each segment for spectrogram
- noverlap: Number of points to overlap between segments

Based on: https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c

Required Dependencies:
--------------------
- numpy: Numerical operations
- pandas: Data processing
- tensorflow: TFRecord creation
- scipy: Signal processing
- matplotlib: Optional visualization

Author: Patrick Mayerhofer
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import glob
from scipy import signal
import matplotlib.pyplot as plt
import functions_running_data_preparation as prep

# Set random seed for reproducibility
np.random.seed(1111)

# Data Processing Configuration
# ---------------------------
data_sampling_fq = 500  # Hz

# Window and Spectrogram Parameters
time_steps = 10000
step = 5000
nperseg = 1000  # 500 for 251,251; 1120 for 223,223
noverlap = 800  # 462 for 251,251; 1080 for 223,223
num_samples = 12  # Number of sensor channels

# Processing Flags
save_flag = 0
plot_flag = 1  # Warning: plots many windows if enabled

# Dataset Configuration
subjects = [100]  # Add more subject IDs as needed
tread_or_overground = 'Treadmill'
trials = [1]
speeds = [2.5, 3.0, 3.5]

# Directory Configuration
# ---------------------
# TODO: Replace this with your project root directory
dir_root = '/path/to/your/project/Data/'
dir_prepared = os.path.join(dir_root, 'Prepared')
dir_csv = os.path.join(dir_prepared, 'csv')
dir_tfr = os.path.join(dir_prepared, 'tfrecords')
dir_spectrogram = os.path.join(dir_tfr, 'windows_10000_spectrogram_fq100')
dir_parquet = os.path.join(dir_prepared, 'parquet')

# Data type specific directories
dir_data_overground = os.path.join(dir_csv, 'Overground')
dir_data_treadmill = os.path.join(dir_csv, 'Treadmill')
dir_data_tfr_overground = os.path.join(dir_spectrogram, 'Overground')
dir_data_tfr_treadmill = os.path.join(dir_spectrogram, 'Treadmill')

# Create required directories
for directory in [dir_prepared, dir_csv, dir_tfr, dir_spectrogram, dir_parquet,
                 dir_data_overground, dir_data_treadmill, 
                 dir_data_tfr_overground, dir_data_tfr_treadmill]:
    os.makedirs(directory, exist_ok=True)

# Get all CSV files
all_files = np.sort(glob.glob(os.path.join(dir_csv, "*/*")))

# Load overview file
overview_file = pd.read_csv(os.path.join(dir_root, 'my_overview_file.csv'))

# creating empty spectrogram_image file
t_length = int((time_steps - nperseg) / (nperseg - noverlap) + 1)
f_length = int((nperseg / 2) + 1)



num_spectrograms_per_subject = int((25000 - time_steps)/(time_steps - step) + 1) * len(speeds) * len(trials)


for subject in subjects:
    ## create a spectrogram (12 layers because 12 features) for each window and save a tfr with all windows
    spectrogram_images = np.empty((12, 223,t_length,12)) #first 12 should be calculated
    i_spectrogram = 0
    sensor = "SENSOR" + "{:03d}".format(subject)
    output_list = []
    dir_files = list()
    print('running: ' + str(subject))
    if tread_or_overground == 'Treadmill':
        for trial in trials:
            for speed in speeds:
                
                # e.g.: SENSOR001_2.5
                if trial == 1:
                    filename = sensor + '_' + str(speed) + '.csv'
                
                # e.g.: SENSOR001_2.5_2
                if trial == 2 or trial == 3:
                    filename = sensor + '_' + str(speed) + '_' + str(trial) + '.csv'
                
                dir_files.append(dir_data_treadmill + filename)  
                
    for dir_file in dir_files:
        # load the file
        df_cur = pd.read_csv(dir_file)
        
        # delete non-usable data
        df_cur = df_cur[df_cur['usabledata'] == True]
        
        # create output_list with moving windows
        score_10k = int(df_cur.score_10k.iloc[0])
        seconds_10k = int(df_cur.seconds_10k.iloc[0])
        subject_id = int(df_cur.subject_id.iloc[0])
        tread_or_overground_bool = int(df_cur.iloc[0,17])
        if df_cur.trial_id.iloc[0] == 1:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0])
        if df_cur.trial_id.iloc[0] == 2:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_2'
        if df_cur.trial_id.iloc[0] ==3:
            filename_and_id = 'SENSOR' + str(subject) + '_' + str(df_cur.speed.iloc[0]) + '_3'
            
                
                
        # creates windows and saves in a dataframe
        file_id = 0
        for i in range(0, len(df_cur)-time_steps + 1, step):
            v = df_cur.iloc[i:(i + time_steps)]
            df_acc_and_angvel = v.iloc[:, 3:15]
            list_samples_cur = df_acc_and_angvel.to_dict('records')
            
            
            for feature in range(len(df_acc_and_angvel.iloc[0])):
                f, t, Sxx = signal.spectrogram(df_acc_and_angvel.iloc[:,feature], data_sampling_fq, nperseg=nperseg, noverlap=noverlap)
                f = f[0:223]
                Sxx = Sxx[0:223,:]
                spectrogram_images[i_spectrogram,:,:,feature] = Sxx
                
                if plot_flag:
                    plt.figure()
                    plt.pcolormesh(t, f, Sxx, shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.title(df_acc_and_angvel.columns[feature] + str(df_cur['speed'].iloc[0]))
            i_spectrogram = i_spectrogram + 1   
    
    score_10k_array = np.full(num_spectrograms_per_subject, score_10k)
    seconds_10k_array = np.full(num_spectrograms_per_subject, seconds_10k)
    subject_id_array = np.full(num_spectrograms_per_subject, filename_and_id)
    filedir = dir_spectrogram + tread_or_overground + '/' + sensor
    
    if save_flag:
        count = prep.write_spectrograms_to_tfr(spectrogram_images, score_10k_array, seconds_10k_array, filedir, subject_id_array)
    
                

    