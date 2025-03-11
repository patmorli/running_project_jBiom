# Data Directory Structure

This directory contains all the data files for the running classification project. Below is the expected structure and description of each directory:

```
Data/
├── Raw/                      # Raw sensor data files
│   └── SENSOR{xxx}/         # Individual sensor data folders (e.g., SENSOR001)
├── Prepared/                 # Processed data files
│   ├── csv/                 # Cleaned CSV files
│   │   ├── Treadmill/      # Treadmill running data
│   │   └── Overground/     # Overground running data
│   ├── tfrecords/          # TensorFlow record files
│   │   └── windows_*       # Windowed data for different configurations
│   └── parquet/            # Parquet format data (if used)
└── Results/                 # Model results and analysis
    ├── model_weights/      # Saved model weights
    └── model_history/      # Training history and metrics

## Data Format

### Raw Data
- Each sensor folder contains CSV files with raw accelerometer and gyroscope data
- Naming convention: SENSOR{xxx}_{speed}[_{trial}].csv
- Example: SENSOR001_2.5.csv, SENSOR001_2.5_2.csv

### Prepared Data
- Cleaned and preprocessed data in various formats (CSV, TFRecord, Parquet)
- Data is split into treadmill and overground categories
- Windows are created for time series analysis

### Results
- Trained model weights are saved in HDF5 format
- Training history and metrics are saved in pickle format

## Note
- Raw data files are not included in the repository due to size constraints
- Please contact the maintainers for access to the raw data
- Sample data may be provided for testing purposes 