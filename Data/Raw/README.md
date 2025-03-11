# Raw Data Directory

This directory should contain your raw sensor data files. Each subject's data should be in a separate directory named `SENSOR{xxx}` where `xxx` is the subject ID (e.g., SENSOR001).

## Expected File Format

Each CSV file should contain the following columns:
- `subject_id`: Unique identifier for each subject
- `trial_id`: Trial number
- `speed`: Running speed in m/s
- `ax`, `ay`, `az`: Accelerometer data
- `gx`, `gy`, `gz`: Gyroscope data
- `tread or overground`: Binary indicator (1 for treadmill, 0 for overground)
- `seconds_10k`: Time in seconds

See the sample data in `Data/Sample/` for an example. 