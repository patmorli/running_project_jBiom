# Running Classification with Spectrograms

For questions, please reach out to Patrick Mayerhofer at pmayerho@sfu.ca

This project processes running data using spectrograms and performs classification using deep learning models. It's designed to analyze running patterns from sensor data collected during treadmill and overground running sessions.

## Features

- Data preparation and preprocessing of running sensor data
- Spectrogram generation from time-series data
- Multiple deep learning models for classification (ResNet50, LSTM)
- Support for both treadmill and overground running data
- TFRecord creation for efficient data handling

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Librosa
- Scipy
- Bottleneck
- Scikit-learn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/running-classification.git
cd running-classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `script_running_data_preparation.py`: Main script for data preparation
- `script_running_classification_spectrogram_v2.py`: Main classification script
- `functions_running_data_preparation.py`: Utility functions for data preparation
- `create_TFRecords_spectrogram2.py`: Creates TFRecord files for TensorFlow
- `functions_my_model_resnet.py`: ResNet50-based model architectures with custom layers, transfer learning capabilities, and configurable layer freezing for fine-tuning
- `functions_recurrent_model.py`: LSTM/RNN model implementations with support for bidirectional layers, attention mechanisms, and variable sequence lengths
- `Colab/run_running_classification_spectrogram2.ipynb`: Google Colab notebook for model training with GPU acceleration. 

## Model Architectures

The project supports two main types of deep learning architectures:

### ResNet-based Models (`functions_my_model_resnet.py`)

The ResNet implementation provides:
- Custom convolutional layers before ResNet integration
- Transfer learning with ImageNet weights
- Configurable layer freezing for fine-tuning
- Enhanced versions with additional conv layers
- Batch normalization and dropout options
- Flexible input shape handling

Best used for:
- Complex spatial feature extraction from spectrograms
- High-resolution pattern analysis
- Transfer learning applications
- Multi-scale feature detection

### Recurrent Models (`functions_recurrent_model.py`)

The LSTM/RNN implementation offers:
- LSTM and GRU cell support
- Bidirectional layer options
- Attention mechanisms for pattern focus
- Time series processing optimizations
- Variable sequence length support
- State management for continuous prediction

Best used for:
- Temporal pattern recognition
- Long-term dependency analysis
- Real-time sequence processing
- Continuous monitoring scenarios

## Usage

### Directory Setup

Before running the scripts, you need to set up your directory structure and configure the paths:

1. Create a directory structure as follows:
```
Your_Project_Root/
├── Code/
│   ├── script_running_data_preparation.py
│   ├── script_running_classification_spectrogram_v2.py
│   ├── functions_running_data_preparation.py
│   └── create_TFRecords_spectrogram2.py
├── Data/
│   ├── Raw/
│   │   └── SENSOR{xxx}/
│   ├── Prepared/
│   │   ├── csv/
│   │   ├── tfrecords/
│   │   └── parquet/
│   └── Results/
│       ├── model_weights/
│       └── model_history/
└── Colab/
    └── run_running_classification_spectrogram2.ipynb
```

2. Update directory paths:
   - Open each script (*.py) file
   - Locate the directory configuration section at the top
   - Replace the existing paths with your project's paths
   - Example:
     ```python
     # Replace this:
     dir_root = '/Users/patrick/Google Drive/My Drive/Running Plantiga Project/'
     # With your path:
     dir_root = '/path/to/your/project/'
     ```

### Running the Pipeline

1. Prepare your data:
```bash
# Configure your data settings in config.py first
python script_running_data_preparation.py
```

The data preparation script will:
- Import raw sensor data from the Raw directory
- Clean and preprocess the data
- Split the data into treadmill and overground categories
- Save the processed data in the Prepared/csv directory

2. Create TFRecords:
```bash
# Adjust spectrogram settings in config.py if needed
python create_TFRecords_spectrogram2.py
```

This step will:
- Convert the processed CSV files into TFRecord format
- Generate spectrograms for each data window
- Save the TFRecords in the Prepared/tfrecords directory

3. Run classification:
```bash
# Local training (CPU/GPU):
python script_running_classification_spectrogram_v2.py

# OR use Google Colab (recommended for GPU acceleration):
# Upload the notebook from Colab/ to Google Colab
# Follow the notebook instructions for mounting Google Drive and running the training
```

Key configuration parameters in `config.py`:
```python
# Data processing
DATA_CONFIG = {
    'subjects': [...],  # List of subject IDs
    'speeds': [2.5, 3.0, 3.5],  # Running speeds
    'trials': [3],  # Number of trials
    'data_sampling_fq': 500,  # Sampling frequency
    'time_steps': 10000,
    'step': 5000
}

# Model settings
MODEL_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'dropout_rate': 0.5
}
```

## Data Format

The project expects sensor data in CSV format with the following columns:
- Accelerometer data (ax, ay, az)
- Gyroscope data (gx, gy, gz)
- Additional metadata (subject_id, trial_id, speed, tread or overground,seconds_10k)

Data should be organized in the following directory structure:
```
Data/
├── Raw/
│   └── SENSOR{xxx}/
├── Prepared/
│   ├── csv/
│   ├── tfrecords/
│   └── parquet/
└── Results/
    ├── model_weights/
    └── model_history/
```

## License

MIT License

Copyright (c) 2024 Running Classification Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 