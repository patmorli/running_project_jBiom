# Google Colab Training Instructions

This directory contains the Colab notebook for training the running classification models with GPU acceleration.

## Setup Instructions

1. Upload `run_running_classification_spectrogram2.ipynb` to Google Colab
2. Configure your Google Drive:
   ```
   project_root/
   ├── Code/
   │   ├── script_running_classification_spectrogram_v2.py
   │   ├── functions_running_data_preparation.py
   │   ├── functions_classification_general.py
   │   ├── functions_my_model_resnet.py
   │   └── functions_recurrent_model.py
   └── Data/
       ├── Prepared/
       │   └── tfrecords/
       └── Results/
           ├── model_weights/
           └── model_history/
   ```

3. Update the project path in the notebook:
   ```python
   PROJECT_ROOT = '/content/drive/My Drive/YOUR_PROJECT_NAME'
   ```

4. Configure model parameters in the CONFIG dictionary:
   ```python
   CONFIG = {
       'epochs': 5000,
       'batch_size': 32,
       'val_split': 0.2,
       # ... other parameters
   }
   ```

## Running the Training

1. Mount your Google Drive
2. Install required packages
3. Run all cells in sequence
4. Monitor training progress
5. Check saved model weights and history in your Results directory

## Model Options

- Model Types:
  - `resnet50_class`: ResNet50 for classification
  - `resnet50_class_more_conv_layers`: ResNet50 with additional conv layers
  - `lstm_model_class`: LSTM model for classification

- Input Shapes:
  - Default: `(126, 40, 12, 1)` for sensor data
  - ResNet: `(126, 40, 3)` for spectrogram input

## Troubleshooting

1. If you get GPU memory errors:
   - Reduce batch size
   - Use a simpler model architecture
   - Clear runtime and restart

2. If you get path errors:
   - Double-check your Google Drive paths
   - Ensure all required files are in place
   - Check file permissions

3. For data loading issues:
   - Verify TFRecord files exist
   - Check data format matches expectations
   - Ensure correct file paths in configuration

## Notes

- Training time varies based on:
  - Dataset size
  - Model complexity
  - GPU availability
  - Batch size
  - Number of epochs

- Best practices:
  - Start with small epochs for testing
  - Use early stopping to prevent overfitting
  - Monitor validation loss
  - Save checkpoints regularly 