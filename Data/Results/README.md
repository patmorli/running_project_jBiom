# Results Directory

This directory stores the outputs from model training and evaluation:

## Structure

- `model_weights/`: Saved model weights in HDF5 format
  - Models are saved with names indicating their configuration
  - Best weights are saved during training using model checkpointing
- `model_history/`: Training history and metrics
  - Contains pickle files with training metrics
  - Includes validation results and performance metrics

## Note

These files are automatically generated during model training. The exact contents will depend on your model configuration and training parameters as set in `config.py`. 