#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For questions, please reach out to Patrick Mayerhofer at pmayerho@sfu.ca

ResNet50-based Model Architectures for Running Classification

This module implements various ResNet50-based architectures for analyzing running patterns
using spectrogram data. It provides three main model variants:
1. Continuous prediction model (regression)
2. Classification model (basic)
3. Enhanced classification model with additional convolutional layers

Key Features:
- Transfer learning with ImageNet weights
- Custom convolutional layers for spectrogram processing
- Configurable dropout for regularization
- Flexible input shapes for different data formats
- Multiple architecture variants for different use cases

References:
- ResNet50 architecture: https://arxiv.org/abs/1512.03385
- Transfer learning guide: https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b

Author: Patrick
"""

import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Conv3D
from tensorflow.keras.optimizers import Adam

def create_my_model_resnet50_cont(input_my_model, input_resnet, weights_to_use, dropout, n_bins):
    """Creates a ResNet50-based model for continuous prediction (regression).
    
    This model is designed for continuous value prediction from spectrogram data.
    It uses transfer learning from ResNet50 with custom preprocessing layers.
    
    Args:
        input_my_model: Keras Input tensor for the model's input layer
        input_resnet: Keras Input tensor for the ResNet50 component
        weights_to_use: Pre-trained weights to use (e.g., 'imagenet' or None)
        dropout: Dropout rate for regularization (0 to 1)
        n_bins: Number of output bins (unused in continuous model but kept for API consistency)
    
    Returns:
        Compiled Keras model ready for training
    
    Architecture:
        1. Input layer
        2. 3D Convolution for initial feature extraction
        3. Pre-trained ResNet50
        4. Flatten layer
        5. Dropout for regularization
        6. Dense output layer with ReLU activation
    """
    # Initialize ResNet50 with pre-trained weights
    res_model_pretrained = keras.applications.ResNet50(
        include_top=False,
        weights=weights_to_use,
        input_tensor=input_resnet
    )
    
    # Build the model
    my_model = Sequential()
    my_model.add(input_my_model)
    my_model.add(Conv3D(filters=1, kernel_size=(1,1,4), strides=(1,1,4), activation='relu'))
    my_model.add(res_model_pretrained)
    my_model.add(Flatten())
    my_model.add(layers.Dropout(dropout))
    my_model.add(layers.Dense(1, activation='relu'))
    my_model.summary()
    
    # Compile with RMSprop optimizer
    my_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
        metrics=['categorical_crossentropy']
    )
    
    return my_model

def create_my_model_resnet50_class(input_my_model, input_resnet, weights_to_use, dropout, n_bins, learning_rate):
    """Creates a ResNet50-based model for classification.
    
    This model performs classification of running patterns using spectrogram data.
    It uses transfer learning from ResNet50 with custom preprocessing layers.
    
    Args:
        input_my_model: Keras Input tensor for the model's input layer
        input_resnet: Keras Input tensor for the ResNet50 component
        weights_to_use: Pre-trained weights to use (e.g., 'imagenet' or None)
        dropout: Dropout rate for regularization (0 to 1)
        n_bins: Number of output classes
        learning_rate: Learning rate for the Adam optimizer
    
    Returns:
        Compiled Keras model ready for training
    
    Architecture:
        1. Input layer
        2. 3D Convolution for initial feature extraction
        3. Pre-trained ResNet50
        4. Flatten layer
        5. Dropout for regularization
        6. Dense output layer with softmax activation
    """
    # Initialize ResNet50 with pre-trained weights
    res_model_pretrained = keras.applications.ResNet50(
        include_top=False,
        weights=weights_to_use,
        input_tensor=input_resnet
    )
    
    # Build the model
    my_model = Sequential()
    my_model.add(input_my_model)
    my_model.add(Conv3D(filters=1, kernel_size=(1,1,4), strides=(1,1,4), activation='relu'))
    my_model.add(res_model_pretrained)
    my_model.add(Flatten())
    my_model.add(layers.Dropout(dropout))
    my_model.add(layers.Dense(n_bins, activation='softmax'))
    my_model.summary()
    
    # Compile with Adam optimizer
    my_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['categorical_crossentropy']
    )
    
    return my_model

def create_my_model_resnet50_class_more_conv_layers(input_my_model, input_resnet, weights_to_use, dropout, n_bins, learning_rate):
    """Creates an enhanced ResNet50-based model with additional convolutional layers.
    
    This model extends the basic classification model with additional convolutional
    layers before the ResNet50 component for more complex feature extraction.
    
    Args:
        input_my_model: Keras Input tensor for the model's input layer
        input_resnet: Keras Input tensor for the ResNet50 component
        weights_to_use: Pre-trained weights to use (e.g., 'imagenet' or None)
        dropout: Dropout rate for regularization (0 to 1)
        n_bins: Number of output classes
        learning_rate: Learning rate for the Adam optimizer
    
    Returns:
        Compiled Keras model ready for training
    
    Architecture:
        1. Input layer
        2. Multiple 3D Convolution layers with increasing filters
        3. Pre-trained ResNet50
        4. Flatten layer
        5. Dropout for regularization
        6. Dense output layer with softmax activation
    
    The additional convolutional layers create a deeper feature hierarchy
    before the ResNet50 component, allowing for more complex pattern recognition.
    """
    # Initialize ResNet50 with pre-trained weights
    res_model_pretrained = keras.applications.ResNet50(
        include_top=False,
        weights=weights_to_use,
        input_tensor=input_resnet
    )
    
    # Build the model with additional conv layers
    my_model = Sequential()
    my_model.add(input_my_model)
    # Progressive feature extraction with increasing filters
    my_model.add(Conv3D(filters=16, kernel_size=(1,1,3), strides=(1,1,1), activation='relu'))
    my_model.add(Conv3D(filters=32, kernel_size=(1,1,3), strides=(1,1,1), activation='relu'))
    my_model.add(Conv3D(filters=64, kernel_size=(1,1,3), strides=(1,1,1), activation='relu'))
    my_model.add(Conv3D(filters=3, kernel_size=(1,1,3), strides=(1,1,1), activation='relu'))
    my_model.add(Conv3D(filters=1, kernel_size=(1,1,2), strides=(1,1,1), activation='relu'))
    my_model.add(res_model_pretrained)
    my_model.add(Flatten())
    my_model.add(layers.Dropout(dropout))
    my_model.add(layers.Dense(n_bins, activation='softmax'))
    my_model.summary()
    
    # Compile with Adam optimizer
    my_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['categorical_crossentropy']
    )
    
    return my_model

"My model creation"
