#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For questions, please reach out to Patrick Mayerhofer at pmayerho@sfu.ca

Recurrent Neural Network Models for Running Classification

This module implements various RNN-based architectures (LSTM, BiLSTM) for analyzing
running patterns using time series sensor data. It provides multiple model variants
for both classification and continuous prediction tasks.

Key Features:
- LSTM and Bidirectional LSTM implementations
- Configurable layer depth and width
- Support for sequence-to-sequence processing
- Flexible input shapes for different data formats
- Multiple architecture variants for different use cases

Author: Patrick
"""

from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import keras
from keras import backend as K
import tensorflow as tf

"""create a custom model for classification.
train X is the input data. layers_nodes is a list where the 
size represents the number of layers and the numbers of nodes in each layer"""
def lstm_model_class(input_my_model, layers_nodes, learning_rate):
    """Creates an LSTM model for classification tasks.
    
    This model implements a deep LSTM architecture with configurable layers
    for classifying running patterns from sequential data.
    
    Args:
        input_my_model: Keras Input tensor for the model's input layer
        layers_nodes: List of integers specifying number of units in each LSTM layer
        learning_rate: Learning rate for the Adam optimizer
    
    Returns:
        Compiled Keras model ready for training
    
    Architecture:
        1. Input layer
        2. Multiple LSTM layers (configurable)
           - Return sequences for all but last LSTM layer
           - ReLU activation
        3. Dense output layer with softmax activation
    
    Example:
        layers_nodes = [64, 32]  # Creates two LSTM layers with 64 and 32 units
    """
    model = Sequential()
    model.add(input_my_model)
    
    # Add LSTM layers based on layers_nodes configuration
    if len(layers_nodes) == 1:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=False))
    else:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=True))
        
        for i in range(len(layers_nodes)-1):
            return_sequences = i < len(layers_nodes)-2
            model.add(LSTM(layers_nodes[i+1], activation='relu', 
                         return_sequences=return_sequences))
    
    # Output layer for classification
    model.add(layers.Dense(3, activation='softmax'))
    
    # Compile with Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['categorical_crossentropy']
    )
    
    print(model.summary())
    return model

def lstm_model_class_try(input_my_model, layers_nodes, learning_rate):
    model = Sequential()
    model.add(input_my_model)
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(layers.Dense(10, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=['categorical_crossentropy'])
    print(model.summary())
    return model

def lstm_model_cont(input_my_model, layers_nodes, dropout):
    """Creates an LSTM model for continuous prediction (regression).
    
    This model implements a deep LSTM architecture with configurable layers
    for continuous value prediction from sequential data.
    
    Args:
        input_my_model: Keras Input tensor for the model's input layer
        layers_nodes: List of integers specifying number of units in each LSTM layer
        dropout: Dropout rate for regularization (0 to 1)
    
    Returns:
        Compiled Keras model ready for training
    
    Architecture:
        1. Input layer
        2. Multiple LSTM layers (configurable)
           - Return sequences for all but last LSTM layer
           - ReLU activation
        3. Dropout layer
        4. Dense output layer with ReLU activation
    """
    model = Sequential()
    model.add(input_my_model)
    
    # Add LSTM layers based on layers_nodes configuration
    if len(layers_nodes) == 1:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=False))
    else:
        model.add(LSTM(layers_nodes[0], activation='relu', return_sequences=True))
        
        for i in range(len(layers_nodes)-1):
            return_sequences = i < len(layers_nodes)-2
            model.add(LSTM(layers_nodes[i+1], activation='relu', 
                         return_sequences=return_sequences))
    
    # Add regularization and output layer
    model.add(layers.Dropout(dropout))    
    model.add(layers.Dense(1, activation='relu'))
    
    def root_mean_squared_error(y_true, y_pred):
        """Custom RMSE loss function."""
        y_true = tf.cast(y_true, tf.float32)
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
    # Compile with Adam optimizer and RMSE loss
    model.compile(
        loss=root_mean_squared_error,
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        metrics=[root_mean_squared_error]
    )
    
    print(model.summary())
    return model
    

"""Create the bidirectional lstm model. Needs X_train and 
y_train to understand the nature of the model input and output.
Input: X_train, ytrain 
Output: The prepared model"""
def bilstm_model_class(X_train, y_train):
    """Creates a Bidirectional LSTM model for classification.
    
    This model implements a bidirectional LSTM architecture for improved
    pattern recognition in both forward and backward directions of the sequence.
    
    Args:
        X_train: Training data to determine input shape
        y_train: Training labels to determine number of classes
    
    Returns:
        Compiled Keras model ready for training
    
    Architecture:
        1. Bidirectional LSTM layer (128 units)
        2. Dropout layer (0.5)
        3. Dense layer with ReLU (128 units)
        4. Output layer with softmax activation
    
    The bidirectional approach allows the model to learn patterns
    in both time directions, which can be beneficial for complex
    temporal dependencies in running patterns.
    """
    model = keras.Sequential()
    
    # Bidirectional LSTM layer
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=128,
                input_shape=[X_train.shape[1], X_train.shape[2]]
            )
        )
    )
    
    # Regularization and dense layers
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    
    # Compile with Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )
    
    return model
