import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import pandas as pd
import numpy as np

def simple_CNN(window_size = 128, num_features = 12,learning_rate=0.001):
    model = Sequential([

        # First Convolutional Layer
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        # Second Convolutional Layer
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        # Third Convolutional Layer
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.summary()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    