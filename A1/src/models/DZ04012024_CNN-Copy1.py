import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.optimizers import Adam
import pandas as pd
import numpy as np

def DZ_CNN(window_size = 128, num_features = 20,learning_rate=0.00005):
    model = Sequential([
		Conv1D(32, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
		BatchNormalization(),
		MaxPooling1D(pool_size=2),
		Dropout(0.25),

		Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
		BatchNormalization(),
		MaxPooling1D(pool_size=2),
		Dropout(0.25),

		Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
		BatchNormalization(),
		MaxPooling1D(pool_size=2),
		Dropout(0.25),

		Flatten(),
		Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
		BatchNormalization(),
		Dropout(0.25),
		Dense(3, activation='softmax')
	])
    model.summary()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    