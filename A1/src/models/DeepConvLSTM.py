# DeepConvLSTM model with 4 convolutional layers followed by 2 LSTM layers

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

def deep_conv_lstm_model(y,window_size = 128, num_features = 12,learning_rate=0.001):
    model = Sequential([
        # Convolutional layers
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        # LSTM layers
        LSTM(100, return_sequences=True),
        LSTM(100),
        # Dense layers (Fully connected layers)
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
]) 
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model