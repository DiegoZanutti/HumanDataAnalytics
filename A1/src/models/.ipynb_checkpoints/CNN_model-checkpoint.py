'''
TensorFlow (v.2.2.0), 
Scikit-Learn,
Keras (v.2.3.1),
Pandas (v.1.0.5),
Numpy
(v.1.18.5). 

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import tensorflow as tf


def read_data(file_path):
    # column_names = ['subject','activity','time_s', 'lw_x','lw_y','lw_z','lh_x','lh_y','lh_z','la_x','la_y','la_z','ra_x','ra_y','ra_z']
    data = pd.read_csv(file_path,header = 0)
    return data


def make_input_data(train_df):
    magnitude_columns = ['magnitude_lw', 'magnitude_lh', 'magnitude_la', 'magnitude_ra']
    activity_column = 'activity'

    # Get the magnitude data
    magnitude_data = train_df[magnitude_columns].values

    # Get the activity data
    activity_data = train_df[activity_column].values

    # Concatenate magnitude and activity data along the channel axis
    input_data = np.concatenate([magnitude_data, activity_data[:, np.newaxis]], axis=-1)

    num_channels = magnitude_data[0].shape[-1]
    print("Number of input channels:", num_channels)


    print('magnitude_data.shape,',magnitude_data.shape)
    # Determine the shape of the magnitude data
    H, W = magnitude_data.shape
    N = 64
    # # Reshape input data to the expected shape (batch_size, height, width, channels)
    # input_data = input_data.reshape(N, H, W, -1)

    # # Convert input data to a TensorFlow tensor
    input_tensor = tf.constant(input_data, dtype=tf.float32)
    return H,W,N,input_tensor


class CNNModel(tf.Module):
    def __init__(self,channels,height,width,num_classes):
        self.conv1_weights = tf.Variable(tf.random.normal([3, 3, channels, 32]))
        self.conv1_biases = tf.Variable(tf.zeros([32]))
        self.conv2_weights = tf.Variable(tf.random.normal([3, 3, 32, 64]))
        self.conv2_biases = tf.Variable(tf.zeros([64]))
        self.fc1_weights = tf.Variable(tf.random.normal([height // 4 * width // 4 * 64, 64]))
        self.fc1_biases = tf.Variable(tf.zeros([64]))
        self.fc2_weights = tf.Variable(tf.random.normal([64, num_classes]))
        self.fc2_biases = tf.Variable(tf.zeros([num_classes]))
        self.height = height
        self.width = width


    def __call__(self, inputs):
        x = tf.nn.conv2d(inputs, self.conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(tf.nn.bias_add(x, self.conv1_biases))
        x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = tf.nn.conv2d(x, self.conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.relu(tf.nn.bias_add(x, self.conv2_biases))
        x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = tf.reshape(x, [-1, self.height // 4 * self.width // 4 * 64])
        x = tf.nn.relu(tf.matmul(x, self.fc1_weights) + self.fc1_biases)

        output = tf.matmul(x, self.fc2_weights) + self.fc2_biases
 




