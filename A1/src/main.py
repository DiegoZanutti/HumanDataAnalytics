from data.data_segmentation import DataSegmentation
from data.data_loader import DataLoader
import pandas as pd
from collections import Counter
import tensorflow as tf
from utils.activity_type import ActivityType
import numpy as np
import random as rn
import datetime
import os
from utils.utils import select_model,train_model,plot

from keras.callbacks import EarlyStopping
import logging
# System and File Operations
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.utils import load_person_df_map, preprocess_data

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.enable_eager_execution()


def reset_seeds():
   os.environ["PYTHONHASHSEED"] = "42"
   np.random.seed(42) 
   rn.seed(12345)
   tf.random.set_seed(1234)


def preprocess_method_1():
    # First method of preprocessing
    data_loader = DataLoader("dataset/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/raw_accelerometry_data")
    data_loader.download_data()
    data_loader.read_files()
    
   
    data_seg = DataSegmentation(window_duration=1.28, overlap=0.5, sampling_rate=100)


    train_data_X,train_data_y = data_seg(data_loader.train_data)
    test_data_X,test_data_y = data_seg(data_loader.test_data)

    label_mapping = ActivityType.create_label_mapping()
    

    # one_hot_encoded_train_y = ActivityType.one_hot(train_data_y, label_mapping)
    # one_hot_encoded_test_y = ActivityType.one_hot(test_data_y, label_mapping)

    
    # final_train_y = one_hot_encoded_train_y.reshape(one_hot_encoded_train_y.shape[0],-1)
    # final_test_y = one_hot_encoded_test_y.reshape(one_hot_encoded_test_y.shape[0],-1)

    
    train_data_y_1d = np.squeeze(train_data_y)
    test_data_y_1d = np.squeeze(test_data_y)

    train_data_y_1d_mapped = np.vectorize(label_mapping.get)(train_data_y_1d)
    test_data_y_1d_mapped = np.vectorize(label_mapping.get)(test_data_y_1d)

    return train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped

def preprocess_method_2():
    WALKING = 1
    DESCENDING = 2
    ASCENDING = 3
    activities_list_to_consider = [WALKING, DESCENDING, ASCENDING]
    person_df_map = load_person_df_map(activities_list_to_consider)
    X, y = preprocess_data(person_df_map)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Reshape input for Conv1D
    num_features = X_train.shape[2]
    window_size=128
    X_train = X_train.reshape((-1, window_size, num_features))
    X_test = X_test.reshape((-1, window_size, num_features))
    return X_train, y_train,X_test,y_test


def calculate_magnitude(data):

    num_accelerometers = 4
    features_per_accelerometer = 3

    # Reshape the data to separate X, Y, Z for each accelerometer
    reshaped_magnitude = data.reshape(data.shape[0], data.shape[1], num_accelerometers, features_per_accelerometer)

    # Calculate the magnitude for each accelerometer
    magnitude = np.sqrt(np.sum(reshaped_magnitude**2, axis=-1))

    displacement_vector = np.diff(data, axis=1)

    # Pad the displacement vector with one sample of value 0 along axis 1
    displacement_vector_padded = np.concatenate([np.zeros((displacement_vector.shape[0], 1, displacement_vector.shape[2])), displacement_vector], axis=1)

    reshaped_displacement = displacement_vector_padded.reshape(data.shape[0], data.shape[1], num_accelerometers, features_per_accelerometer)
	
    # Calculate the magnitude of the displacement vector
    displacement_magnitude = np.sqrt(np.sum(reshaped_displacement**2, axis=-1))

    return magnitude, displacement_magnitude


def main():
    LABELS = [
        "Walking",
        "Descending Stairs",
        "Ascending Stairs"
    ]
    epochs = 20
    batch_size = 16
    learning_rate = 0.00005
    num_runs = 1
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"training_log_{current_time}.txt"
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()


    # Choose one of the two methods of preprocessing
    method = 1
    if method == 1:
        train_data_X,train_data_y_1d_mapped,test_data_X,test_data_y_1d_mapped = preprocess_method_1()
    else:
        X_train,y_train,X_test,y_test = preprocess_method_2()

    magnitude_result_train, displacement_result_train = calculate_magnitude(train_data_X)
    train_data_X_full = np.concatenate([train_data_X, magnitude_result_train, displacement_result_train],axis=-1)
    
    magnitude_result_test, displacement_result_test = calculate_magnitude(test_data_X)
    test_data_X_full = np.concatenate([test_data_X, magnitude_result_test, displacement_result_test],axis=-1)

    models = ["DZ_CNN"] #"LSTM_CNN", "Dual_LSTM", "DeepConvLSTM3",
    for model_name in models:
        with open(filename, "w") as file:
            logger.info(
            "Training Log\n"
            f"Date and Time: {datetime.datetime.now()}\n"
            f"Running the training process {num_runs} times\n\n"
            f"Seed for this run is: {42}, {12345}, {1234}\n\n"
            f"Training epoch: {epochs}, learning rate: {learning_rate}, "
            f"batch_size = {batch_size}, model is: {model_name} with new way of segmenting data\n"
            )
            for i in range(num_runs):
                tf.compat.v1.enable_eager_execution()
                reset_seeds() 
                # tf.compat.v1.disable_eager_execution()  # Or enable, depending on your requirement
                # loss,accuracy,precision,recall,f1= train_model(model_name,X_train,y_train,X_test,y_test) #first method
                loss,accuracy,precision,recall,f1= train_model(model_name,train_data_X_full[:,:,12:],train_data_y_1d_mapped,test_data_X_full[:,:,12:],test_data_y_1d_mapped,batch_size, epochs)
                # loss,accuracy,precision,recall,f1= train_model(model_name,X_train,y_train,X_test,y_test) #second method
                logger.info(f"Run {i+1}: \n"
                            f"Accuracy = {accuracy}\n"
                            f"Precision = {precision}\n"
                            f"Recall = {recall}\n"
                            f"F1 = {f1}\n"
                            f"Loss = {loss}\n")
                tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
