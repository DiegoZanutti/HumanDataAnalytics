import os
import pandas as pd
import numpy as np
import subprocess
from math import floor
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, folder_path, subject_split_ratio=0.8, random_state=42):
        self.train_data = None
        self.test_data = None
        self.folder_path = folder_path
        self.subject_split_ratio = subject_split_ratio 
        self.random_state = random_state

    def download_data(self):
        if os.path.exists(self.folder_path):
            print(f"Dataset folder '{self.folder_path}' already exists. Skipping download.")
        else:
            download_command = "wget -c --timeout 10 https://physionet.org/static/published-projects/accelerometry-walk-climb-drive/labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0.zip -O dataset.zip"
            unzip_command = "unzip -d dataset -q dataset.zip"
            subprocess.run(download_command, shell=True, check=True)
            subprocess.run(unzip_command, shell=True, check=True)
            print("`Download and extraction of dataset complete!")

    def read_files(self):
        if self.folder_path is None:
            print("Error: Folder path is not set. Please run download_data first.")
            return

        subjects = os.listdir(self.folder_path)
        num_train_subjects = floor(len(subjects) * self.subject_split_ratio)  # 25
        train_subjects, test_subjects = train_test_split(subjects, train_size=num_train_subjects, random_state=self.random_state, shuffle=True)
        train_arrays = []
        test_arrays = []
        for subject in subjects:    
            file_path = os.path.join(self.folder_path, subject)
            try:

                df = pd.read_csv(file_path) 
                # remove the rows with activity label 99 or 77 or 4
                df = df[(df['activity'] != 99) & (df['activity'] != 77) & (df['activity'] != 4)]
                # print("剩下的活动类型有：",df['activity'].unique())
                array = df.to_numpy() # convert df to numpy array
                
                if subject in train_subjects: 
                    train_arrays.append(array) # append to train data list
                else:
                    test_arrays.append(array)
            except Exception as e:
                print(f"Error processing file '{file_path}': {str(e)}")
        self.train_data = np.concatenate(train_arrays) if train_arrays else None
        self.test_data = np.concatenate(test_arrays) if test_arrays else None
    






  