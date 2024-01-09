import pandas as pd
import numpy as np
from data.data_processor import VectorMagnitude, Standardize
class DataSegmentation:
    """
    Class for segmenting data based on window size and overlap.
    """

    def __init__(self, window_duration, overlap, sampling_rate):
        """
        Initializes the DataSegmentation.

        Args:
            window_duration: Window size for segmentation, in seconds. ->2.56s
            overlap (float): Overlap between windows, as a fraction. -> 0.5
            sampling_rate (int): Sampling rate of the data, in Hz. -> 50HZ 
        """
        self.window_duration = window_duration 
        self.overlap = overlap
        self.sampling_rate = sampling_rate

    def __call__(self, data):
        """
        Segments the data based on initialized window size and overlap.

        Args:
            df (pd.DataFrame): DataFrame with data to segment.

        Returns:
            pd.DataFrame: Segmented DataFrame.
        """
        # convert df to numpy array
        window_size_samples = int(self.window_duration * self.sampling_rate) # 2.56 * 50 = 128 (number of samples in a window)
        shift_length = int(window_size_samples * self.overlap) # 128 * 0.5 = 64 (shift length)


        data_segments = []
        label_segments = []
        start_index = 0
        standardizer = Standardize()
        max_len = len(data)
        print("all data:",max_len)
        
        while start_index + window_size_samples <= len(data):
            data_segment = data[start_index:start_index + window_size_samples]
            label_segments.append(data_segment[0, 0]) # first column is label
            # make a copy of data_segment

            data_segment_new = standardizer(data_segment[:, 2:]) # standardize all columns except first two
            data_segments.append(data_segment_new) 
            start_index += shift_length


        print("Data Segments:",len(data_segments))
        segmented_data = np.array(data_segments)
        segmented_labels = np.array(label_segments).reshape(-1, 1)

        return segmented_data, segmented_labels
    
    def one_hot(y_, label_mapping):
        """
        Function to encode output labels from number indexes to one-hot encoding,
        considering non-sequential and non-zero-based class labels.

        Args:
            y_ (numpy array): Array of labels.
            label_mapping (dict): Mapping from actual labels to zero-based indexes.

        Returns:
            numpy array: One-hot encoded labels.
        """
        # Map labels to zero-based indexes
        mapped_labels = np.vectorize(label_mapping.get)(y_)

        # One-hot encoding
        n_values = len(label_mapping)
        return np.eye(n_values)[mapped_labels.astype(np.int32)]
