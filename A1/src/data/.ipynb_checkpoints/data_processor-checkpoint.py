import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, medfilt


class VectorMagnitude:
    """
    Class to compute vector magnitude from accelerometer data.
    """

    def __init__(self, columns):
        """
        Initializes the VectorMagnitude with specified columns.

        Args:
            columns (list): List of column prefixes to calculate magnitude for.
        """
        self.columns = columns

    def __call__(self, df):
        """
        Calculates the vector magnitude for the specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with accelerometer data.

        Returns:
            pd.DataFrame: DataFrame with added magnitude columns.
        """
        df_transformed = df.copy()
        for col in self.columns:
            x_col, y_col, z_col = f'{col}_x', f'{col}_y', f'{col}_z'
            magnitude_col = f'magnitude_{col}'
            df_transformed[magnitude_col] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
        return df_transformed

class Standardize:
    """
    Class for standardizing columns in a DataFrame or numpy array.
    """

    def __init__(self):
        """
        Initializes the Standardize class.
        """
        self.scaler = StandardScaler()

    def __call__(self, data):
        """
        Standardizes the specified columns in the DataFrame or numpy array.

        Args:
            data (pd.DataFrame or np.ndarray): DataFrame or numpy array with data to standardize.

        Returns:
            pd.DataFrame or np.ndarray: DataFrame or numpy array with standardized columns.
        """
        if isinstance(data, np.ndarray):
            return self.scaler.fit_transform(data)
        else:
            raise TypeError("Input data must be a pandas DataFrame or numpy ndarray.")

class ButterLowPassFilter:
    """
    Class to apply a Butterworth low-pass filter to DataFrame columns.
    """

    def __init__(self, columns, cutoff, fs, order=3):
        """
        Initializes the ButterLowPassFilter.

        Args:
            columns (list): List of column names to apply the filter.
            cutoff (float): Cutoff frequency for the filter.
            fs (float): Sampling frequency of the data.
            order (int): Order of the filter.
        """
        self.columns = columns
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def __call__(self, df):
        """
        Applies Butterworth low-pass filter to specified columns in DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with data to filter.

        Returns:
            pd.DataFrame: DataFrame with filtered data.
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        
        df_filtered = df.copy()
        for column in self.columns:
            df_filtered[column] = filtfilt(b, a, df[column].values)
        return df_filtered

class ApplyNoiseFilter:
    """
    Class to apply noise filtering (median and Butterworth low-pass) to DataFrame columns.
    """

    def __init__(self, columns, median_kernel_size=5, cutoff_freq=0.2, fs=100, order=3):
        """
        Initializes the ApplyNoiseFilter.

        Args:
            columns (list): List of column names to apply the filter.
            median_kernel_size (int): Kernel size for median filter.
            cutoff_freq (float): Cutoff frequency for Butterworth filter.
            fs (float): Sampling frequency of the data.
            order (int): Order of the Butterworth filter.
        """
        self.columns = columns
        self.median_kernel_size = median_kernel_size
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order

    def __call__(self, df):
        """
        Applies noise filtering to the specified columns in DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with data to filter.

        Returns:
            pd.DataFrame: DataFrame with filtered data.
        """
        df_filtered = df.copy()
        for column in self.columns:
            df_filtered[column] = medfilt(df[column], kernel_size=self.median_kernel_size)
        
        butter_lowpass_filter = ButterLowPassFilter(self.columns, self.cutoff_freq, self.fs, self.order)
        return butter_lowpass_filter(df_filtered)


