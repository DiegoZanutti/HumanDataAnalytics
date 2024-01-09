import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, medfilt


class VectorMagnitude:
    """
    Class to compute vector magnitude from accelerometer data.
    """

    def __init__(self, columns):
        self.columns = columns

    def __call__(self, df):
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