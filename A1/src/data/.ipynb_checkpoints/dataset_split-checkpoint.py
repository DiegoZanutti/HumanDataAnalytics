# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np

# class X_Y_Split:
#     """
#     Class for splitting data into X and y.
#     """

#     def __init__(self,target_column_index):
#         """
#         Initializes the X_Y_Split.

#         Args:
#             target_column (str): Name of the target column.
#         """
#         self.target_column_index = target_column_index 

#     def __call__(self, data):
#         """
#         Splits the data into X and y.

#         Args:
#             data (pd.DataFrame): DataFrame with data to split.

#         Returns:
#             tuple: Two DataFrames representing X and y.
#         """
#         index = [0,self.target_column_index]
#         X = np.delete(data,index)
#         y = data[self.target_column_index]
#         return X, y

#     #TODO: customize this function to one_hot encode the y_train and y_test
#     def one_hot(y_): 
#         """
#         Function to encode output labels from number indexes.

#         E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
#         """
#         y_ = y_.reshape(len(y_))
#         n_values = int(np.max(y_)) + 1
#         return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
