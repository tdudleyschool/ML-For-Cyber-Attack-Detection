# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
# Pre-processing program to extract window-related normalized entropy from Netflow files

# Importing necessary libraries
import pandas as pd
import numpy as np
import datetime
import h5py
from scipy.stats import mode

# Setting window width and stride
window_width = 120  # seconds
window_stride = 60  # seconds

# Printing "Import data" to indicate the start of data import
print("Import data")

# Reading the Netflow file into a pandas DataFrame
data = pd.read_csv('/content/drive/My Drive/REU2023/CyberAttackDataset/capture20110815-2.binetflow')

# Printing "Preprocessing" to indicate the start of the preprocessing step
print("Preprocessing")

# Function to normalize a column in the DataFrame
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)

    dt[column] = (dt[column]-mean) / std

# Converting 'StartTime' column to datetime and converting it to nanoseconds
data['StartTime'] = pd.to_datetime(data['StartTime']).astype(np.int64) * 1e-9

# Finding the minimum 'StartTime' value
datetime_start = data['StartTime'].min()

# Calculating 'Window_lower' and 'Window_upper_excl' values based on window width and stride
data['Window_lower'] = (data['StartTime'] - datetime_start - window_width) / window_stride + 1
data['Window_lower'].clip(lower=0, inplace=True)
data['Window_upper_excl'] = (data['StartTime'] - datetime_start) / window_stride + 1
data = data.astype({"Window_lower": int, "Window_upper_excl": int})

# Dropping 'StartTime' column from the DataFrame
data.drop('StartTime', axis=1, inplace=True)

# Factorizing the 'Label' column and storing the labels for later use
data['Label'], labels = pd.factorize(data['Label'].str.slice(0, 15))

# Function to calculate the normalized entropy for a DataFrame column
def RU(df):
    if df.shape[0] == 1:
        return 1.0
    else:
        proba = df.value_counts() / df.shape[0]
        h = proba * np.log10(proba)
        return -h.sum() / np.log10(df.shape[0])

# Creating an empty DataFrame to store the extracted data
X = pd.DataFrame()

# Calculating the number of windows
nb_windows = data['Window_upper_excl'].max()
print(nb_windows)

# Looping over each window
for i in range(0, nb_windows):
    # Grouping the data based on 'SrcAddr' within the current window and applying the 'RU' function to calculate normalized entropy
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('SrcAddr')
    X = X.append(gb.agg({'Sport': [RU],
                         'DstAddr': [RU],
                         'Dport': [RU]}).reset_index())
    print(X.shape)

# Deleting the original 'data' DataFrame to save memory
del(data)

# Renaming the columns of the extracted data DataFrame
X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]

# Getting the columns to normalize by removing the 'SrcAddr_' column
columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('SrcAddr_')

# Normalizing the selected columns in the extracted data DataFrame
normalize_column(X, columns_to_normalize)

# Printing the shape and information of the extracted data DataFrame
with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)

# Saving the extracted data DataFrame to an HDF5 file
X.drop('SrcAddr_', axis=1).to_hdf('data_window3_botnet3.h5', key="data", mode="w")

# Saving the 'SrcAddr' column to a NumPy array
np.save("data_window_botnet3_id3.npy", X['SrcAddr_'])

# Saving the labels to a NumPy array
np.save("data_window_botnet3_labels3.npy", labels)
