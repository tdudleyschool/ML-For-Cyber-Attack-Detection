# Last update: 07/04/2023
"""
Pre-processing program to extract window-related data from Netflow files

Parameters
----------
window_width  : window width in seconds
window_stride : window stride in seconds
data          : pandas DataFrame of the Netflow file

Return
----------
Create 3 output files:
- data_window_botnetx.h5         : DataFrame with the extracted data: Sport, DstAddr, Dport,
                                   Dur (sum, mean, std, max, median), TotBytes (sum, mean, std, max, median),
                                   SrcBytes (sum, mean, std, max, median)
- data_window_botnetx_id.npy     : Numpy array containing SrcAddr
- data_window_botnetx_labels.npy : Numpy array containing Label
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import datetime
import h5py
from scipy.stats import mode

# Set window width and stride
window_width = 120  # seconds
window_stride = 60  # seconds

# Print a message to indicate data import
print("Import data")
data = pd.read_csv('/content/drive/My Drive/REU2023/CyberAttackDataset/capture20110815-2.binetflow')

# Preprocessing
print("Preprocessing")

# Define a function to normalize a column in the DataFrame
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)

    dt[column] = (dt[column] - mean) / std

# Convert the 'StartTime' column to datetime and convert it to nanoseconds
data['StartTime'] = pd.to_datetime(data['StartTime']).astype(np.int64) * 1e-9
datetime_start = data['StartTime'].min()

# Calculate the lower and upper bounds of the window for each record
data['Window_lower'] = (data['StartTime'] - datetime_start - window_width) / window_stride + 1
data['Window_lower'].clip(lower=0, inplace=True)
data['Window_upper_excl'] = (data['StartTime'] - datetime_start) / window_stride + 1
data = data.astype({"Window_lower": int, "Window_upper_excl": int})

# Drop the 'StartTime' column from the DataFrame
data.drop('StartTime', axis=1, inplace=True)

# Factorize the 'Label' column and store the labels
data['Label'], labels = pd.factorize(data['Label'].str.slice(0, 15))

# Create an empty DataFrame to store the extracted data
X = pd.DataFrame()

# Calculate the number of windows
nb_windows = data['Window_upper_excl'].max()
print(nb_windows)

# Iterate over each window
for i in range(0, nb_windows):
    print("Iteration:", i)

    # Group the data by 'SrcAddr' within the current window
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('SrcAddr')

    # Append the aggregated statistics to the output DataFrame
    X = X.append(gb.size().to_frame(name='counts').join(gb.agg({'Sport': 'nunique',
                                                                'DstAddr': 'nunique',
                                                                'Dport': 'nunique',
                                                                'Dur': ['sum', 'mean', 'std', 'max', 'median'],
                                                                'TotBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                                                'SrcBytes': ['sum', 'mean', 'std', 'max', 'median'],
                                                                'Label': lambda x: x.mode().iat[0]})).reset_index().assign(window_id=i))
    print(X.shape)

# Delete the original data DataFrame to save memory
del(data)

# Rename the columns in the output DataFrame
X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]

# Fill NaN values with -1 in the output DataFrame
X.fillna(-1, inplace=True)

# Normalize the selected columns in the output DataFrame
columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('SrcAddr')
columns_to_normalize.remove('Label_<lambda>')
columns_to_normalize.remove('window_id')
normalize_column(X, columns_to_normalize)

# Print the shape and data types of the output DataFrame
with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)

# Save the processed data to output files
X.drop('SrcAddr', axis=1).to_hdf('data_window_botnet3.h5', key="data", mode="w")
np.save("data_window_botnet3_id.npy", X['SrcAddr'])
np.save("data_window_botnet3_labels.npy", labels)
