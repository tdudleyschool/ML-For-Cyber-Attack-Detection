# Last update: 7/04/2023
"""
Dimensionality reduction algorithms using PCA and t-SNE
Parameters
----------
data_window.h5         : extracted data from preprocessing1.py
data_window3.h5        : extracted data from preprocessing2.py
data_window_labels.npy : label numpy array from preprocessing1.py
Return
----------
Plot 2D representation of the data thanks to PCA or t-SNE
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py

from sklearn import model_selection, manifold, decomposition

# Import data
print("Import data")

# Load data from HDF5 file and reset index
X = pd.read_hdf('data_window_botnet3.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3_botnet3.h5', key='data')
X2.reset_index(drop=True, inplace=True)

# Join the two datasets
X = X.join(X2)

# Remove 'window_id' column
X.drop('window_id', axis=1, inplace=True)

# Extract the labels
y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

# Load the labels from a numpy array
labels = np.load("data_window_botnet3_labels.npy", allow_pickle=True)

print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

# Create binary labels for a specific category
y_bin6 = y == np.where(labels == 'flow=From-Botne')[0][0]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)

print("y", np.unique(y, return_counts=True))
print("y_train", np.unique(X_train, return_counts=True))
print("y_test", np.unique(y_test, return_counts=True))

# Perform t-SNE dimensionality reduction
print("t-SNE")  # Beware: this is very time-consuming
clf = manifold.TSNE(n_components=2, random_state=123456)
clf.fit(X[['Dport_nunique', 'TotBytes_sum', 'Dur_sum', 'Dur_mean', 'TotBytes_std']])

# Print the embedded points
print(clf.embedding_)

# Select positive class points for plotting
y_plot = np.where(y_bin6 == True)[0]
print(len(y_plot))

# Randomly select negative class points for plotting
y_plot2 = np.random.choice(np.where(y_bin6 == False)[0], size=len(y_plot) * 100, replace=False)
print(len(y_plot2))

# Combine the indices of positive and negative class points
index = list(y_plot) + list(y_plot2)
print(len(index))

# Plot the t-SNE representation
plt.scatter(clf.embedding_[index, 0], clf.embedding_[index, 1], c=y[index])
plt.colorbar()
plt.show()

# Perform PCA dimensionality reduction
print("PCA")
clf = decomposition.PCA(n_components=2, random_state=123456)
clf.fit(X[['Dport_nunique', 'TotBytes_sum', 'Dur_sum', 'Dur_mean', 'TotBytes_std']].transpose())

# Print the principal components and explained variance ratios
print(clf.components_)
print(clf.explained_variance_ratio_)

# Select positive class points for plotting
y_plot = np.where(y_bin6 == True)[0]
print(len(y_plot))

y_plot2 = np.random.choice(np.where(y_bin6 == False)[0], size=len(y_plot) * 100, replace=False)
print(len(y_plot2))

index = list(y_plot) + list(y_plot2)
print(len(index))

# Plot the PCA representation
plt.scatter(clf.components_[0, index], clf.components_[1, index], c=y[index])
plt.colorbar()
plt.show()
