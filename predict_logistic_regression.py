# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Updated By: Sukhraj Singh
# Last update: 02/07/2023
"""
Perform a statistic analysis of the Logistic regression classifier.

Parameters
----------
data_window_botnetx.h5         : extracted data from preprocessing1.py
data_window3_botnetx.h5        : extracted data from preprocessing2.py
data_window_botnetx_labels.npy : label numpy array from preprocessing1.py
nb_prediction                  : number of predictions to perform

Return
----------
Print train and test mean accuracy, precison, recall, f1
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py

from sklearn import model_selection, feature_selection, utils, ensemble, linear_model, metrics

print("Import data")

# Read data from HDF file
X = pd.read_hdf('data_window_botnet3.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3_botnet3.h5', key='data')
X2.reset_index(drop=True, inplace=True)

# Join the two datasets
X = X.join(X2)

# Drop unnecessary columns
X.drop('window_id', axis=1, inplace=True)

# Extract the target variable y
y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

# Load labels and print them
labels = np.load("data_window_botnet3_labels.npy", allow_pickle=True)
print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

# Create binary target variable y_bin6
y_bin6 = y == np.where(labels == 'flow=From-Botne')[0][0]
print("y", np.unique(y, return_counts=True))

## Train

nb_prediction = 50
np.random.seed(seed=123456)
tab_seed = np.random.randint(0, 1000000000, nb_prediction)
print(tab_seed)

# Arrays to store performance metrics
tab_train_precision = np.array([0.] * nb_prediction)
tab_train_recall = np.array([0.] * nb_prediction)
tab_train_fbeta_score = np.array([0.] * nb_prediction)

tab_test_precision = np.array([0.] * nb_prediction)
tab_test_recall = np.array([0.] * nb_prediction)
tab_test_fbeta_score = np.array([0.] * nb_prediction)

# Perform predictions for each seed
for i in range(0, nb_prediction):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=tab_seed[i])

    # Upsample the minority class in the training set
    X_train_new, y_train_new = utils.resample(X_train, y_train, n_samples=X_train.shape[0] * 10, random_state=tab_seed[i])

    print(i)
    print("y_train", np.unique(y_train_new, return_counts=True))
    print("y_test", np.unique(y_test, return_counts=True))

    # Logistic Regression Classifier
    clf = linear_model.LogisticRegression(penalty='l2', C=550, random_state=tab_seed[i], multi_class="auto", class_weight={0: 0.044, 1: 1 - 0.044}, solver="lbfgs", max_iter=1000, verbose=0)
    clf.fit(X_train_new, y_train_new)

    # Predictions on the training set
    y_pred_train = clf.predict(X_train_new)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_train_new, y_pred_train)
    tab_train_precision[i] = precision[1]
    tab_train_recall[i] = recall[1]
    tab_train_fbeta_score[i] = fbeta_score[1]

    # Predictions on the test set
    y_pred_test = clf.predict(X_test)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred_test)
    tab_test_precision[i] = precision[1]
    tab_test_recall[i] = recall[1]
    tab_test_fbeta_score[i] = fbeta_score[1]

# Print performance metrics for the training and test sets
print("Train")
print("precision =", tab_train_precision.mean(), tab_train_precision.std(), tab_train_precision)
print("recall =", tab_train_recall.mean(), tab_train_recall.std(), tab_train_recall)
print("fbeta_score =", tab_train_fbeta_score.mean(), tab_train_fbeta_score.std(), tab_train_fbeta_score)

print("Test")
print("precision =", tab_test_precision.mean(), tab_test_precision.std(), tab_test_precision)
print("recall =", tab_test_recall.mean(), tab_test_recall.std(), tab_test_recall)
print("fbeta_score =", tab_test_fbeta_score.mean(), tab_test_fbeta_score.std(), tab_test_fbeta_score)
