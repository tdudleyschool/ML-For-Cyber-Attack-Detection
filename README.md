# ML-For-Cyber-Attack-Detection
_Author: Tafari Dudley, Michael Porter & Sukhraj Singh_  
_Last update: 07/04/2023_  
This project is related to code used for the Advanced Security project "__Cyber Attack Detection thanks to Machine Learning Algorithms__". The authors of this work are Antoine Delplace, Sheryl Hermoso, and Kristofer Anandita.
## Perpose
The aim of this project was to use various machine learning algorithms on the Netflow database to detect cyberattacks. As a result, six algorithms were tested. 
## Methods
The project was performed on Google Collaborator. All Python modules were included within the ide. To repeat this in Google Collaborator, the `capture20110815-2.binetflow` must be imported from google drive before running the rest of the code. 
### Libraries Used
- Numpy
- Pandas
- Scipy
- Datetime
- h5py
- Matplotlib
- Scikit-learn
- Tensorflow -- `predict_Neural_Network.py`
### File Descriptions
- `preprocessing1.py` and `preprocessing2.py` are the files used to synthesize data from the Netflow files in a usable format. It achieves this by performing specific tasks such as data importation, data transformation, feature extraction, normalization, and saving the processed data for further analysis or modeling.

- `feature_extraction.py` Feature extraction attempts to reduce the number of features by creating new ones from existing features. The extraction technique used here is the time window schema. time window taken is 2 minutes, and a stride of 1 minute was used.
   
- `dimensionality_reduction.py` try to decrease the number of features using embedded methods or dimensionality reduction techniques. Both PCA and t-SNE techniques are used to reduce dimensionality. PCA is a linear technique that captures global variance in the data, while t-SNE is a nonlinear technique that focuses on preserving local structures and clusters in the data.

- `predict_Gradiant_Boosting_Algorithm_Analysis.py` Analysis of the dataset with Gradiant Boosting.
  
- `predict_Support_Vector_Machine_Analysis.py` Analyses the dataset using Support Vector Machine.
  
- `predict_logistic_regression.py` and `predict_statistic_analysis.py` are the two models Sukhraj chose to test the data sets on.

- `predict_Neural_Network.py` creates a training model using a neural network.
  
- `predict_Random_Forest.py` creates a training model using a random forest.
## Results
The algorithms were analyzed based on precision, recall, and f1 score. Precision measures the proportion of correctly predicted positive instances out of all the cases predicted as positive. It measures the amount of correctly classified botnets in comparison to misclassified ones. Recall measures the proportion of correctly predicted positive instances out of all positive instances. It will show the amount of correctly predicted botnets compared to every botnet in the system. Finally, the f1 score is a combination of both, making it the most reliable metric.

- The Gradient Boosting algorithm had a precision of 0.925, a recall of 0.649, and an F-beta score of 0.727.
- The Support Vector Machine algorithm had a precision of 0.949, a recall of 0.925, and an F-beta score of 0.937
- The Neural Network algorithm had a precision of 0.87, a recall of 0.49, and an F-beta score of 0.64.
- The Random Forest algorithm had a precision of 1.0, a recall of 1.0, and an F-beta score of 1.0. **Random Forest was shown to be the best algorithm in both the original paper and our observations.**
- The statistic analysis F-beta score was 0.70.
- The logistic regression F-beta score was 0.42. Similar to the original analysis performed in the reference paper, this algorithm performed the worst.

Note: The reference paper shows that each algorithm was less accurate for the smaller datasets, such as the one used here. 

## References
1. A. Delplace, S. Hermoso and K. Anandita. "Cyber Attack Detection thanks to Machine Learning Algorithms", _Advanced Security Report at the University of Queensland_, May 2019. [arXiv:2001.06309](https://arxiv.org/abs/2001.06309)

