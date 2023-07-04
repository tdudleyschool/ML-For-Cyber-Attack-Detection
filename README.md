# ML-For-Cyber-Attack-Detection
_Author: Tafari Dudley, Michael Porter & Sukhraj Singh_  
_Last update: 07/04/2023_  
This project is related to code used for the Advanced Security project "__Cyber Attack Detection thanks to Machine Learning Algorithms__". This work has been carried out by Antoine Delplace, Sheryl Hermoso, and Kristofer Anandita.
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
### File Descriptions
- `preprocessing1.py` and `preprocessing2.py` are the files used to synthesize data from the Netflow files in a usable format. It achieves this by performing specific tasks such as data importation, data transformation, feature extraction, normalization, and saving the processed data for further analysis or modeling.

- `feature_extraction.py` Feature extraction attempts to reduce the number of features by creating new ones from existing features. The extraction technique used here is the time window schema. time window taken is 2 minutes, and a stride of 1 minute was used.
   
- `dimensionality_reduction.py` try to decrease the number of features using embedded methods or dimensionality reduction techniques. Both PCA and t-SNE techniques are used to reduce dimensionality. PCA is a linear technique that captures global variance in the data, while t-SNE is a nonlinear technique that focuses on preserving local structures and clusters in the data.

- `predict_Gradiant_Boosting_Algorithm_Analysis.py` Analysis the dataset with Gradiant Boosting.

- `predict_logistic_regression.py` and `predict_statistic_analysis.py` are the two models Sukhraj choose to test the data sets on.

