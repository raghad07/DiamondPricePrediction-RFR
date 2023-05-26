# shai_ai

Diamond Price Prediction using Random Forest Regression
This repository contains the code for predicting the price of diamonds using Random Forest Regression. The goal of this project is to build a machine learning model that can accurately predict the price of diamonds based on various features such as carat, cut, color, clarity, depth, table, and dimensions.

# Project Structure
The repository is structured as follows:

# data: 
This directory contains the dataset files used for training and testing the model.
notebooks: This directory contains Jupyter notebooks with the code for data exploration, model training, evaluation, and prediction.
models: This directory stores the trained models for future use.
utils: This directory includes utility functions used throughout the project.
Requirements
To run the code, you will need the following dependencies:

# Python 3.x
Jupyter Notebook
Pandas
NumPy
Scikit-learn
Getting Started
To get started, follow these steps:

# Clone the repository to your local machine.
Navigate to the notebooks directory.
Open the Jupyter notebooks and run each cell to execute the code sequentially.
Data
The dataset used for this project is stored in the data directory. The dataset contains information about the diamonds, including their carat weight, cut, color, clarity, depth, table, and dimensions. The dataset is split into a training set and a test set for model evaluation.

# Model Training
The Random Forest Regression model is used for predicting diamond prices. The model is trained on the training dataset using the features mentioned above. Grid search is performed to tune the hyperparameters of the model, including the number of estimators and maximum features.

# Model Evaluation
The performance of the trained model is evaluated on the test dataset using the root mean squared error (RMSE) metric. Additionally, the feature importances are analyzed to determine the significance of each feature in predicting the diamond prices.

# Prediction
The trained model is then used to predict the prices of diamonds in the test dataset. The predictions are saved in a CSV file for further analysis or submission.

# Conclusion
This project demonstrates the application of Random Forest Regression in predicting diamond prices. The code provided can be used as a reference for building similar regression models or for predicting diamond prices using different machine learning algorithms.

