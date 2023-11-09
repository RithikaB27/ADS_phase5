# ADS_phase5
# Covid-19 Vaccines Analysis Project

## Overview
This project analyzes Covid-19 vaccination progress worldwide. The dataset used is sourced from [Kaggle](https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress).

## Code Compilation
- The analysis code is written in Python.
- Ensure you have the required libraries installed by running `pip install pandas matplotlib scikit-learn numpy`.

## How to Run

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress"
data = pd.read_csv(url)

# Data preprocessing

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress"
data = pd.read_csv(url)

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Handle missing values
# Example: Replace missing numerical values with the mean, and missing categorical values with the most frequent value
numerical_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(exclude=['number']).columns

imputer_numerical = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_numerical.fit_transform(data[numerical_cols])

imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])

# Perform feature scaling (if necessary)
# Example: Standardize numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Display the first few rows of the preprocessed data
print("\nPreprocessed Data:")
print(data.head())

# Save the preprocessed data to a new CSV file
data.to_csv("preprocessed_data.csv", index=False)

# Code for linear regression example
# X and y should be replaced with your features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Print the mean squared error
print(f"Mean Squared Error: {mse}")


## Dataset Source
The dataset is available on Kaggle: [Covid World Vaccination Progress](https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress).

## Results
- Key findings and recommendations are documented in the project report.
- Additional visualizations and insights can be found in the 'results' folder.

## Sharing
This project is open-source 
