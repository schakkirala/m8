import sys

from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import make_scorer

# This module provides functions to load datasets either as pandas DataFrames or Dask DataFrames.
import pandas as pd
import dask.dataframe as dd

import numpy as np
import typing as t
from preprocessing import load_dataset

TARGET_COLUMN = "Qualified"

def run_training() -> None:
    
    # read training data
    data = load_dataset(file_name="DriversLiceneseData.csv", use_dask=False)

    # Check if the target column is present
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")


    # split data into features and target

    X = data.drop(columns=['Qualified'] if 'Qualified' in data.columns else [])
    X.columns = X.columns.str.strip()  
    #if 'Applicant ID' in X.columns:
    #    print ("Column 'Applicant ID' found in features.")
    #else:
    #    print ("Column 'Applicant ID' not found in features. Available columns:", X.columns)
    
    y = data['Qualified'] 

    print(X.dtypes)
    print(X.head())
    # Convert categorical columns to numerical using one-hot encoding
    #X = pd.get_dummies(X, drop_first=True)

    print("Target variable 'Qualified' unique values:", y.unique())
    # Ensure that the target variable is numeric
    #print(y.dtypes)
    #print(y.head())
   


    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    y_pred = model.predict(X)

    # Check manually
    from sklearn.metrics import accuracy_score
    print("Accuracy:", accuracy_score(y, y_pred))


if __name__ == "__main__":
    print("Starting training process...")
    run_training()
