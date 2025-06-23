import sys

from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


import pandas as pd
import dask.dataframe as dd

import typing as t

DATASET_DIR = "dataset"

# This module provides functions to load datasets either as pandas DataFrames or Dask DataFrames.
def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    df = df.dropna(how='all')  # Drop rows where all elements are NaN
    df = onehot_encode(df)

    return df

def load_dataset(*, file_name: str, use_dask: bool = False) -> t.Union[pd.DataFrame, dd.DataFrame]:
    if use_dask:
        df_dask = dd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
        df_dask = df_dask.dropna(how='all')  # Drop rows where all elements are NaN
        df_dask = onehot_encode(df_dask.compute())
        df_dask = dd.from_pandas(df_dask, npartitions=1)  # Convert back to Dask DataFrame
        return df_dask
    return _load_raw_dataset(file_name=file_name)

def onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to encode.
    
    Returns:
    pd.DataFrame: The one-hot encoded DataFrame.
    """
    pdd = pd.get_dummies(df, columns=['Applicant ID', 'Gender', 'Age Group', 'Race', 'Training', 'Reactions'])
    print(pdd.columns)
    return pdd