import pandas as pd
import numpy as np


def find_loc_null(df):
    """
    Find the location of null values in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to search for null values.

    Returns
    -------
    loc_null : list
        List of tuples with the index and column name of each null value.
    """
    loc_null = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if pd.isnull(df.iloc[i,j]):
                loc_null.append((i, df.columns[j]))
    return loc_null

def get_sample_with_missing_values(df, start_date, end_date, column, days_missing=1):   
    """
    Get a sample of the dataframe between two dates. Add missing values to the sample at the halfway mark.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to sample.
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    end_date : str
        End date in the format 'YYYY-MM-DD'.

    Returns
    -------
    sample : pd.DataFrame
        Sampled dataframe.
    sample_missing : pd.DataFrame
        Sampled dataframe with missing values.
    """
    sample = df.copy()
    sample = sample.loc[start_date:end_date, column].to_frame()
    sample_missing=sample.copy()
    missing_start = sample.index[int(len(sample)/2)]
    missing_end = sample.index[int(len(sample)/2)+24*days_missing-1]
    print(f"Missing values from {missing_start} to {missing_end}")
    sample_missing.loc[missing_start:missing_end] = np.nan
    dates_missing = sample_missing.index[(sample_missing[column].isnull())]
    return sample, sample_missing, dates_missing