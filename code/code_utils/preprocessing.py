import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

def load_auction_data(file_paths:list):
    """
    Load day-ahead auction and balancing mechanism data from a CSV file paths, merge these dataframes, and parse the index as datetime.
    The first row of the merged dataframe is removed, and the data is converted to numeric.

    Seperate the reference price forecasts contianed in the datasets from the actual price and volume data.

    Parameters
    ----------
    file_paths : list
        List of file paths to the CSV files to be loaded.
    Returns
    -------
    auction_data : pd.DataFrame
        Merged dataframe with the auction data.
    reference_forecasts : pd.DataFrame
        Dataframe with the reference forecasts.
    """
    df1=pd.read_csv(file_paths[0], delimiter=';', index_col=0)
    df2=pd.read_csv(file_paths[1], delimiter=';', index_col=0)

    # remove duplicate rows with same datetime (from daylight savings)
    df1=df1[~df1.index.duplicated(keep='first')]
    df2=df2[~df2.index.duplicated(keep='first')]

    auction_data=df1.merge(df2, left_index=True, right_index=True, how='left')
    # remove first row
    auction_data=auction_data.iloc[1:]
    # convert data to numeric
    auction_data=auction_data.apply(pd.to_numeric, errors='coerce')
    # parse dates
    auction_data.index=auction_data.index.str.replace(r'[\[\]]', '', regex=True)
    auction_data.index=pd.to_datetime(auction_data.index, format='%d/%m/%Y %H:%M')
    auction_data.index.name='date'
    auction_data=auction_data.asfreq('h')

    # seperate reference price forecasts from actual price and volume data
    reference_forecasts=auction_data.loc[:,auction_data.columns.str.contains('forecast')]

    auction_data=auction_data.loc[:,~auction_data.columns.str.contains('forecast')]

    return auction_data, reference_forecasts


def load_forecasts(file_path):
    """
    Load forecasts of energy fundamentals and price forecasts data from a CSV file, 
    parse the index as datetime, and convert data to numeric.

    Parameters
    ----------
    file_paths : list
        List of file paths to the CSV files to be loaded.
    Returns
    -------
    fundamentals : pd.DataFrame
        Dataframe with the energy fundamentals data.
    """
    df=pd.read_csv(file_path, delimiter=';', index_col=0)

    # remove duplicate rows with same datetime (from daylight savings)
    df=df[~df.index.duplicated(keep='first')]
    # remove first row
    df=df.iloc[1:]
    # convert data to numeric
    df=df.apply(pd.to_numeric, errors='coerce')
    # parse dates
    df.index=df.index.str.replace(r'[\[\]]', '', regex=True)
    df.index=pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
    df.index.name='date'
    df=df.asfreq('h')

    # columns containing 'previous' are not forecasts but actual previous values of the auction data 
    # we will construct our own price indicators from past auction data
    forecasts=df[df.columns[~df.columns.str.contains('previous')]]

    return forecasts

def STL_decompose_data(df, column, period:int):
    """
    Decompose a time series using STL decomposition. 
    Results are stored in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the time series to decompose.
    column : str
        Column name of the time series to decompose.
    period : int
        Period of the time series.

    Returns
    -------
    result : STL
        STL decomposition result.
    """
    # Interpolate missing values first in order to decompose, these interpolated values will
    # only slightly influence the seasonal and trand components
    stl = STL(df[column].interpolate(), period=period)
    result = stl.fit()
    return result

class Imputer:
    """
    Class to impute missing values using different methods.

    Parameters
    ----------
    method : str
        The imputation method to use. Options are:
        - 'periodic_rolling_mean'
        - 'STL_decomposition'
        - 'STL_ARIMA'
    **kwargs : dict
        Additional arguments for the specified imputation method.
    """

    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def impute(self, df, column):
        """
        Apply the specified imputation method to the column.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the specified column imputed.
        """
        if self.method == 'periodic_rolling_mean':
            return self.impute_with_periodic_rolling_mean(df, column, **self.kwargs)
        elif self.method == 'STL_decomposition':
            return self.impute_with_STL_decomposition(df, column, **self.kwargs)
        elif self.method == 'STL_ARIMA':
            return self.impute_with_STL_ARIMA(df, column, **self.kwargs)
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

    @staticmethod
    def impute_with_periodic_rolling_mean(df, column, window_start=2, window_end=2, period=24):
        """
        Impute missing values using a periodic rolling mean.

        Parameters
        ----------
        window_start : int, optional
            The number of values to consider before the missing value. Default is 2.
        window_end : int, optional
            The number of values to consider after the missing value. Default is 2.
        period : int, optional
            The period to consider for the rolling mean. Default is 24.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        imputed_df = df.copy()
        imputed_df = imputed_df.reset_index()
        series = imputed_df[column].copy()

        # Iterate over missing values
        for idx in imputed_df[imputed_df[column].isna()].index:
            # Select every nth value around the missing index
            start = max(0, idx - window_start * period)
            end = min(idx + window_end * period + 1, len(series) - 1)
            values = series.iloc[start:end:period].dropna()

            # Compute the mean and impute the value
            if not values.empty:
                imputed_df.loc[idx, column] = values.mean()

        imputed_df = imputed_df.set_index(df.index.name)
        return imputed_df

    @staticmethod
    def impute_with_STL_decomposition(df, column, period=24, iterations=1):
        """
        Impute values using STL decomposition.

        Parameters
        ----------
        period : int
            Period of the seasonal component.
        iterations : int, optional
            Number of iterations for imputation. Default is 1.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        imputed_df = df.copy()
        missing_indices = imputed_df[imputed_df[column].isna()].index

        for _ in range(iterations):
            res = STL_decompose_data(imputed_df, column, period)
            impute_values = res.seasonal + res.trend
            imputed_df.loc[missing_indices, column] = impute_values.loc[missing_indices]

        return imputed_df

    @staticmethod
    def impute_with_STL_ARIMA(df, column, period=24, ARIMA_order=(1, 0, 1), iterations=1):
        """
        Impute values using STL decomposition + ARIMA.

        Parameters
        ----------
        period : int
            Period of the seasonal component.
        ARIMA_order : tuple, optional
            Order of the ARIMA model. Default is (1, 0, 1).
        iterations : int, optional
            Number of iterations for imputation. Default is 1.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        imputed_df = df.copy()
        missing_indices = imputed_df[imputed_df[column].isna()].index

        for _ in range(iterations):
            res = STL_decompose_data(imputed_df, column, period)
            resid = res.resid
            arima_model = ARIMA(resid, order=ARIMA_order)
            arima_fit = arima_model.fit()

            resid_pred = arima_fit.predict(start=missing_indices[0], end=missing_indices[-1])
            resid_interp = resid.copy()
            resid_interp[missing_indices] = resid_pred[missing_indices]

            impute_values = res.seasonal + res.trend + resid_interp
            imputed_df.loc[missing_indices, column] = impute_values.loc[missing_indices]

        return imputed_df

def preprocess_data(df):
    """
    Preprocess the data by imputing missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to preprocess.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """

    for col in df.columns:
        imputer = Imputer('STL_decomposition', **{'period': 24, 'iterations':1})
        df.loc[:,col] = imputer.impute(df, col).loc[:,col]

    return df