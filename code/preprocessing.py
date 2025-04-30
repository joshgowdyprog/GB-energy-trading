import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn import metrics

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

def plot_imputations(df, imputed_df_dict, dates_missing):
    """
    Plot the original dataframe and the imputed dataframes for various methods of imputing.
    """
    plt.figure(figsize=(20,5))
    plt.grid()
    plt.plot(df.loc[dates_missing], label='original')
    for key, val in imputed_df_dict.items():
        plt.plot(val.loc[dates_missing], label="imputed with"+key)
    plt.legend(loc='best')

def evaluate_imputations(df, imputed_df_dict, dates_missing):
    """
    Evaluate the imputation of missing values in a dataframe for a dictionary of 
    imputation methods.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    df_missing : pd.DataFrame
        Dataframe with missing values.
    imputed_df : pd.DataFrame
        Dataframe with imputed values.
    dates_missing : pd.DatetimeIndex
        Dates of the missing values.

    Returns
    -------
    mse : float
        Mean squared error of the imputation.
    """
    rmse_dict = imputed_df_dict.copy()
    for key,val in rmse_dict.items():
        rmse_dict[key]=np.sqrt(metrics.mean_squared_error(df.loc[dates_missing], val.loc[dates_missing]))
    rmse_df = pd.DataFrame(list(rmse_dict.items()), columns=['Method', 'RMSE'])
    rmse_df.set_index('Method', inplace=True)
    rmse_df.sort_values(by='RMSE', ascending=True, inplace=True)
    return rmse_df

def test_imputations(df, cols:list, sample_dates:list, methods:list):
    """
    Test the imputation methods on a dataframe with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to test.
    cols : list
        List of columns to test.
    dates : list
        List of dates to test.

    Returns
    -------
    None
    """
    from utils import get_sample_with_missing_values

    results={}
    for col in cols:
        samples=[]
        for sample_date_start, sample_date_end in sample_dates:
            sample, sample_missing, dates_missing = get_sample_with_missing_values(df, sample_date_start, sample_date_end, col)
            sample_imputed_dict = {}
            for method, kwargs in methods:
                imputer = Imputer(method, **kwargs)
                sample_imputed_dict[method+'_'+str(list(kwargs.values()))] = imputer.impute(sample_missing, col)  
            
            eval_df=evaluate_imputations(sample, sample_imputed_dict, dates_missing)
            samples.append(eval_df)

        sample_means_rmse=pd.concat(samples).groupby(level=0).mean()
        sample_means_rmse=sample_means_rmse.sort_values(by='RMSE', ascending=True)
        results[col]=sample_means_rmse
        
    return results

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