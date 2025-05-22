import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_lag_timeseries(data, targets, lag=24):
    """
    Generate a lag time series for each column in the DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    lag : int
        The number of periods to use for the lag of the time series.
    Returns
    -------
    pd.DataFrame
        The input data with the lagged time series added.
    """
    for col in targets:
        data[col+f'_lag{lag}']=data[col].shift(lag)

    return data

def generate_close_timeseries(data, targets):
    """
    Generate a most recent price at days close for each column in the DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    Returns
    -------
    pd.DataFrame
        The input data with the close time series added.
    """
    for col in targets:
        for row in data[col].index:
            data.loc[row, col+'_close']=(data[col].shift(row.hour+1)).loc[row]
    return data

def pre_engineer_timeseries(data):
    """
    Pre-engineer the time series data by creating new features based on the auction prices and spreads.
    Generate lagged and close time series for the specified targets.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    Returns
    -------
    data : pd.DataFrame
        The input data with the new features added.
    """
    data.loc[:,'auction_price_spread']=data['price_second_auction']-data['price_first_auction']
    data.loc[:,'system_price_spread']=data['system_price']-data['price_first_auction']

    targets=['price_first_auction', 'price_second_auction',  'system_price', 'auction_price_spread', 'system_price_spread']
    df=data[targets].copy()
    df.columns=['price1', 'price2', 'price3', 'spread1', 'spread2']
    df['log_price1'] = np.log(df['price1']+100)
    df['log_price2'] = np.log(df['price2']+100)
    df['log_price3'] = np.log(df['price3']+100)
    targets=['price1', 'price2', 'price3','log_price1','log_price2', 'log_price3', 'spread1', 'spread2']

    # standardize the data
    coeffs={'col':[], 'mean':[], 'std':[]}
    for col in targets:
        coeffs['col'].append(col)
        mean = df[col].mean()
        coeffs['mean'].append(mean)
        std = df[col].std()
        coeffs['std'].append(std)
        df[col] = (df[col] - mean) / std
    
    coeffs = pd.DataFrame(coeffs)
    coeffs=coeffs.set_index('col')

    df=generate_lag_timeseries(df, targets, 24)
    df=generate_lag_timeseries(df, targets, 72)
    df=generate_lag_timeseries(df, targets, 24*7)
    df=generate_close_timeseries(df, targets)

    return df, coeffs

def calculate_PCMA(data, feature, window=24):
    """
    Calculate the Proportional Change Moving Average for a given feature and window.
    """
    pc= (data[feature]-data[feature].shift(24))/data[feature].shift(24)
    pcma= pc.rolling(window).mean()
    return pcma

def calculate_ATR(data, feature, window=24, lag=24):
    """
    Calculate the Average True Range (ATR) for a given feature in the data and window.

    The ATR is a measure of volatility and is calculated as the average of the true range over a specified window.
    The True Range is defined as the maximum of the following three values:
    A) The difference between the current high and low prices within window.
    B) The difference between the current high and a previous price i.e. lagged current price.
    C) The difference between the current low and a previous price i.e. lagged current price.

    When lag<window then option A) is always the maximum.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate ATR for.
    window : int
        The number of periods to use for the ATR calculation.

    Returns
    -------
    pd.Series
        The ATR values for the specified feature.
    """
    high = data[feature].rolling(window).max()
    low = data[feature].rolling(window).min()
    lagged = data[feature].shift(lag)
    
    tr = pd.concat([high - low, abs(high - lagged), abs(low - lagged)], axis=1).max(axis=1)
    
    atr = tr.rolling(window).mean()
    
    return atr

def calculate_MAD(data, feature, window=24):
    """
    Calculate the Moving Average Deviation (MAD) for a given feature in the data.
    This measures the deviation of the current price from the moving average of the price. 

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate MAD for.
    window : int
        The number of periods to use for the MAD calculation.

    Returns
    -------
    pd.Series
        The MAD values for the specified feature.
    """
    pma= data[feature].rolling(window).mean()
    mad = (data[feature]-pma)/pma
    
    return mad

def calculate_ADX(data, feature, window=24):
    """
    Calculate the Average Directional Index (ADX) for a given feature in the data and window size.
    The ADX is a measure of trend strength and is calculated using the True Range and Directional Movement.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate ADX for.
    window : int
        The number of periods to use for the ADX calculation.

    Returns
    -------
    pd.Series
        The ADX values for the specified feature.
    """
    high = data[feature].rolling(window).max()
    low = data[feature].rolling(window).min()
    
    tr = high - low

    atr= tr.rolling(window).mean()
    
    pc = data[feature] - data[feature].shift(window)
    
    pc_plus = pc.copy()
    pc_plus[pc_plus < 0] = 0
    pc_minus = pc.copy()
    pc_minus[pc_minus > 0] = 0
    pc_plus = pc_plus.rolling(window).mean()
    pc_minus = pc_minus.rolling(window).mean()

    DX_plus = (pc_plus / atr)
    DX_minus = (pc_minus / atr)

    adx= abs(DX_plus - DX_minus)/(abs(DX_plus+DX_minus))

    return adx

def calculate_PR(data, feature, window=24):
    """
    Calculate the Proportional Range (PR) for a given feature in the data and window size.
    The PR is a measure as a proportion of the price within its range over the specified window.    

    """
    high = data[feature].rolling(window).max()
    low = data[feature].rolling(window).min()
    
    pr = (high-data[feature]) / (high - low)
    
    return pr

def calculate_RSI(data, feature, window=24):
    """
    Calculate the Relative Strength Index (RSI) for a given feature in the data and window size.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate RSI for.
    window : int
        The number of periods to use for the RSI calculation.

    Returns
    -------
    pd.Series
        The RSI values for the specified feature.
    """
    pc= data[feature]-data[feature].shift(window)
    gain = pc.copy()
    gain[gain < 0] = 0
    loss = pc.copy()
    loss[loss > 0] = 0
    gain_ma = gain.rolling(window).mean()
    loss_ma = loss.rolling(window).mean()
    loss_ma[loss_ma==0]=1e-10
    ds=1-gain_ma/loss_ma

    rsi = 1 - (1 / ds)

    return rsi

def calculate_MACD(data, feature, window_smaller=12, window_larger=24):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given feature in the data and window sizes.
    IS the diffeence in the moving averages of the feature between the specified window sizes.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate MACD for.
    window_smaller : int
        The number of periods to use for the smaller_window.
    window_larger : int
        The number of periods to use for the larger_window.

    Returns
    -------
    pd.Series
        The MACD values for the specified feature.
    """
    pma_smaller= data[feature].rolling(window_smaller).mean()
    pma_larger= data[feature].rolling(window_larger).mean()
    macd = pma_smaller - pma_larger
    
    return macd

def calculate_MOM(data, feature, lag=1):
    """
    Calculate the Momentum (MOM) for a given feature in the data and lag.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate PMOM for.
    lag : int
        The number of periods to use for the lag of the PMOM calculation.

    Returns
    -------
    pd.Series
        The PMOM values for the specified feature.
    """
    mom = data[feature] - data[feature].shift(lag)
    
    return mom

def calculate_RSTDEV(data, feature, window=24):
    """
    Calculate the Rolling Standard Deviation (RSTDEV) for a given feature in the data and window size.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate RSTDEV for.
    window : int
        The number of periods to use for the RSTDEV calculation.

    Returns
    -------
    pd.Series
        The RSTDEV values for the specified feature.
    """
    rstdev = data[feature].rolling(window).std()
    
    return rstdev

def calculate_ZSCORE(data, feature, window=24):
    """
    Calculate the Z-Score for a given feature in the data and window size.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    feature : str
        The name of the feature to calculate ZSCORE for.
    window : int
        The number of periods to use for the ZSCORE calculation.

    Returns
    -------
    pd.Series
        The ZSCORE values for the specified feature.
    """
    mean = data[feature].rolling(window).mean()
    std = data[feature].rolling(window).std()
    
    zscore = (data[feature] - mean) / std
    
    return zscore

def make_new_price_indicators(data, features, indicator_functions):
    """
    Create new indicators based on the lagged timeseries using a flexible approach.
    Accepts a dictionary of indicator functions and their respective keyword arguments.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing the features.
    features : list
        The list of features to create indicators for.
    indicator_functions : dict
        A dictionary where keys are indicator names and values are tuples of the form 
        (function, kwargs) where `function` is the indicator function and `kwargs` is a 
        dictionary of keyword arguments for the function.

    Returns
    -------
    data : pd.DataFrame
        The input data with the new indicators added.
    """
    indicators = {}
    for feature in features:
        for indicator_name, func, kwargs in indicator_functions:
            indicators[f'{feature}_{indicator_name}'] = eval(func)(data, feature, **kwargs)

    new_data = pd.concat([data, pd.DataFrame(indicators, index=data.index)], axis=1)
    return new_data

def transform_data_to_PCA(X, n_components):
    """
    Transform the data to PCA space using the specified number of components.
    This function standardizes the data before applying PCA.
    
    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with features to apply PCA on.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    X_pca : pd.DataFrame
        Transformed data in PCA space.
    """
    X_std=StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    
    return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index)
