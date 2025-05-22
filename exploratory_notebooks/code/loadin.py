import pandas as pd


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