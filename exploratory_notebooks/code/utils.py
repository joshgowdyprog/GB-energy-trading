import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def calculate_VIF(features):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the dataframe.
    This is used to detect multicollinearity in regression models.

    Parameters
    ----------
    features : pd.DataFrame
        Dataframe with features to calculate VIF for.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with features and their corresponding VIF values.
    """
    vif = pd.DataFrame()
    vif["features"] = features.columns[1:]
    vif["vif_Factor"] = [variance_inflation_factor(np.array(features[features.columns[1:]].values, dtype=float), i) 
                     for i in range(features.shape[1]-1)]

    return vif[vif.vif_Factor>10].sort_values(by='vif_Factor',ascending=False)

def calculate_explained_variance(X):
    """
    Calculate the portion of explained variance
    by each principal component in the PCA. Also outputs a dataframe with the
    principal components and their corresponding features.
    This function standardizes the data before applying PCA.
    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with features to apply PCA on.
    Returns
    -------
    explained_variance : np.ndarray
        Array with the explained variance ratio for each principal component.
    pc_df : pd.DataFrame
        Dataframe with the principal components and their corresponding features.
    """
    X_std=StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X_std)
    explained_variance = pca.explained_variance_ratio_
    pc_df = pd.DataFrame(pca.components_,
                        columns=X.columns,
                        index=[f"PC{i+1}" for i in range(len(X.columns))])

    return explained_variance, pc_df

def find_low_variance_features(principal_components, podium=10, threshold=0.1):
    """
    Find low variance features in the principal components.
    This function checks the first 'podium' number of principal components and
    returns the features that have a variance lower than 'threshold' in all of them.

    Parameters
    ----------
    principal_components : pd.DataFrame
        Dataframe with the principal components.
    podium : int
        Number of principal components to check.
    threshold : float
        Threshold for low variance. Features with variance lower than this value
        in all 'podium' principal components will be returned.
    Returns
    -------
    low_variance_cols : list
        List of features with low variance in the specified principal components.
    """
    low_variance_cols = []
    top_components=principal_components.iloc[0:podium]
    for col in top_components.columns:
        if (top_components[col]<threshold).all():
            low_variance_cols.append(col)
    return low_variance_cols