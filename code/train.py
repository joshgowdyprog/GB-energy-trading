from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

def timeseries_train_test_split(X, y, n_splits=5):
    """
    Split the data into training and testing sets for time series data.
    This function uses the TimeSeriesSplit from sklearn to create
    a time series cross-validation object. It splits the data into
    nested sequential training and testing sets, ensuring that the
    training set always precedes the testing set in time.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with features to split.
    y : pd.Series
        Series with target variable to split.
    n_splits : int
        Number of splits for the time series cross-validation.
        Default is 5.
    Returns
    -------
    -------
    cv_folds : dict
        Dictionary with the training and testing sets for each fold.
        Each key is the fold number and the value is a tuple with
        (X_train, X_test, y_train, y_test).
    """
    tscv = TimeSeriesSplit(n_splits)
    cv_folds={}
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        # use train_index and test_index to split your data
        cv_folds[i] = (X.iloc[train_index], y.iloc[train_index], X.iloc[test_index],
                                    y.iloc[test_index])
    return cv_folds

def evaluate_model(y_pred, y_test):
    # use RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    # use R^2
    r2 = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    # use adjusted R^2
    n = len(y_test)
    p = y_pred.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    # use MAE
    mae = np.mean(np.abs(y_pred - y_test))
    # use MAPE
    mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

    return {
        'RMSE': rmse,
        'R^2': r2,
        'Adjusted R^2': r2_adj,
        'MAE': mae,
        'MAPE': mape
    }

def train_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, evaluate_model(y_pred, y_test)

def instantiate_model(model_name, kwargs):

    if model_name == 'LinearRegression':
        model = LinearRegression(**kwargs)
    elif model_name == 'Lasso':
        model = Lasso(**kwargs)
    elif model_name == 'MLPRegressor':
        model = MLPRegressor(**kwargs)
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(**kwargs)
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(**kwargs)
    return model

def compare_models(model_config, cv_folds):
    """
    Compare models using cross-validation
    """
    trained_models={}
    evals={}
    for model_name, kwargs in model_config:
        print(f"Model: {model_name}")
        model=instantiate_model(model_name, kwargs)
        eval_cv=[]
        model_cv=[]
        for fold, (X_train, y_train, X_test, y_test) in cv_folds.items():
            trained_model, eval_result=train_evaluate_model(model, X_train, y_train, X_test, y_test)
            model_cv.append(trained_model)
            eval_cv.append(eval_result)
            eval_results=pd.DataFrame(eval_cv)
        trained_models[model]=model_cv
        evals[model]=eval_results
    return trained_models, evals