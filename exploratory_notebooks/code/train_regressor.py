from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

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
    expanding training and testing sets, ensuring that the
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

def evaluate_model(y_pred, X_test, y_test):
    # use RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    # use R^2
    r2 = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    # use adjusted R^2
    n = len(y_test)
    p = X_test.shape[1]  # Number of predictors (features)
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
    
    # Use the last 20% of the training data as validation data 
    X_valid, y_valid = X_train[-int(len(X_train)*0.2):], y_train[-int(len(X_train)*0.2):]
    X_train, y_train = X_train[:-int(len(X_train)*0.2)], y_train[:-int(len(X_train)*0.2)]
    model.fit(X_train, y_train, 
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=50,
            eval_metric="rmse",
            verbose=True)
    y_pred = model.predict(X_test)
    y_valid = model.predict(X_train)
    return model, y_pred, y_valid

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_valid = model.predict(X_train)
    return model, evaluate_model(y_pred, X_test, y_test), evaluate_model(y_valid, X_train, y_train)

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

def compare_models(model_config, cv_folds, target):
    """
    Compare models using cross-validation
    """
    trained_models={}
    evals={}
    for model_name, kwargs in model_config:
        print(f"Model: {model_name}")
        eval_cv=[]
        valid_cv=[]
        model_cv=[]
        for fold, (X_train, y_train, X_test, y_test) in cv_folds.items():
            print(f"Validating Fold {fold+1} of {len(cv_folds)}:")
            model=instantiate_model(model_name, kwargs)
            trained_model, eval_result, valid_result=train_evaluate_model(model, X_train, y_train[target], X_test, y_test[target])
            model_cv.append(trained_model)
            eval_cv.append(eval_result)
            valid_cv.append(valid_result)
            eval_results=pd.DataFrame(eval_cv).merge(pd.DataFrame(valid_cv), left_index=True, right_index=True, suffixes=('_test', '_train'))
        trained_models[model_name]=model_cv
        evals[model_name]=eval_results
    return trained_models, evals


def make_XGB_objective(target, cv_folds, trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500)
    }

    model = model = xgb.XGBRegressor(**params)
    
    eval_cv=[]
    for (X_train, y_train, X_test, y_test) in cv_folds.values():
        _, eval_result, _ = train_evaluate_model(model, X_train, y_train[target], X_test, y_test[target])
        eval_cv.append(eval_result)
    return np.min([result['RMSE'] for result in eval_cv])