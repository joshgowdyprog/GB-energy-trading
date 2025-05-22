import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

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

def plot_model_validation(cv_folds, model):
    train_accuracies=[]
    test_accuracies=[]
    train_auc=[]
    test_auc=[]
    for i,fold in cv_folds.items():
        (X_train, y_train, X_test, y_test) = fold
        model.fit(X_train, y_train['auction_spread_dir'])
        test_pred = model.predict(X_test)
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        train_pred = model.predict(X_train)
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        
        train_accuracies.append(accuracy_score(y_train['auction_spread_dir'], train_pred))
        test_accuracies.append(accuracy_score(y_test['auction_spread_dir'], test_pred))
        train_auc.append(roc_auc_score(y_train['auction_spread_dir'], train_pred_proba))
        test_auc.append(roc_auc_score(y_test['auction_spread_dir'], test_pred_proba))

    plt.figure(figsize=(20, 5))
    plt.grid()
    plt.plot(train_accuracies, label='train accuracy')
    plt.plot(test_accuracies, label='test accuracy')
    plt.plot(train_auc, label='train auc')
    plt.plot(test_auc, label='test auc')
    plt.legend(loc='best')
    plt.title('Train and Test Accuracy and AUC for Logistic Regression')
    plt.show()

