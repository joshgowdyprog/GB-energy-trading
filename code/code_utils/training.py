import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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
    cv_folds=[]
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        # use train_index and test_index to split data
        cv_folds.append([X.iloc[train_index], y.iloc[train_index], X.iloc[test_index],
                                    y.iloc[test_index]])
    return cv_folds

def plot_model_validation(cv_folds, model):
    """
    Plot the model validation results for each fold.
    This function takes the cross-validation folds and the model
    used for training. It trains the model on each fold and
    calculates the accuracy and AUC for both training and testing sets.
    It then plots the results for each fold.
    """
    train_accuracies=[]
    test_accuracies=[]
    train_auc=[]
    test_auc=[]
    for fold in cv_folds:
        (X_train, y_train, X_test, y_test) = fold
        model.fit(X_train, y_train.values.ravel())
        test_pred = model.predict(X_test)
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        train_pred = model.predict(X_train)
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        
        train_accuracies.append(accuracy_score(y_train, train_pred))
        test_accuracies.append(accuracy_score(y_test, test_pred))
        train_auc.append(roc_auc_score(y_train, train_pred_proba))
        test_auc.append(roc_auc_score(y_test, test_pred_proba))

    plt.figure(figsize=(20, 5))
    plt.grid()
    plt.plot(train_accuracies, label='train accuracy')
    plt.plot(test_accuracies, label='test accuracy')
    plt.plot(train_auc, label='train auc')
    plt.plot(test_auc, label='test auc')
    plt.legend(loc='best')
    plt.title('Train and Test Accuracy and AUC')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba):
    """
    Plot the ROC curve for the model predictions.
    This function takes the true labels and the predicted probabilities
    and plots the ROC curve. It also calculates the AUC score
    and returns it along with the false positive rate, true positive rate,
    and thresholds used to calculate the ROC curve.
    """
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
    return roc_auc, fpr, tpr, thresholds_roc

def objective(cv_fold, score, trial):
    """
    Objective function for Optuna optimization.
    This function defines the hyperparameters to be optimized
    and the model to be used. It also defines the scoring metric
    to be used for optimization. The function returns the score
    for the given hyperparameters.
    Parameters
    ----------
    target : str
        Target variable to be predicted.
    cv_folds : dict
        Dictionary with the training and testing sets for each fold.
        Each key is the fold number and the value is a tuple with
        (X_train, y_train, X_test, y_test).
    score : str
        Scoring metric to be used for optimization.
        Can be 'roc_auc' or 'accuracy'.
    trial : optuna.trial.Trial
        Optuna trial object to be used for optimization.
    Returns
    -------
    -------
    score : float
        Score for the given hyperparameters.
    """
    if score not in ['roc_auc', 'accuracy']:
        raise ValueError("score must be 'roc_auc' or 'accuracy'")
    
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.01, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 4),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "subsample": trial.suggest_float("subsample", 0.5, 0.7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.7),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.7),
        "reg_lambda": trial.suggest_float("reg_lambda", 2, 5),
        "eval_metric": "logloss"
    }
    
    (X_train, y_train, X_test, y_test) = cv_fold
    model = XGBClassifier(**params)
    model.fit(X_train, y_train.values.ravel())
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    if score == 'roc_auc':
        roc_auc = auc(fpr, tpr)
        return roc_auc
    elif score == 'accuracy':
        j_scores = tpr - fpr
        best_thresh = thresholds[j_scores.argmax()]  
        y_pred = (y_pred_proba > best_thresh).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    

