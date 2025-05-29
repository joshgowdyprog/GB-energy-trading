import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from code_utils.preprocessing import load_auction_data, load_forecasts, preprocess_data
from code_utils.feature_engineering import (
    generate_lag_timeseries,
    generate_close_timeseries,
    make_new_price_indicators
)
from code_utils.training import (
    timeseries_train_test_split,
    plot_model_validation,
    plot_roc_curve,
)
from code_utils.ensemble import WeightedEnsembleClassifier


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_model(model_config):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type == "xgboost":
        return XGBClassifier(**params)
    elif model_type == "logistic_regression":
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def save_model(model, config, params_dict, dataset_dict):

    # save accuracies and weight and threshold to csv
    results_df = pd.DataFrame({
        "model_1": [config["model_1"]["type"]],
        "model_2": [config["model_2"]["type"]],
        "optimum_weight_model_1": [params_dict["best_weight"]],
        "optimum_prediction_threshold": [params_dict["best_thresh"]],
        "validation_start_date": [dataset_dict["y_valid"].index[0].strftime("%Y:%m:%d")],
        "validation_accuracy": [params_dict["valid_accuracy"]],
        "holdout_start_date": [dataset_dict["y_holdout"].index[0].strftime("%Y:%m:%d")],
        "holdout_accuracy": [params_dict["unseen_acc"]]
    })
    results_path = config["model"]["results_path"]
    print(f"Saving results to: {results_path}")
    results_df.to_csv(results_path, index=False)

    # save model
    model_path= config["model"]["model_path"]
    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)

    # save training, validation, and holdout data for futher testing
    print(f"Saving training, validation, and holdout data to CSV files in folder: {config['data']['processed_data']}")
    dataset_dict["X_train"].to_csv(config["data"]["processed_data"]+'X_train.csv')
    dataset_dict["y_train"].to_csv(config["data"]["processed_data"]+'y_train.csv')
    dataset_dict["X_valid"].to_csv(config["data"]["processed_data"]+'X_valid.csv')
    dataset_dict["y_valid"].to_csv(config["data"]["processed_data"]+'y_valid.csv')
    dataset_dict["X_holdout"].to_csv(config["data"]["processed_data"]+'X_holdout.csv')
    dataset_dict["y_holdout"].to_csv(config["data"]["processed_data"]+'y_holdout.csv')

    # modify the trades_config.yaml file
    with open('code/trades_config.yaml', 'r') as f:
        trades_config = yaml.safe_load(f)
    trades_config['model']['model_path'] = model_path
    trades_config['model']['results_path'] = results_path
    trades_config['data']['processed_data'] = config["data"]["processed_data"]
    with open('code/trades_config.yaml', 'w') as f:
        yaml.safe_dump(trades_config, f)

def train_model(config):
    # load and preprocess auction data
    auction_data, _ = load_auction_data(config["data"]["auction_files"])
    auction_data = preprocess_data(auction_data)

    auction_data["auction_price_spread"] = (
        auction_data["price_second_auction"] - auction_data["price_first_auction"]
    )
    auction_data["system_price_spread"] = (
        auction_data["system_price"] - auction_data["price_first_auction"]
    )
    auction_data["auction_spread_dir"] = (auction_data["auction_price_spread"] > 0).astype(int)
    auction_data["system_spread_dir"] = (auction_data["system_price_spread"] > 0).astype(int)
    
    # load and preprocess forecasted energy fundamentals data
    fundamentals = load_forecasts(config["data"]["forecast_file"])
    fundamentals = preprocess_data(fundamentals)
    fundamentals = fundamentals.drop(columns=['availability', 'within_day_availability', 'long_term_wind_over_demand', 'long_term_wind_over_margin',
       'long_term_solar_over_demand', 'long_term_solar_over_margin','margin_over_demand'])

    # generate lagged and close timeseries
    for lag in config["features"]["lags"]:
        auction_data = generate_lag_timeseries(auction_data, auction_data.columns, lag)
    auction_data = generate_close_timeseries(auction_data, auction_data.columns)

    # make indicators from auction data
    if config["features"]["use_indicators"]:
        auction_data = make_new_price_indicators(
            auction_data,
            config["features"]["timeseries_columns"],
            config["features"]["indicators"]
        )

    # combine and rescale features
    data=pd.concat([auction_data, fundamentals], axis=1)

    # use the second half of the available data (most relebvant for the current/future state of the market)
    data = data.iloc[int(data.shape[0]/2):]

    data=data.dropna(axis=0, how='any')
    y=data[config["target"]]

    # drop all future (i.e. not lagged) columns from features
    X=data.drop(columns=['auction_spread_dir',
                        'system_spread_dir', 
                        'price_second_auction', 
                        'system_price', 
                        'traded_volume_second_auction', 
                        'auction_price_spread',
                        'system_price_spread'])

    # scale data with a standard scaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled = scaler.fit_transform(X_scaled)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Train/test split
    X_holdout = X_scaled.iloc[int(config["train"]["holdout_frac"]*X_scaled.shape[0]):]
    y_holdout = y.iloc[int(config["train"]["holdout_frac"]*X_scaled.shape[0]):]
    print(f"Holdout dates: {y_holdout.index[0]} to {y_holdout.index[-1]}")
    cv_folds = timeseries_train_test_split(
        X_scaled.iloc[:int(config["train"]["holdout_frac"]*X_scaled.shape[0])],
        y.iloc[:int(config["train"]["holdout_frac"]*X_scaled.shape[0])],
        n_splits=config["train"]["n_splits"],
    )
    X_train, y_train, X_valid, y_valid = cv_folds[-1]
    print(f"Training dates: {y_train.index[0]} to {y_train.index[-1]}")
    print(f"Validation dates: {y_valid.index[0]} to {y_valid.index[-1]}")


    # Model selection
    model_1 = get_model(config["model_1"])
    model_1.fit(X_train, y_train)
    model_2 = get_model(config["model_2"])
    model_2.fit(X_train, y_train)

    # ensemble vaidation and evaluation
    best_acc = -np.inf
    best_weight = None
    best_thresh = None

    model_weights = np.linspace(0, 1, 20)
    thresholds = np.linspace(0, 1, 200)

    for weight in model_weights:
        ensemble = WeightedEnsembleClassifier(models=[model_1, model_2], weights=[weight, 1 - weight])
        y_pred_proba = ensemble.predict_proba(X_valid)[:, 1]
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            acc = accuracy_score(y_valid, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_weight = weight
                best_thresh = thresh

    print(f"Best weight: {best_weight}, Best threshold: {best_thresh}.")

    # Refit ensemble with best weight and evaluate
    ensemble = WeightedEnsembleClassifier(models=[model_1, model_2], weights=[best_weight, 1 - best_weight])
    y_pred_proba = ensemble.predict_proba(X_valid)[:, 1]
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    acc = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy (best): {acc:.4f}")

    unseen_pred_proba = ensemble.predict_proba(X_holdout)[:, 1]
    unseen_y_pred = (unseen_pred_proba >= best_thresh).astype(int)
    unseen_acc = accuracy_score(y_holdout, unseen_y_pred)
    print(f"Holdout Accuracy: {unseen_acc:.4f}")

    # save model and results
    params_dict = {
        "best_weight": best_weight,
        "best_thresh": best_thresh,
        "valid_accuracy": acc,
        "unseen_acc": unseen_acc
    }
    dataset_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_holdout": X_holdout,
        "y_holdout": y_holdout
    }
    save_model(ensemble, config, params_dict, dataset_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier to forecast price movements")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_model(config)
