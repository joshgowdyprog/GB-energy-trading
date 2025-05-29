import argparse
import yaml
import pandas as pd
import numpy as np
import joblib

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def make_trades(config):

    # load model
    model = joblib.load(config["model"]["path"])
    
    # load training, vallidation and holdout data
    X_train = pd.read_csv(config["data"]["processed_data"]+'X_train.csv', index_col=0)
    y_train = pd.read_csv(config["data"]["processed_data"]+'y_train.csv', index_col=0)
    X_valid = pd.read_csv(config["data"]["processed_data"]+'X_valid.csv', index_col=0)
    y_valid = pd.read_csv(config["data"]["processed_data"]+'y_valid.csv', index_col=0)
    X_holdout = pd.read_csv(config["data"]["processed_data"]+'X_holdout.csv', index_col=0)
    y_holdout = pd.read_csv(config["data"]["processed_data"]+'y_holdout.csv', index_col=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make trades according to strategy")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    make_trades(config)
