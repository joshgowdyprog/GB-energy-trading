import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

from code_utils.trading import TradingStrategy

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def make_trades(config):

    # load model
    model = joblib.load(config['model']['model_path'])
    results = pd.read_csv(config['model']['results_path'])
        
    # load training, vallidation and holdout data
    X_train = pd.read_csv(config['data']['processed_data']+'X_train.csv', index_col=0)
    y_train = pd.read_csv(config['data']['processed_data']+'y_train.csv', index_col=0)
    X_valid = pd.read_csv(config['data']['processed_data']+'X_valid.csv', index_col=0)
    y_valid = pd.read_csv(config['data']['processed_data']+'y_valid.csv', index_col=0)
    X_holdout = pd.read_csv(config['data']['processed_data']+'X_holdout.csv', index_col=0)
    y_holdout = pd.read_csv(config['data']['processed_data']+'y_holdout.csv', index_col=0)

    # load auction data
    auction_data = pd.read_csv(config['data']['processed_data']+'auction_data.csv', index_col=0)
    auction_data=auction_data[['price_first_auction', 'price_second_auction', 'auction_price_spread',
        'auction_spread_dir']]
    
    # check accuracy on training, validation and holdout data
    # use the best threshold from the results
    best_threshold = results.loc[0,'optimum_prediction_threshold']
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    y_valid_pred = (y_valid_proba >= best_threshold).astype(int)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)  
    y_holdout_proba = model.predict_proba(X_holdout)[:, 1]
    y_holdout_pred = (y_holdout_proba >= best_threshold).astype(int)
    holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {valid_accuracy:.4f}")
    print(f"Holdout accuracy: {holdout_accuracy:.4f}")

    # combine all sets
    X=pd.concat([X_train, X_valid, X_holdout], axis=0)
    y=pd.concat([y_train, y_valid, y_holdout], axis=0)
    y_proba=model.predict_proba(X)[:, 1]
    y_pred = (y_proba>=best_threshold).astype(int)
    overall_accuracy=accuracy_score(y, y_pred)
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    if config.get('strategy').get('strategy_start_date', None) is not None:
        strategy_start_date = config['strategy']['strategy_start_date']
    else:
        strategy_start_date = None

    print(f"Using strategy start date: {strategy_start_date}")

    strat = TradingStrategy(
        config, auction_data, y_proba, start_date=strategy_start_date, 
    )
    order_book = strat.perform_strategy(strategy=config['strategy']['name'], best_threshold=best_threshold)
    strat.plot_returns(title='Cumulative Profit from Trades')
    strat.plot_distribution_of_returns()
    roi_dict=strat.compute_roi()
    sharpe_dict=strat.compute_sharpe_ratio()

    # save all results
    order_book.to_csv(config['data']['processed_data']+'order_book.csv')
    strat_results = pd.DataFrame(roi_dict, index=[0])
    strat_results = pd.concat([strat_results, pd.DataFrame(sharpe_dict, index=[0])], axis=1)
    strat_results.to_csv(config.get('strategy').get('strategy_folder', 'strategies/')+'strategy_results.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make trades according to strategy")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    make_trades(config)
