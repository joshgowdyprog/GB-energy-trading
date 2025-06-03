import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

class TradingStrategy:
    def __init__(self, config_dict, auction_data, y_proba=None, start_date=None):

        self.config = config_dict

        self.fee = self.config.get('strategy').get('fee', 5)
        self.sizing_exponent = self.config.get('strategy').get('sizing_exponent', 2)
        self.max_vol = self.config.get('strategy').get('max_vol', 10)
        self.threshold_delta = self.config.get('strategy').get('threshold_delta', 0.15)

        self.y_proba = y_proba
        self.start_date = str(start_date)
        if self.start_date is None:
            self.order_book = auction_data.copy()
            if y_proba is not None:
                self.order_book["y_proba"] = y_proba
        else:
            self.order_book = auction_data.loc[self.start_date:].copy()
            if y_proba is not None:
                start_idx = auction_data.index.get_loc(self.start_date)
                self.order_book["y_proba"] = y_proba[start_idx:]

    def define_baseline_signal(self):
        self.order_book['auction_spread_dir_lag24'] = self.order_book['auction_spread_dir'].shift(24)
        self.signal = np.where(self.order_book['auction_spread_dir_lag24'] == 1, 'long',
                       np.where(self.order_book['auction_spread_dir_lag24'] == 0, 'short', 'hold'))

    def define_main_signal(self, best_threshold):
        upper_threshold = best_threshold + self.threshold_delta
        lower_threshold = best_threshold - self.threshold_delta
        self.signal = np.where(self.order_book["y_proba"] > upper_threshold, 'long',
                       np.where(self.order_book["y_proba"] < lower_threshold, 'short', 'hold'))

    def add_signal(self):
        self.order_book['trade_signal'] = self.signal

    def make_trade(self):
        self.order_book['trade'] = self.order_book['trade_signal']

    def calculate_profit(self, use_sizing=False):
        if not use_sizing:
            self.order_book['position_size'] = self.max_vol
        else:
            self.order_book['position_size'] = np.where(
                self.order_book['trade'] == 'long',
                (self.order_book["y_proba"] ** self.sizing_exponent) * self.max_vol,
                np.where(
                    self.order_book['trade'] == 'short',
                    ((1 - self.order_book["y_proba"]) ** self.sizing_exponent) * self.max_vol,
                    0
                )
            )
        self.order_book['profit'] = np.where(
                self.order_book['trade'] == 'long',
                (self.order_book['price_second_auction'] - self.order_book['price_first_auction']) * self.order_book['position_size'] - self.fee,
                np.where(
                    self.order_book['trade'] == 'short',
                    (self.order_book['price_first_auction'] - self.order_book['price_second_auction']) * self.order_book['position_size'] - self.fee,
                    0
                )
            )
        #calulate % returns per period
        self.order_book['profit_pct'] = 100*self.order_book['profit'] / (self.order_book['price_first_auction'] * self.order_book['position_size'])
        
        # calculate cumulative profit
        self.order_book['cumulative_profit'] = self.order_book['profit'].cumsum()

    def plot_returns(self, title='Cumulative Profit from Trades'):
        self.order_book['cumulative_profit'].plot(
            figsize=(12, 6), title=title, grid=True
        )
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit (GBP)')
        # save the plot
        plt.savefig(self.config.get('strategy').get('strategy_folder', 'strategies/')+'cumulative_profit.png')

    def plot_distribution_of_returns(self, title='Histogram of Profits from Trades'):
        plt.figure(figsize=(12, 6))
        self.order_book[self.order_book['profit']!=0]['profit'].hist(bins=100)
        plt.grid(True)
        plt.title(title)
        plt.xlabel('Profit per Trade')
        plt.ylabel('Frequency')
        # save the plot
        plt.savefig(self.config.get('strategy').get('strategy_folder', 'strategies/')+'hist_profit.png')

    def compute_roi(self):
        if 'position_size' in self.order_book.columns:
            invested_cash = (self.order_book['position_size'] * self.order_book['price_first_auction'])[self.order_book['trade'] != 'hold'].sum()
        else:
            invested_cash = (self.max_vol * self.order_book['price_first_auction'])[self.order_book['trade'] != 'hold'].sum()
        total_profit = self.order_book['cumulative_profit'].iloc[-1]
        roi = 100 * total_profit / invested_cash if invested_cash != 0 else float('nan')
        time_held = self.order_book.shape[0] / (24 * 365)
        annualised_roi = ((1 + roi / 100) ** (1 / time_held) - 1) * 100 if time_held > 0 else float('nan')

        print(f"Invested Cash: {invested_cash:.2f} GBP")
        print(f"Total Profit: {total_profit:.2f} GBP")
        print(f"Return on Investment (ROI): {roi:.4f} %")
        print(f"Time Held in years: {time_held}")
        print(f"Annualised ROI: {annualised_roi:.4f} %")
        return {
            "invested_cash": invested_cash,
            "total_profit": total_profit,
            "roi": roi,
            "time_held": time_held,
            "annualised_roi": annualised_roi
        }
    
    def compute_sharpe_ratio(self):

        #find daily returns by resampling the hourly profit
        profit_df = self.order_book['profit_pct'].to_frame()
        if not isinstance(profit_df.index, pd.DatetimeIndex):
            profit_df.index = pd.to_datetime(profit_df.index)
        daily_returns = profit_df.resample('D').sum().dropna()

        mean_return = daily_returns.mean()['profit_pct']
        std_return = daily_returns.std()['profit_pct']
    
        annualised_sharpe_ratio = mean_return / std_return * np.sqrt(365)

        print(f"Mean Daily Return: {mean_return:.4f} %")
        print(f"Standard Deviation of Daily Returns: {std_return:.4f} %")
        print(f"Sharpe Ratio: {annualised_sharpe_ratio:.4f}")
        return {'mean_return': mean_return,
                'std_return': std_return,
                'annualised_sharpe_ratio': annualised_sharpe_ratio}


    def perform_strategy(self, strategy='baseline', best_threshold=None):
        if strategy == 'baseline':
            self.define_baseline_signal()
            self.add_signal()
            self.make_trade()
            self.calculate_profit(use_sizing=False)
        elif strategy == 'main':
            if best_threshold is None:
                raise ValueError("best_threshold must be provided for main strategy.")
            self.define_main_signal(best_threshold)
            self.add_signal()
            self.make_trade()
            self.calculate_profit(use_sizing=True)
        else:
            raise ValueError("Unknown strategy type. Use 'baseline' or 'main'.")
        return self.order_book