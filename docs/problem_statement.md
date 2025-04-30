# Problem Statement

## Build a price and spread forecaster.

1. Build a ML learning model forecaster of the first auction price. We need this price in order to determine at what level we enter the market regardless of whether we buy or sell.

2. Build a separate ML learning model forecaster of the auction price spread between the first and second aucitons, auction_price_spread =  price_second_auction - price_first_auction AND the system price spread between the system price in the balancing mechanism and the first auction, system_price_spread = system_price - first_auction_price.

## Build and Test Strategies

1. use these forecasts to predict profit from a given strategy.
2. build a cuistom naive strategy as baseline
3. optimise a strategy to maximise profit
4. model risk? Minimise risk?

