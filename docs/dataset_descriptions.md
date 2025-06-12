* *auction_data.csv* - This file contains the data for the two auctions in the GB day-ahead markets and contains actual clearing prices and traded volumes for these auctions. These auctions are both hourly, ie.e. the second auction has been coarse grained. A final column contains a reference forecast for the first auctions clearing price. Volumes in MW for the hour period, prices in GBP/MWh for the energy generated in that hour period.

* *forecast_features.csv* - These are forcasts of the various fundamentals which influence wholesale energy price i.e. the clearning price in the day-ahead auctions. Also includes some technical indicators including the previous days energy price.

* *system_prices.csv* - This dataset has the actual hourly system prices from the balancing mechanism as well as for reference the forecasted price range of these system prices. Prices in GBP/MWh for the energy generated in that hour period.

Datasets gathered from Kaggle.

In all the above datasets the source and modelling method for forecasts of the features (i.e. fundamentals such as demand and renewable generation forecasts) is not known and should be used carefully. Also the reference price forecasts are a result of an unknown model (what this repo replicates) and so should basically be ignored.

Similarly the exact sources of actual price and volume data which are the key inputs for model training are unknown although unlikely to be completely incorrect.

In summary, these datasets shuld be regarded as toy datasets!