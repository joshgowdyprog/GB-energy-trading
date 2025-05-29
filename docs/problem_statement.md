# Problem Statement

## Build a price and spread forecaster.

1. Build a ML learning model forecaster of the direction of spread between the first and second auction prices. We need this in order to determine whether we buy to sell or sell to buy. I.e. long versus short.

## Build and Test Strategies

1. use these forecasts to decide when to trade and if so whether to trade long or short
2. decision to trade can be based off thresholds in the model prediction probabilities
3. sizing can be based also on the model probabilities
4. build a cuistom naive strategy as baseline
5. optimise a strategy to maximise profit on test set
6. see how it performs on unseen data

