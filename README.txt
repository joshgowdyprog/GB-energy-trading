Trading on GB day-ahead energy markets
==============================

This repo contains code and documentation for a ML model which forecasts the movements of spreads between daily morning and afternoon GB energy auctions and runs a custom trading strategy on these spreads. The ML model uses 2021-2022 data for various energy fundamentals and custom technical indicators as features and uses Logistic and XGBoost classifiers with careful hyper-parameter tuning to make predictions. The strategy uses these predicitons and makes trades achieving very good performance on unseen historical data. Annualised ROI >16%, Sharpe ratio of 8. Would merit further back-testing and improvements.

