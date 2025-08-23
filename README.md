Forex Intraday Levels & Predictive Modeling
Overview

This project explores intraday price behavior in EUR/USD around the U.S. trading session.
The core focus is data engineering and feature extraction, building a clean dataset that captures daily Fair Value (FV) zones, L1 support/resistance bands, and volatility conditions, before applying Machine Learning for predictive tasks.

The workflow is split into two parts:

Data Engineering (majority of work)

Modeling (planned extension with ML)

Data Engineering

Input: 1-minute EUR/USD historical data (2021–2025).

Preprocessing:

Removed missing values, duplicates, and zero-volume rows.

Converted all timestamps into both UTC and NY session time (with DST handling).

Mapped each row to a U.S. trading day (date_us_open).

Feature Creation:

Session markers: NY open, warmup window, session close.

Fair Value (FV) zone: Derived from warmup or extended windows.

L1 bands: Support/resistance zones around FV (absolute or ATR-scaled).

Volatility flag: Pre-US volatility filter to skip noisy days.

Trading-ready dataset:

can_trade_now (post-warmup, low-vol days)

*_prev_active (levels carried forward for actual trading decisions).

Visualization:

TradingView-style daily bands.

Continuous 24h Close price + FV/L1 zones per U.S. session.

Export to Excel for manual inspection of random periods.

Modeling (planned)

The ML part will build on the engineered dataset.

Target idea: Predict whether price will touch FV/L1 levels after US open.

Methods: Baseline logistic regression → Neural Nets (RNN/CNN for time series).

Data split: ~600–800 trades (train/test).

Focus: Interpretability and risk-adjusted performance, not overfitting.

Contribution

This project emphasizes data quality and feature robustness first, then experiments with predictive modeling. The engineered dataset already provides value for backtesting trading strategies without ML.
