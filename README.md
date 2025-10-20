Quantitative Trading Machine Learning Models

This repository contains a series of hands-on projects exploring quantitative trading using machine learning. The goal was to build, test, and improve trading strategies with multiple stocks, using feature engineering, LightGBM models, meta-labeling, and enhanced backtesting methods.

🏗️ Project Overview

The project workflow covers the following stages:

Multi-Stock Data Collection

Feature Engineering

Labeling Future Returns

Base Machine Learning Model Training (LightGBM)

Signal Filtering & Backtesting Enhancements

Meta-Labeling

Time-Series Cross Validation

Multi-Stock Meta-Labeling

📊 1. Multi-Stock Data Collection

Data sourced from Yahoo Finance using yfinance.

Stocks: AAPL, MSFT, GOOG, AMZN, TSLA

Date range: 2018-01-01 → 2023-01-01

OHLCV data retrieved and stored as individual DataFrames per ticker.

import yfinance as yf
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
data = yf.download(tickers, start='2018-01-01', end='2023-01-01', group_by='ticker', auto_adjust=True)

⚙️ 2. Feature Engineering

Features designed for predictive trading models:

Feature	Description
Return_1d	1-day return
Return_5d	5-day return
SMA_5 / SMA_20	Short-term & long-term moving averages
RSI_14	14-day Relative Strength Index
Volatility_5d	5-day rolling standard deviation of returns
SMA_Ratio	SMA_5 / SMA_20 (price momentum signal)

These features were computed per stock, then combined into a multi-index DataFrame (Date, Ticker).

df['RSI_14'] = 100 - (100 / (1 + rs))
df['Volatility_5d'] = df['Return_1d'].rolling(5).std()
df['SMA_Ratio'] = df['SMA_5'] / (df['SMA_20'] + 1e-10)

🎯 3. Labeling Future Returns

Future 5-day return computed: Future_5d_Return = Close.shift(-5) / Close - 1

Binary Label for ML classification:

1 → price increases in 5 days

0 → price decreases or remains the same

💻 4. Base Machine Learning Model (LightGBM)

Trained LightGBM classifier to predict probability of 5-day price increase.

Time-based train-test split to prevent lookahead bias.

from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)


Evaluation metrics included: Accuracy, ROC AUC, Classification Report

Output: predicted probabilities and class labels.

📈 5. Signal Filtering & Backtesting

Filtered model predictions using a probability threshold (e.g., 0.7) to focus on high-confidence trades.

Applied realistic trading adjustments:

Commission per trade: 5 bps

Slippage per trade: 3 bps

Adjustable position sizing

Computed cumulative strategy returns vs. benchmark (all trades).

Visualized using matplotlib.

🔄 6. Meta-Labeling

Secondary ML model to filter profitable trades from high-confidence base signals.

Features included:

Base model probability

Rolling volatility

Rolling return trend

Meta-model trained using LightGBM, evaluated with Accuracy, ROC AUC, and Classification Report.

⏳ 7. Time-Series Cross Validation

Applied TimeSeriesSplit to validate model performance across sequential folds.

Ensures no leakage from future data.

Metrics collected per fold: Accuracy and ROC AUC.

🌐 8. Multi-Stock Meta-Labeling

Extended meta-labeling to all tickers, pooling confident trades.

Balanced the dataset by upsampling minority class to improve model learning.

Resulting meta-model predicts profitable trades across multiple stocks.

🔧 Key Libraries & Tools

yfinance – Stock OHLCV data

pandas, numpy – Data manipulation & feature engineering

scikit-learn – ML utilities & evaluation metrics

lightgbm – Gradient boosting classification

matplotlib – Visualization

📌 Key Learnings

Feature engineering is critical for time-series trading models.

Meta-labeling improves signal quality and reduces false positives.

Time-based train-test splits prevent lookahead bias.

Multi-stock modeling allows portfolio-level insights.

Backtesting with slippage and commission produces more realistic performance.

🔗 Next Steps / Enhancements

Reinforcement learning agent for limit order book simulation.

Ensemble models combining multiple strategies.

Hyperparameter tuning using Optuna or GridSearchCV.

Extend to intraday / high-frequency data.
