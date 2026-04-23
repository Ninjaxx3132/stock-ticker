# Stock Prediction Bot Starter

This project gives you a starting point for an automatic stock trading simulator and a local browser UI.

It does **not** guarantee profit, and it should be used for learning, paper trading, and careful backtesting before any real money is involved.

## What it does

- Downloads historical stock data with `yfinance`
- Builds simple technical features
- Trains a machine learning classifier to estimate next-day direction
- Backtests an automatic strategy with entry filters, exits, and portfolio reporting
- Saves the trained model so you can reuse it later
- Opens a local browser simulator so you can preview how it behaves
- Includes slippage, commissions, stop-losses, take-profit logic, trailing stops, and volatility-based position sizing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train and backtest

```bash
py -3 stock_bot.py --symbols AAPL,MSFT,NVDA,GOOGL,AMZN --start 2018-01-01 --end 2026-01-01
```

## Open the simulator

```bash
py -3 -m streamlit run simulator_app.py
```

Or on Windows PowerShell:

```powershell
.\run_site.ps1
```

## Publish it as a site

This project is ready to deploy as a public Streamlit app.

The quickest path is:

1. Put the project in a GitHub repository.
2. Deploy it from [Streamlit Community Cloud](https://share.streamlit.io).
3. Use `simulator_app.py` as the main app file.

More detailed steps are in [DEPLOY.md](C:\Users\jksnw\OneDrive\Documents\stock predictor\DEPLOY.md).

The app opens in your browser and lets you:

- enter 5 to 10 tickers
- choose a date range
- view portfolio and per-stock equity curves
- compare the strategy to buy-and-hold
- inspect recent buy and sell decisions with reasons and costs

## Command-line output

The script prints:

- model accuracy
- precision
- recall
- profit/loss
- buy-and-hold return
- win rate
- max drawdown
- trading costs

## Notes

- Start with paper trading only.
- This simulator already includes transaction costs and slippage, but it is still only a research tool.
- No backtest can guarantee future profit.
