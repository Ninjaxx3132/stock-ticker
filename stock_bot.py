from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


MODEL_DIR = Path("models")
INITIAL_CAPITAL = 10_000.0
COMMISSION_RATE = 0.0005
SLIPPAGE_RATE = 0.001
MAX_POSITION_SIZE = 0.25
RISK_BUDGET_PER_TRADE = 0.01
STOP_ATR_MULTIPLE = 2.0
TAKE_PROFIT_ATR_MULTIPLE = 3.0
TRAILING_STOP_ATR_MULTIPLE = 2.5
MAX_HOLD_DAYS = 20
ANNUAL_TRADING_DAYS = 252


@dataclass
class BacktestResult:
    accuracy: float
    precision: float
    recall: float
    strategy_return: float
    buy_and_hold_return: float
    latest_probability: float
    latest_action: str
    rows_used: int
    buy_count: int
    sell_count: int
    trade_count: int
    final_position: int
    total_cost_pct: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_return: float
    final_equity: float
    buy_hold_final_equity: float
    test_frame: pd.DataFrame


@dataclass
class SimulationResult:
    symbol: str
    start: str
    end: str
    model_path: Path
    train_rows: int
    backtest: BacktestResult


def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No market data returned for {symbol}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna().copy()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["atr_14"] = compute_atr(df, 14)
    df["atr_pct"] = df["atr_14"] / df["Close"]
    df["volume_change"] = df["Volume"].pct_change(1)
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["trend_gap"] = (df["sma_20"] - df["sma_50"]) / df["Close"]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna().copy()


def feature_columns() -> list[str]:
    return [
        "return_1d",
        "return_5d",
        "return_20d",
        "sma_5",
        "sma_20",
        "sma_50",
        "ema_10",
        "volatility_10",
        "volatility_20",
        "rsi_14",
        "atr_pct",
        "volume_change",
        "high_low_range",
        "trend_gap",
    ]


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Not enough rows to create train/test split.")
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def train_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(train_df[feature_columns()], train_df["target"])
    return model


def entry_signal(probability: float, row: pd.Series) -> bool:
    score = 0
    score += int(probability >= 0.54)
    score += int(row["sma_5"] > row["sma_20"])
    score += int(row["sma_20"] > row["sma_50"])
    score += int(45 <= row["rsi_14"] <= 67)
    score += int(row["return_20d"] > 0)
    score += int(row["atr_pct"] < 0.05)
    return score >= 5


def exit_signal(probability: float, row: pd.Series) -> bool:
    score = 0
    score += int(probability < 0.48)
    score += int(row["sma_5"] < row["sma_20"])
    score += int(row["rsi_14"] > 72)
    score += int(row["return_5d"] < -0.03)
    return score >= 2


def compute_position_size(row: pd.Series) -> float:
    atr_pct = float(row["atr_pct"])
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return 0.0
    stop_distance_pct = max(atr_pct * STOP_ATR_MULTIPLE, 0.01)
    size = RISK_BUDGET_PER_TRADE / stop_distance_pct
    return float(min(MAX_POSITION_SIZE, max(0.05, size)))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def sharpe_ratio(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(ANNUAL_TRADING_DAYS) * daily_returns.mean() / daily_returns.std(ddof=0))


def generate_trade_actions(test_df: pd.DataFrame, probabilities: np.ndarray) -> tuple[pd.DataFrame, list[float]]:
    next_day_returns = test_df["Close"].shift(-1) / test_df["Close"] - 1

    in_position = False
    entry_price = 0.0
    peak_price = 0.0
    hold_days = 0
    position_size = 0.0
    total_cost_pct = 0.0

    actions: list[str] = []
    action_reason: list[str] = []
    positions: list[int] = []
    position_sizes: list[float] = []
    strategy_returns: list[float] = []
    costs: list[float] = []
    trade_returns: list[float] = []

    for idx, (timestamp, row) in enumerate(test_df.iterrows()):
        probability = float(probabilities[idx])
        next_return = float(next_day_returns.loc[timestamp]) if pd.notna(next_day_returns.loc[timestamp]) else 0.0
        close_price = float(row["Close"])
        atr_value = float(row["atr_14"])
        action = "HOLD"
        reason = "No change"
        cost_pct = 0.0

        if in_position:
            hold_days += 1
            peak_price = max(peak_price, close_price)
            stop_price = entry_price - STOP_ATR_MULTIPLE * atr_value
            take_profit_price = entry_price + TAKE_PROFIT_ATR_MULTIPLE * atr_value
            trailing_stop_price = peak_price - TRAILING_STOP_ATR_MULTIPLE * atr_value

            if close_price <= stop_price:
                action = "SELL"
                reason = "Stop loss"
            elif close_price >= take_profit_price:
                action = "SELL"
                reason = "Take profit"
            elif close_price <= trailing_stop_price:
                action = "SELL"
                reason = "Trailing stop"
            elif hold_days >= MAX_HOLD_DAYS:
                action = "SELL"
                reason = "Max hold reached"
            elif exit_signal(probability, row):
                action = "SELL"
                reason = "Trend weakened"

        if not in_position and entry_signal(probability, row):
            action = "BUY"
            reason = "Trend and model aligned"
            position_size = compute_position_size(row)
            entry_price = close_price
            peak_price = close_price
            hold_days = 0
            in_position = True
            cost_pct = position_size * (COMMISSION_RATE + SLIPPAGE_RATE)
        elif in_position and action == "SELL":
            in_position = False
            hold_days = 0
            cost_pct = position_size * (COMMISSION_RATE + SLIPPAGE_RATE)
            realized_return = ((close_price / entry_price) - 1) * position_size - cost_pct
            trade_returns.append(realized_return)
            position_size = 0.0

        exposure = position_size if in_position else 0.0
        daily_strategy_return = exposure * next_return - cost_pct
        total_cost_pct += cost_pct

        actions.append(action)
        action_reason.append(reason)
        positions.append(1 if in_position else 0)
        position_sizes.append(exposure)
        strategy_returns.append(daily_strategy_return)
        costs.append(cost_pct)

    detailed_test = test_df.copy()
    detailed_test["predicted_probability"] = probabilities
    detailed_test["action"] = actions
    detailed_test["action_reason"] = action_reason
    detailed_test["position"] = positions
    detailed_test["position_size"] = position_sizes
    detailed_test["next_day_return"] = next_day_returns.fillna(0)
    detailed_test["trade_cost"] = costs
    detailed_test["strategy_return"] = strategy_returns
    return detailed_test, trade_returns


def backtest(model: RandomForestClassifier, test_df: pd.DataFrame) -> BacktestResult:
    probabilities = model.predict_proba(test_df[feature_columns()])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    detailed_test, trade_returns = generate_trade_actions(test_df, probabilities)

    strategy_curve = (1 + pd.Series(detailed_test["strategy_return"], index=test_df.index)).cumprod()
    buy_hold_curve = (1 + detailed_test["next_day_return"]).cumprod()
    detailed_test["predicted_direction"] = predictions
    detailed_test["strategy_equity_curve"] = strategy_curve
    detailed_test["buy_hold_equity_curve"] = buy_hold_curve
    final_equity = INITIAL_CAPITAL * float(strategy_curve.iloc[-1])
    buy_hold_final_equity = INITIAL_CAPITAL * float(buy_hold_curve.iloc[-1])

    daily_returns = detailed_test["strategy_return"]
    wins = [trade for trade in trade_returns if trade > 0]

    return BacktestResult(
        accuracy=float(accuracy_score(test_df["target"], predictions)),
        precision=float(precision_score(test_df["target"], predictions, zero_division=0)),
        recall=float(recall_score(test_df["target"], predictions, zero_division=0)),
        strategy_return=float(strategy_curve.iloc[-1] - 1),
        buy_and_hold_return=float(buy_hold_curve.iloc[-1] - 1),
        latest_probability=float(probabilities[-1]),
        latest_action=str(detailed_test["action"].iloc[-1]),
        rows_used=len(test_df),
        buy_count=int((detailed_test["action"] == "BUY").sum()),
        sell_count=int((detailed_test["action"] == "SELL").sum()),
        trade_count=int(detailed_test["action"].isin(["BUY", "SELL"]).sum()),
        final_position=int(detailed_test["position"].iloc[-1]),
        total_cost_pct=float(detailed_test["trade_cost"].sum()),
        max_drawdown=max_drawdown(strategy_curve),
        sharpe_ratio=sharpe_ratio(daily_returns),
        win_rate=float(len(wins) / len(trade_returns)) if trade_returns else 0.0,
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        final_equity=final_equity,
        buy_hold_final_equity=buy_hold_final_equity,
        test_frame=detailed_test,
    )


def save_model(model: RandomForestClassifier, symbol: str) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    output_path = MODEL_DIR / f"{symbol.upper()}_model.joblib"
    joblib.dump(model, output_path)
    return output_path


def run_simulation(symbol: str, start: str, end: str) -> SimulationResult:
    raw_data = download_data(symbol, start, end)
    dataset = build_features(raw_data)
    train_df, test_df = split_train_test(dataset)
    model = train_model(train_df)
    results = backtest(model, test_df)
    model_path = save_model(model, symbol)

    return SimulationResult(
        symbol=symbol.upper(),
        start=start,
        end=end,
        model_path=model_path,
        train_rows=len(train_df),
        backtest=results,
    )


def run_portfolio_simulation(symbols: list[str], start: str, end: str) -> tuple[list[SimulationResult], pd.DataFrame]:
    simulations: list[SimulationResult] = []
    strategy_columns: list[pd.Series] = []
    benchmark_columns: list[pd.Series] = []

    for symbol in symbols:
        cleaned_symbol = symbol.strip().upper()
        if not cleaned_symbol:
            continue
        simulation = run_simulation(cleaned_symbol, start, end)
        simulations.append(simulation)
        frame = simulation.backtest.test_frame
        strategy_columns.append(frame["strategy_return"].rename(cleaned_symbol))
        benchmark_columns.append(frame["next_day_return"].rename(f"{cleaned_symbol}_bench"))

    if not simulations:
        raise ValueError("No valid stock symbols were provided.")

    strategy_returns = pd.concat(strategy_columns, axis=1).fillna(0)
    benchmark_returns = pd.concat(benchmark_columns, axis=1).fillna(0)

    portfolio = pd.DataFrame(index=strategy_returns.index)
    portfolio["portfolio_return"] = strategy_returns.mean(axis=1)
    portfolio["portfolio_equity_curve"] = (1 + portfolio["portfolio_return"]).cumprod()
    portfolio["benchmark_return"] = benchmark_returns.mean(axis=1)
    portfolio["benchmark_equity_curve"] = (1 + portfolio["benchmark_return"]).cumprod()
    return simulations, portfolio


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and backtest an advanced stock trading simulator.")
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers, for example AAPL,MSFT,NVDA")
    parser.add_argument("--start", default="2018-01-01", help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date in YYYY-MM-DD")
    args = parser.parse_args()

    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    simulations, portfolio = run_portfolio_simulation(symbols, args.start, args.end)

    print(f"Stocks tested: {', '.join(sim.symbol for sim in simulations)}")
    print(f"Portfolio return: {format_pct(float(portfolio['portfolio_equity_curve'].iloc[-1] - 1))}")
    print(f"Portfolio benchmark: {format_pct(float(portfolio['benchmark_equity_curve'].iloc[-1] - 1))}")
    print("")

    for simulation in simulations:
        results = simulation.backtest
        print(f"Symbol: {simulation.symbol}")
        print(f"Training rows: {simulation.train_rows}")
        print(f"Rows used: {results.rows_used}")
        print(f"Accuracy: {results.accuracy:.4f}")
        print(f"Precision: {results.precision:.4f}")
        print(f"Recall: {results.recall:.4f}")
        print(f"Profit/Loss: {format_pct(results.strategy_return)}")
        print(f"Buy and hold: {format_pct(results.buy_and_hold_return)}")
        print(f"Win rate: {format_pct(results.win_rate)}")
        print(f"Average trade: {format_pct(results.avg_trade_return)}")
        print(f"Max drawdown: {format_pct(results.max_drawdown)}")
        print(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
        print(f"Trading costs: {format_pct(results.total_cost_pct)}")
        print(f"Final equity: ${results.final_equity:,.2f}")
        print(f"Latest action: {results.latest_action}")
        print(f"Saved model: {simulation.model_path}")
        print("")

    print("Warning: this is a research simulator, not a guarantee of profit.")


if __name__ == "__main__":
    main()
