from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_bot import INITIAL_CAPITAL, format_pct, run_portfolio_simulation


st.set_page_config(page_title="Stock Bot Simulator", page_icon="ST", layout="wide")


def build_portfolio_chart(portfolio_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=portfolio_frame.index,
            y=portfolio_frame["portfolio_equity_curve"],
            mode="lines",
            name="Strategy Portfolio",
            line={"width": 3, "color": "#0f766e"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=portfolio_frame.index,
            y=portfolio_frame["benchmark_equity_curve"],
            mode="lines",
            name="Buy and Hold Basket",
            line={"width": 2, "color": "#7c3aed", "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        height=400,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    return figure


def build_stock_chart(test_frame: pd.DataFrame, symbol: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=test_frame.index,
            y=test_frame["strategy_equity_curve"],
            mode="lines",
            name=f"{symbol} Strategy",
            line={"width": 3, "color": "#1d4ed8"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test_frame.index,
            y=test_frame["buy_hold_equity_curve"],
            mode="lines",
            name="Buy and Hold",
            line={"width": 2, "color": "#7c3aed", "dash": "dash"},
        )
    )
    buys = test_frame[test_frame["action"] == "BUY"]
    sells = test_frame[test_frame["action"] == "SELL"]
    figure.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["strategy_equity_curve"],
            mode="markers",
            name="Buy",
            marker={"size": 8, "color": "#16a34a", "symbol": "diamond"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["strategy_equity_curve"],
            mode="markers",
            name="Sell",
            marker={"size": 8, "color": "#dc2626", "symbol": "x"},
        )
    )
    figure.update_layout(
        title=f"{symbol} Advanced Trade Simulation",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        height=430,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    return figure


st.title("Advanced Automatic Stock Trading Simulator")
st.caption("This version uses automatic entries, exits, costs, stops, and position sizing to make the backtest more realistic.")

with st.sidebar:
    st.header("Simulation Settings")
    symbols_input = st.text_area(
        "Stock tickers",
        value="AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,AMD",
        height=120,
        help="Enter 5 to 10 comma-separated stocks.",
    )
    start = st.date_input("Start date", value=pd.Timestamp("2018-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today())
    run_clicked = st.button("Run advanced simulation", type="primary", use_container_width=True)

st.info(
    "This simulator still cannot predict future profits, but it is more realistic than the earlier version because "
    "it includes stop rules, trade costs, slippage, and volatility-based sizing."
)

if run_clicked:
    raw_symbols = [symbol.strip().upper() for symbol in symbols_input.replace("\n", ",").split(",") if symbol.strip()]
    if len(raw_symbols) < 5 or len(raw_symbols) > 10:
        st.error("Please enter between 5 and 10 stock tickers.")
    else:
        with st.spinner("Running advanced multi-stock backtest..."):
            try:
                simulations, portfolio = run_portfolio_simulation(raw_symbols, str(start), str(end))
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
            else:
                portfolio_return = float(portfolio["portfolio_equity_curve"].iloc[-1] - 1)
                benchmark_return = float(portfolio["benchmark_equity_curve"].iloc[-1] - 1)
                profitable_count = sum(sim.backtest.strategy_return > 0 for sim in simulations)

                top_metrics = st.columns(5)
                top_metrics[0].metric("Stocks Tested", str(len(simulations)))
                top_metrics[1].metric("Strategy P/L", format_pct(portfolio_return))
                top_metrics[2].metric("Benchmark P/L", format_pct(benchmark_return))
                top_metrics[3].metric("Profitable Stocks", f"{profitable_count}/{len(simulations)}")
                top_metrics[4].metric("Base Capital", f"${INITIAL_CAPITAL:,.0f}")

                st.plotly_chart(build_portfolio_chart(portfolio), use_container_width=True)

                summary_rows = []
                for simulation in simulations:
                    results = simulation.backtest
                    summary_rows.append(
                        {
                            "Symbol": simulation.symbol,
                            "P/L": format_pct(results.strategy_return),
                            "Buy & Hold": format_pct(results.buy_and_hold_return),
                            "Win Rate": format_pct(results.win_rate),
                            "Avg Trade": format_pct(results.avg_trade_return),
                            "Max Drawdown": format_pct(results.max_drawdown),
                            "Sharpe": f"{results.sharpe_ratio:.2f}",
                            "Costs": format_pct(results.total_cost_pct),
                            "Final Equity": f"${results.final_equity:,.0f}",
                            "Latest Action": results.latest_action,
                        }
                    )

                st.subheader("Per-Stock Results")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                tabs = st.tabs([simulation.symbol for simulation in simulations])
                for tab, simulation in zip(tabs, simulations):
                    with tab:
                        results = simulation.backtest
                        st.plotly_chart(build_stock_chart(results.test_frame, simulation.symbol), use_container_width=True)

                        detail_columns = st.columns(5)
                        detail_columns[0].metric("P/L", format_pct(results.strategy_return))
                        detail_columns[1].metric("Win Rate", format_pct(results.win_rate))
                        detail_columns[2].metric("Max Drawdown", format_pct(results.max_drawdown))
                        detail_columns[3].metric("Trade Costs", format_pct(results.total_cost_pct))
                        detail_columns[4].metric("Latest Action", results.latest_action)

                        recent = results.test_frame[
                            [
                                "Close",
                                "predicted_probability",
                                "action",
                                "action_reason",
                                "position_size",
                                "trade_cost",
                                "next_day_return",
                            ]
                        ].tail(12).copy()
                        recent.index.name = "Date"
                        recent["predicted_probability"] = recent["predicted_probability"].map(lambda value: f"{value:.3f}")
                        recent["position_size"] = recent["position_size"].map(lambda value: f"{value:.1%}")
                        recent["trade_cost"] = recent["trade_cost"].map(lambda value: f"{value:.3%}")
                        recent["next_day_return"] = recent["next_day_return"].map(lambda value: f"{value:.2%}")
                        st.dataframe(recent, use_container_width=True)
else:
    st.subheader("What changed")
    st.write("The strategy is now stricter about entering trades and quicker to protect capital once a trade is open.")
    st.write("It also measures costs and risk so the results look closer to what real trading might feel like.")
