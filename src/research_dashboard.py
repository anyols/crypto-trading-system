from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_BACKTEST_DIR = Path("data/final_strategy_backtests_rediscovered_nextbar")
DEFAULT_REPORT_DIR = Path("reports/final_strategy_report_rediscovered_nextbar")


# ─────────────────────────────────────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_file(folder: Path, pattern: str) -> Optional[Path]:
    files = sorted(folder.glob(pattern))
    if not files:
        return None
    return files[-1]


def require_file(folder: Path, pattern: str) -> Path:
    path = find_latest_file(folder, pattern)
    if path is None:
        raise FileNotFoundError(f"No file found in {folder} matching {pattern}")
    return path


@st.cache_data
def load_backtest_outputs(backtest_dir_str: str, report_dir_str: str) -> dict[str, pd.DataFrame]:
    backtest_dir = Path(backtest_dir_str)
    report_dir = Path(report_dir_str)

    metrics_path = require_file(backtest_dir, "final_strategy_metrics_*.csv")
    equity_path = require_file(backtest_dir, "final_strategy_equity_*.csv")
    trades_path = require_file(backtest_dir, "final_strategy_trades_*.csv")
    asset_path = require_file(backtest_dir, "final_strategy_asset_contributions_*.csv")
    definitions_path = require_file(backtest_dir, "final_strategy_definitions_*.csv")

    metrics = pd.read_csv(metrics_path)
    equity = pd.read_csv(equity_path, parse_dates=["time"])
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    asset = pd.read_csv(asset_path)
    definitions = pd.read_csv(definitions_path)

    rolling_path = find_latest_file(report_dir, "rolling_window_summary*.csv")
    if rolling_path is not None:
        rolling = pd.read_csv(rolling_path)
    else:
        rolling = calculate_rolling_window_summary(equity, windows=[10, 30, 90])

    return {
        "metrics": metrics,
        "equity": equity,
        "trades": trades,
        "asset": asset,
        "definitions": definitions,
        "rolling": rolling,
        "paths": pd.DataFrame(
            [
                {"name": "metrics", "path": str(metrics_path)},
                {"name": "equity", "path": str(equity_path)},
                {"name": "trades", "path": str(trades_path)},
                {"name": "asset", "path": str(asset_path)},
                {"name": "definitions", "path": str(definitions_path)},
                {"name": "rolling", "path": str(rolling_path) if rolling_path else "computed in app"},
            ]
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calculations
# ─────────────────────────────────────────────────────────────────────────────

def calculate_drawdown(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("time").copy()
    out["running_max"] = out["equity"].cummax()
    out["drawdown_pct"] = (out["equity"] / out["running_max"] - 1.0) * 100.0
    return out


def calculate_rolling_window_summary(equity: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []

    equity = equity.copy()
    equity["time"] = pd.to_datetime(equity["time"], utc=True)

    for strategy_id, group in equity.groupby("strategy_id"):
        group = group.sort_values("time").set_index("time")
        daily = group["equity"].resample("1D").last().dropna()

        if daily.empty:
            continue

        for window in windows:
            rolling_return = (daily / daily.shift(window) - 1.0).dropna() * 100.0

            if rolling_return.empty:
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "window_days": window,
                        "num_windows": 0,
                        "avg_return_pct": 0.0,
                        "median_return_pct": 0.0,
                        "positive_window_rate_pct": 0.0,
                        "best_return_pct": 0.0,
                        "worst_return_pct": 0.0,
                    }
                )
                continue

            rows.append(
                {
                    "strategy_id": strategy_id,
                    "window_days": window,
                    "num_windows": int(len(rolling_return)),
                    "avg_return_pct": float(rolling_return.mean()),
                    "median_return_pct": float(rolling_return.median()),
                    "positive_window_rate_pct": float((rolling_return > 0).mean() * 100.0),
                    "best_return_pct": float(rolling_return.max()),
                    "worst_return_pct": float(rolling_return.min()),
                }
            )

    return pd.DataFrame(rows)


def build_daily_rolling_returns(equity: pd.DataFrame, window_days: int) -> pd.DataFrame:
    frames = []

    for strategy_id, group in equity.groupby("strategy_id"):
        daily = (
            group.sort_values("time")
            .set_index("time")["equity"]
            .resample("1D")
            .last()
            .dropna()
        )

        if daily.empty:
            continue

        rolling = (daily / daily.shift(window_days) - 1.0) * 100.0
        part = rolling.dropna().reset_index()
        part.columns = ["time", "rolling_return_pct"]
        part["strategy_id"] = strategy_id
        part["window_days"] = window_days
        frames.append(part)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def format_percent(x: float) -> str:
    return f"{x:.2f}%"


def format_float(x: float) -> str:
    return f"{x:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curves(equity: pd.DataFrame) -> go.Figure:
    data = equity.copy()
    data = data.sort_values(["strategy_id", "time"])
    data["normalized_equity"] = data.groupby("strategy_id")["equity"].transform(
        lambda s: s / s.iloc[0]
    )

    fig = px.line(
        data,
        x="time",
        y="normalized_equity",
        color="strategy_id",
        title="Normalized Equity Curves",
        hover_data=["equity", "cash", "unrealized_pnl", "gross_exposure_pct"],
    )
    fig.update_layout(yaxis_title="Growth of $1", xaxis_title="Time")
    return fig


def plot_drawdowns(equity: pd.DataFrame) -> go.Figure:
    drawdowns = (
        equity.groupby("strategy_id", group_keys=False)
        .apply(calculate_drawdown)
        .reset_index(drop=True)
    )

    fig = px.line(
        drawdowns,
        x="time",
        y="drawdown_pct",
        color="strategy_id",
        title="Drawdown Curves",
        hover_data=["equity", "drawdown_pct"],
    )
    fig.update_layout(yaxis_title="Drawdown (%)", xaxis_title="Time")
    return fig


def plot_risk_return(metrics: pd.DataFrame) -> go.Figure:
    data = metrics.copy()
    size = data["sharpe_ratio"].clip(lower=0.1)
    data["bubble_size"] = size

    fig = px.scatter(
        data,
        x="annualized_volatility_pct",
        y="annualized_return_pct",
        color="category",
        size="bubble_size",
        text="strategy_id",
        hover_data=[
            "strategy_name",
            "total_return_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown_pct",
            "profit_factor",
            "num_trades",
            "win_rate_pct",
        ],
        title="Risk vs Return",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
    )
    return fig


def plot_metric_bars(metrics: pd.DataFrame, metric: str, title: str) -> go.Figure:
    data = metrics.sort_values(metric, ascending=False).copy()

    fig = px.bar(
        data,
        x="strategy_id",
        y=metric,
        color="category",
        title=title,
        hover_data=["strategy_name", "category"],
    )
    fig.update_layout(xaxis_title="Strategy", yaxis_title=title)
    return fig


def plot_rolling_returns(equity: pd.DataFrame, window_days: int) -> go.Figure:
    rolling = build_daily_rolling_returns(equity, window_days)

    fig = px.line(
        rolling,
        x="time",
        y="rolling_return_pct",
        color="strategy_id",
        title=f"Rolling {window_days}-Day Returns",
    )
    fig.update_layout(yaxis_title="Return (%)", xaxis_title="Time")
    return fig


def plot_trade_distribution(trades: pd.DataFrame) -> go.Figure:
    fig = px.box(
        trades,
        x="strategy_id",
        y="trade_return_pct",
        color="strategy_id",
        points="outliers",
        title="Trade Return Distribution",
    )
    fig.update_layout(yaxis_title="Trade Return (%)")
    return fig


def plot_asset_contribution(asset: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        asset,
        x="strategy_id",
        y="net_pnl",
        color="product_id",
        title="Asset Contribution to Net PnL",
        hover_data=["num_trades", "win_rate_pct", "fees_paid"],
    )
    fig.update_layout(yaxis_title="Net PnL")
    return fig


def plot_trade_side_summary(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return go.Figure()

    summary = (
        trades.groupby(["strategy_id", "side"], as_index=False)
        .agg(
            net_pnl=("net_pnl", "sum"),
            trades=("net_pnl", "count"),
            win_rate_pct=("net_pnl", lambda s: (s > 0).mean() * 100.0),
        )
    )

    fig = px.bar(
        summary,
        x="strategy_id",
        y="net_pnl",
        color="side",
        title="Long/Short Contribution to Net PnL",
        hover_data=["trades", "win_rate_pct"],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backtest-dir", type=str, default=str(DEFAULT_BACKTEST_DIR))
    parser.add_argument("--report-dir", type=str, default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    st.set_page_config(
        page_title="Crypto Strategy Research Report",
        page_icon="📈",
        layout="wide",
    )

    data = load_backtest_outputs(args.backtest_dir, args.report_dir)

    metrics = data["metrics"]
    equity = data["equity"]
    trades = data["trades"]
    asset = data["asset"]
    definitions = data["definitions"]
    rolling = data["rolling"]

    st.title("Crypto Volatility/Momentum Strategy Research Report")

    st.caption(
        "Final report dashboard for exploratory and clean rediscovered crypto strategies. "
        "Backtests use next-bar execution, transaction fees, and a one-year chronological holdout."
    )

    with st.sidebar:
        st.header("Data")
        st.write("Backtest directory:")
        st.code(args.backtest_dir)
        st.write("Report directory:")
        st.code(args.report_dir)

        strategy_options = sorted(metrics["strategy_id"].unique())
        selected_strategies = st.multiselect(
            "Strategies to display",
            options=strategy_options,
            default=strategy_options,
        )

        st.divider()
        st.subheader("Methodology labels")
        st.markdown(
            """
            - **Exploratory V3:** strong but contaminated by earlier iteration.
            - **R1/R2/R4:** selected using pre-holdout data only.
            - **Ablations:** test whether volatility filters add value.
            - **Buy & hold:** passive benchmark.
            """
        )

    metrics_view = metrics[metrics["strategy_id"].isin(selected_strategies)].copy()
    equity_view = equity[equity["strategy_id"].isin(selected_strategies)].copy()
    trades_view = trades[trades["strategy_id"].isin(selected_strategies)].copy()
    asset_view = asset[asset["strategy_id"].isin(selected_strategies)].copy()
    rolling_view = rolling[rolling["strategy_id"].isin(selected_strategies)].copy()

    tab_summary, tab_methodology, tab_performance, tab_rolling, tab_trades, tab_experiments, tab_files = st.tabs(
        [
            "Executive Summary",
            "Methodology",
            "Performance",
            "Rolling Windows",
            "Trades & Assets",
            "Experiment Log",
            "Files",
        ]
    )

    with tab_summary:
        st.header("Executive Summary")

        col1, col2, col3, col4 = st.columns(4)

        best_return = metrics_view.sort_values("total_return_pct", ascending=False).iloc[0]
        best_sharpe = metrics_view.sort_values("sharpe_ratio", ascending=False).iloc[0]
        worst_dd = metrics_view.sort_values("max_drawdown_pct", ascending=True).iloc[0]
        best_pf = metrics_view.sort_values("profit_factor", ascending=False).iloc[0]

        col1.metric(
            "Best total return",
            best_return["strategy_id"],
            format_percent(best_return["total_return_pct"]),
        )
        col2.metric(
            "Best Sharpe",
            best_sharpe["strategy_id"],
            format_float(best_sharpe["sharpe_ratio"]),
        )
        col3.metric(
            "Worst drawdown observed",
            worst_dd["strategy_id"],
            format_percent(worst_dd["max_drawdown_pct"]),
        )
        col4.metric(
            "Best profit factor",
            best_pf["strategy_id"],
            format_float(best_pf["profit_factor"]),
        )

        st.subheader("Main conclusion")

        st.markdown(
            """
            The project started as a short-term crypto trading competition research system,
            but the strongest early strategy, **V3**, was contaminated by iterative selection.
            Instead of pretending it was clean, the process was rebuilt:

            1. Reserve the final year as holdout.
            2. Rediscover short-term volatility/momentum strategies using only pre-holdout data.
            3. Freeze rediscovered candidates.
            4. Test them on the final year using next-bar execution and fees.

            The clean rediscovered strategies were lower-return than V3, but they remained positive
            on the final holdout and beat the passive crypto benchmark. The strongest research finding
            is that **volatility conditioning improved extreme selloff rebound signals**.
            """
        )

        st.subheader("Ranked metrics")
        display_cols = [
            "strategy_id",
            "category",
            "total_return_pct",
            "annualized_return_pct",
            "annualized_volatility_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown_pct",
            "num_trades",
            "win_rate_pct",
            "profit_factor",
            "exposure_time_pct",
            "avg_gross_exposure_pct",
        ]
        display_cols = [c for c in display_cols if c in metrics_view.columns]
        st.dataframe(
            metrics_view.sort_values(["sharpe_ratio", "total_return_pct"], ascending=False)[display_cols],
            use_container_width=True,
        )

        st.plotly_chart(plot_risk_return(metrics_view), use_container_width=True)

    with tab_methodology:
        st.header("Methodology")

        st.markdown(
            """
            ## Research question

            Do realized-volatility regimes improve short-term crypto momentum/reversion signals?

            ## Data

            - Coinbase OHLCV candles
            - 15-minute execution timeframe
            - Assets: BTC, ETH, SOL, DOGE, XRP
            - Final holdout: last one year
            - Execution: signal at candle close, entry at next candle open
            - Fees: 0.01%
            - Slippage in this run: 0 bps

            ## Validation discipline

            The early V3 strategy was developed during competition-style exploration,
            so it is **not treated as clean holdout evidence**.

            Clean rediscovered candidates were selected using only pre-holdout data.
            The final holdout was then used once to evaluate those frozen strategies.
            """
        )

        st.subheader("Strategy definitions")
        definition_cols = [
            "strategy_id",
            "name",
            "category",
            "side",
            "lookback_hours",
            "horizon_minutes",
            "momentum_lower",
            "momentum_upper",
            "vol_lower",
            "vol_upper",
            "filter_name",
            "description",
        ]
        definition_cols = [c for c in definition_cols if c in definitions.columns]
        st.dataframe(definitions[definition_cols], use_container_width=True)

        st.subheader("Interpretation labels")
        st.markdown(
            """
            - **A_v3_baseline:** exploratory competition-derived benchmark.
            - **R1/R2/R4:** clean rediscovered strategy candidates.
            - **R1/R3 baselines:** ablation tests.
            - **F_buy_and_hold_equal_weight:** passive benchmark.
            """
        )

    with tab_performance:
        st.header("Performance")

        st.plotly_chart(plot_equity_curves(equity_view), use_container_width=True)
        st.plotly_chart(plot_drawdowns(equity_view), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_metric_bars(metrics_view, "total_return_pct", "Total Return (%)"),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_metric_bars(metrics_view, "sharpe_ratio", "Sharpe Ratio"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_metric_bars(metrics_view, "max_drawdown_pct", "Max Drawdown (%)"),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_metric_bars(metrics_view, "profit_factor", "Profit Factor"),
                use_container_width=True,
            )

        st.subheader("Ablation interpretation")
        st.markdown(
            """
            The key ablation is **R1 with volatility filter vs R1 without volatility filter**.
            If the no-vol version performs materially worse, the volatility regime is not just decoration;
            it is improving the tradability of the selloff rebound signal.
            """
        )

    with tab_rolling:
        st.header("Rolling Window Analysis")

        st.markdown(
            """
            Rolling windows answer a different question from full-year return.

            - **10-day windows:** competition-style consistency.
            - **30-day windows:** monthly short-term robustness.
            - **90-day windows:** quarterly persistence.
            """
        )

        st.subheader("Rolling summary table")
        st.dataframe(
            rolling_view.sort_values(["window_days", "avg_return_pct"], ascending=[True, False]),
            use_container_width=True,
        )

        for window in [10, 30, 90]:
            st.plotly_chart(plot_rolling_returns(equity_view, window), use_container_width=True)

        st.warning(
            "Many clean rediscovered strategies are selective. A median 10-day return of 0% often means "
            "there were no trades in many 10-day windows, not that every window had flat active performance."
        )

    with tab_trades:
        st.header("Trades & Asset Contribution")

        st.plotly_chart(plot_trade_distribution(trades_view), use_container_width=True)
        st.plotly_chart(plot_asset_contribution(asset_view), use_container_width=True)
        st.plotly_chart(plot_trade_side_summary(trades_view), use_container_width=True)

        st.subheader("Trade summary")
        if not trades_view.empty:
            trade_summary = (
                trades_view.groupby(["strategy_id", "side"], as_index=False)
                .agg(
                    trades=("net_pnl", "count"),
                    net_pnl=("net_pnl", "sum"),
                    avg_trade_return_pct=("trade_return_pct", "mean"),
                    median_trade_return_pct=("trade_return_pct", "median"),
                    win_rate_pct=("net_pnl", lambda s: (s > 0).mean() * 100.0),
                    avg_holding_hours=("holding_hours", "mean"),
                    fees_paid=("total_fees", "sum"),
                )
                .sort_values(["strategy_id", "net_pnl"], ascending=[True, False])
            )
            st.dataframe(trade_summary, use_container_width=True)

        st.subheader("Asset contribution table")
        st.dataframe(asset_view, use_container_width=True)

    with tab_experiments:
        st.header("Experiment Log")

        st.markdown(
            """
            ## 1. Breakout / Donchian / trend strategy family

            **Status:** rejected.

            These strategies traded too frequently, suffered poor drawdowns,
            and did not provide robust performance after costs.

            ## 2. Automatic probability bucket selector

            **Status:** rejected.

            The broad automatic selector found unstable buckets and overfit.
            It was useful for discovery but not reliable as a final strategy.

            ## 3. Core2 / V3

            **Status:** exploratory benchmark.

            V3 performed strongly, but it was developed through iterative competition-style testing.
            Because strategy changes were made after looking at intermediate results,
            it is not clean final holdout proof.

            ## 4. Development-only rediscovery

            **Status:** accepted methodology.

            A new rediscovery pipeline reserved the final year as untouched holdout,
            then searched for short-term volatility/momentum strategies using only pre-holdout data.

            ## 5. Clean rediscovered candidates

            **Status:** final research candidates.

            R1/R2/R4 were selected without using the final year and then tested on the final holdout.
            They produced lower but more defensible returns than V3.

            ## 6. Ablation tests

            **Status:** key research evidence.

            The R1 no-vol baseline tested whether volatility filtering mattered.
            The result showed that volatility conditioning materially improved the selloff rebound signal.

            ## 7. Passive benchmark

            **Status:** benchmark.

            Equal-weight buy-and-hold suffered large drawdowns and negative return during the holdout.
            """
        )

    with tab_files:
        st.header("Loaded files")
        st.dataframe(data["paths"], use_container_width=True)

        st.subheader("Raw metrics")
        st.dataframe(metrics, use_container_width=True)

        st.subheader("Raw rolling summary")
        st.dataframe(rolling, use_container_width=True)


if __name__ == "__main__":
    main()