from __future__ import annotations

import argparse
import json
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(backtest_dir: str, report_dir: str) -> dict:
    cells = [
        markdown_cell(
            """# Crypto Volatility/Momentum Strategy Research Report

This notebook documents a systematic crypto strategy research project.

The project started as a short-term crypto trading competition tool, but evolved into a broader research framework for testing whether realized-volatility regimes improve short-term crypto momentum/reversion signals.

The report separates:

1. exploratory competition-derived research,
2. clean development-only rediscovery,
3. final one-year holdout evaluation,
4. ablation tests,
5. benchmark comparison.

The goal is not to pretend a backtest is proof. The goal is to show a disciplined research process.
"""
        ),
        markdown_cell(
            """## Methodology Summary

### Data

- Coinbase OHLCV candles
- 15-minute execution bars
- Assets: BTC, ETH, SOL, DOGE, XRP
- Final holdout: last one year
- Execution: signal at candle close, entry at next candle open
- Fees: 0.01%

### Key issue discovered

The original V3 strategy performed well, but was developed through iterative competition-style testing. That means it is contaminated by selection bias.

To correct this, a new rediscovery process was run:

1. reserve the last year as untouched holdout,
2. use only pre-holdout data to discover strategies,
3. freeze candidates,
4. test frozen candidates on the final holdout.
"""
        ),
        code_cell(
            f"""from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BACKTEST_DIR = Path(r"{backtest_dir}")
REPORT_DIR = Path(r"{report_dir}")

def latest(folder, pattern):
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {{pattern}} in {{folder}}")
    return files[-1]

metrics_path = latest(BACKTEST_DIR, "final_strategy_metrics_*.csv")
equity_path = latest(BACKTEST_DIR, "final_strategy_equity_*.csv")
trades_path = latest(BACKTEST_DIR, "final_strategy_trades_*.csv")
asset_path = latest(BACKTEST_DIR, "final_strategy_asset_contributions_*.csv")
definitions_path = latest(BACKTEST_DIR, "final_strategy_definitions_*.csv")

rolling_candidates = sorted(REPORT_DIR.glob("rolling_window_summary*.csv"))
rolling_path = rolling_candidates[-1] if rolling_candidates else None

metrics = pd.read_csv(metrics_path)
equity = pd.read_csv(equity_path, parse_dates=["time"])
trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
asset = pd.read_csv(asset_path)
definitions = pd.read_csv(definitions_path)

metrics_path, equity_path, trades_path, asset_path, definitions_path, rolling_path
"""
        ),
        code_cell(
            """def calculate_rolling_window_summary(equity, windows):
    rows = []

    equity = equity.copy()
    equity["time"] = pd.to_datetime(equity["time"], utc=True)

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

        for window in windows:
            rolling_return = (daily / daily.shift(window) - 1.0).dropna() * 100.0

            if rolling_return.empty:
                rows.append({
                    "strategy_id": strategy_id,
                    "window_days": window,
                    "num_windows": 0,
                    "avg_return_pct": 0.0,
                    "median_return_pct": 0.0,
                    "positive_window_rate_pct": 0.0,
                    "best_return_pct": 0.0,
                    "worst_return_pct": 0.0,
                })
                continue

            rows.append({
                "strategy_id": strategy_id,
                "window_days": window,
                "num_windows": int(len(rolling_return)),
                "avg_return_pct": float(rolling_return.mean()),
                "median_return_pct": float(rolling_return.median()),
                "positive_window_rate_pct": float((rolling_return > 0).mean() * 100.0),
                "best_return_pct": float(rolling_return.max()),
                "worst_return_pct": float(rolling_return.min()),
            })

    return pd.DataFrame(rows)

if rolling_path is not None:
    rolling = pd.read_csv(rolling_path)
else:
    rolling = calculate_rolling_window_summary(equity, [10, 30, 90])

rolling.head()
"""
        ),
        markdown_cell("## Strategy Definitions"),
        code_cell(
            """definition_cols = [
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
definitions[definition_cols]
"""
        ),
        markdown_cell(
            """## Final Holdout Metrics

The key distinction:

- **A_v3_baseline** is exploratory and competition-derived.
- **R1/R2/R4** are clean rediscovered candidates selected without using the final holdout.
- **Ablation baselines** test whether volatility filters matter.
- **Buy & hold** is the passive benchmark.
"""
        ),
        code_cell(
            """display_cols = [
    "strategy_id",
    "strategy_name",
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
    "fees_paid",
    "exposure_time_pct",
    "avg_gross_exposure_pct",
]
display_cols = [c for c in display_cols if c in metrics.columns]

metrics_sorted = metrics.sort_values(
    ["sharpe_ratio", "total_return_pct"],
    ascending=False,
)

metrics_sorted[display_cols]
"""
        ),
        markdown_cell(
            """## Normalized Equity Curves

This chart shows the growth of $1 for each strategy.

The exploratory V3 strategy is expected to look strong, but it should not be treated as clean proof because of selection bias.
The clean rediscovered candidates are the main confirmatory strategies.
"""
        ),
        code_cell(
            """eq = equity.copy()
eq = eq.sort_values(["strategy_id", "time"])
eq["normalized_equity"] = eq.groupby("strategy_id")["equity"].transform(lambda s: s / s.iloc[0])

fig = px.line(
    eq,
    x="time",
    y="normalized_equity",
    color="strategy_id",
    title="Normalized Equity Curves",
    hover_data=["equity", "cash", "unrealized_pnl", "gross_exposure_pct"],
)
fig.update_layout(yaxis_title="Growth of $1")
fig.show()
"""
        ),
        markdown_cell(
            """## Drawdown Curves

Drawdown shows the pain of the strategy.

A strategy with high return and huge drawdown is fragile. The clean candidates are judged not only on return, but also on drawdown control.
"""
        ),
        code_cell(
            """dd = equity.copy()
dd = dd.sort_values(["strategy_id", "time"])
dd["running_max"] = dd.groupby("strategy_id")["equity"].cummax()
dd["drawdown_pct"] = (dd["equity"] / dd["running_max"] - 1.0) * 100.0

fig = px.line(
    dd,
    x="time",
    y="drawdown_pct",
    color="strategy_id",
    title="Drawdown Curves",
    hover_data=["equity", "drawdown_pct"],
)
fig.update_layout(yaxis_title="Drawdown (%)")
fig.show()
"""
        ),
        markdown_cell("## Risk vs Return"),
        code_cell(
            """m = metrics.copy()
m["bubble_size"] = m["sharpe_ratio"].clip(lower=0.1)

fig = px.scatter(
    m,
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
    ],
    title="Risk vs Return",
)
fig.update_traces(textposition="top center")
fig.update_layout(
    xaxis_title="Annualized Volatility (%)",
    yaxis_title="Annualized Return (%)",
)
fig.show()
"""
        ),
        markdown_cell("## Return, Risk, and Risk-Adjusted Metrics"),
        code_cell(
            """metric_cols = [
    "total_return_pct",
    "annualized_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "win_rate_pct",
]
metric_cols = [c for c in metric_cols if c in metrics.columns]

metric_long = metrics.melt(
    id_vars=["strategy_id", "category"],
    value_vars=metric_cols,
    var_name="metric",
    value_name="value",
)

fig = px.bar(
    metric_long,
    x="strategy_id",
    y="value",
    color="metric",
    barmode="group",
    title="Strategy Metrics Comparison",
)
fig.show()
"""
        ),
        markdown_cell(
            """## Rolling Window Analysis

This section answers the competition-style question:

> If the strategy were evaluated over repeated 10-day, 30-day, and 90-day windows, how consistent was it?

A median 10-day return of 0% for a selective strategy usually means many windows had no trades.
"""
        ),
        code_cell(
            """rolling.sort_values(["window_days", "avg_return_pct"], ascending=[True, False])
"""
        ),
        code_cell(
            """def build_daily_rolling_returns(equity, window_days):
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

        rolling_returns = (daily / daily.shift(window_days) - 1.0) * 100.0
        part = rolling_returns.dropna().reset_index()
        part.columns = ["time", "rolling_return_pct"]
        part["strategy_id"] = strategy_id
        part["window_days"] = window_days
        frames.append(part)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

rolling_10 = build_daily_rolling_returns(equity, 10)
rolling_30 = build_daily_rolling_returns(equity, 30)
rolling_90 = build_daily_rolling_returns(equity, 90)
"""
        ),
        code_cell(
            """fig = px.line(
    rolling_10,
    x="time",
    y="rolling_return_pct",
    color="strategy_id",
    title="Rolling 10-Day Returns",
)
fig.show()
"""
        ),
        code_cell(
            """fig = px.line(
    rolling_30,
    x="time",
    y="rolling_return_pct",
    color="strategy_id",
    title="Rolling 30-Day Returns",
)
fig.show()
"""
        ),
        code_cell(
            """fig = px.line(
    rolling_90,
    x="time",
    y="rolling_return_pct",
    color="strategy_id",
    title="Rolling 90-Day Returns",
)
fig.show()
"""
        ),
        markdown_cell(
            """## Trade Return Distribution

This chart shows whether a strategy's performance comes from many small edges or a few large outliers.
"""
        ),
        code_cell(
            """fig = px.box(
    trades,
    x="strategy_id",
    y="trade_return_pct",
    color="strategy_id",
    points="outliers",
    title="Trade Return Distribution",
)
fig.show()
"""
        ),
        markdown_cell("## Asset Contribution"),
        code_cell(
            """fig = px.bar(
    asset,
    x="strategy_id",
    y="net_pnl",
    color="product_id",
    title="Asset Contribution to Net PnL",
    hover_data=["num_trades", "win_rate_pct", "fees_paid"],
)
fig.show()
"""
        ),
        markdown_cell("## Long / Short Contribution"),
        code_cell(
            """side_summary = (
    trades.groupby(["strategy_id", "side"], as_index=False)
    .agg(
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
        win_rate_pct=("net_pnl", lambda s: (s > 0).mean() * 100.0),
        avg_trade_return_pct=("trade_return_pct", "mean"),
        avg_holding_hours=("holding_hours", "mean"),
    )
    .sort_values(["strategy_id", "net_pnl"], ascending=[True, False])
)

fig = px.bar(
    side_summary,
    x="strategy_id",
    y="net_pnl",
    color="side",
    title="Long/Short Net PnL Contribution",
    hover_data=["trades", "win_rate_pct", "avg_trade_return_pct", "avg_holding_hours"],
)
fig.show()

side_summary
"""
        ),
        markdown_cell(
            """## Experiment Log

### Breakout / Donchian / trend strategy family

Rejected. These strategies underperformed, had poor drawdowns, and did not reliably beat buy-and-hold after costs.

### Automatic probability bucket selector

Rejected. The broad automatic selector found unstable buckets and overfit.

### Core2 / V3

Strong exploratory result, but contaminated because it was refined during competition-style research after inspecting intermediate outcomes. Kept as an exploratory benchmark only.

### Development-only rediscovery

Accepted methodology. The final year was reserved as holdout. R1/R2/R4 were selected using only pre-holdout development data.

### Ablation tests

Important evidence. The no-volatility baseline tested whether volatility filtering mattered.

### Buy-and-hold

Passive benchmark. Useful because it shows how much drawdown passive crypto exposure suffered during the holdout.
"""
        ),
        markdown_cell(
            """## Final Interpretation Template

Fill this section after reviewing the charts.

Suggested conclusion:

The project found that the exploratory V3 strategy achieved the strongest performance, but because it was developed through iterative competition-style testing, it is not treated as clean final proof. The clean rediscovered strategies selected only on pre-holdout data produced lower but defensible positive returns on the final year under next-bar execution and fees. The strongest research insight is that volatility filtering materially improved the extreme selloff rebound signal relative to a momentum-only baseline, while upside continuation below BTC trend remained profitable even without a volatility filter.

Limitations:

- historical backtest only,
- no order book simulation,
- slippage sensitivity still limited,
- clean strategies are selective and low-frequency,
- future performance is not guaranteed.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final crypto strategy research notebook."
    )
    parser.add_argument(
        "--backtest-dir",
        type=str,
        default="data/final_strategy_backtests_rediscovered_nextbar",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/final_strategy_report_rediscovered_nextbar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/crypto_strategy_research_report.ipynb"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook(
        backtest_dir=args.backtest_dir,
        report_dir=args.report_dir,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Saved notebook -> {args.output}")


if __name__ == "__main__":
    main()