from __future__ import annotations

import argparse
import json
from pathlib import Path


def nb_cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None if cell_type == "code" else None,
    }


def markdown(source: str) -> dict:
    return nb_cell("markdown", source)


def code(source: str) -> dict:
    cell = nb_cell("code", source)
    cell["execution_count"] = None
    return cell


def build_notebook(backtest_dir: str) -> dict:
    cells = [
        markdown(
            """# Final Crypto Strategy Report

This notebook compares frozen non-ML crypto strategies on a one-year final holdout.

The evaluation emphasizes not only raw return, but also:

- drawdown
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- profit factor
- win rate
- trade frequency
- fees
- exposure
- asset contribution

The backtest uses realistic next-bar execution:

> Signal forms at candle close. Entry occurs at the next candle open.
"""
        ),
        code(
            f"""from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

BACKTEST_DIR = Path(r"{backtest_dir}")

def latest(pattern):
    files = sorted(BACKTEST_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]

metrics_path = latest("final_strategy_metrics_*.csv")
equity_path = latest("final_strategy_equity_*.csv")
trades_path = latest("final_strategy_trades_*.csv")
asset_path = latest("final_strategy_asset_contributions_*.csv")
defs_path = latest("final_strategy_definitions_*.csv")

metrics = pd.read_csv(metrics_path)
equity = pd.read_csv(equity_path, parse_dates=["time"])
trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
asset = pd.read_csv(asset_path)
definitions = pd.read_csv(defs_path)

metrics_path, equity_path, trades_path, asset_path, defs_path
"""
        ),
        markdown("## Strategy Definitions"),
        code(
            """definitions[[
    "strategy_id",
    "name",
    "category",
    "side",
    "lookback_hours",
    "horizon_minutes",
    "description",
]]"""
        ),
        markdown("## Final Holdout Metrics"),
        code(
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

metrics_sorted = metrics.sort_values(
    ["sharpe_ratio", "total_return_pct"],
    ascending=False,
)

metrics_sorted[display_cols]"""
        ),
        markdown(
            """## Normalized Equity Curves

This chart shows the growth of $1 invested in each strategy over the final holdout period.
"""
        ),
        code(
            """eq = equity.copy()
eq = eq.sort_values(["strategy_id", "time"])
eq["normalized_equity"] = eq.groupby("strategy_id")["equity"].transform(lambda s: s / s.iloc[0])

fig = px.line(
    eq,
    x="time",
    y="normalized_equity",
    color="strategy_id",
    title="Normalized Equity Curves",
)
fig.show()
"""
        ),
        markdown(
            """## Drawdown Curves

Drawdown is often more important than return. A strategy that makes money but suffers huge drawdowns is fragile.
"""
        ),
        code(
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
)
fig.show()
"""
        ),
        markdown("## Risk vs Return"),
        code(
            """fig = px.scatter(
    metrics,
    x="annualized_volatility_pct",
    y="annualized_return_pct",
    color="strategy_id",
    size=metrics["sharpe_ratio"].clip(lower=0.1),
    hover_data=[
        "strategy_name",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown_pct",
        "profit_factor",
        "num_trades",
    ],
    title="Risk vs Return",
)
fig.show()
"""
        ),
        markdown("## Risk-Adjusted Metrics"),
        code(
            """risk_metrics = metrics.melt(
    id_vars=["strategy_id"],
    value_vars=[
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor",
    ],
    var_name="metric",
    value_name="value",
)

fig = px.bar(
    risk_metrics,
    x="strategy_id",
    y="value",
    color="metric",
    barmode="group",
    title="Risk-Adjusted Metrics",
)
fig.show()
"""
        ),
        markdown("## Return and Drawdown Comparison"),
        code(
            """comparison = metrics.melt(
    id_vars=["strategy_id"],
    value_vars=[
        "total_return_pct",
        "annualized_return_pct",
        "max_drawdown_pct",
    ],
    var_name="metric",
    value_name="value",
)

fig = px.bar(
    comparison,
    x="strategy_id",
    y="value",
    color="metric",
    barmode="group",
    title="Return and Drawdown Comparison",
)
fig.show()
"""
        ),
        markdown("## Rolling Returns"),
        code(
            """daily = (
    equity.sort_values("time")
    .set_index("time")
    .groupby("strategy_id")["equity"]
    .resample("1D")
    .last()
    .dropna()
    .reset_index()
)

def add_rolling_return(df, window):
    out = df.copy()
    out[f"rolling_{window}d_return_pct"] = (
        out.groupby("strategy_id")["equity"]
        .transform(lambda s: (s / s.shift(window) - 1.0) * 100.0)
    )
    return out

daily_30 = add_rolling_return(daily, 30)

fig = px.line(
    daily_30.dropna(subset=["rolling_30d_return_pct"]),
    x="time",
    y="rolling_30d_return_pct",
    color="strategy_id",
    title="Rolling 30-Day Returns",
)
fig.show()
"""
        ),
        code(
            """daily_90 = add_rolling_return(daily, 90)

fig = px.line(
    daily_90.dropna(subset=["rolling_90d_return_pct"]),
    x="time",
    y="rolling_90d_return_pct",
    color="strategy_id",
    title="Rolling 90-Day Returns",
)
fig.show()
"""
        ),
        markdown("## Trade Return Distribution"),
        code(
            """fig = px.box(
    trades,
    x="strategy_id",
    y="trade_return_pct",
    color="strategy_id",
    points=False,
    title="Trade Return Distribution by Strategy",
)
fig.show()
"""
        ),
        markdown("## Asset Contribution"),
        code(
            """fig = px.bar(
    asset,
    x="strategy_id",
    y="net_pnl",
    color="product_id",
    title="Asset Contribution to Net PnL",
)
fig.show()
"""
        ),
        markdown("## Trade Summary"),
        code(
            """trade_summary = (
    trades.groupby(["strategy_id", "side"], as_index=False)
    .agg(
        trades=("net_pnl", "count"),
        net_pnl=("net_pnl", "sum"),
        avg_trade_return_pct=("trade_return_pct", "mean"),
        median_trade_return_pct=("trade_return_pct", "median"),
        win_rate_pct=("net_pnl", lambda s: (s > 0).mean() * 100),
        avg_holding_hours=("holding_hours", "mean"),
        fees_paid=("total_fees", "sum"),
    )
    .sort_values(["strategy_id", "net_pnl"], ascending=[True, False])
)

trade_summary"""
        ),
        markdown("## Key Interpretation Template"),
        markdown(
            """Use this section to write the final interpretation.

Suggested structure:

1. Which strategy had the best total return?
2. Which strategy had the best risk-adjusted return?
3. Which strategy had the lowest drawdown?
4. Did volatility-conditioned strategies outperform momentum-only?
5. Did the strategies beat buy-and-hold?
6. Which assets contributed most to PnL?
7. What are the main limitations?
8. What would be improved next?

Important limitation:

> This is still a historical backtest. It is not evidence of guaranteed future performance.
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
        description="Generate an interactive Jupyter notebook for final strategy results."
    )
    parser.add_argument(
        "--backtest-dir",
        type=str,
        default="data/final_strategy_backtests_nextbar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/final_strategy_report.ipynb"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook(args.backtest_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Saved notebook -> {args.output}")


if __name__ == "__main__":
    main()