from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_single_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[-1]


def load_inputs(backtest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = find_single_file(backtest_dir, "final_strategy_metrics_*.csv")
    equity_path = find_single_file(backtest_dir, "final_strategy_equity_*.csv")
    trades_path = find_single_file(backtest_dir, "final_strategy_trades_*.csv")
    asset_path = find_single_file(backtest_dir, "final_strategy_asset_contributions_*.csv")

    metrics = pd.read_csv(metrics_path)
    equity = pd.read_csv(equity_path, parse_dates=["time"])
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"]) if trades_path.exists() else pd.DataFrame()
    asset = pd.read_csv(asset_path) if asset_path.exists() else pd.DataFrame()

    return metrics, equity, trades, asset


def plot_equity_curves(equity: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(14, 8))

    for strategy_id, group in equity.groupby("strategy_id"):
        group = group.sort_values("time")
        if group.empty:
            continue
        normalized = group["equity"] / group["equity"].iloc[0]
        plt.plot(group["time"], normalized, label=strategy_id)

    plt.title("Normalized Equity Curves")
    plt.xlabel("Time")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves_normalized.png", dpi=200)
    plt.close()


def plot_drawdown_curves(equity: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(14, 8))

    for strategy_id, group in equity.groupby("strategy_id"):
        group = group.sort_values("time")
        if group.empty:
            continue
        dd = group["equity"] / group["equity"].cummax() - 1.0
        plt.plot(group["time"], dd * 100.0, label=strategy_id)

    plt.title("Drawdown Curves")
    plt.xlabel("Time")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown_curves.png", dpi=200)
    plt.close()


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    metrics = metrics.copy().sort_values("sharpe_ratio", ascending=False)

    charts = [
        ("total_return_pct", "Total Return (%)", "metric_total_return.png"),
        ("annualized_return_pct", "Annualized Return (%)", "metric_annualized_return.png"),
        ("annualized_volatility_pct", "Annualized Volatility (%)", "metric_annualized_volatility.png"),
        ("max_drawdown_pct", "Max Drawdown (%)", "metric_max_drawdown.png"),
        ("sharpe_ratio", "Sharpe Ratio", "metric_sharpe.png"),
        ("sortino_ratio", "Sortino Ratio", "metric_sortino.png"),
        ("calmar_ratio", "Calmar Ratio", "metric_calmar.png"),
        ("profit_factor", "Profit Factor", "metric_profit_factor.png"),
        ("win_rate_pct", "Win Rate (%)", "metric_win_rate.png"),
        ("num_trades", "Number of Trades", "metric_num_trades.png"),
    ]

    for col, title, filename in charts:
        if col not in metrics.columns:
            continue

        plt.figure(figsize=(12, 6))
        plt.bar(metrics["strategy_id"], metrics[col])
        plt.title(title)
        plt.xlabel("Strategy")
        plt.ylabel(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def plot_risk_return_scatter(metrics: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 8))

    x = metrics["annualized_volatility_pct"]
    y = metrics["annualized_return_pct"]

    plt.scatter(x, y)

    for _, row in metrics.iterrows():
        plt.annotate(row["strategy_id"], (row["annualized_volatility_pct"], row["annualized_return_pct"]))

    plt.title("Risk vs Return")
    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_return_scatter.png", dpi=200)
    plt.close()


def build_daily_equity(equity: pd.DataFrame) -> pd.DataFrame:
    daily_frames = []

    for strategy_id, group in equity.groupby("strategy_id"):
        part = (
            group.sort_values("time")
            .set_index("time")[["equity"]]
            .resample("1D")
            .last()
            .dropna()
            .rename(columns={"equity": strategy_id})
        )
        daily_frames.append(part)

    if not daily_frames:
        return pd.DataFrame()

    daily = pd.concat(daily_frames, axis=1)
    return daily


def plot_rolling_returns(equity: pd.DataFrame, output_dir: Path, window_days: int) -> None:
    daily = build_daily_equity(equity)
    if daily.empty:
        return

    plt.figure(figsize=(14, 8))

    for strategy_id in daily.columns:
        series = daily[strategy_id].dropna()
        if len(series) <= window_days:
            continue
        rolling = series / series.shift(window_days) - 1.0
        plt.plot(rolling.index, rolling * 100.0, label=strategy_id)

    plt.title(f"Rolling {window_days}-Day Returns")
    plt.xlabel("Time")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"rolling_{window_days}d_returns.png", dpi=200)
    plt.close()


def plot_trade_return_boxplot(trades: pd.DataFrame, output_dir: Path) -> None:
    if trades.empty:
        return

    grouped = trades.groupby("strategy_id")["trade_return_pct"]
    labels = []
    values = []

    for strategy_id, series in grouped:
        labels.append(strategy_id)
        values.append(series.dropna().values)

    if not values:
        return

    plt.figure(figsize=(12, 6))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.title("Trade Return Distribution by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Trade Return (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "trade_return_boxplot.png", dpi=200)
    plt.close()


def plot_asset_contribution(asset: pd.DataFrame, output_dir: Path) -> None:
    if asset.empty:
        return

    pivot = asset.pivot(index="strategy_id", columns="product_id", values="net_pnl").fillna(0.0)
    pivot = pivot.sort_index()

    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(pivot))

    for col in pivot.columns:
        plt.bar(pivot.index, pivot[col].values, bottom=bottom, label=col)
        bottom = bottom + pivot[col].values

    plt.title("Asset Contribution to Net PnL")
    plt.xlabel("Strategy")
    plt.ylabel("Net PnL")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "asset_contribution_stacked.png", dpi=200)
    plt.close()


def save_summary_tables(
    metrics: pd.DataFrame,
    trades: pd.DataFrame,
    asset: pd.DataFrame,
    output_dir: Path,
) -> None:
    metrics_sorted = metrics.sort_values(["sharpe_ratio", "total_return_pct"], ascending=False).copy()

    keep_cols = [
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
        "avg_trade_return_pct",
        "median_trade_return_pct",
        "best_trade_return_pct",
        "worst_trade_return_pct",
        "profit_factor",
        "fees_paid",
        "avg_holding_hours",
        "exposure_time_pct",
        "avg_gross_exposure_pct",
        "top_asset_by_pnl",
        "top_asset_pnl",
    ]
    keep_cols = [c for c in keep_cols if c in metrics_sorted.columns]

    metrics_sorted[keep_cols].to_csv(output_dir / "strategy_summary_table.csv", index=False)

    if not trades.empty:
        trade_summary = (
            trades.groupby("strategy_id", as_index=False)
            .agg(
                num_trades=("strategy_id", "count"),
                avg_trade_return_pct=("trade_return_pct", "mean"),
                median_trade_return_pct=("trade_return_pct", "median"),
                best_trade_return_pct=("trade_return_pct", "max"),
                worst_trade_return_pct=("trade_return_pct", "min"),
                gross_pnl=("gross_pnl", "sum"),
                total_fees=("total_fees", "sum"),
                net_pnl=("net_pnl", "sum"),
            )
        )
        trade_summary.to_csv(output_dir / "trade_summary_table.csv", index=False)

    if not asset.empty:
        asset.sort_values(["strategy_id", "net_pnl"], ascending=[True, False]).to_csv(
            output_dir / "asset_contribution_table.csv",
            index=False,
        )

    markdown_path = output_dir / "final_strategy_report_summary.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# Final Strategy Holdout Report\n\n")
        f.write("## Ranked Summary\n\n")
        f.write(metrics_sorted[keep_cols].to_markdown(index=False))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visuals and summary tables for final strategy comparison.")
    parser.add_argument("--backtest-dir", type=Path, default=Path("data/final_strategy_backtests"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/final_strategy_report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics, equity, trades, asset = load_inputs(args.backtest_dir)

    plot_equity_curves(equity, args.output_dir)
    plot_drawdown_curves(equity, args.output_dir)
    plot_metric_bars(metrics, args.output_dir)
    plot_risk_return_scatter(metrics, args.output_dir)
    plot_rolling_returns(equity, args.output_dir, 10)
    plot_rolling_returns(equity, args.output_dir, 30)
    plot_rolling_returns(equity, args.output_dir, 90)
    plot_trade_return_boxplot(trades, args.output_dir)
    plot_asset_contribution(asset, args.output_dir)
    save_summary_tables(metrics, trades, asset, args.output_dir)

    print("Saved report outputs to:", args.output_dir)


if __name__ == "__main__":
    main()