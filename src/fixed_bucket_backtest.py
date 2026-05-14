"""
Fixed stable-bucket backtest for the probability-table strategy.

Run from project root:

    python -m src.fixed_bucket_backtest --timeframe 15m

Cleaner V2 run with only the two strongest rules:

    python -m src.fixed_bucket_backtest \
        --timeframe 15m \
        --rule-set core2 \
        --initial-equity 100000 \
        --position-pct 1.0 \
        --fee-rate 0.0001 \
        --slippage-rate 0 \
        --test-days 10 \
        --step-days 10 \
        --output-dir data/fixed_bucket_backtests_core2

Rule sets:
    all4:
        LONG  24h lookback, vol 40-50, momentum <-4%
        LONG  24h lookback, vol 60-70, momentum -4% to -2%
        SHORT 24h lookback, vol 40-50, momentum 0.25% to 1%
        SHORT 24h lookback, vol 90-100, momentum >4%

    core2:
        LONG  24h lookback, vol 40-50, momentum <-4%
        SHORT 24h lookback, vol 40-50, momentum 0.25% to 1%

    core2_v3:
        LONG  core2 long rule, but excludes SOL
        SHORT core2 short rule, but excludes XRP

Purpose:
    The broad walk-forward bucket selector overfit.
    The stricter automatic selector improved but still selected unstable buckets.
    This file tests fixed bucket hypotheses across rolling 10-day windows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.probability_table import ProbabilityTableConfig, build_probability_dataset
from src.probability_signal_backtest import (
    ProbabilitySignalBacktestConfig,
    run_backtest_for_product,
)
from src.strategy import load_raw_candles


Side = Literal["long", "short"]
RuleSet = Literal["all4", "core2", "core2_v3"]


@dataclass(frozen=True)
class FixedBucketRule:
    name: str
    side: Side
    lookback_hours: float
    horizon_minutes: int
    vol_bucket: int
    momentum_bucket: str
    description: str
    products: list[str] | None = None


@dataclass(frozen=True)
class FixedBucketBacktestConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    rule_set: RuleSet
    initial_equity: float
    position_pct: float
    fee_rate: float
    slippage_rate: float
    threshold_pct: float
    percentile_window: int
    bucket_size: int
    test_days: int
    step_days: int
    warmup_days: int
    products: list[str] | None
    allow_longs: bool
    allow_shorts: bool


# ─────────────────────────────────────────────────────────────────────────────
# Fixed bucket definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_fixed_rules(rule_set: RuleSet) -> list[FixedBucketRule]:
    """Return fixed bucket rules for the selected rule set."""
    core_rules = [
        FixedBucketRule(
            name="LONG_24h_vol40_50_mom_lt_minus4",
            side="long",
            lookback_hours=24.0,
            horizon_minutes=240,
            vol_bucket=40,
            momentum_bucket="<-4%",
            description="Long after extreme 24h selloff in mid volatility regime.",
        ),
        FixedBucketRule(
            name="SHORT_24h_vol40_50_mom_0p25_to_1",
            side="short",
            lookback_hours=24.0,
            horizon_minutes=240,
            vol_bucket=40,
            momentum_bucket="0.25% to 1%",
            description="Short mild 24h rise in mid volatility regime.",
        ),
    ]

    if rule_set == "core2":
        return core_rules

    if rule_set == "core2_v3":
        return [
            FixedBucketRule(
                name="LONG_24h_vol40_50_mom_lt_minus4_no_SOL",
                side="long",
                lookback_hours=24.0,
                horizon_minutes=240,
                vol_bucket=40,
                momentum_bucket="<-4%",
                description="Core long rule, restricted to BTC, ETH, XRP, and DOGE. Excludes SOL longs.",
                products=["BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD"],
            ),
            FixedBucketRule(
                name="SHORT_24h_vol40_50_mom_0p25_to_1_no_XRP",
                side="short",
                lookback_hours=24.0,
                horizon_minutes=240,
                vol_bucket=40,
                momentum_bucket="0.25% to 1%",
                description="Core short rule, restricted to BTC, ETH, SOL, and DOGE. Excludes XRP shorts.",
                products=["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
            ),
        ]

    if rule_set == "all4":
        return core_rules + [
            FixedBucketRule(
                name="LONG_24h_vol60_70_mom_minus4_to_minus2",
                side="long",
                lookback_hours=24.0,
                horizon_minutes=240,
                vol_bucket=60,
                momentum_bucket="-4% to -2%",
                description="Long after strong 24h selloff in elevated volatility regime.",
            ),
            FixedBucketRule(
                name="SHORT_24h_vol90_100_mom_gt_4",
                side="short",
                lookback_hours=24.0,
                horizon_minutes=240,
                vol_bucket=90,
                momentum_bucket=">4%",
                description="Short extreme 24h pump in maximum volatility regime.",
            ),
        ]

    raise ValueError(f"Unknown rule_set: {rule_set}")


# ─────────────────────────────────────────────────────────────────────────────
# Windows
# ─────────────────────────────────────────────────────────────────────────────

def make_test_windows(
    candles: pd.DataFrame,
    config: FixedBucketBacktestConfig,
) -> pd.DataFrame:
    min_time = candles["time"].min()
    max_time = candles["time"].max()

    first_test_start = min_time + pd.Timedelta(days=config.warmup_days)
    last_test_start = max_time - pd.Timedelta(days=config.test_days)

    windows: list[dict[str, object]] = []
    window_id = 1
    start = first_test_start

    while start <= last_test_start:
        end = start + pd.Timedelta(days=config.test_days)
        windows.append(
            {
                "window_id": window_id,
                "test_start": start,
                "test_end": end,
                "test_days_actual": (end - start).total_seconds() / 86400,
            }
        )
        window_id += 1
        start = start + pd.Timedelta(days=config.step_days)

    if not windows:
        raise ValueError(
            "No test windows created. Download more history or reduce warmup/test days."
        )

    return pd.DataFrame(windows)


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

def build_base_dataset(
    candles: pd.DataFrame,
    config: FixedBucketBacktestConfig,
) -> pd.DataFrame:
    """
    Build the probability dataset used for fixed rules.

    Current fixed rules all use 24h lookback and 240m horizon.
    """
    table_config = ProbabilityTableConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        timeframe=config.timeframe,
        lookback_hours=24.0,
        horizon_minutes=240,
        threshold_pct=config.threshold_pct,
        percentile_window=config.percentile_window,
        min_samples_per_bucket=1,
        bucket_size=config.bucket_size,
    )

    dataset = build_probability_dataset(candles, table_config)
    return dataset.sort_values(["product_id", "time"]).reset_index(drop=True)


def rule_mask(dataset: pd.DataFrame, rule: FixedBucketRule) -> pd.Series:
    mask = (
        (dataset["vol_bucket"].astype("Int64") == rule.vol_bucket)
        & (dataset["return_lookback_bucket"].astype(str) == rule.momentum_bucket)
    )

    if rule.products is not None:
        mask &= dataset["product_id"].isin(rule.products)

    return mask.fillna(False)


def build_fixed_signals(
    dataset: pd.DataFrame,
    rules: list[FixedBucketRule],
    config: FixedBucketBacktestConfig,
) -> pd.DataFrame:
    signal_frames: list[pd.DataFrame] = []

    for rule in rules:
        if rule.side == "long" and not config.allow_longs:
            continue
        if rule.side == "short" and not config.allow_shorts:
            continue

        matches = dataset.loc[rule_mask(dataset, rule)].copy()

        if matches.empty:
            continue

        matches["side"] = rule.side
        matches["signal_key"] = (
            f"{rule.name} | vol={rule.vol_bucket}-{rule.vol_bucket + config.bucket_size} "
            f"| momentum={rule.momentum_bucket}"
        )
        matches["signal_score"] = 1.0
        matches["candidate_table_type"] = "fixed_vol_momentum_bucket"
        matches["candidate_samples"] = np.nan
        matches["candidate_p_up"] = np.nan
        matches["candidate_p_down"] = np.nan
        matches["candidate_up_down_edge"] = np.nan
        matches["candidate_down_up_edge"] = np.nan
        matches["candidate_avg_future_return_pct"] = np.nan
        matches["candidate_lookback_hours"] = rule.lookback_hours
        matches["candidate_horizon_minutes"] = rule.horizon_minutes
        matches["fixed_rule_name"] = rule.name
        matches["fixed_rule_description"] = rule.description

        keep_cols = [
            "time",
            "product_id",
            "side",
            "signal_key",
            "signal_score",
            "candidate_table_type",
            "candidate_samples",
            "candidate_p_up",
            "candidate_p_down",
            "candidate_up_down_edge",
            "candidate_down_up_edge",
            "candidate_avg_future_return_pct",
            "candidate_lookback_hours",
            "candidate_horizon_minutes",
            "lookback_bars",
            "horizon_bars",
            "close",
            "vol_bucket",
            "return_lookback_bucket",
            "return_lookback_pct",
            "realized_vol_percentile",
            "fixed_rule_name",
            "fixed_rule_description",
        ]

        signal_frames.append(matches[keep_cols])

    if not signal_frames:
        return pd.DataFrame()

    signals = pd.concat(signal_frames, ignore_index=True)

    side_priority = {"long": 2, "short": 1}
    signals["side_priority"] = signals["side"].map(side_priority).fillna(0)

    signals = (
        signals.sort_values(
            ["product_id", "time", "side_priority"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["product_id", "time"], keep="first")
        .drop(columns=["side_priority"])
        .sort_values(["product_id", "time"])
        .reset_index(drop=True)
    )

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Backtest by rolling 10-day windows
# ─────────────────────────────────────────────────────────────────────────────

def get_candle_slice(
    candles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return candles[(candles["time"] >= start) & (candles["time"] < end)].copy()


def run_single_window(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    window: pd.Series,
    config: FixedBucketBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_id = int(window["window_id"])
    test_start = window["test_start"]
    test_end = window["test_end"]

    test_candles = get_candle_slice(candles, test_start, test_end)
    test_signals = signals[
        (signals["time"] >= test_start) & (signals["time"] < test_end)
    ].copy()

    if config.products:
        test_candles = test_candles[
            test_candles["product_id"].isin(config.products)
        ].copy()
        test_signals = test_signals[
            test_signals["product_id"].isin(config.products)
        ].copy()

    backtest_config = ProbabilitySignalBacktestConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        timeframe=config.timeframe,
        threshold_pct=config.threshold_pct,
        percentile_window=config.percentile_window,
        bucket_size=config.bucket_size,
        initial_equity=config.initial_equity,
        position_pct=config.position_pct,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        allow_longs=config.allow_longs,
        allow_shorts=config.allow_shorts,
    )

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []

    product_ids = sorted(test_candles["product_id"].dropna().unique())

    for product_id in product_ids:
        trades, equity, metrics = run_backtest_for_product(
            candles=test_candles,
            signals=test_signals,
            product_id=product_id,
            config=backtest_config,
        )

        metrics.update(
            {
                "window_id": window_id,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        if not trades.empty:
            trades["window_id"] = window_id
            trades["test_start"] = test_start
            trades["test_end"] = test_end
            all_trades.append(trades)

        if not equity.empty:
            equity["window_id"] = window_id
            equity["test_start"] = test_start
            equity["test_end"] = test_end
            all_equity.append(equity)

        all_metrics.append(metrics)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)

    return trades_df, equity_df, metrics_df


def run_fixed_bucket_backtest(
    candles: pd.DataFrame,
    config: FixedBucketBacktestConfig,
) -> dict[str, pd.DataFrame]:
    candles = candles.copy().sort_values(["product_id", "time"]).reset_index(drop=True)

    if config.products:
        candles = candles[candles["product_id"].isin(config.products)].copy()

    rules = build_fixed_rules(config.rule_set)
    dataset = build_base_dataset(candles, config)
    signals = build_fixed_signals(dataset, rules, config)
    windows = make_test_windows(candles, config)

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    print(f"Rule set: {config.rule_set}")
    print(f"Rules: {len(rules)}")
    print(f"Built {len(signals)} fixed-rule signals")
    print(f"Created {len(windows)} rolling test windows")

    for _, window in windows.iterrows():
        print(
            f"Window {int(window['window_id'])}: "
            f"{window['test_start']} -> {window['test_end']}"
        )

        trades, equity, metrics = run_single_window(candles, signals, window, config)

        if not trades.empty:
            all_trades.append(trades)
        if not equity.empty:
            all_equity.append(equity)
        if not metrics.empty:
            all_metrics.append(metrics)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

    window_summary = build_window_summary(metrics_df)
    overall_summary = build_overall_summary(window_summary, metrics_df)
    rule_summary = build_rule_summary(trades_df)

    return {
        "rules": pd.DataFrame([rule.__dict__ for rule in rules]),
        "signals": signals,
        "windows": windows,
        "trades": trades_df,
        "equity": equity_df,
        "metrics": metrics_df,
        "window_summary": window_summary,
        "overall_summary": overall_summary,
        "rule_summary": rule_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────────────

def build_window_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    return (
        metrics.groupby("window_id")
        .agg(
            test_start=("test_start", "first"),
            test_end=("test_end", "first"),
            avg_total_return_pct=("total_return_pct", "mean"),
            median_total_return_pct=("total_return_pct", "median"),
            avg_buy_hold_return_pct=("buy_hold_return_pct", "mean"),
            avg_excess_return_pct=("excess_return_vs_buy_hold_pct", "mean"),
            total_trades=("num_trades", "sum"),
            total_long_trades=("long_trades", "sum"),
            total_short_trades=("short_trades", "sum"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_sharpe_ratio=("sharpe_ratio", "mean"),
            avg_exposure_time_pct=("exposure_time_pct", "mean"),
            total_fees=("total_fees", "sum"),
            products_tested=("product_id", "nunique"),
        )
        .reset_index()
    )


def build_overall_summary(
    window_summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    if window_summary.empty:
        return pd.DataFrame()

    row = {
        "num_windows": int(len(window_summary)),
        "positive_window_rate_pct": float(
            (window_summary["avg_total_return_pct"] > 0).mean() * 100.0
        ),
        "beat_buy_hold_window_rate_pct": float(
            (window_summary["avg_excess_return_pct"] > 0).mean() * 100.0
        ),
        "avg_window_return_pct": float(window_summary["avg_total_return_pct"].mean()),
        "median_window_return_pct": float(window_summary["avg_total_return_pct"].median()),
        "best_window_return_pct": float(window_summary["avg_total_return_pct"].max()),
        "worst_window_return_pct": float(window_summary["avg_total_return_pct"].min()),
        "avg_window_buy_hold_return_pct": float(
            window_summary["avg_buy_hold_return_pct"].mean()
        ),
        "avg_window_excess_return_pct": float(
            window_summary["avg_excess_return_pct"].mean()
        ),
        "avg_window_trades": float(window_summary["total_trades"].mean()),
        "avg_window_drawdown_pct": float(window_summary["avg_max_drawdown_pct"].mean()),
        "avg_window_sharpe": float(window_summary["avg_sharpe_ratio"].mean()),
        "total_trades_all_windows": int(window_summary["total_trades"].sum()),
        "total_long_trades_all_windows": int(window_summary["total_long_trades"].sum()),
        "total_short_trades_all_windows": int(window_summary["total_short_trades"].sum()),
        "asset_rows": int(len(metrics)),
    }

    return pd.DataFrame([row])


def build_rule_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    summary = (
        trades.groupby(["side", "signal_key"])
        .agg(
            trades=("net_pnl", "count"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda x: float((x > 0).mean() * 100.0)),
            avg_return_pct=("return_pct", "mean"),
            total_fees=("total_fees", "sum"),
            best_trade=("net_pnl", "max"),
            worst_trade=("net_pnl", "min"),
        )
        .reset_index()
    )

    wins = (
        trades[trades["net_pnl"] > 0]
        .groupby(["side", "signal_key"])["net_pnl"]
        .sum()
    )
    losses = (
        trades[trades["net_pnl"] < 0]
        .groupby(["side", "signal_key"])["net_pnl"]
        .sum()
    )

    idx = summary.set_index(["side", "signal_key"]).index
    summary["gross_profit"] = idx.map(wins).fillna(0.0).to_numpy()
    summary["gross_loss"] = idx.map(losses).fillna(0.0).to_numpy()

    summary["profit_factor"] = np.where(
        summary["gross_loss"] < 0,
        summary["gross_profit"] / summary["gross_loss"].abs(),
        np.where(summary["gross_profit"] > 0, np.inf, 0.0),
    )

    return summary.sort_values("net_pnl", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Save / CLI
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    results: dict[str, pd.DataFrame],
    config: FixedBucketBacktestConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    product_suffix = (
        "all"
        if not config.products
        else "_".join(p.replace("-", "") for p in config.products)
    )

    if config.allow_longs and config.allow_shorts:
        side_suffix = "ls"
    elif config.allow_longs:
        side_suffix = "longonly"
    elif config.allow_shorts:
        side_suffix = "shortonly"
    else:
        side_suffix = "nosides"

    suffix = (
        f"{config.timeframe}_fixed_{config.rule_set}_24h_240m_"
        f"test_{config.test_days}d_step_{config.step_days}d_"
        f"pos_{config.position_pct:g}x_{side_suffix}_{product_suffix}"
    ).replace(".", "p")

    paths = {
        "rules": config.output_dir / f"fixed_bucket_rules_{suffix}.csv",
        "signals": config.output_dir / f"fixed_bucket_signals_{suffix}.csv",
        "windows": config.output_dir / f"fixed_bucket_windows_{suffix}.csv",
        "trades": config.output_dir / f"fixed_bucket_trades_{suffix}.csv",
        "equity": config.output_dir / f"fixed_bucket_equity_{suffix}.csv",
        "metrics": config.output_dir / f"fixed_bucket_metrics_{suffix}.csv",
        "window_summary": config.output_dir / f"fixed_bucket_window_summary_{suffix}.csv",
        "overall_summary": config.output_dir / f"fixed_bucket_overall_summary_{suffix}.csv",
        "rule_summary": config.output_dir / f"fixed_bucket_rule_summary_{suffix}.csv",
    }

    for key, path in paths.items():
        results[key].to_csv(path, index=False)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest fixed stable probability buckets.")

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fixed_bucket_backtests"))
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--rule-set", type=str, choices=["all4", "core2", "core2_v3"], default="all4")

    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)

    parser.add_argument("--threshold-pct", type=float, default=0.30)
    parser.add_argument("--percentile-window", type=int, default=200)
    parser.add_argument("--bucket-size", type=int, default=10)

    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--warmup-days", type=int, default=30)

    parser.add_argument("--products", nargs="+", default=None)
    parser.add_argument("--no-longs", action="store_true")
    parser.add_argument("--no-shorts", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FixedBucketBacktestConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        rule_set=args.rule_set,
        initial_equity=args.initial_equity,
        position_pct=args.position_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        test_days=args.test_days,
        step_days=args.step_days,
        warmup_days=args.warmup_days,
        products=args.products,
        allow_longs=not args.no_longs,
        allow_shorts=not args.no_shorts,
    )

    if not config.allow_longs and not config.allow_shorts:
        raise ValueError("Both longs and shorts are disabled. Enable at least one side.")

    candles = load_raw_candles(config.input_dir, config.timeframe)
    results = run_fixed_bucket_backtest(candles, config)
    paths = save_outputs(results, config)

    print("\nFixed-bucket overall summary:")
    if results["overall_summary"].empty:
        print("No overall summary generated.")
    else:
        print(results["overall_summary"].to_string(index=False))

    print("\nFixed-bucket window summary:")
    if results["window_summary"].empty:
        print("No window summary generated.")
    else:
        display_cols = [
            "window_id",
            "test_start",
            "test_end",
            "avg_total_return_pct",
            "avg_buy_hold_return_pct",
            "avg_excess_return_pct",
            "total_trades",
            "total_long_trades",
            "total_short_trades",
            "avg_win_rate_pct",
            "avg_profit_factor",
            "avg_max_drawdown_pct",
            "avg_sharpe_ratio",
        ]
        display_cols = [col for col in display_cols if col in results["window_summary"].columns]
        print(results["window_summary"][display_cols].to_string(index=False))

    print("\nFixed-bucket rule summary:")
    if results["rule_summary"].empty:
        print("No rule summary generated.")
    else:
        display_cols = [
            "side",
            "signal_key",
            "trades",
            "net_pnl",
            "win_rate_pct",
            "profit_factor",
            "avg_return_pct",
            "worst_trade",
        ]
        display_cols = [col for col in display_cols if col in results["rule_summary"].columns]
        print(results["rule_summary"][display_cols].to_string(index=False))

    print("\nSaved fixed-bucket outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
