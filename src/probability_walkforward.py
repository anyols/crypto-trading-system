"""
Walk-forward validation for table-driven probability signals.

Run from project root:

    python -m src.probability_walkforward --timeframe 15m

Competition-style example:

    python -m src.probability_walkforward \
        --timeframe 15m \
        --train-days 180 \
        --test-days 10 \
        --step-days 10 \
        --lookback-hours-list 4 6 12 24 \
        --horizon-minutes-list 240 \
        --fee-rate 0.0001 \
        --slippage-rate 0 \
        --initial-equity 100000 \
        --position-pct 1.0

Purpose:
    The previous probability-signal backtest was in-sample:
        1. It found the best probability buckets on the full dataset.
        2. It backtested those same buckets on the full dataset.

    That is useful for discovery, but it is optimistic.

    This file performs walk-forward validation:
        1. Train on older historical data.
        2. Select bullish/bearish probability buckets from that train period only.
        3. Test those buckets on the next unseen 10-15 day window.
        4. Repeat across time.

    This is much closer to the actual competition situation.

Important:
    This is still an independent-per-asset backtest, not a single-account portfolio
    simulator. It validates signal quality across assets and time windows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.probability_table import (
    ProbabilityTableConfig,
    build_probability_dataset,
    build_signal_candidates,
    summarize_by_momentum_bucket,
    summarize_by_vol_and_momentum_bucket,
    summarize_by_vol_bucket,
)
from src.probability_signal_backtest import (
    ProbabilitySignalBacktestConfig,
    build_signals_for_candidates,
    run_backtest_for_product,
)
from src.strategy import load_raw_candles


@dataclass(frozen=True)
class WalkForwardConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str

    train_days: int
    test_days: int
    step_days: int
    warmup_days: int

    lookback_hours_list: list[float]
    horizon_minutes_list: list[int]
    threshold_pct: float
    percentile_window: int
    bucket_size: int

    table_types: list[str]
    min_samples: int
    min_edge: float
    min_avg_return_pct: float
    top_n_per_side: int

    initial_equity: float
    position_pct: float
    fee_rate: float
    slippage_rate: float
    allow_longs: bool
    allow_shorts: bool


# ─────────────────────────────────────────────────────────────────────────────
# Window creation
# ─────────────────────────────────────────────────────────────────────────────

def make_walkforward_windows(
    candles: pd.DataFrame,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """
    Create chronological train/test windows.

    Example with train_days=180, test_days=10, step_days=10:
        train: day 0-180, test: day 180-190
        train: day 10-190, test: day 190-200
        train: day 20-200, test: day 200-210
        ...
    """
    min_time = candles["time"].min()
    max_time = candles["time"].max()

    first_test_start = min_time + pd.Timedelta(days=config.train_days)
    last_possible_test_start = max_time - pd.Timedelta(days=config.test_days)

    windows: list[dict[str, object]] = []
    test_start = first_test_start
    window_id = 1

    while test_start <= last_possible_test_start:
        test_end = test_start + pd.Timedelta(days=config.test_days)
        train_end = test_start
        train_start = train_end - pd.Timedelta(days=config.train_days)

        if train_start < min_time:
            train_start = min_time

        windows.append(
            {
                "window_id": window_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_days_actual": (train_end - train_start).total_seconds() / 86400,
                "test_days_actual": (test_end - test_start).total_seconds() / 86400,
            }
        )

        window_id += 1
        test_start = test_start + pd.Timedelta(days=config.step_days)

    if not windows:
        raise ValueError(
            "No walk-forward windows could be created. "
            "Use fewer train_days/test_days or download more history."
        )

    return pd.DataFrame(windows)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection from train period only
# ─────────────────────────────────────────────────────────────────────────────

def build_train_candidates_for_combo(
    train_candles: pd.DataFrame,
    config: WalkForwardConfig,
    lookback_hours: float,
    horizon_minutes: int,
) -> pd.DataFrame:
    """
    Build candidate states from one train-period lookback/horizon combo.
    """
    table_config = ProbabilityTableConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        timeframe=config.timeframe,
        lookback_hours=lookback_hours,
        horizon_minutes=horizon_minutes,
        threshold_pct=config.threshold_pct,
        percentile_window=config.percentile_window,
        min_samples_per_bucket=config.min_samples,
        bucket_size=config.bucket_size,
    )

    dataset = build_probability_dataset(train_candles, table_config)

    if dataset.empty:
        return pd.DataFrame()

    vol_table = summarize_by_vol_bucket(dataset, table_config)
    momentum_table = summarize_by_momentum_bucket(dataset, table_config)
    vol_momentum_table = summarize_by_vol_and_momentum_bucket(dataset, table_config)

    candidates = build_signal_candidates(
        vol_table=vol_table,
        momentum_table=momentum_table,
        vol_momentum_table=vol_momentum_table,
        config=table_config,
    )

    return candidates


def select_candidates_from_train(
    train_candles: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select bullish and bearish candidates using the train period only.
    """
    candidate_frames: list[pd.DataFrame] = []

    for lookback_hours in config.lookback_hours_list:
        for horizon_minutes in config.horizon_minutes_list:
            try:
                candidates = build_train_candidates_for_combo(
                    train_candles=train_candles,
                    config=config,
                    lookback_hours=lookback_hours,
                    horizon_minutes=horizon_minutes,
                )
            except ValueError as exc:
                print(
                    f"Skipping lookback={lookback_hours:g}h horizon={horizon_minutes}m: {exc}"
                )
                continue

            if not candidates.empty:
                candidate_frames.append(candidates)

    if not candidate_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    filtered = all_candidates[all_candidates["table_type"].isin(config.table_types)].copy()

    bullish = filtered[
        (filtered["samples"] >= config.min_samples)
        & (filtered["up_down_edge"] >= config.min_edge)
        & (filtered["avg_future_return_pct"] >= config.min_avg_return_pct)
    ].copy()

    bearish = filtered[
        (filtered["samples"] >= config.min_samples)
        & (filtered["down_up_edge"] >= config.min_edge)
        & (filtered["avg_future_return_pct"] <= -config.min_avg_return_pct)
    ].copy()

    if not bullish.empty:
        bullish = (
            bullish.sort_values(
                ["bull_score_weighted", "up_down_edge", "avg_future_return_pct", "samples"],
                ascending=False,
            )
            .head(config.top_n_per_side)
            .copy()
        )
        bullish["side"] = "long"

    if not bearish.empty:
        bearish = (
            bearish.sort_values(
                ["bear_score_weighted", "down_up_edge", "avg_future_return_pct", "samples"],
                ascending=[False, False, True, False],
            )
            .head(config.top_n_per_side)
            .copy()
        )
        bearish["side"] = "short"

    return bullish.reset_index(drop=True), bearish.reset_index(drop=True), all_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Test window evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_candle_slice(
    candles: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return candles[(candles["time"] >= start) & (candles["time"] < end)].copy()


def build_test_signals(
    candles: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    window: pd.Series,
    config: WalkForwardConfig,
    backtest_config: ProbabilitySignalBacktestConfig,
) -> pd.DataFrame:
    """
    Build signals for the test window.

    We include warmup candles before the test window so rolling volatility and
    percentile features are available at the start of the test.

    We also include horizon padding after the test window so build_probability_dataset
    does not drop the final test rows due to missing future_return. The future return
    is not used for signal generation; it only prevents row dropping inside the shared
    dataset builder.
    """
    if selected_candidates.empty:
        return pd.DataFrame()

    test_start = window["test_start"]
    test_end = window["test_end"]

    max_horizon_minutes = int(max(selected_candidates["horizon_minutes"].max(), 0))

    context_start = test_start - pd.Timedelta(days=config.warmup_days)
    context_end = test_end + pd.Timedelta(minutes=max_horizon_minutes)

    context_candles = get_candle_slice(candles, context_start, context_end)

    if context_candles.empty:
        return pd.DataFrame()

    signals = build_signals_for_candidates(
        candles=context_candles,
        selected_candidates=selected_candidates,
        backtest_config=backtest_config,
    )

    if signals.empty:
        return signals

    signals = signals[
        (signals["time"] >= test_start)
        & (signals["time"] < test_end)
    ].copy()

    return signals.reset_index(drop=True)


def run_single_walkforward_window(
    candles: pd.DataFrame,
    window: pd.Series,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train candidate buckets on the train period and test them on the unseen test period.
    """
    window_id = int(window["window_id"])

    train_candles = get_candle_slice(candles, window["train_start"], window["train_end"])
    test_candles = get_candle_slice(candles, window["test_start"], window["test_end"])

    bullish, bearish, all_train_candidates = select_candidates_from_train(
        train_candles=train_candles,
        config=config,
    )

    selected = pd.concat(
        [
            bullish if config.allow_longs else pd.DataFrame(),
            bearish if config.allow_shorts else pd.DataFrame(),
        ],
        ignore_index=True,
    )

    for df in [bullish, bearish, all_train_candidates, selected]:
        if not df.empty:
            df["window_id"] = window_id
            df["train_start"] = window["train_start"]
            df["train_end"] = window["train_end"]
            df["test_start"] = window["test_start"]
            df["test_end"] = window["test_end"]

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

    if selected.empty:
        metrics = build_empty_window_metrics(candles=test_candles, window=window, config=config)
        return pd.DataFrame(), pd.DataFrame(), metrics, selected, pd.DataFrame()

    signals = build_test_signals(
        candles=candles,
        selected_candidates=selected,
        window=window,
        config=config,
        backtest_config=backtest_config,
    )

    if signals.empty:
        metrics = build_empty_window_metrics(candles=test_candles, window=window, config=config)
        return pd.DataFrame(), pd.DataFrame(), metrics, selected, signals

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float | int | str | pd.Timestamp]] = []

    product_ids = sorted(test_candles["product_id"].dropna().unique())

    for product_id in product_ids:
        trades, equity, metrics = run_backtest_for_product(
            candles=test_candles,
            signals=signals,
            product_id=product_id,
            config=backtest_config,
        )

        metrics.update(
            {
                "window_id": window_id,
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "selected_bullish_candidates": len(bullish),
                "selected_bearish_candidates": len(bearish),
            }
        )

        if not trades.empty:
            trades["window_id"] = window_id
            trades["train_start"] = window["train_start"]
            trades["train_end"] = window["train_end"]
            trades["test_start"] = window["test_start"]
            trades["test_end"] = window["test_end"]
            all_trades.append(trades)

        if not equity.empty:
            equity["window_id"] = window_id
            equity["train_start"] = window["train_start"]
            equity["train_end"] = window["train_end"]
            equity["test_start"] = window["test_start"]
            equity["test_end"] = window["test_end"]
            all_equity.append(equity)

        all_metrics.append(metrics)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)

    return trades_df, equity_df, metrics_df, selected, signals


def build_empty_window_metrics(
    candles: pd.DataFrame,
    window: pd.Series,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for product_id in sorted(candles["product_id"].dropna().unique()):
        rows.append(
            {
                "product_id": product_id,
                "window_id": int(window["window_id"]),
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "initial_equity": config.initial_equity,
                "ending_equity": config.initial_equity,
                "total_return_pct": 0.0,
                "buy_hold_return_pct": 0.0,
                "excess_return_vs_buy_hold_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "exposure_time_pct": 0.0,
                "num_trades": 0,
                "long_trades": 0,
                "short_trades": 0,
                "win_rate_pct": 0.0,
                "avg_trade_return_pct": 0.0,
                "profit_factor": 0.0,
                "avg_holding_bars": 0.0,
                "total_fees": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "selected_bullish_candidates": 0,
                "selected_bearish_candidates": 0,
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def build_window_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    summary = (
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
            selected_bullish_candidates=("selected_bullish_candidates", "max"),
            selected_bearish_candidates=("selected_bearish_candidates", "max"),
            products_tested=("product_id", "nunique"),
        )
        .reset_index()
    )

    return summary


def build_overall_summary(window_summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if window_summary.empty:
        return pd.DataFrame()

    positive_windows = (window_summary["avg_total_return_pct"] > 0).mean() * 100.0
    beat_buy_hold_windows = (window_summary["avg_excess_return_pct"] > 0).mean() * 100.0

    row = {
        "num_windows": int(len(window_summary)),
        "positive_window_rate_pct": float(positive_windows),
        "beat_buy_hold_window_rate_pct": float(beat_buy_hold_windows),
        "avg_window_return_pct": float(window_summary["avg_total_return_pct"].mean()),
        "median_window_return_pct": float(window_summary["avg_total_return_pct"].median()),
        "best_window_return_pct": float(window_summary["avg_total_return_pct"].max()),
        "worst_window_return_pct": float(window_summary["avg_total_return_pct"].min()),
        "avg_window_buy_hold_return_pct": float(window_summary["avg_buy_hold_return_pct"].mean()),
        "avg_window_excess_return_pct": float(window_summary["avg_excess_return_pct"].mean()),
        "avg_window_trades": float(window_summary["total_trades"].mean()),
        "avg_window_drawdown_pct": float(window_summary["avg_max_drawdown_pct"].mean()),
        "avg_window_sharpe": float(window_summary["avg_sharpe_ratio"].mean()),
        "total_trades_all_windows": int(window_summary["total_trades"].sum()),
        "total_long_trades_all_windows": int(window_summary["total_long_trades"].sum()),
        "total_short_trades_all_windows": int(window_summary["total_short_trades"].sum()),
        "avg_selected_bullish_candidates": float(window_summary["selected_bullish_candidates"].mean()),
        "avg_selected_bearish_candidates": float(window_summary["selected_bearish_candidates"].mean()),
        "asset_rows": int(len(metrics)),
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_walkforward(candles: pd.DataFrame, config: WalkForwardConfig) -> dict[str, pd.DataFrame]:
    candles = candles.copy().sort_values(["product_id", "time"]).reset_index(drop=True)
    windows = make_walkforward_windows(candles, config)

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_selected_candidates: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []

    print(f"Created {len(windows)} walk-forward windows")

    for _, window in windows.iterrows():
        print(
            f"Window {int(window['window_id'])}: "
            f"train {window['train_start']} -> {window['train_end']} | "
            f"test {window['test_start']} -> {window['test_end']}"
        )

        trades, equity, metrics, selected_candidates, signals = run_single_walkforward_window(
            candles=candles,
            window=window,
            config=config,
        )

        if not trades.empty:
            all_trades.append(trades)
        if not equity.empty:
            all_equity.append(equity)
        if not metrics.empty:
            all_metrics.append(metrics)
        if not selected_candidates.empty:
            all_selected_candidates.append(selected_candidates)
        if not signals.empty:
            signals["window_id"] = int(window["window_id"])
            signals["test_start"] = window["test_start"]
            signals["test_end"] = window["test_end"]
            all_signals.append(signals)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    selected_candidates_df = (
        pd.concat(all_selected_candidates, ignore_index=True)
        if all_selected_candidates
        else pd.DataFrame()
    )
    signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()

    window_summary = build_window_summary(metrics_df)
    overall_summary = build_overall_summary(window_summary, metrics_df)

    return {
        "windows": windows,
        "trades": trades_df,
        "equity": equity_df,
        "metrics": metrics_df,
        "selected_candidates": selected_candidates_df,
        "signals": signals_df,
        "window_summary": window_summary,
        "overall_summary": overall_summary,
    }


def save_outputs(results: dict[str, pd.DataFrame], config: WalkForwardConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"{config.timeframe}_train_{config.train_days}d_"
        f"test_{config.test_days}d_step_{config.step_days}d_"
        f"pos_{config.position_pct:g}x"
    ).replace(".", "p")

    paths = {
        "windows": config.output_dir / f"walkforward_windows_{suffix}.csv",
        "trades": config.output_dir / f"walkforward_trades_{suffix}.csv",
        "equity": config.output_dir / f"walkforward_equity_{suffix}.csv",
        "metrics": config.output_dir / f"walkforward_metrics_{suffix}.csv",
        "selected_candidates": config.output_dir / f"walkforward_selected_candidates_{suffix}.csv",
        "signals": config.output_dir / f"walkforward_signals_{suffix}.csv",
        "window_summary": config.output_dir / f"walkforward_window_summary_{suffix}.csv",
        "overall_summary": config.output_dir / f"walkforward_overall_summary_{suffix}.csv",
    }

    for key, path in paths.items():
        results[key].to_csv(path, index=False)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for probability-table signals.")

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/probability_walkforward"))
    parser.add_argument("--timeframe", type=str, default="15m")

    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--warmup-days", type=int, default=30)

    parser.add_argument("--lookback-hours-list", type=float, nargs="+", default=[4.0, 6.0, 12.0, 24.0])
    parser.add_argument("--horizon-minutes-list", type=int, nargs="+", default=[240])
    parser.add_argument("--threshold-pct", type=float, default=0.30)
    parser.add_argument("--percentile-window", type=int, default=200)
    parser.add_argument("--bucket-size", type=int, default=10)

    parser.add_argument(
        "--table-types",
        nargs="+",
        default=["vol_momentum_bucket"],
        choices=["vol_bucket", "momentum_bucket", "vol_momentum_bucket"],
    )
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--min-edge", type=float, default=0.12)
    parser.add_argument("--min-avg-return-pct", type=float, default=0.15)
    parser.add_argument("--top-n-per-side", type=int, default=50)

    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--no-longs", action="store_true")
    parser.add_argument("--no-shorts", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = WalkForwardConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        warmup_days=args.warmup_days,
        lookback_hours_list=args.lookback_hours_list,
        horizon_minutes_list=args.horizon_minutes_list,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        table_types=args.table_types,
        min_samples=args.min_samples,
        min_edge=args.min_edge,
        min_avg_return_pct=args.min_avg_return_pct,
        top_n_per_side=args.top_n_per_side,
        initial_equity=args.initial_equity,
        position_pct=args.position_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        allow_longs=not args.no_longs,
        allow_shorts=not args.no_shorts,
    )

    candles = load_raw_candles(config.input_dir, config.timeframe)
    results = run_walkforward(candles, config)
    paths = save_outputs(results, config)

    print("\nWalk-forward overall summary:")
    if results["overall_summary"].empty:
        print("No overall summary generated.")
    else:
        print(results["overall_summary"].to_string(index=False))

    print("\nWalk-forward window summary:")
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
            "selected_bullish_candidates",
            "selected_bearish_candidates",
        ]
        display_cols = [col for col in display_cols if col in results["window_summary"].columns]
        print(results["window_summary"][display_cols].to_string(index=False))

    print("\nSaved walk-forward outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
