"""
Conditional probability grid search for volatility / momentum crypto prediction.

Run from project root:

    python -m src.probability_table --timeframe 15m

Recommended competition-oriented run:

    python -m src.probability_table --timeframe 15m --lookback-hours-list 4 6 12 24 --horizon-minutes-list 30 60 120 240

Before running on 15m data, download candles:

    python -m src.historical_data --days 365 --granularity 900

Purpose:
    Search many combinations of:
        - past lookback window: 4h, 6h, 12h, 24h
        - future prediction horizon: 30m, 1h, 2h, 4h

    For each combination, the script builds conditional probability tables:
        1. Volatility bucket only
        2. Momentum bucket only
        3. Volatility bucket + momentum bucket
        4. Asset + volatility bucket
        5. Hour of day

    It also creates a combined leaderboard of the strongest bullish and bearish
    conditional signals across all tested combinations.

Core idea:
    For every asset and timestamp:
        1. Calculate past realized volatility over the chosen lookback window.
        2. Calculate recent return over the same lookback window.
        3. Measure future return over the chosen horizon.
        4. Classify future move as UP / FLAT / DOWN.
        5. Rank conditional states by P(UP), P(DOWN), edge, and average return.

No trading is done here. This file only creates research tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.strategy import load_raw_candles


TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "6h": 360,
    "1d": 1440,
}


@dataclass(frozen=True)
class ProbabilityTableConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    lookback_hours: float
    horizon_minutes: int
    threshold_pct: float
    percentile_window: int
    min_samples_per_bucket: int
    bucket_size: int


@dataclass(frozen=True)
class GridSearchConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    lookback_hours_list: list[float]
    horizon_minutes_list: list[int]
    threshold_pct: float
    percentile_window: int
    min_samples_per_bucket: int
    bucket_size: int
    top_n: int


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def timeframe_to_minutes(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_TO_MINUTES:
        valid = ", ".join(TIMEFRAME_TO_MINUTES)
        raise ValueError(f"Unsupported timeframe: {timeframe}. Valid options: {valid}")

    return TIMEFRAME_TO_MINUTES[timeframe]


def format_number_for_filename(value: float | int) -> str:
    return f"{value:g}".replace(".", "p")


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the latest value inside the rolling window.

    Example:
        percentile = 10 means current volatility is very low compared with recent history.
        percentile = 90 means current volatility is very high compared with recent history.
    """

    def percentile_of_last(values: np.ndarray) -> float:
        last = values[-1]

        if np.isnan(last):
            return np.nan

        clean = values[~np.isnan(values)]

        if len(clean) == 0:
            return np.nan

        return 100.0 * (clean <= last).sum() / len(clean)

    return series.rolling(window=window, min_periods=window).apply(
        percentile_of_last,
        raw=True,
    )


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def make_vol_bucket(percentile: pd.Series, bucket_size: int) -> pd.Series:
    """
    Convert volatility percentile into buckets.

    Example with bucket_size=10:
        0-10, 10-20, ..., 90-100
    """
    bucket = (np.floor(percentile / bucket_size) * bucket_size).clip(0, 100 - bucket_size)
    return bucket.astype("Int64")


def classify_future_return(future_return: pd.Series, threshold_pct: float) -> pd.Series:
    """
    Classify future return into UP / DOWN / FLAT.

    threshold_pct is expressed in percent.

    Example:
        threshold_pct = 0.30 means:
            UP   if future return > +0.30%
            DOWN if future return < -0.30%
            FLAT otherwise
    """
    threshold = threshold_pct / 100.0

    labels = pd.Series("FLAT", index=future_return.index, dtype="object")
    labels[future_return > threshold] = "UP"
    labels[future_return < -threshold] = "DOWN"

    return labels


def add_momentum_bucket(data: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket the recent lookback return.

    These buckets are deliberately coarse. If they are too fine, the table will
    look impressive but be statistically fragile.
    """
    result = data.copy()

    result["return_lookback_bucket"] = pd.cut(
        result["return_lookback_pct"],
        bins=[-np.inf, -4.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 4.0, np.inf],
        labels=[
            "<-4%",
            "-4% to -2%",
            "-2% to -1%",
            "-1% to -0.25%",
            "-0.25% to 0.25%",
            "0.25% to 1%",
            "1% to 2%",
            "2% to 4%",
            ">4%",
        ],
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_probability_dataset(
    candles: pd.DataFrame,
    config: ProbabilityTableConfig,
) -> pd.DataFrame:
    """
    Build one row per asset per timestamp with:
        - realized volatility over the lookback window
        - volatility percentile
        - recent lookback return
        - future return over the horizon
        - UP / FLAT / DOWN label
        - supporting features for later analysis
    """
    timeframe_minutes = timeframe_to_minutes(config.timeframe)

    lookback_bars = int(round((config.lookback_hours * 60) / timeframe_minutes))
    horizon_bars = int(round(config.horizon_minutes / timeframe_minutes))

    if lookback_bars < 2:
        raise ValueError(
            f"lookback_hours={config.lookback_hours} is too small for timeframe={config.timeframe}."
        )

    if horizon_bars < 1:
        raise ValueError(
            f"horizon_minutes={config.horizon_minutes} is smaller than one {config.timeframe} candle. "
            f"Use a lower timeframe or a larger horizon."
        )

    one_hour_bars = max(1, int(round(60 / timeframe_minutes)))
    four_hour_bars = max(1, int(round(240 / timeframe_minutes)))

    frames: list[pd.DataFrame] = []

    candles = candles.sort_values(["product_id", "time"]).reset_index(drop=True)

    for product_id, group in candles.groupby("product_id", sort=False):
        g = group.copy().sort_values("time").reset_index(drop=True)

        g["log_return"] = np.log(g["close"] / g["close"].shift(1))

        # Main volatility feature: realized volatility over the chosen lookback window.
        g["realized_vol"] = g["log_return"].rolling(
            window=lookback_bars,
            min_periods=lookback_bars,
        ).std()

        g["realized_vol_percentile"] = rolling_percentile_rank(
            g["realized_vol"],
            window=config.percentile_window,
        )

        g["vol_bucket"] = make_vol_bucket(
            g["realized_vol_percentile"],
            bucket_size=config.bucket_size,
        )

        # Supporting context features. return_lookback is the momentum/extreme-move feature
        # that we will compare against volatility.
        g["return_lookback"] = g["close"] / g["close"].shift(lookback_bars) - 1.0
        g["return_lookback_pct"] = g["return_lookback"] * 100.0
        g["return_1h"] = g["close"] / g["close"].shift(one_hour_bars) - 1.0
        g["return_4h"] = g["close"] / g["close"].shift(four_hour_bars) - 1.0
        g["return_1h_pct"] = g["return_1h"] * 100.0
        g["return_4h_pct"] = g["return_4h"] * 100.0

        g["ema_50"] = ema(g["close"], span=50)
        g["ema_200"] = ema(g["close"], span=200)
        g["price_vs_ema50_pct"] = (g["close"] / g["ema_50"] - 1.0) * 100.0
        g["price_vs_ema200_pct"] = (g["close"] / g["ema_200"] - 1.0) * 100.0

        volume_mean = g["volume"].rolling(window=20, min_periods=20).mean()
        volume_std = g["volume"].rolling(window=20, min_periods=20).std()
        g["volume_zscore_20"] = (g["volume"] - volume_mean) / volume_std.replace(0, np.nan)

        # Future target.
        g["future_close"] = g["close"].shift(-horizon_bars)
        g["future_return"] = g["future_close"] / g["close"] - 1.0
        g["future_return_pct"] = g["future_return"] * 100.0
        g["future_direction"] = classify_future_return(
            g["future_return"],
            threshold_pct=config.threshold_pct,
        )

        g["lookback_bars"] = lookback_bars
        g["horizon_bars"] = horizon_bars
        g["lookback_hours"] = config.lookback_hours
        g["horizon_minutes"] = config.horizon_minutes
        g["threshold_pct"] = config.threshold_pct
        g["timeframe"] = config.timeframe

        # Time features for later analysis / ML.
        g["hour_utc"] = g["time"].dt.hour
        g["day_of_week"] = g["time"].dt.dayofweek

        frames.append(g)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = add_momentum_bucket(dataset)

    required = [
        "realized_vol",
        "realized_vol_percentile",
        "vol_bucket",
        "return_lookback",
        "return_lookback_bucket",
        "future_return",
        "future_direction",
    ]

    dataset = dataset.dropna(subset=required).reset_index(drop=True)

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Generic table helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_direction_probabilities(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()

    for col in ["UP", "FLAT", "DOWN"]:
        if col not in table.columns:
            table[col] = 0

    table["p_up"] = table["UP"] / table["samples"]
    table["p_flat"] = table["FLAT"] / table["samples"]
    table["p_down"] = table["DOWN"] / table["samples"]
    table["up_down_edge"] = table["p_up"] - table["p_down"]
    table["down_up_edge"] = table["p_down"] - table["p_up"]

    return table


def direction_count_table(dataset: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    return (
        dataset.pivot_table(
            index=index_cols,
            columns="future_direction",
            values="time",
            aggfunc="count",
            fill_value=0,
            observed=True,
        )
        .reset_index()
    )


def add_run_metadata(table: pd.DataFrame, config: ProbabilityTableConfig, table_type: str) -> pd.DataFrame:
    table = table.copy()
    table["table_type"] = table_type
    table["timeframe"] = config.timeframe
    table["lookback_hours"] = config.lookback_hours
    table["horizon_minutes"] = config.horizon_minutes
    table["threshold_pct"] = config.threshold_pct
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Probability tables
# ─────────────────────────────────────────────────────────────────────────────

def summarize_by_vol_bucket(dataset: pd.DataFrame, config: ProbabilityTableConfig) -> pd.DataFrame:
    grouped = dataset.groupby("vol_bucket", observed=True)

    summary = grouped.agg(
        samples=("future_direction", "count"),
        avg_future_return_pct=("future_return_pct", "mean"),
        median_future_return_pct=("future_return_pct", "median"),
        std_future_return_pct=("future_return_pct", "std"),
        avg_realized_vol=("realized_vol", "mean"),
        avg_vol_percentile=("realized_vol_percentile", "mean"),
    ).reset_index()

    table = summary.merge(direction_count_table(dataset, ["vol_bucket"]), on="vol_bucket", how="left")
    table = add_direction_probabilities(table)

    table["bucket_label"] = table["vol_bucket"].astype(str) + "-" + (
        table["vol_bucket"] + config.bucket_size
    ).astype(str)
    table["tradable_sample"] = table["samples"] >= config.min_samples_per_bucket

    table = add_run_metadata(table, config, table_type="vol_bucket")

    ordered_cols = [
        "table_type",
        "timeframe",
        "lookback_hours",
        "horizon_minutes",
        "threshold_pct",
        "bucket_label",
        "vol_bucket",
        "samples",
        "tradable_sample",
        "p_up",
        "p_flat",
        "p_down",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "median_future_return_pct",
        "std_future_return_pct",
        "avg_realized_vol",
        "avg_vol_percentile",
        "UP",
        "FLAT",
        "DOWN",
    ]

    return table[ordered_cols].sort_values("vol_bucket").reset_index(drop=True)


def summarize_by_momentum_bucket(dataset: pd.DataFrame, config: ProbabilityTableConfig) -> pd.DataFrame:
    grouped = dataset.groupby("return_lookback_bucket", observed=True)

    summary = grouped.agg(
        samples=("future_direction", "count"),
        avg_future_return_pct=("future_return_pct", "mean"),
        median_future_return_pct=("future_return_pct", "median"),
        std_future_return_pct=("future_return_pct", "std"),
        avg_return_lookback_pct=("return_lookback_pct", "mean"),
    ).reset_index()

    table = summary.merge(
        direction_count_table(dataset, ["return_lookback_bucket"]),
        on="return_lookback_bucket",
        how="left",
    )
    table = add_direction_probabilities(table)
    table["tradable_sample"] = table["samples"] >= config.min_samples_per_bucket
    table = add_run_metadata(table, config, table_type="momentum_bucket")

    ordered_cols = [
        "table_type",
        "timeframe",
        "lookback_hours",
        "horizon_minutes",
        "threshold_pct",
        "return_lookback_bucket",
        "samples",
        "tradable_sample",
        "p_up",
        "p_flat",
        "p_down",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "median_future_return_pct",
        "std_future_return_pct",
        "avg_return_lookback_pct",
        "UP",
        "FLAT",
        "DOWN",
    ]

    return table[ordered_cols].reset_index(drop=True)


def summarize_by_asset_and_vol_bucket(dataset: pd.DataFrame, config: ProbabilityTableConfig) -> pd.DataFrame:
    grouped = dataset.groupby(["product_id", "vol_bucket"], observed=True)

    summary = grouped.agg(
        samples=("future_direction", "count"),
        avg_future_return_pct=("future_return_pct", "mean"),
        median_future_return_pct=("future_return_pct", "median"),
        std_future_return_pct=("future_return_pct", "std"),
        avg_realized_vol=("realized_vol", "mean"),
        avg_vol_percentile=("realized_vol_percentile", "mean"),
    ).reset_index()

    table = summary.merge(
        direction_count_table(dataset, ["product_id", "vol_bucket"]),
        on=["product_id", "vol_bucket"],
        how="left",
    )
    table = add_direction_probabilities(table)
    table["bucket_label"] = table["vol_bucket"].astype(str) + "-" + (
        table["vol_bucket"] + config.bucket_size
    ).astype(str)
    table["tradable_sample"] = table["samples"] >= config.min_samples_per_bucket
    table = add_run_metadata(table, config, table_type="asset_vol_bucket")

    ordered_cols = [
        "table_type",
        "timeframe",
        "lookback_hours",
        "horizon_minutes",
        "threshold_pct",
        "product_id",
        "bucket_label",
        "vol_bucket",
        "samples",
        "tradable_sample",
        "p_up",
        "p_flat",
        "p_down",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "median_future_return_pct",
        "std_future_return_pct",
        "avg_realized_vol",
        "avg_vol_percentile",
        "UP",
        "FLAT",
        "DOWN",
    ]

    return table[ordered_cols].sort_values(["product_id", "vol_bucket"]).reset_index(drop=True)


def summarize_by_vol_and_momentum_bucket(dataset: pd.DataFrame, config: ProbabilityTableConfig) -> pd.DataFrame:
    grouped = dataset.groupby(["vol_bucket", "return_lookback_bucket"], observed=True)

    summary = grouped.agg(
        samples=("future_direction", "count"),
        avg_future_return_pct=("future_return_pct", "mean"),
        median_future_return_pct=("future_return_pct", "median"),
        std_future_return_pct=("future_return_pct", "std"),
        avg_return_lookback_pct=("return_lookback_pct", "mean"),
        avg_realized_vol=("realized_vol", "mean"),
        avg_vol_percentile=("realized_vol_percentile", "mean"),
    ).reset_index()

    table = summary.merge(
        direction_count_table(dataset, ["vol_bucket", "return_lookback_bucket"]),
        on=["vol_bucket", "return_lookback_bucket"],
        how="left",
    )
    table = add_direction_probabilities(table)
    table["bucket_label"] = table["vol_bucket"].astype(str) + "-" + (
        table["vol_bucket"] + config.bucket_size
    ).astype(str)
    table["tradable_sample"] = table["samples"] >= config.min_samples_per_bucket
    table = add_run_metadata(table, config, table_type="vol_momentum_bucket")

    ordered_cols = [
        "table_type",
        "timeframe",
        "lookback_hours",
        "horizon_minutes",
        "threshold_pct",
        "bucket_label",
        "vol_bucket",
        "return_lookback_bucket",
        "samples",
        "tradable_sample",
        "p_up",
        "p_flat",
        "p_down",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "median_future_return_pct",
        "std_future_return_pct",
        "avg_return_lookback_pct",
        "avg_realized_vol",
        "avg_vol_percentile",
        "UP",
        "FLAT",
        "DOWN",
    ]

    return table[ordered_cols].sort_values(["vol_bucket", "return_lookback_bucket"]).reset_index(drop=True)


def summarize_by_hour(dataset: pd.DataFrame, config: ProbabilityTableConfig) -> pd.DataFrame:
    grouped = dataset.groupby("hour_utc", observed=True)

    summary = grouped.agg(
        samples=("future_direction", "count"),
        avg_future_return_pct=("future_return_pct", "mean"),
        median_future_return_pct=("future_return_pct", "median"),
    ).reset_index()

    table = summary.merge(direction_count_table(dataset, ["hour_utc"]), on="hour_utc", how="left")
    table = add_direction_probabilities(table)
    table["tradable_sample"] = table["samples"] >= config.min_samples_per_bucket
    table = add_run_metadata(table, config, table_type="hour")

    ordered_cols = [
        "table_type",
        "timeframe",
        "lookback_hours",
        "horizon_minutes",
        "threshold_pct",
        "hour_utc",
        "samples",
        "tradable_sample",
        "p_up",
        "p_flat",
        "p_down",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "median_future_return_pct",
        "UP",
        "FLAT",
        "DOWN",
    ]

    return table[ordered_cols].sort_values("hour_utc").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Signal leaderboard across grid
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_candidates(
    vol_table: pd.DataFrame,
    momentum_table: pd.DataFrame,
    vol_momentum_table: pd.DataFrame,
    config: ProbabilityTableConfig,
) -> pd.DataFrame:
    """
    Normalize different table types into one comparable candidate table.

    A candidate is not automatically a trade. It is a conditional state that may
    be worth backtesting.
    """
    candidate_frames: list[pd.DataFrame] = []

    vol_candidates = vol_table.copy()
    vol_candidates["signal_key"] = "vol=" + vol_candidates["bucket_label"].astype(str)
    candidate_frames.append(vol_candidates)

    mom_candidates = momentum_table.copy()
    mom_candidates["signal_key"] = "momentum=" + mom_candidates["return_lookback_bucket"].astype(str)
    candidate_frames.append(mom_candidates)

    vm_candidates = vol_momentum_table.copy()
    vm_candidates["signal_key"] = (
        "vol=" + vm_candidates["bucket_label"].astype(str)
        + " | momentum=" + vm_candidates["return_lookback_bucket"].astype(str)
    )
    candidate_frames.append(vm_candidates)

    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)

    candidates["bull_score"] = candidates["up_down_edge"] * candidates["avg_future_return_pct"]
    candidates["bear_score"] = candidates["down_up_edge"] * (-candidates["avg_future_return_pct"])

    # Penalize tiny sample sizes without hiding them. This is not a formal statistic;
    # it is just a practical ranking score.
    candidates["sample_weight"] = np.sqrt(candidates["samples"].clip(lower=1))
    candidates["bull_score_weighted"] = candidates["bull_score"] * candidates["sample_weight"]
    candidates["bear_score_weighted"] = candidates["bear_score"] * candidates["sample_weight"]

    candidates["bullish_candidate"] = (
        (candidates["tradable_sample"])
        & (candidates["up_down_edge"] > 0)
        & (candidates["avg_future_return_pct"] > 0)
    )

    candidates["bearish_candidate"] = (
        (candidates["tradable_sample"])
        & (candidates["down_up_edge"] > 0)
        & (candidates["avg_future_return_pct"] < 0)
    )

    keep_cols = [
        "table_type",
        "timeframe",
        "lookback_hours",
        "horizon_minutes",
        "threshold_pct",
        "signal_key",
        "samples",
        "tradable_sample",
        "p_up",
        "p_flat",
        "p_down",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "median_future_return_pct",
        "bull_score_weighted",
        "bear_score_weighted",
        "bullish_candidate",
        "bearish_candidate",
    ]

    existing = [col for col in keep_cols if col in candidates.columns]
    return candidates[existing].reset_index(drop=True)


def build_grid_leaderboards(all_candidates: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    bullish = (
        all_candidates[all_candidates["bullish_candidate"]]
        .sort_values(
            ["bull_score_weighted", "up_down_edge", "avg_future_return_pct", "samples"],
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    bearish = (
        all_candidates[all_candidates["bearish_candidate"]]
        .sort_values(
            ["bear_score_weighted", "down_up_edge", "avg_future_return_pct", "samples"],
            ascending=[False, False, True, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return bullish, bearish


# ─────────────────────────────────────────────────────────────────────────────
# Running one config and whole grid
# ─────────────────────────────────────────────────────────────────────────────

def run_single_probability_config(
    candles: pd.DataFrame,
    config: ProbabilityTableConfig,
) -> dict[str, pd.DataFrame]:
    dataset = build_probability_dataset(candles, config)
    vol_table = summarize_by_vol_bucket(dataset, config)
    momentum_table = summarize_by_momentum_bucket(dataset, config)
    asset_vol_table = summarize_by_asset_and_vol_bucket(dataset, config)
    vol_momentum_table = summarize_by_vol_and_momentum_bucket(dataset, config)
    hour_table = summarize_by_hour(dataset, config)
    candidates = build_signal_candidates(vol_table, momentum_table, vol_momentum_table, config)

    return {
        "dataset": dataset,
        "vol_table": vol_table,
        "momentum_table": momentum_table,
        "asset_vol_table": asset_vol_table,
        "vol_momentum_table": vol_momentum_table,
        "hour_table": hour_table,
        "candidates": candidates,
    }


def run_probability_grid_search(candles: pd.DataFrame, grid_config: GridSearchConfig) -> dict[str, pd.DataFrame]:
    all_vol_tables: list[pd.DataFrame] = []
    all_momentum_tables: list[pd.DataFrame] = []
    all_asset_vol_tables: list[pd.DataFrame] = []
    all_vol_momentum_tables: list[pd.DataFrame] = []
    all_hour_tables: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []

    timeframe_minutes = timeframe_to_minutes(grid_config.timeframe)

    for lookback_hours in grid_config.lookback_hours_list:
        for horizon_minutes in grid_config.horizon_minutes_list:
            if horizon_minutes < timeframe_minutes:
                print(
                    f"Skipping lookback={lookback_hours}h horizon={horizon_minutes}m: "
                    f"horizon is smaller than one {grid_config.timeframe} candle."
                )
                continue

            config = ProbabilityTableConfig(
                input_dir=grid_config.input_dir,
                output_dir=grid_config.output_dir,
                timeframe=grid_config.timeframe,
                lookback_hours=lookback_hours,
                horizon_minutes=horizon_minutes,
                threshold_pct=grid_config.threshold_pct,
                percentile_window=grid_config.percentile_window,
                min_samples_per_bucket=grid_config.min_samples_per_bucket,
                bucket_size=grid_config.bucket_size,
            )

            print(f"Running lookback={lookback_hours:g}h horizon={horizon_minutes}m")
            result = run_single_probability_config(candles, config)

            all_vol_tables.append(result["vol_table"])
            all_momentum_tables.append(result["momentum_table"])
            all_asset_vol_tables.append(result["asset_vol_table"])
            all_vol_momentum_tables.append(result["vol_momentum_table"])
            all_hour_tables.append(result["hour_table"])
            all_candidates.append(result["candidates"])

    if not all_candidates:
        raise ValueError("No grid configurations were run. Check timeframe and horizon settings.")

    vol_tables = pd.concat(all_vol_tables, ignore_index=True)
    momentum_tables = pd.concat(all_momentum_tables, ignore_index=True)
    asset_vol_tables = pd.concat(all_asset_vol_tables, ignore_index=True)
    vol_momentum_tables = pd.concat(all_vol_momentum_tables, ignore_index=True)
    hour_tables = pd.concat(all_hour_tables, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)

    bullish_leaderboard, bearish_leaderboard = build_grid_leaderboards(candidates, grid_config.top_n)

    return {
        "vol_tables": vol_tables,
        "momentum_tables": momentum_tables,
        "asset_vol_tables": asset_vol_tables,
        "vol_momentum_tables": vol_momentum_tables,
        "hour_tables": hour_tables,
        "candidates": candidates,
        "bullish_leaderboard": bullish_leaderboard,
        "bearish_leaderboard": bearish_leaderboard,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Saving and CLI
# ─────────────────────────────────────────────────────────────────────────────

def save_grid_outputs(results: dict[str, pd.DataFrame], config: GridSearchConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lookbacks = "_".join(format_number_for_filename(x) for x in config.lookback_hours_list)
    horizons = "_".join(format_number_for_filename(x) for x in config.horizon_minutes_list)

    suffix = (
        f"{config.timeframe}_lookbacks_{lookbacks}h_"
        f"horizons_{horizons}m_threshold_{format_number_for_filename(config.threshold_pct)}pct"
    )

    paths = {
        "vol_tables": config.output_dir / f"grid_probability_by_vol_bucket_{suffix}.csv",
        "momentum_tables": config.output_dir / f"grid_probability_by_momentum_bucket_{suffix}.csv",
        "asset_vol_tables": config.output_dir / f"grid_probability_by_asset_vol_bucket_{suffix}.csv",
        "vol_momentum_tables": config.output_dir / f"grid_probability_by_vol_momentum_{suffix}.csv",
        "hour_tables": config.output_dir / f"grid_probability_by_hour_{suffix}.csv",
        "candidates": config.output_dir / f"grid_signal_candidates_{suffix}.csv",
        "bullish_leaderboard": config.output_dir / f"grid_bullish_signal_leaderboard_{suffix}.csv",
        "bearish_leaderboard": config.output_dir / f"grid_bearish_signal_leaderboard_{suffix}.csv",
    }

    for key, path in paths.items():
        results[key].to_csv(path, index=False)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search conditional probability tables for volatility/momentum and future returns."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw OHLCV CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/probability_tables"),
        help="Directory where probability table outputs will be saved.",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        choices=list(TIMEFRAME_TO_MINUTES.keys()),
        help="Timeframe label in raw CSV filenames.",
    )

    parser.add_argument(
        "--lookback-hours-list",
        type=float,
        nargs="+",
        default=[4.0, 6.0, 12.0, 24.0],
        help="Past lookback windows to test, in hours.",
    )

    parser.add_argument(
        "--horizon-minutes-list",
        type=int,
        nargs="+",
        default=[30, 60, 120, 240],
        help="Future horizons to test, in minutes.",
    )

    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.30,
        help="Future return threshold for UP/DOWN classification, in percent.",
    )

    parser.add_argument(
        "--percentile-window",
        type=int,
        default=200,
        help="Rolling window used to rank current volatility versus recent volatility history.",
    )

    parser.add_argument(
        "--min-samples-per-bucket",
        type=int,
        default=100,
        help="Minimum sample count for a bucket to be considered statistically usable.",
    )

    parser.add_argument(
        "--bucket-size",
        type=int,
        default=10,
        help="Volatility percentile bucket size. 10 means 0-10, 10-20, etc.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top bullish and bearish candidate rows to save/print.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grid_config = GridSearchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        lookback_hours_list=args.lookback_hours_list,
        horizon_minutes_list=args.horizon_minutes_list,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        min_samples_per_bucket=args.min_samples_per_bucket,
        bucket_size=args.bucket_size,
        top_n=args.top_n,
    )

    candles = load_raw_candles(grid_config.input_dir, grid_config.timeframe)
    results = run_probability_grid_search(candles, grid_config)
    paths = save_grid_outputs(results, grid_config)

    print("\nTop bullish conditional signals:")
    bullish = results["bullish_leaderboard"]
    if bullish.empty:
        print("No bullish candidates met the requirements.")
    else:
        display_cols = [
            "table_type",
            "timeframe",
            "lookback_hours",
            "horizon_minutes",
            "signal_key",
            "samples",
            "p_up",
            "p_down",
            "up_down_edge",
            "avg_future_return_pct",
            "bull_score_weighted",
        ]
        print(bullish[display_cols].head(20).to_string(index=False))

    print("\nTop bearish conditional signals:")
    bearish = results["bearish_leaderboard"]
    if bearish.empty:
        print("No bearish candidates met the requirements.")
    else:
        display_cols = [
            "table_type",
            "timeframe",
            "lookback_hours",
            "horizon_minutes",
            "signal_key",
            "samples",
            "p_up",
            "p_down",
            "down_up_edge",
            "avg_future_return_pct",
            "bear_score_weighted",
        ]
        print(bearish[display_cols].head(20).to_string(index=False))

    print("\nSaved probability grid-search outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
