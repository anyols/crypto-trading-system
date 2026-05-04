"""
Volatility Compression Breakout strategy signal generator.

Run from project root:

    python -m src.strategy --timeframe 1h

This file generates entry signals only. It does not execute trades and does not
run a full backtest yet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.indicators import add_volatility_strategy_features


@dataclass(frozen=True)
class VolatilityBreakoutConfig:
    btc_product_id: str = "BTC-USD"

    # Volatility compression condition
    max_vol_percentile: float = 20.0

    # Breakout confirmation
    volume_multiplier: float = 1.0

    # Risk model placeholders for later backtesting
    stop_atr_multiplier: float = 2.0
    take_profit_r_multiple: float = 3.0


def load_raw_candles(input_dir: Path, timeframe: str) -> pd.DataFrame:
    """
    Load all raw Coinbase OHLCV CSV files for a given timeframe.

    Expected filename example:
        BTC-USD_1h_20260404_20260504.csv
    """
    pattern = f"*-USD_{timeframe}_*.csv"
    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching pattern: {pattern}"
        )

    frames: list[pd.DataFrame] = []

    for file in files:
        if "empty" in file.name.lower():
            continue

        df = pd.read_csv(file, parse_dates=["time"])
        frames.append(df)

    if not frames:
        raise ValueError(f"No non-empty candle files found in {input_dir}")

    candles = pd.concat(frames, ignore_index=True)

    required_cols = {"time", "product_id", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(candles.columns)

    if missing:
        raise ValueError(f"Missing required columns in candle data: {sorted(missing)}")

    candles = (
        candles.drop_duplicates(subset=["time", "product_id"])
        .sort_values(["product_id", "time"])
        .reset_index(drop=True)
    )

    return candles


def add_btc_regime_filter(
    features: pd.DataFrame,
    btc_product_id: str = "BTC-USD",
) -> pd.DataFrame:
    """
    Add BTC market regime columns to every asset.

    We use BTC as a market filter:
        BTC close > BTC EMA 200

    For altcoins, this avoids taking long signals when the broader crypto market
    is weak.
    """
    btc = features[features["product_id"] == btc_product_id].copy()

    if btc.empty:
        raise ValueError(
            f"BTC product {btc_product_id} not found. "
            "The BTC file is required for the market regime filter."
        )

    btc = btc[["time", "close", "ema_200"]].rename(
        columns={
            "close": "btc_close",
            "ema_200": "btc_ema_200",
        }
    )

    btc = btc.sort_values("time").reset_index(drop=True)

    merged_frames: list[pd.DataFrame] = []

    for product_id, group in features.groupby("product_id", sort=False):
        group = group.sort_values("time").reset_index(drop=True)

        merged = pd.merge_asof(
            group,
            btc,
            on="time",
            direction="backward",
        )

        merged_frames.append(merged)

    result = pd.concat(merged_frames, ignore_index=True)
    result["btc_bull_regime"] = result["btc_close"] > result["btc_ema_200"]

    return result


def generate_volatility_breakout_signals(
    features: pd.DataFrame,
    config: VolatilityBreakoutConfig,
) -> pd.DataFrame:
    """
    Generate long-only volatility compression breakout signals.

    Long signal logic:

        1. Realized volatility percentile is below threshold
        2. Price closes above previous Donchian high
        3. Volume is above its moving average
        4. Asset is in an uptrend
        5. BTC is in an uptrend
    """
    data = add_btc_regime_filter(
        features=features,
        btc_product_id=config.btc_product_id,
    )

    data["low_volatility"] = (
        data["realized_vol_percentile_200"] <= config.max_vol_percentile
    )

    data["price_breakout"] = data["close"] > data["donchian_high_20"]

    data["volume_confirmed"] = (
        data["volume"] > data["volume_sma_20"] * config.volume_multiplier
    )

    data["asset_bull_regime"] = (
        (data["close"] > data["ema_200"]) &
        (data["ema_50"] > data["ema_200"])
    )

    conditions = [
        data["low_volatility"],
        data["price_breakout"],
        data["volume_confirmed"],
        data["asset_bull_regime"],
        data["btc_bull_regime"],
    ]

    data["long_signal"] = pd.concat(conditions, axis=1).all(axis=1)

    # Risk levels for later backtesting.
    data["entry_price"] = data["close"]
    data["stop_price"] = data["entry_price"] - (
        config.stop_atr_multiplier * data["atr_14"]
    )

    data["risk_per_unit"] = data["entry_price"] - data["stop_price"]

    data["take_profit_price"] = data["entry_price"] + (
        config.take_profit_r_multiple * data["risk_per_unit"]
    )

    # If ATR is unavailable, the signal is not tradable.
    data.loc[data["atr_14"].isna(), "long_signal"] = False
    data.loc[data["risk_per_unit"] <= 0, "long_signal"] = False

    return data


def summarize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a simple signal count summary by asset.
    """
    summary = (
        signals.groupby("product_id")
        .agg(
            rows=("time", "count"),
            long_signals=("long_signal", "sum"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
    )

    summary["signal_rate_pct"] = (
        100.0 * summary["long_signals"] / summary["rows"]
    )

    return summary.sort_values("long_signals", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate volatility compression breakout signals."
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
        default=Path("data/signals"),
        help="Directory where signal CSV files will be saved.",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe label in the raw CSV filename, e.g. 1m, 5m, 1h, 1d.",
    )

    parser.add_argument(
        "--max-vol-percentile",
        type=float,
        default=20.0,
        help="Maximum realized volatility percentile allowed for compression.",
    )

    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=1.0,
        help="Volume must be above volume_sma_20 times this multiplier.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candles = load_raw_candles(
        input_dir=args.input_dir,
        timeframe=args.timeframe,
    )

    print(f"Loaded {len(candles):,} candle rows.")
    print(f"Products: {sorted(candles['product_id'].unique())}")

    features = add_volatility_strategy_features(candles)

    config = VolatilityBreakoutConfig(
        max_vol_percentile=args.max_vol_percentile,
        volume_multiplier=args.volume_multiplier,
    )

    signals = generate_volatility_breakout_signals(
        features=features,
        config=config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / f"volatility_breakout_signals_{args.timeframe}.csv"
    signals.to_csv(output_path, index=False)

    summary = summarize_signals(signals)
    summary_path = args.output_dir / f"volatility_breakout_summary_{args.timeframe}.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSignal summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved full signals to: {output_path}")
    print(f"Saved summary to: {summary_path}")

    recent_signals = signals[signals["long_signal"]].tail(10)

    if recent_signals.empty:
        print("\nNo long signals found.")
    else:
        print("\nMost recent long signals:")
        print(
            recent_signals[
                [
                    "time",
                    "product_id",
                    "close",
                    "realized_vol_percentile_200",
                    "donchian_high_20",
                    "volume",
                    "volume_sma_20",
                    "entry_price",
                    "stop_price",
                    "take_profit_price",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()