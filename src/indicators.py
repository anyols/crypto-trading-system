"""
Indicator functions for the crypto volatility strategy.

These functions do not fetch data and do not trade.
They only transform clean OHLCV data into research features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log returns:

        log_return[t] = ln(close[t] / close[t-1])
    """
    result = df.copy()
    result["log_return"] = np.log(result["close"] / result["close"].shift(1))
    return result


def realized_volatility(
    close: pd.Series,
    window: int = 20,
    annualize: bool = False,
    periods_per_year: int | None = None,
) -> pd.Series:
    """
    Rolling realized volatility based on log returns.

    For early research, non-annualized volatility is usually enough because
    we care about relative volatility compression versus the asset's own history.
    """
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=window, min_periods=window).std()

    if annualize:
        if periods_per_year is None:
            raise ValueError("periods_per_year is required when annualize=True.")
        vol = vol * np.sqrt(periods_per_year)

    return vol


def rolling_percentile_rank(series: pd.Series, window: int = 200) -> pd.Series:
    """
    Rolling percentile rank of the most recent value inside each rolling window.

    Example:
        value is in the bottom 20% of its last 200 values -> percentile <= 20
    """

    def percentile_of_last_value(values: np.ndarray) -> float:
        last_value = values[-1]

        if np.isnan(last_value):
            return np.nan

        clean = values[~np.isnan(values)]

        if len(clean) == 0:
            return np.nan

        return 100.0 * (clean <= last_value).sum() / len(clean)

    return series.rolling(window=window, min_periods=window).apply(
        percentile_of_last_value,
        raw=True,
    )


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range.

    True Range is the max of:
        high - low
        abs(high - previous close)
        abs(low - previous close)
    """
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat(
        [high_low, high_prev_close, low_prev_close],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window=window, min_periods=window).mean()


def donchian_high(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Previous N-bar high.

    We shift by 1 to avoid lookahead bias.
    Using the current candle's high for a breakout signal would be cheating.
    """
    return df["high"].rolling(window=window, min_periods=window).max().shift(1)


def donchian_low(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Previous N-bar low.

    Shifted by 1 to avoid lookahead bias.
    """
    return df["low"].rolling(window=window, min_periods=window).min().shift(1)


def volume_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling average volume."""
    return df["volume"].rolling(window=window, min_periods=window).mean()


def add_volatility_strategy_features(
    df: pd.DataFrame,
    vol_window: int = 20,
    vol_percentile_window: int = 200,
    ema_fast_window: int = 50,
    ema_slow_window: int = 200,
    atr_window: int = 14,
    donchian_window: int = 20,
    volume_window: int = 20,
) -> pd.DataFrame:
    """
    Add all features required for the first volatility compression breakout model.
    """
    required_cols = {"time", "product_id", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result = result.sort_values(["product_id", "time"]).reset_index(drop=True)

    feature_frames: list[pd.DataFrame] = []

    for product_id, group in result.groupby("product_id", sort=False):
        g = group.copy()

        g["log_return"] = np.log(g["close"] / g["close"].shift(1))

        g["realized_vol_20"] = realized_volatility(
            close=g["close"],
            window=vol_window,
            annualize=False,
        )

        g["realized_vol_percentile_200"] = rolling_percentile_rank(
            series=g["realized_vol_20"],
            window=vol_percentile_window,
        )

        g["ema_50"] = ema(g["close"], ema_fast_window)
        g["ema_200"] = ema(g["close"], ema_slow_window)

        g["atr_14"] = atr(g, atr_window)

        g["donchian_high_20"] = donchian_high(g, donchian_window)
        g["donchian_low_20"] = donchian_low(g, donchian_window)

        g["volume_sma_20"] = volume_sma(g, volume_window)

        feature_frames.append(g)

    return pd.concat(feature_frames, ignore_index=True)