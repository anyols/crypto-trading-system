"""
Experiment runner for strategy-variant testing.

Run from project root:

    python -m src.experiments --timeframe 1h

Example with your current research settings:

    python -m src.experiments --timeframe 1h --max-vol-percentile 30 --volume-multiplier 0.8

Purpose:
    The original volatility-compression breakout model failed. Exit-mode tests showed
    that slow EMA200 exits worked better than fixed take-profit exits.

    This file now tests one more focused iteration:
        - keep the best exit style: 3 ATR initial stop + EMA200 exit
        - test whether the entry logic is the real weakness

    This is ablation testing. It is not random optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, ExitMode, run_backtests
from src.indicators import add_volatility_strategy_features
from src.strategy import add_btc_regime_filter, load_raw_candles


@dataclass(frozen=True)
class StrategyVariant:
    """
    Defines one controlled strategy variant.
    """

    name: str
    description: str

    # Signal filters
    use_breakout: bool = True
    use_volatility_filter: bool = False
    use_volume_filter: bool = False
    use_asset_trend_filter: bool = False
    use_btc_filter: bool = False
    use_price_above_ema50_filter: bool = False
    use_rsi_reclaim_filter: bool = False

    # Exit/risk settings
    stop_atr_multiplier: float = 2.0
    take_profit_r_multiple: float = 3.0
    exit_mode: ExitMode = "fixed_r_target"
    trailing_atr_multiplier: float = 3.0


@dataclass(frozen=True)
class ExperimentSettings:
    max_vol_percentile: float = 30.0
    volume_multiplier: float = 0.8
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.005
    max_position_pct: float = 1.0
    fee_rate: float = 0.006
    slippage_rate: float = 0.001
    rsi_window: int = 14
    rsi_reclaim_level: float = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_default_variants() -> list[StrategyVariant]:
    """
    Build structured experiments.

    The point is not to find pretty results by accident.
    The point is to isolate which assumptions are useful.
    """
    return [
        StrategyVariant(
            name="A_full_original",
            description=(
                "Original model: volatility compression + Donchian breakout + volume confirmation + "
                "asset trend filter + BTC trend filter, fixed 3R target."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="B_breakout_only",
            description=(
                "Plain Donchian breakout. Tests whether the breakout trigger has raw edge "
                "before adding filters."
            ),
            use_breakout=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="C_breakout_asset_trend",
            description=(
                "Breakout plus asset trend filter. Tests whether only trading breakouts when "
                "the asset is above EMA 200 with EMA 50 > EMA 200 improves quality."
            ),
            use_breakout=True,
            use_asset_trend_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="D_breakout_btc_filter",
            description=(
                "Breakout plus BTC trend filter. Tests whether broad crypto-market regime "
                "improves breakout performance."
            ),
            use_breakout=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="E_breakout_volatility",
            description=(
                "Breakout plus volatility compression only. Tests whether low realized volatility "
                "improves breakout quality or creates fake-breakout traps."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="F_breakout_volume",
            description=(
                "Breakout plus volume confirmation only. Tests whether above-average volume "
                "filters weak breakouts."
            ),
            use_breakout=True,
            use_volume_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="G_breakout_trend_btc",
            description=(
                "Breakout + asset trend + BTC trend. Removes volatility and volume filters. "
                "Tests a cleaner trend-following baseline."
            ),
            use_breakout=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="H_full_without_volatility",
            description=(
                "Full model without volatility compression: breakout + volume + asset trend + BTC trend. "
                "Directly tests whether volatility compression adds value."
            ),
            use_breakout=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="I_full_without_volume",
            description=(
                "Full model without volume confirmation: volatility compression + breakout + asset trend + BTC trend. "
                "Tests whether the volume filter helps or kills too many good setups."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="J_full_tp_2R",
            description=(
                "Original model with 2R take-profit. Tests whether 3R is too ambitious."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=2.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="K_full_tp_1_5R",
            description=(
                "Original model with 1.5R take-profit. Tests whether faster profit taking "
                "fits the breakout behavior better."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=1.5,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="L_full_wider_stop",
            description=(
                "Original model with wider 3 ATR stop and fixed 3R target. Tests whether "
                "2 ATR was too tight for crypto noise."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="M_trend_btc_tp_2R",
            description=(
                "Breakout + asset trend + BTC trend with 2R target. Cleaner trend-breakout "
                "baseline without volatility or volume filters."
            ),
            use_breakout=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=2.0,
            take_profit_r_multiple=2.0,
            exit_mode="fixed_r_target",
        ),
        StrategyVariant(
            name="N_full_ema50_exit",
            description=(
                "Original full entry model, no fixed take-profit. Uses 3 ATR initial stop, "
                "then exits when close falls below EMA 50. Tests whether medium-term trend exits "
                "let winners run better than fixed targets."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="ema50_exit",
        ),
        StrategyVariant(
            name="O_full_ema200_exit",
            description=(
                "Original full entry model, no fixed take-profit. Uses 3 ATR initial stop, "
                "then exits when close falls below EMA 200. Tests whether slower trend exits "
                "capture larger crypto moves."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="ema200_exit",
        ),
        StrategyVariant(
            name="P_full_atr_trailing",
            description=(
                "Original full entry model, no fixed take-profit. Uses 3 ATR initial stop "
                "and 3 ATR trailing stop. Tests whether dynamic trailing exits capture trends "
                "better than fixed R targets."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="atr_trailing_stop",
            trailing_atr_multiplier=3.0,
        ),
        StrategyVariant(
            name="Q_trend_btc_atr_trailing",
            description=(
                "Breakout + asset trend + BTC trend with 3 ATR trailing stop. Removes volatility "
                "and volume filters to test a cleaner trend-following version that lets winners run."
            ),
            use_breakout=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="atr_trailing_stop",
            trailing_atr_multiplier=3.0,
        ),
        StrategyVariant(
            name="R_full_donchian_low_exit",
            description=(
                "Original full entry model, no fixed take-profit. Uses 3 ATR initial stop, "
                "then exits when close falls below the previous 20-bar low. Tests a structural "
                "price-action exit."
            ),
            use_breakout=True,
            use_volatility_filter=True,
            use_volume_filter=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="donchian_low_exit",
        ),
        StrategyVariant(
            name="S_trend_btc_donchian_low_exit",
            description=(
                "Breakout + asset trend + BTC trend with Donchian-low exit. Removes volatility "
                "and volume filters, then exits on structural breakdown."
            ),
            use_breakout=True,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="donchian_low_exit",
        ),
        StrategyVariant(
            name="T_low_vol_trend_ema200_exit",
            description=(
                "Low-volatility compression + asset trend + BTC trend, no Donchian breakout, "
                "3 ATR initial stop and EMA200 exit. Tests whether volatility compression is useful "
                "without requiring a breakout trigger."
            ),
            use_breakout=False,
            use_volatility_filter=True,
            use_volume_filter=False,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="ema200_exit",
        ),
        StrategyVariant(
            name="U_trend_only_ema200_exit",
            description=(
                "Asset trend + BTC trend only, no volatility, breakout, or volume filter, "
                "3 ATR initial stop and EMA200 exit. This is the dumb-but-important baseline: "
                "if it wins, the fancy filters are probably hurting."
            ),
            use_breakout=False,
            use_volatility_filter=False,
            use_volume_filter=False,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="ema200_exit",
        ),
        StrategyVariant(
            name="V_pullback_reclaim_ema200_exit",
            description=(
                "Asset trend + BTC trend + RSI reclaim above 50, no Donchian breakout, "
                "3 ATR initial stop and EMA200 exit. Tests pullback/momentum-recovery entries "
                "instead of buying breakouts."
            ),
            use_breakout=False,
            use_volatility_filter=False,
            use_volume_filter=False,
            use_asset_trend_filter=True,
            use_btc_filter=True,
            use_rsi_reclaim_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="ema200_exit",
        ),
        StrategyVariant(
            name="W_ema50_breakout_ema200_exit",
            description=(
                "Donchian breakout + price above EMA50 + BTC trend, 3 ATR initial stop and EMA200 exit. "
                "Tests a simpler breakout variant without volatility and volume filters."
            ),
            use_breakout=True,
            use_volatility_filter=False,
            use_volume_filter=False,
            use_asset_trend_filter=False,
            use_btc_filter=True,
            use_price_above_ema50_filter=True,
            stop_atr_multiplier=3.0,
            take_profit_r_multiple=3.0,
            exit_mode="ema200_exit",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers for experiment-only features
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder-style RSI.

    This is kept here instead of src.indicators.py because RSI is only used for
    the experimental pullback-reclaim variant for now.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ─────────────────────────────────────────────────────────────────────────────
# Signal construction
# ─────────────────────────────────────────────────────────────────────────────

def add_base_signal_columns(
    features: pd.DataFrame,
    settings: ExperimentSettings,
) -> pd.DataFrame:
    """
    Add reusable boolean condition columns used by all variants.
    """
    data = add_btc_regime_filter(features).copy()
    data = data.sort_values(["product_id", "time"]).reset_index(drop=True)

    data["low_volatility"] = (
        data["realized_vol_percentile_200"] <= settings.max_vol_percentile
    )

    data["price_breakout"] = data["close"] > data["donchian_high_20"]

    data["volume_confirmed"] = (
        data["volume"] > data["volume_sma_20"] * settings.volume_multiplier
    )

    data["asset_bull_regime"] = (
        (data["close"] > data["ema_200"])
        & (data["ema_50"] > data["ema_200"])
    )

    data["btc_bull_regime"] = data["btc_close"] > data["btc_ema_200"]

    enriched_frames: list[pd.DataFrame] = []

    for _, group in data.groupby("product_id", sort=False):
        g = group.copy()
        g["rsi_14"] = calculate_rsi(g["close"], settings.rsi_window)
        g["rsi_reclaim_50"] = (
            (g["rsi_14"].shift(1) < settings.rsi_reclaim_level)
            & (g["rsi_14"] >= settings.rsi_reclaim_level)
        )
        g["price_above_ema50"] = g["close"] > g["ema_50"]
        enriched_frames.append(g)

    return pd.concat(enriched_frames, ignore_index=True)


def apply_variant_signal(
    base_data: pd.DataFrame,
    variant: StrategyVariant,
) -> pd.DataFrame:
    """
    Create long_signal for one strategy variant.
    """
    data = base_data.copy()

    conditions: list[pd.Series] = []

    if variant.use_breakout:
        conditions.append(data["price_breakout"])

    if variant.use_volatility_filter:
        conditions.append(data["low_volatility"])

    if variant.use_volume_filter:
        conditions.append(data["volume_confirmed"])

    if variant.use_asset_trend_filter:
        conditions.append(data["asset_bull_regime"])

    if variant.use_btc_filter:
        conditions.append(data["btc_bull_regime"])

    if variant.use_price_above_ema50_filter:
        conditions.append(data["price_above_ema50"])

    if variant.use_rsi_reclaim_filter:
        conditions.append(data["rsi_reclaim_50"])

    if not conditions:
        raise ValueError(f"Variant {variant.name} has no signal conditions.")

    data["long_signal"] = pd.concat(conditions, axis=1).all(axis=1)

    data.loc[data["atr_14"].isna(), "long_signal"] = False
    data.loc[data["ema_200"].isna(), "long_signal"] = False

    if variant.use_breakout:
        data.loc[data["donchian_high_20"].isna(), "long_signal"] = False

    if variant.use_rsi_reclaim_filter:
        data.loc[data["rsi_14"].isna(), "long_signal"] = False

    data["entry_price"] = data["close"]
    data["stop_price"] = data["entry_price"] - (
        variant.stop_atr_multiplier * data["atr_14"]
    )
    data["risk_per_unit"] = data["entry_price"] - data["stop_price"]
    data["take_profit_price"] = data["entry_price"] + (
        variant.take_profit_r_multiple * data["risk_per_unit"]
    )

    data["variant"] = variant.name
    data["variant_description"] = variant.description

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Experiment execution
# ─────────────────────────────────────────────────────────────────────────────

def summarize_variant_signals(
    signals: pd.DataFrame,
    variant: StrategyVariant,
) -> pd.DataFrame:
    """
    Count signals by product for one variant.
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

    summary["signal_rate_pct"] = 100.0 * summary["long_signals"] / summary["rows"]
    summary["variant"] = variant.name
    summary["variant_description"] = variant.description
    summary["exit_mode"] = variant.exit_mode
    summary["stop_atr_multiplier"] = variant.stop_atr_multiplier
    summary["take_profit_r_multiple"] = variant.take_profit_r_multiple
    summary["trailing_atr_multiplier"] = variant.trailing_atr_multiplier
    summary["uses_volatility_filter"] = variant.use_volatility_filter
    summary["uses_volume_filter"] = variant.use_volume_filter
    summary["uses_asset_trend_filter"] = variant.use_asset_trend_filter
    summary["uses_btc_filter"] = variant.use_btc_filter
    summary["uses_price_above_ema50_filter"] = variant.use_price_above_ema50_filter
    summary["uses_rsi_reclaim_filter"] = variant.use_rsi_reclaim_filter

    return summary


def run_single_variant(
    base_data: pd.DataFrame,
    variant: StrategyVariant,
    settings: ExperimentSettings,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate signals and run the backtest for one variant.
    """
    variant_signals = apply_variant_signal(base_data, variant)

    backtest_config = BacktestConfig(
        initial_equity=settings.initial_equity,
        risk_per_trade=settings.risk_per_trade,
        max_position_pct=settings.max_position_pct,
        fee_rate=settings.fee_rate,
        slippage_rate=settings.slippage_rate,
        stop_atr_multiplier=variant.stop_atr_multiplier,
        take_profit_r_multiple=variant.take_profit_r_multiple,
        exit_mode=variant.exit_mode,
        trailing_atr_multiplier=variant.trailing_atr_multiplier,
    )

    trades, equity_curves, metrics = run_backtests(
        signals=variant_signals,
        config=backtest_config,
        timeframe=timeframe,
    )

    for df in [trades, equity_curves, metrics]:
        if not df.empty:
            df["variant"] = variant.name
            df["variant_description"] = variant.description
            df["exit_mode"] = variant.exit_mode
            df["stop_atr_multiplier"] = variant.stop_atr_multiplier
            df["take_profit_r_multiple"] = variant.take_profit_r_multiple
            df["trailing_atr_multiplier"] = variant.trailing_atr_multiplier
            df["uses_volatility_filter"] = variant.use_volatility_filter
            df["uses_volume_filter"] = variant.use_volume_filter
            df["uses_asset_trend_filter"] = variant.use_asset_trend_filter
            df["uses_btc_filter"] = variant.use_btc_filter
            df["uses_price_above_ema50_filter"] = variant.use_price_above_ema50_filter
            df["uses_rsi_reclaim_filter"] = variant.use_rsi_reclaim_filter

    signal_summary = summarize_variant_signals(variant_signals, variant)

    return trades, equity_curves, metrics, signal_summary


def run_experiments(
    candles: pd.DataFrame,
    settings: ExperimentSettings,
    timeframe: str,
    variants: list[StrategyVariant] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all strategy variants and combine outputs.
    """
    if variants is None:
        variants = build_default_variants()

    features = add_volatility_strategy_features(candles)
    base_data = add_base_signal_columns(features, settings)

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_signal_summaries: list[pd.DataFrame] = []

    for variant in variants:
        print(f"Running variant: {variant.name}")

        trades, equity_curves, metrics, signal_summary = run_single_variant(
            base_data=base_data,
            variant=variant,
            settings=settings,
            timeframe=timeframe,
        )

        if not trades.empty:
            all_trades.append(trades)
        if not equity_curves.empty:
            all_equity.append(equity_curves)
        if not metrics.empty:
            all_metrics.append(metrics)
        if not signal_summary.empty:
            all_signal_summaries.append(signal_summary)

    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame()
    )
    equity_df = (
        pd.concat(all_equity, ignore_index=True)
        if all_equity
        else pd.DataFrame()
    )
    metrics_df = (
        pd.concat(all_metrics, ignore_index=True)
        if all_metrics
        else pd.DataFrame()
    )
    signal_summary_df = (
        pd.concat(all_signal_summaries, ignore_index=True)
        if all_signal_summaries
        else pd.DataFrame()
    )

    return trades_df, equity_df, metrics_df, signal_summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ─────────────────────────────────────────────────────────────────────────────

def build_variant_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Build a variant-level summary across products.

    This is not a portfolio backtest. It averages independent asset backtests.
    """
    if metrics.empty:
        return pd.DataFrame()

    leaderboard = (
        metrics.groupby("variant")
        .agg(
            avg_total_return_pct=("total_return_pct", "mean"),
            median_total_return_pct=("total_return_pct", "median"),
            avg_buy_hold_return_pct=("buy_hold_return_pct", "mean"),
            avg_excess_return_pct=("excess_return_vs_buy_hold_pct", "mean"),
            total_trades=("num_trades", "sum"),
            avg_win_rate_pct=("win_rate_pct", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
            avg_sharpe_ratio=("sharpe_ratio", "mean"),
            products_tested=("product_id", "nunique"),
        )
        .reset_index()
    )

    metadata_cols = [
        "variant",
        "variant_description",
        "exit_mode",
        "stop_atr_multiplier",
        "take_profit_r_multiple",
        "trailing_atr_multiplier",
        "uses_volatility_filter",
        "uses_volume_filter",
        "uses_asset_trend_filter",
        "uses_btc_filter",
        "uses_price_above_ema50_filter",
        "uses_rsi_reclaim_filter",
    ]

    metadata = (
        metrics[[col for col in metadata_cols if col in metrics.columns]]
        .drop_duplicates(subset=["variant"])
    )

    leaderboard = leaderboard.merge(metadata, on="variant", how="left")

    leaderboard = leaderboard.sort_values(
        ["avg_excess_return_pct", "avg_total_return_pct"],
        ascending=False,
    ).reset_index(drop=True)

    return leaderboard


def build_product_variant_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Total return by product and variant.
    """
    if metrics.empty:
        return pd.DataFrame()

    return (
        metrics.pivot_table(
            index="variant",
            columns="product_id",
            values="total_return_pct",
            aggfunc="mean",
        )
        .reset_index()
    )


def save_experiment_outputs(
    output_dir: Path,
    timeframe: str,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: pd.DataFrame,
    signal_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    product_matrix: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "trades": output_dir / f"experiment_trades_{timeframe}.csv",
        "equity": output_dir / f"experiment_equity_{timeframe}.csv",
        "metrics": output_dir / f"experiment_metrics_{timeframe}.csv",
        "signal_summary": output_dir / f"experiment_signal_summary_{timeframe}.csv",
        "leaderboard": output_dir / f"experiment_leaderboard_{timeframe}.csv",
        "product_matrix": output_dir / f"experiment_product_matrix_{timeframe}.csv",
    }

    trades.to_csv(paths["trades"], index=False)
    equity.to_csv(paths["equity"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    signal_summary.to_csv(paths["signal_summary"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    product_matrix.to_csv(paths["product_matrix"], index=False)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strategy-variant experiments.")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw OHLCV CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Directory where experiment outputs will be saved.",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe label in raw CSV filenames, e.g. 1m, 5m, 1h, 1d.",
    )

    parser.add_argument(
        "--max-vol-percentile",
        type=float,
        default=30.0,
        help="Volatility percentile threshold for variants that use compression.",
    )

    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=0.8,
        help="Volume multiplier for variants that use volume confirmation.",
    )

    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="Starting equity for each independent product backtest.",
    )

    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.005,
        help="Fraction of equity risked per trade. 0.005 = 0.5%.",
    )

    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=1.0,
        help="Maximum notional exposure per trade as a fraction of equity. 1.0 = 100%.",
    )

    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.006,
        help="Fee rate per side. 0.006 = 0.6%.",
    )

    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.001,
        help="Adverse slippage per side. 0.001 = 0.1%.",
    )

    parser.add_argument(
        "--rsi-window",
        type=int,
        default=14,
        help="RSI window used by pullback-reclaim variants.",
    )

    parser.add_argument(
        "--rsi-reclaim-level",
        type=float,
        default=50.0,
        help="RSI reclaim level used by pullback-reclaim variants.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = ExperimentSettings(
        max_vol_percentile=args.max_vol_percentile,
        volume_multiplier=args.volume_multiplier,
        initial_equity=args.initial_equity,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        rsi_window=args.rsi_window,
        rsi_reclaim_level=args.rsi_reclaim_level,
    )

    candles = load_raw_candles(args.input_dir, args.timeframe)

    trades, equity, metrics, signal_summary = run_experiments(
        candles=candles,
        settings=settings,
        timeframe=args.timeframe,
    )

    leaderboard = build_variant_leaderboard(metrics)
    product_matrix = build_product_variant_matrix(metrics)

    paths = save_experiment_outputs(
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        trades=trades,
        equity=equity,
        metrics=metrics,
        signal_summary=signal_summary,
        leaderboard=leaderboard,
        product_matrix=product_matrix,
    )

    print("\nVariant leaderboard:")

    if leaderboard.empty:
        print("No leaderboard generated.")
    else:
        display_cols = [
            "variant",
            "avg_total_return_pct",
            "avg_buy_hold_return_pct",
            "avg_excess_return_pct",
            "total_trades",
            "avg_win_rate_pct",
            "avg_profit_factor",
            "avg_max_drawdown_pct",
            "avg_sharpe_ratio",
            "exit_mode",
        ]
        display_cols = [col for col in display_cols if col in leaderboard.columns]
        print(leaderboard[display_cols].to_string(index=False))

    print("\nSaved experiment outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
