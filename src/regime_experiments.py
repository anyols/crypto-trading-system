"""
Regime-allocation experiment runner.

Run from project root:

    python -m src.regime_experiments --timeframe 1h

Example:

    python -m src.regime_experiments --timeframe 1h --position-pct 1.0

Purpose:
    The volatility-compression breakout strategy family failed because it traded too
    often, suffered fee drag, and failed to capture large crypto upside.

    This file tests a different strategy family:

        Volatility-Regime Allocation

    Instead of trying to time every breakout, the model decides whether each asset
    should be risk-on or risk-off:

        risk-on  = hold spot exposure
        risk-off = move to cash

    This is designed to:
        - reduce churn
        - capture longer crypto trends
        - avoid the worst high-volatility / trend-breakdown regimes
        - compare directly against buy-and-hold

Backtest assumptions:
    - long-only spot exposure
    - one position per asset
    - independent backtest per asset
    - enter on next candle open after risk-on transition
    - exit on next candle open after risk-off transition
    - fees and slippage included
    - no leverage
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import (
    calculate_buy_and_hold_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    periods_per_year_from_timeframe,
)
from src.indicators import add_volatility_strategy_features
from src.strategy import add_btc_regime_filter, load_raw_candles


@dataclass(frozen=True)
class RegimeVariant:
    name: str
    description: str
    entry_rule: str
    exit_rule: str


@dataclass(frozen=True)
class RegimeExperimentSettings:
    initial_equity: float = 10_000.0
    position_pct: float = 1.0
    fee_rate: float = 0.006
    slippage_rate: float = 0.001
    vol_entry_max_percentile: float = 80.0
    vol_exit_min_percentile: float = 90.0
    low_vol_entry_max_percentile: float = 30.0


@dataclass
class OpenRegimePosition:
    product_id: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    entry_fee: float
    entry_cash_equity: float
    entry_index: int


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_variants() -> list[RegimeVariant]:
    """
    Build a small, disciplined set of regime-allocation variants.

    These are not random parameter guesses. Each one tests a clean idea.
    """
    return [
        RegimeVariant(
            name="RA_A_trend_asset_btc",
            description=(
                "Trend-only regime allocation. Hold when asset close > EMA200 and BTC close > BTC EMA200. "
                "Exit when either trend breaks. This is the core risk-on/risk-off baseline."
            ),
            entry_rule="asset_trend and btc_trend",
            exit_rule="not asset_trend or not btc_trend",
        ),
        RegimeVariant(
            name="RA_B_trend_vol_danger_filter",
            description=(
                "Trend allocation with volatility danger filter. Enter only when asset and BTC trends are bullish "
                "and realized-volatility percentile is below the entry threshold. Exit on trend break or extreme volatility spike."
            ),
            entry_rule="asset_trend and btc_trend and vol_pct < vol_entry_max",
            exit_rule="not asset_trend or not btc_trend or vol_pct > vol_exit_min",
        ),
        RegimeVariant(
            name="RA_C_strong_trend",
            description=(
                "Stronger trend allocation. Hold only when asset close > EMA200, EMA50 > EMA200, and BTC is bullish. "
                "This tests whether stronger trend confirmation improves risk-adjusted performance."
            ),
            entry_rule="asset_strong_trend and btc_trend",
            exit_rule="not asset_trend or not btc_trend or ema50 < ema200",
        ),
        RegimeVariant(
            name="RA_D_low_vol_accumulation",
            description=(
                "Low-volatility accumulation regime. Enter when asset and BTC trends are bullish and volatility percentile is low. "
                "Exit on trend break or volatility shock. Tests whether low-volatility regimes are good accumulation windows."
            ),
            entry_rule="asset_trend and btc_trend and vol_pct < low_vol_entry_max",
            exit_rule="not asset_trend or not btc_trend or vol_pct > vol_exit_min",
        ),
        RegimeVariant(
            name="RA_E_asset_trend_only",
            description=(
                "Asset-only trend baseline. Hold when asset close > EMA200. No BTC filter, no volatility filter. "
                "This tests whether the BTC filter helps or hurts."
            ),
            entry_rule="asset_trend",
            exit_rule="not asset_trend",
        ),
        RegimeVariant(
            name="RA_F_btc_gated_strong_trend_no_vol",
            description=(
                "BTC-gated strong trend without volatility filter. Hold when asset is in strong trend and BTC is bullish. "
                "This isolates trend confirmation without volatility timing."
            ),
            entry_rule="asset_strong_trend and btc_trend",
            exit_rule="not asset_strong_trend or not btc_trend",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

def prepare_regime_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature columns required by regime strategies.
    """
    features = add_volatility_strategy_features(candles)
    data = add_btc_regime_filter(features).copy()
    data = data.sort_values(["product_id", "time"]).reset_index(drop=True)

    data["asset_trend"] = data["close"] > data["ema_200"]
    data["asset_strong_trend"] = (
        (data["close"] > data["ema_200"])
        & (data["ema_50"] > data["ema_200"])
    )
    data["btc_trend"] = data["btc_close"] > data["btc_ema_200"]
    data["ema50_below_ema200"] = data["ema_50"] < data["ema_200"]

    return data


def add_regime_signals(
    data: pd.DataFrame,
    variant: RegimeVariant,
    settings: RegimeExperimentSettings,
) -> pd.DataFrame:
    """
    Create stateful risk_on, long_signal, and exit_signal columns.

    long_signal:
        True only when the system transitions from risk-off to risk-on.

    exit_signal:
        True only when the system transitions from risk-on to risk-off.
    """
    result_frames: list[pd.DataFrame] = []

    for product_id, group in data.groupby("product_id", sort=False):
        g = group.copy().sort_values("time").reset_index(drop=True)

        vol_pct = g["realized_vol_percentile_200"]

        if variant.name == "RA_A_trend_asset_btc":
            entry_condition = g["asset_trend"] & g["btc_trend"]
            exit_condition = (~g["asset_trend"]) | (~g["btc_trend"])

        elif variant.name == "RA_B_trend_vol_danger_filter":
            entry_condition = (
                g["asset_trend"]
                & g["btc_trend"]
                & (vol_pct < settings.vol_entry_max_percentile)
            )
            exit_condition = (
                (~g["asset_trend"])
                | (~g["btc_trend"])
                | (vol_pct > settings.vol_exit_min_percentile)
            )

        elif variant.name == "RA_C_strong_trend":
            entry_condition = g["asset_strong_trend"] & g["btc_trend"]
            exit_condition = (
                (~g["asset_trend"])
                | (~g["btc_trend"])
                | g["ema50_below_ema200"]
            )

        elif variant.name == "RA_D_low_vol_accumulation":
            entry_condition = (
                g["asset_trend"]
                & g["btc_trend"]
                & (vol_pct < settings.low_vol_entry_max_percentile)
            )
            exit_condition = (
                (~g["asset_trend"])
                | (~g["btc_trend"])
                | (vol_pct > settings.vol_exit_min_percentile)
            )

        elif variant.name == "RA_E_asset_trend_only":
            entry_condition = g["asset_trend"]
            exit_condition = ~g["asset_trend"]

        elif variant.name == "RA_F_btc_gated_strong_trend_no_vol":
            entry_condition = g["asset_strong_trend"] & g["btc_trend"]
            exit_condition = (~g["asset_strong_trend"]) | (~g["btc_trend"])

        else:
            raise ValueError(f"Unknown regime variant: {variant.name}")

        entry_condition = entry_condition.fillna(False)
        exit_condition = exit_condition.fillna(False)

        risk_on_values: list[bool] = []
        long_signal_values: list[bool] = []
        exit_signal_values: list[bool] = []

        risk_on = False

        for entry_ok, exit_ok in zip(entry_condition, exit_condition):
            long_signal = False
            exit_signal = False

            if not risk_on and bool(entry_ok):
                risk_on = True
                long_signal = True
            elif risk_on and bool(exit_ok):
                risk_on = False
                exit_signal = True

            risk_on_values.append(risk_on)
            long_signal_values.append(long_signal)
            exit_signal_values.append(exit_signal)

        g["risk_on"] = risk_on_values
        g["long_signal"] = long_signal_values
        g["exit_signal"] = exit_signal_values
        g["variant"] = variant.name
        g["variant_description"] = variant.description
        g["entry_rule"] = variant.entry_rule
        g["exit_rule"] = variant.exit_rule

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Regime allocation backtester
# ─────────────────────────────────────────────────────────────────────────────

def enter_position(
    product_id: str,
    row: pd.Series,
    row_index: int,
    equity: float,
    settings: RegimeExperimentSettings,
) -> tuple[OpenRegimePosition | None, float]:
    """
    Enter long exposure at the current candle open.

    Position size is allocation-based, not risk-per-trade based.
    If position_pct = 1.0, deploy almost all equity, reserving cash for fees.
    """
    if equity <= 0:
        return None, equity

    raw_entry_price = float(row["open"])

    if raw_entry_price <= 0:
        return None, equity

    entry_price = raw_entry_price * (1.0 + settings.slippage_rate)

    capital_to_deploy = equity * settings.position_pct
    notional = capital_to_deploy / (1.0 + settings.fee_rate)
    qty = notional / entry_price
    entry_fee = notional * settings.fee_rate

    cash_after_entry = equity - notional - entry_fee

    if qty <= 0 or cash_after_entry < -1e-8:
        return None, equity

    position = OpenRegimePosition(
        product_id=product_id,
        entry_time=row["time"],
        entry_price=entry_price,
        qty=qty,
        entry_fee=entry_fee,
        entry_cash_equity=equity,
        entry_index=row_index,
    )

    return position, cash_after_entry


def exit_position(
    position: OpenRegimePosition,
    row: pd.Series,
    row_index: int,
    cash: float,
    settings: RegimeExperimentSettings,
    reason: str,
) -> tuple[dict[str, float | str | pd.Timestamp], float]:
    raw_exit_price = float(row["open"])
    exit_price = raw_exit_price * (1.0 - settings.slippage_rate)

    proceeds = position.qty * exit_price
    exit_fee = proceeds * settings.fee_rate
    cash_after_exit = cash + proceeds - exit_fee

    gross_pnl = (exit_price - position.entry_price) * position.qty
    net_pnl = gross_pnl - position.entry_fee - exit_fee

    notional_entry = position.entry_price * position.qty
    return_pct = (net_pnl / notional_entry) * 100.0 if notional_entry > 0 else 0.0

    trade = {
        "product_id": position.product_id,
        "entry_time": position.entry_time,
        "exit_time": row["time"],
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "qty": position.qty,
        "entry_fee": position.entry_fee,
        "exit_fee": exit_fee,
        "total_fees": position.entry_fee + exit_fee,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "exit_reason": reason,
        "holding_bars": int(row_index - position.entry_index + 1),
    }

    return trade, cash_after_exit


def run_regime_backtest_for_product(
    signals: pd.DataFrame,
    product_id: str,
    variant: RegimeVariant,
    settings: RegimeExperimentSettings,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    data = (
        signals[signals["product_id"] == product_id]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(data) < 2:
        return pd.DataFrame(), pd.DataFrame(), empty_metrics(product_id, variant, settings)

    cash = settings.initial_equity
    position: OpenRegimePosition | None = None
    pending_entry = False
    pending_exit = False

    trades: list[dict[str, float | str | pd.Timestamp]] = []
    equity_records: list[dict[str, float | str | pd.Timestamp | bool]] = []

    for i, row in data.iterrows():
        # Execute pending exit first. Do not allow same-bar exit and re-entry.
        exited_this_bar = False

        if position is not None and pending_exit:
            trade, cash = exit_position(
                position=position,
                row=row,
                row_index=i,
                cash=cash,
                settings=settings,
                reason="regime_exit",
            )
            trades.append(trade)
            position = None
            pending_exit = False
            exited_this_bar = True

        if position is None and pending_entry and not exited_this_bar:
            position, cash = enter_position(
                product_id=product_id,
                row=row,
                row_index=i,
                equity=cash,
                settings=settings,
            )
            pending_entry = False

        if position is not None:
            mark_price = float(row["close"])
            position_value = position.qty * mark_price
            equity = cash + position_value
            unrealized_pnl = (mark_price - position.entry_price) * position.qty
            in_position = True
        else:
            equity = cash
            unrealized_pnl = 0.0
            in_position = False

        equity_records.append(
            {
                "time": row["time"],
                "product_id": product_id,
                "equity": equity,
                "cash": cash,
                "unrealized_pnl": unrealized_pnl,
                "close": float(row["close"]),
                "in_position": in_position,
                "risk_on": bool(row.get("risk_on", False)),
                "variant": variant.name,
                "variant_description": variant.description,
                "entry_rule": variant.entry_rule,
                "exit_rule": variant.exit_rule,
            }
        )

        # Schedule next-bar actions based on signals observed at this candle close.
        if i < len(data) - 1:
            if position is None and bool(row.get("long_signal", False)):
                pending_entry = True
            elif position is not None and bool(row.get("exit_signal", False)):
                pending_exit = True

    # Force close at final candle open/close equivalent if still in position.
    if position is not None:
        last_row = data.iloc[-1]
        trade, cash = exit_position(
            position=position,
            row=last_row,
            row_index=len(data) - 1,
            cash=cash,
            settings=settings,
            reason="end_of_data",
        )
        trades.append(trade)

        if equity_records:
            equity_records[-1]["equity"] = cash
            equity_records[-1]["cash"] = cash
            equity_records[-1]["unrealized_pnl"] = 0.0
            equity_records[-1]["in_position"] = False

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records)

    metrics = calculate_regime_metrics(
        product_id=product_id,
        variant=variant,
        trades=trades_df,
        equity_curve=equity_df,
        product_data=data,
        settings=settings,
        timeframe=timeframe,
    )

    return trades_df, equity_df, metrics


def empty_metrics(
    product_id: str,
    variant: RegimeVariant,
    settings: RegimeExperimentSettings,
) -> dict[str, float | int | str]:
    return {
        "product_id": product_id,
        "variant": variant.name,
        "variant_description": variant.description,
        "entry_rule": variant.entry_rule,
        "exit_rule": variant.exit_rule,
        "initial_equity": settings.initial_equity,
        "ending_equity": settings.initial_equity,
        "total_return_pct": 0.0,
        "buy_hold_return_pct": 0.0,
        "excess_return_vs_buy_hold_pct": 0.0,
        "num_trades": 0,
        "win_rate_pct": 0.0,
        "avg_trade_return_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "exposure_time_pct": 0.0,
        "avg_holding_bars": 0.0,
        "total_fees": 0.0,
    }


def calculate_regime_metrics(
    product_id: str,
    variant: RegimeVariant,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    product_data: pd.DataFrame,
    settings: RegimeExperimentSettings,
    timeframe: str,
) -> dict[str, float | int | str]:
    ending_equity = (
        float(equity_curve["equity"].iloc[-1])
        if not equity_curve.empty
        else settings.initial_equity
    )

    total_return = ending_equity / settings.initial_equity - 1.0
    buy_hold_return = calculate_buy_and_hold_return(product_data)

    max_drawdown = (
        calculate_max_drawdown(equity_curve["equity"])
        if not equity_curve.empty
        else 0.0
    )

    sharpe = calculate_sharpe_ratio(
        equity_curve=equity_curve,
        periods_per_year=periods_per_year_from_timeframe(timeframe),
    )

    exposure_time = (
        float(equity_curve["in_position"].mean() * 100.0)
        if not equity_curve.empty and "in_position" in equity_curve.columns
        else 0.0
    )

    base = {
        "product_id": product_id,
        "variant": variant.name,
        "variant_description": variant.description,
        "entry_rule": variant.entry_rule,
        "exit_rule": variant.exit_rule,
        "initial_equity": settings.initial_equity,
        "ending_equity": ending_equity,
        "total_return_pct": total_return * 100.0,
        "buy_hold_return_pct": buy_hold_return * 100.0,
        "excess_return_vs_buy_hold_pct": (total_return - buy_hold_return) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
        "exposure_time_pct": exposure_time,
    }

    if trades.empty:
        base.update(
            {
                "num_trades": 0,
                "win_rate_pct": 0.0,
                "avg_trade_return_pct": 0.0,
                "profit_factor": 0.0,
                "avg_holding_bars": 0.0,
                "total_fees": 0.0,
            }
        )
        return base

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]

    gross_profit = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["net_pnl"].sum()) if not losses.empty else 0.0

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    base.update(
        {
            "num_trades": int(len(trades)),
            "win_rate_pct": float((trades["net_pnl"] > 0).mean() * 100.0),
            "avg_trade_return_pct": float(trades["return_pct"].mean()),
            "profit_factor": float(profit_factor),
            "avg_holding_bars": float(trades["holding_bars"].mean()),
            "total_fees": float(trades["total_fees"].sum()),
        }
    )

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Experiment orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_regime_experiments(
    candles: pd.DataFrame,
    settings: RegimeExperimentSettings,
    timeframe: str,
    variants: list[RegimeVariant] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if variants is None:
        variants = build_regime_variants()

    base_features = prepare_regime_features(candles)

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float | int | str]] = []
    all_signal_summary: list[pd.DataFrame] = []

    product_ids = sorted(base_features["product_id"].dropna().unique())

    for variant in variants:
        print(f"Running regime variant: {variant.name}")
        signals = add_regime_signals(base_features, variant, settings)

        signal_summary = summarize_regime_signals(signals, variant)
        all_signal_summary.append(signal_summary)

        for product_id in product_ids:
            trades, equity, metrics = run_regime_backtest_for_product(
                signals=signals,
                product_id=product_id,
                variant=variant,
                settings=settings,
                timeframe=timeframe,
            )

            if not trades.empty:
                trades["variant"] = variant.name
                trades["variant_description"] = variant.description
                trades["entry_rule"] = variant.entry_rule
                trades["exit_rule"] = variant.exit_rule
                all_trades.append(trades)

            if not equity.empty:
                all_equity.append(equity)

            all_metrics.append(metrics)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)
    signal_summary_df = (
        pd.concat(all_signal_summary, ignore_index=True)
        if all_signal_summary
        else pd.DataFrame()
    )

    return trades_df, equity_df, metrics_df, signal_summary_df


def summarize_regime_signals(signals: pd.DataFrame, variant: RegimeVariant) -> pd.DataFrame:
    summary = (
        signals.groupby("product_id")
        .agg(
            rows=("time", "count"),
            long_signals=("long_signal", "sum"),
            exit_signals=("exit_signal", "sum"),
            risk_on_rate_pct=("risk_on", lambda x: float(x.mean() * 100.0)),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
    )

    summary["variant"] = variant.name
    summary["variant_description"] = variant.description
    summary["entry_rule"] = variant.entry_rule
    summary["exit_rule"] = variant.exit_rule

    return summary


def build_regime_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
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
            avg_exposure_time_pct=("exposure_time_pct", "mean"),
            avg_holding_bars=("avg_holding_bars", "mean"),
            total_fees=("total_fees", "sum"),
            products_tested=("product_id", "nunique"),
        )
        .reset_index()
    )

    metadata = (
        metrics[["variant", "variant_description", "entry_rule", "exit_rule"]]
        .drop_duplicates(subset=["variant"])
    )

    leaderboard = leaderboard.merge(metadata, on="variant", how="left")

    leaderboard = leaderboard.sort_values(
        ["avg_excess_return_pct", "avg_total_return_pct"],
        ascending=False,
    ).reset_index(drop=True)

    return leaderboard


def build_regime_product_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
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


def save_regime_outputs(
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
        "trades": output_dir / f"regime_trades_{timeframe}.csv",
        "equity": output_dir / f"regime_equity_{timeframe}.csv",
        "metrics": output_dir / f"regime_metrics_{timeframe}.csv",
        "signal_summary": output_dir / f"regime_signal_summary_{timeframe}.csv",
        "leaderboard": output_dir / f"regime_leaderboard_{timeframe}.csv",
        "product_matrix": output_dir / f"regime_product_matrix_{timeframe}.csv",
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
    parser = argparse.ArgumentParser(description="Run volatility-regime allocation experiments.")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw OHLCV CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/regime_experiments"),
        help="Directory where regime experiment outputs will be saved.",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe label in raw CSV filenames, e.g. 1m, 5m, 1h, 1d.",
    )

    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        help="Starting equity for each independent asset backtest.",
    )

    parser.add_argument(
        "--position-pct",
        type=float,
        default=1.0,
        help="Fraction of equity allocated when risk-on. 1.0 = 100% spot exposure.",
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
        "--vol-entry-max-percentile",
        type=float,
        default=80.0,
        help="Maximum volatility percentile allowed for broad risk-on entry in volatility-filtered variants.",
    )

    parser.add_argument(
        "--vol-exit-min-percentile",
        type=float,
        default=90.0,
        help="Volatility percentile above which volatility-filtered variants exit risk-on exposure.",
    )

    parser.add_argument(
        "--low-vol-entry-max-percentile",
        type=float,
        default=30.0,
        help="Maximum volatility percentile for low-volatility accumulation variants.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = RegimeExperimentSettings(
        initial_equity=args.initial_equity,
        position_pct=args.position_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        vol_entry_max_percentile=args.vol_entry_max_percentile,
        vol_exit_min_percentile=args.vol_exit_min_percentile,
        low_vol_entry_max_percentile=args.low_vol_entry_max_percentile,
    )

    candles = load_raw_candles(args.input_dir, args.timeframe)

    trades, equity, metrics, signal_summary = run_regime_experiments(
        candles=candles,
        settings=settings,
        timeframe=args.timeframe,
    )

    leaderboard = build_regime_leaderboard(metrics)
    product_matrix = build_regime_product_matrix(metrics)

    paths = save_regime_outputs(
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        trades=trades,
        equity=equity,
        metrics=metrics,
        signal_summary=signal_summary,
        leaderboard=leaderboard,
        product_matrix=product_matrix,
    )

    print("\nRegime allocation leaderboard:")

    if leaderboard.empty:
        print("No leaderboard generated.")
    else:
        display_cols = [
            "variant",
            "avg_total_return_pct",
            "avg_buy_hold_return_pct",
            "avg_excess_return_pct",
            "total_trades",
            "avg_profit_factor",
            "avg_max_drawdown_pct",
            "avg_sharpe_ratio",
            "avg_exposure_time_pct",
            "total_fees",
        ]
        display_cols = [col for col in display_cols if col in leaderboard.columns]
        print(leaderboard[display_cols].to_string(index=False))

    print("\nSaved regime experiment outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
