"""
Backtest table-driven probability signals, with long and short support.

Run from project root:

    python -m src.probability_signal_backtest --timeframe 15m

Typical competition-oriented run:

    python -m src.probability_signal_backtest \
        --timeframe 15m \
        --min-samples 500 \
        --min-edge 0.12 \
        --min-avg-return-pct 0.15 \
        --table-types vol_momentum_bucket \
        --position-pct 1.0

Optional: restrict to 4h horizon signals only:

    python -m src.probability_signal_backtest \
        --timeframe 15m \
        --allowed-horizons 240

Purpose:
    Use the probability-table grid-search results to generate actual long/short
    trades and test whether the conditional edge survives execution costs.

Core logic:
    1. Load grid_signal_candidates_*.csv from data/probability_tables.
    2. Select bullish states:
           samples >= min_samples
           P(up) - P(down) >= min_edge
           avg_future_return_pct >= min_avg_return_pct
    3. Select bearish states:
           samples >= min_samples
           P(down) - P(up) >= min_edge
           avg_future_return_pct <= -min_avg_return_pct
    4. Rebuild historical state features from candle data.
    5. Enter long/short on the next candle open when state matches a selected bucket.
    6. Hold for that candidate's horizon, then exit.
    7. Avoid overlapping positions per asset.

This is not ML yet. It is a probability-table strategy backtest.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.backtest import (
    calculate_buy_and_hold_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    periods_per_year_from_timeframe,
)
from src.probability_table import ProbabilityTableConfig, build_probability_dataset
from src.strategy import load_raw_candles


Side = Literal["long", "short"]
ExitReason = Literal["time_exit", "end_of_data"]


@dataclass(frozen=True)
class CandidateSelectionConfig:
    candidates_path: Path | None
    probability_tables_dir: Path
    table_types: list[str]
    allowed_lookbacks: list[float] | None
    allowed_horizons: list[int] | None
    min_samples: int
    min_edge: float
    min_avg_return_pct: float
    top_n_per_side: int


@dataclass(frozen=True)
class ProbabilitySignalBacktestConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    threshold_pct: float
    percentile_window: int
    bucket_size: int
    initial_equity: float
    position_pct: float
    fee_rate: float
    slippage_rate: float
    allow_longs: bool
    allow_shorts: bool


@dataclass
class ProbabilityPosition:
    product_id: str
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    entry_fee: float
    entry_equity: float
    entry_index: int
    target_exit_index: int
    horizon_bars: int
    signal_key: str
    signal_score: float
    p_up: float
    p_down: float
    avg_future_return_pct: float
    lookback_hours: float
    horizon_minutes: int


# ─────────────────────────────────────────────────────────────────────────────
# Candidate loading / parsing
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_candidates_file(probability_tables_dir: Path) -> Path:
    files = sorted(
        probability_tables_dir.glob("grid_signal_candidates_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"No grid_signal_candidates_*.csv file found in {probability_tables_dir}. "
            "Run src.probability_table first."
        )

    return files[0]


def parse_vol_bucket(signal_key: str) -> int | None:
    match = re.search(r"vol=(\d+)-(\d+)", signal_key)
    if not match:
        return None
    return int(match.group(1))


def parse_momentum_bucket(signal_key: str) -> str | None:
    match = re.search(r"momentum=(.+)$", signal_key)
    if not match:
        return None
    return match.group(1).strip()


def load_and_select_candidates(
    config: CandidateSelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    candidates_path = config.candidates_path or find_latest_candidates_file(
        config.probability_tables_dir
    )

    candidates = pd.read_csv(candidates_path)

    required_cols = {
        "table_type",
        "lookback_hours",
        "horizon_minutes",
        "signal_key",
        "samples",
        "up_down_edge",
        "down_up_edge",
        "avg_future_return_pct",
        "p_up",
        "p_down",
    }
    missing = required_cols - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidate file is missing required columns: {sorted(missing)}"
        )

    filtered = candidates[candidates["table_type"].isin(config.table_types)].copy()

    if config.allowed_lookbacks:
        filtered = filtered[filtered["lookback_hours"].isin(config.allowed_lookbacks)]

    if config.allowed_horizons:
        filtered = filtered[filtered["horizon_minutes"].isin(config.allowed_horizons)]

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

    if "bull_score_weighted" not in bullish.columns:
        bullish["bull_score_weighted"] = (
            bullish["up_down_edge"]
            * bullish["avg_future_return_pct"]
            * np.sqrt(bullish["samples"])
        )

    if "bear_score_weighted" not in bearish.columns:
        bearish["bear_score_weighted"] = (
            bearish["down_up_edge"]
            * (-bearish["avg_future_return_pct"])
            * np.sqrt(bearish["samples"])
        )

    bullish = (
        bullish.sort_values(
            ["bull_score_weighted", "up_down_edge", "avg_future_return_pct", "samples"],
            ascending=False,
        )
        .head(config.top_n_per_side)
        .copy()
    )

    bearish = (
        bearish.sort_values(
            ["bear_score_weighted", "down_up_edge", "avg_future_return_pct", "samples"],
            ascending=[False, False, True, False],
        )
        .head(config.top_n_per_side)
        .copy()
    )

    bullish["side"] = "long"
    bearish["side"] = "short"

    for df in [bullish, bearish]:
        if not df.empty:
            df["parsed_vol_bucket"] = df["signal_key"].apply(parse_vol_bucket)
            df["parsed_momentum_bucket"] = df["signal_key"].apply(parse_momentum_bucket)

    return bullish.reset_index(drop=True), bearish.reset_index(drop=True), candidates_path


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation from selected candidates
# ─────────────────────────────────────────────────────────────────────────────

def candidate_condition(dataset: pd.DataFrame, candidate: pd.Series) -> pd.Series:
    table_type = str(candidate["table_type"])
    signal_key = str(candidate["signal_key"])

    condition = pd.Series(True, index=dataset.index)

    if table_type in {"vol_bucket", "vol_momentum_bucket"}:
        vol_bucket = parse_vol_bucket(signal_key)
        if vol_bucket is None:
            raise ValueError(f"Could not parse volatility bucket from signal_key={signal_key}")
        condition &= dataset["vol_bucket"].astype("Int64") == int(vol_bucket)

    if table_type in {"momentum_bucket", "vol_momentum_bucket"}:
        momentum_bucket = parse_momentum_bucket(signal_key)
        if momentum_bucket is None:
            raise ValueError(f"Could not parse momentum bucket from signal_key={signal_key}")
        condition &= dataset["return_lookback_bucket"].astype(str) == momentum_bucket

    return condition.fillna(False)


def build_signals_for_candidates(
    candles: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    backtest_config: ProbabilitySignalBacktestConfig,
) -> pd.DataFrame:
    """
    Build one signal row per matching asset/timestamp.

    If multiple candidates fire at the same asset/timestamp, keep the strongest
    absolute weighted score.
    """
    if selected_candidates.empty:
        return pd.DataFrame()

    signal_frames: list[pd.DataFrame] = []

    combos = (
        selected_candidates[["lookback_hours", "horizon_minutes"]]
        .drop_duplicates()
        .sort_values(["lookback_hours", "horizon_minutes"])
    )

    for _, combo in combos.iterrows():
        lookback_hours = float(combo["lookback_hours"])
        horizon_minutes = int(combo["horizon_minutes"])

        candidates_for_combo = selected_candidates[
            (selected_candidates["lookback_hours"] == lookback_hours)
            & (selected_candidates["horizon_minutes"] == horizon_minutes)
        ].copy()

        dataset_config = ProbabilityTableConfig(
            input_dir=backtest_config.input_dir,
            output_dir=backtest_config.output_dir,
            timeframe=backtest_config.timeframe,
            lookback_hours=lookback_hours,
            horizon_minutes=horizon_minutes,
            threshold_pct=backtest_config.threshold_pct,
            percentile_window=backtest_config.percentile_window,
            min_samples_per_bucket=1,
            bucket_size=backtest_config.bucket_size,
        )

        dataset = build_probability_dataset(candles, dataset_config)

        for _, candidate in candidates_for_combo.iterrows():
            condition = candidate_condition(dataset, candidate)
            matches = dataset[condition].copy()

            if matches.empty:
                continue

            side = str(candidate["side"])
            score_col = "bull_score_weighted" if side == "long" else "bear_score_weighted"
            score = float(candidate.get(score_col, 0.0))

            matches["side"] = side
            matches["signal_key"] = candidate["signal_key"]
            matches["signal_score"] = score
            matches["candidate_table_type"] = candidate["table_type"]
            matches["candidate_samples"] = candidate["samples"]
            matches["candidate_p_up"] = candidate["p_up"]
            matches["candidate_p_down"] = candidate["p_down"]
            matches["candidate_up_down_edge"] = candidate["up_down_edge"]
            matches["candidate_down_up_edge"] = candidate["down_up_edge"]
            matches["candidate_avg_future_return_pct"] = candidate["avg_future_return_pct"]
            matches["candidate_lookback_hours"] = lookback_hours
            matches["candidate_horizon_minutes"] = horizon_minutes

            signal_frames.append(
                matches[
                    [
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
                    ]
                ]
            )

    if not signal_frames:
        return pd.DataFrame()

    signals = pd.concat(signal_frames, ignore_index=True)

    # If multiple candidate rules fire at the same asset/time, keep strongest.
    signals = (
        signals.sort_values("signal_score", ascending=False)
        .drop_duplicates(subset=["product_id", "time"], keep="first")
        .sort_values(["product_id", "time"])
        .reset_index(drop=True)
    )

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Long / short backtest
# ─────────────────────────────────────────────────────────────────────────────

def enter_position(
    product_id: str,
    row: pd.Series,
    signal: pd.Series,
    row_index: int,
    equity: float,
    config: ProbabilitySignalBacktestConfig,
) -> tuple[ProbabilityPosition | None, float]:
    side: Side = str(signal["side"])  # type: ignore[assignment]

    if side == "long" and not config.allow_longs:
        return None, equity

    if side == "short" and not config.allow_shorts:
        return None, equity

    raw_entry_price = float(row["open"])
    if raw_entry_price <= 0 or equity <= 0:
        return None, equity

    if side == "long":
        entry_price = raw_entry_price * (1.0 + config.slippage_rate)
    else:
        entry_price = raw_entry_price * (1.0 - config.slippage_rate)

    notional = equity * config.position_pct
    qty = notional / entry_price
    entry_fee = notional * config.fee_rate
    equity_after_fee = equity - entry_fee

    if qty <= 0 or equity_after_fee <= 0:
        return None, equity

    horizon_bars = int(signal["horizon_bars"])
    target_exit_index = row_index + horizon_bars

    position = ProbabilityPosition(
        product_id=product_id,
        side=side,
        entry_time=row["time"],
        entry_price=entry_price,
        qty=qty,
        entry_fee=entry_fee,
        entry_equity=equity,
        entry_index=row_index,
        target_exit_index=target_exit_index,
        horizon_bars=horizon_bars,
        signal_key=str(signal["signal_key"]),
        signal_score=float(signal["signal_score"]),
        p_up=float(signal["candidate_p_up"]),
        p_down=float(signal["candidate_p_down"]),
        avg_future_return_pct=float(signal["candidate_avg_future_return_pct"]),
        lookback_hours=float(signal["candidate_lookback_hours"]),
        horizon_minutes=int(signal["candidate_horizon_minutes"]),
    )

    return position, equity_after_fee


def mark_unrealized_pnl(position: ProbabilityPosition, mark_price: float) -> float:
    if position.side == "long":
        return (mark_price - position.entry_price) * position.qty

    return (position.entry_price - mark_price) * position.qty


def exit_position(
    position: ProbabilityPosition,
    row: pd.Series,
    row_index: int,
    equity_before_exit: float,
    config: ProbabilitySignalBacktestConfig,
    reason: ExitReason,
) -> tuple[dict[str, float | str | pd.Timestamp], float]:
    raw_exit_price = float(row["open"])

    if position.side == "long":
        exit_price = raw_exit_price * (1.0 - config.slippage_rate)
        gross_pnl = (exit_price - position.entry_price) * position.qty
    else:
        exit_price = raw_exit_price * (1.0 + config.slippage_rate)
        gross_pnl = (position.entry_price - exit_price) * position.qty

    exit_notional = exit_price * position.qty
    exit_fee = exit_notional * config.fee_rate
    net_pnl = gross_pnl - position.entry_fee - exit_fee
    ending_equity = equity_before_exit + gross_pnl - exit_fee

    entry_notional = position.entry_price * position.qty
    return_pct = (net_pnl / entry_notional) * 100.0 if entry_notional > 0 else 0.0

    trade = {
        "product_id": position.product_id,
        "side": position.side,
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
        "signal_key": position.signal_key,
        "signal_score": position.signal_score,
        "candidate_p_up": position.p_up,
        "candidate_p_down": position.p_down,
        "candidate_avg_future_return_pct": position.avg_future_return_pct,
        "lookback_hours": position.lookback_hours,
        "horizon_minutes": position.horizon_minutes,
    }

    return trade, ending_equity


def run_backtest_for_product(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    product_id: str,
    config: ProbabilitySignalBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    product = (
        candles[candles["product_id"] == product_id]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )

    product_signals = signals[signals["product_id"] == product_id].copy()
    signals_by_time = {
        row["time"]: row for _, row in product_signals.iterrows()
    }

    equity = config.initial_equity
    position: ProbabilityPosition | None = None
    pending_signal: pd.Series | None = None

    trades: list[dict[str, float | str | pd.Timestamp]] = []
    equity_records: list[dict[str, float | str | pd.Timestamp | bool]] = []

    if len(product) < 2:
        empty_equity = pd.DataFrame()
        return pd.DataFrame(), empty_equity, calculate_metrics(
            product_id=product_id,
            trades=pd.DataFrame(),
            equity_curve=empty_equity,
            product_data=product,
            config=config,
        )

    for i, row in product.iterrows():
        exited_this_bar = False

        # 1. Time-based exit at current candle open.
        if position is not None and i >= position.target_exit_index:
            trade, equity = exit_position(
                position=position,
                row=row,
                row_index=i,
                equity_before_exit=equity,
                config=config,
                reason="time_exit",
            )
            trades.append(trade)
            position = None
            exited_this_bar = True

        # 2. Enter pending signal at current candle open.
        if position is None and pending_signal is not None and not exited_this_bar:
            position, equity = enter_position(
                product_id=product_id,
                row=row,
                signal=pending_signal,
                row_index=i,
                equity=equity,
                config=config,
            )
            pending_signal = None

        # 3. Mark-to-market at current close.
        if position is not None:
            unrealized_pnl = mark_unrealized_pnl(position, float(row["close"]))
            mark_equity = equity + unrealized_pnl
            in_position = True
            current_side = position.side
        else:
            unrealized_pnl = 0.0
            mark_equity = equity
            in_position = False
            current_side = "flat"

        equity_records.append(
            {
                "time": row["time"],
                "product_id": product_id,
                "equity": mark_equity,
                "realized_equity": equity,
                "unrealized_pnl": unrealized_pnl,
                "close": float(row["close"]),
                "in_position": in_position,
                "side": current_side,
            }
        )

        # 4. Schedule next-bar entry from current close signal, only if flat.
        if position is None and i < len(product) - 1:
            signal = signals_by_time.get(row["time"])
            if signal is not None:
                pending_signal = signal

    # Force close at final candle if still open.
    if position is not None:
        last_row = product.iloc[-1]
        trade, equity = exit_position(
            position=position,
            row=last_row,
            row_index=len(product) - 1,
            equity_before_exit=equity,
            config=config,
            reason="end_of_data",
        )
        trades.append(trade)

        if equity_records:
            equity_records[-1]["equity"] = equity
            equity_records[-1]["realized_equity"] = equity
            equity_records[-1]["unrealized_pnl"] = 0.0
            equity_records[-1]["in_position"] = False
            equity_records[-1]["side"] = "flat"

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_records)
    metrics = calculate_metrics(product_id, trades_df, equity_curve, product, config)

    return trades_df, equity_curve, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(
    product_id: str,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    product_data: pd.DataFrame,
    config: ProbabilitySignalBacktestConfig,
) -> dict[str, float | int | str]:
    ending_equity = (
        float(equity_curve["equity"].iloc[-1])
        if not equity_curve.empty
        else config.initial_equity
    )

    total_return = ending_equity / config.initial_equity - 1.0
    buy_hold_return = calculate_buy_and_hold_return(product_data)
    max_drawdown = (
        calculate_max_drawdown(equity_curve["equity"])
        if not equity_curve.empty
        else 0.0
    )
    sharpe = calculate_sharpe_ratio(
        equity_curve,
        periods_per_year=periods_per_year_from_timeframe(config.timeframe),
    )
    exposure = (
        float(equity_curve["in_position"].mean() * 100.0)
        if not equity_curve.empty and "in_position" in equity_curve.columns
        else 0.0
    )

    base: dict[str, float | int | str] = {
        "product_id": product_id,
        "initial_equity": config.initial_equity,
        "ending_equity": ending_equity,
        "total_return_pct": total_return * 100.0,
        "buy_hold_return_pct": buy_hold_return * 100.0,
        "excess_return_vs_buy_hold_pct": (total_return - buy_hold_return) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
        "exposure_time_pct": exposure,
    }

    if trades.empty:
        base.update(
            {
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
            "long_trades": int((trades["side"] == "long").sum()),
            "short_trades": int((trades["side"] == "short").sum()),
            "win_rate_pct": float((trades["net_pnl"] > 0).mean() * 100.0),
            "avg_trade_return_pct": float(trades["return_pct"].mean()),
            "profit_factor": float(profit_factor),
            "avg_holding_bars": float(trades["holding_bars"].mean()),
            "total_fees": float(trades["total_fees"].sum()),
            "best_trade": float(trades["net_pnl"].max()),
            "worst_trade": float(trades["net_pnl"].min()),
        }
    )

    return base


def build_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    row = {
        "avg_total_return_pct": metrics["total_return_pct"].mean(),
        "median_total_return_pct": metrics["total_return_pct"].median(),
        "avg_buy_hold_return_pct": metrics["buy_hold_return_pct"].mean(),
        "avg_excess_return_pct": metrics["excess_return_vs_buy_hold_pct"].mean(),
        "total_trades": metrics["num_trades"].sum(),
        "total_long_trades": metrics["long_trades"].sum(),
        "total_short_trades": metrics["short_trades"].sum(),
        "avg_win_rate_pct": metrics["win_rate_pct"].mean(),
        "avg_profit_factor": metrics["profit_factor"].mean(),
        "avg_max_drawdown_pct": metrics["max_drawdown_pct"].mean(),
        "avg_sharpe_ratio": metrics["sharpe_ratio"].mean(),
        "avg_exposure_time_pct": metrics["exposure_time_pct"].mean(),
        "total_fees": metrics["total_fees"].sum(),
        "products_tested": metrics["product_id"].nunique(),
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration / saving
# ─────────────────────────────────────────────────────────────────────────────

def run_probability_signal_backtest(
    candles: pd.DataFrame,
    bullish_candidates: pd.DataFrame,
    bearish_candidates: pd.DataFrame,
    config: ProbabilitySignalBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.concat([bullish_candidates, bearish_candidates], ignore_index=True)

    if selected.empty:
        raise ValueError("No candidates selected. Loosen thresholds or check candidate file.")

    signals = build_signals_for_candidates(candles, selected, config)

    if signals.empty:
        raise ValueError("Selected candidates generated no historical signals.")

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float | int | str]] = []

    product_ids = sorted(candles["product_id"].dropna().unique())

    for product_id in product_ids:
        trades, equity, metrics = run_backtest_for_product(
            candles=candles,
            signals=signals,
            product_id=product_id,
            config=config,
        )

        if not trades.empty:
            all_trades.append(trades)
        if not equity.empty:
            all_equity.append(equity)
        all_metrics.append(metrics)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)
    leaderboard = build_leaderboard(metrics_df)

    return trades_df, equity_df, metrics_df, leaderboard, signals


def save_outputs(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    signals: pd.DataFrame,
    bullish_candidates: pd.DataFrame,
    bearish_candidates: pd.DataFrame,
    config: ProbabilitySignalBacktestConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{config.timeframe}_probability_signal"

    paths = {
        "trades": config.output_dir / f"prob_signal_trades_{suffix}.csv",
        "equity": config.output_dir / f"prob_signal_equity_{suffix}.csv",
        "metrics": config.output_dir / f"prob_signal_metrics_{suffix}.csv",
        "leaderboard": config.output_dir / f"prob_signal_leaderboard_{suffix}.csv",
        "signals": config.output_dir / f"prob_signal_generated_signals_{suffix}.csv",
        "bullish_candidates": config.output_dir / f"prob_signal_selected_bullish_candidates_{suffix}.csv",
        "bearish_candidates": config.output_dir / f"prob_signal_selected_bearish_candidates_{suffix}.csv",
    }

    trades.to_csv(paths["trades"], index=False)
    equity.to_csv(paths["equity"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    leaderboard.to_csv(paths["leaderboard"], index=False)
    signals.to_csv(paths["signals"], index=False)
    bullish_candidates.to_csv(paths["bullish_candidates"], index=False)
    bearish_candidates.to_csv(paths["bearish_candidates"], index=False)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_float_list(values: list[str] | None) -> list[float] | None:
    if not values:
        return None
    return [float(v) for v in values]


def parse_int_list(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest long/short table-driven probability signals."
    )

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/probability_backtests"))
    parser.add_argument("--probability-tables-dir", type=Path, default=Path("data/probability_tables"))
    parser.add_argument("--candidates-path", type=Path, default=None)

    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--threshold-pct", type=float, default=0.30)
    parser.add_argument("--percentile-window", type=int, default=200)
    parser.add_argument("--bucket-size", type=int, default=10)

    parser.add_argument(
        "--table-types",
        nargs="+",
        default=["vol_momentum_bucket"],
        choices=["vol_bucket", "momentum_bucket", "vol_momentum_bucket"],
        help="Candidate table types allowed for signal selection.",
    )

    parser.add_argument("--allowed-lookbacks", nargs="+", default=None)
    parser.add_argument("--allowed-horizons", nargs="+", default=None)

    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--min-edge", type=float, default=0.12)
    parser.add_argument("--min-avg-return-pct", type=float, default=0.15)
    parser.add_argument("--top-n-per-side", type=int, default=50)

    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--position-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.006)
    parser.add_argument("--slippage-rate", type=float, default=0.001)

    parser.add_argument("--no-longs", action="store_true", help="Disable long trades.")
    parser.add_argument("--no-shorts", action="store_true", help="Disable short trades.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selection_config = CandidateSelectionConfig(
        candidates_path=args.candidates_path,
        probability_tables_dir=args.probability_tables_dir,
        table_types=args.table_types,
        allowed_lookbacks=parse_float_list(args.allowed_lookbacks),
        allowed_horizons=parse_int_list(args.allowed_horizons),
        min_samples=args.min_samples,
        min_edge=args.min_edge,
        min_avg_return_pct=args.min_avg_return_pct,
        top_n_per_side=args.top_n_per_side,
    )

    backtest_config = ProbabilitySignalBacktestConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        initial_equity=args.initial_equity,
        position_pct=args.position_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        allow_longs=not args.no_longs,
        allow_shorts=not args.no_shorts,
    )

    bullish_candidates, bearish_candidates, candidates_path = load_and_select_candidates(
        selection_config
    )

    print(f"Using candidate file: {candidates_path}")
    print(f"Selected bullish candidates: {len(bullish_candidates)}")
    print(f"Selected bearish candidates: {len(bearish_candidates)}")

    if bullish_candidates.empty and backtest_config.allow_longs:
        print("Warning: no bullish candidates selected.")
    if bearish_candidates.empty and backtest_config.allow_shorts:
        print("Warning: no bearish candidates selected.")

    candles = load_raw_candles(backtest_config.input_dir, backtest_config.timeframe)

    trades, equity, metrics, leaderboard, signals = run_probability_signal_backtest(
        candles=candles,
        bullish_candidates=bullish_candidates if backtest_config.allow_longs else pd.DataFrame(),
        bearish_candidates=bearish_candidates if backtest_config.allow_shorts else pd.DataFrame(),
        config=backtest_config,
    )

    paths = save_outputs(
        trades=trades,
        equity=equity,
        metrics=metrics,
        leaderboard=leaderboard,
        signals=signals,
        bullish_candidates=bullish_candidates,
        bearish_candidates=bearish_candidates,
        config=backtest_config,
    )

    print("\nProbability signal leaderboard:")
    if leaderboard.empty:
        print("No leaderboard generated.")
    else:
        display_cols = [
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
            "avg_exposure_time_pct",
            "total_fees",
        ]
        print(leaderboard[display_cols].to_string(index=False))

    print("\nPer-asset metrics:")
    metric_cols = [
        "product_id",
        "total_return_pct",
        "buy_hold_return_pct",
        "excess_return_vs_buy_hold_pct",
        "num_trades",
        "long_trades",
        "short_trades",
        "win_rate_pct",
        "profit_factor",
        "max_drawdown_pct",
        "sharpe_ratio",
        "exposure_time_pct",
    ]
    print(metrics[metric_cols].to_string(index=False))

    print("\nSaved probability signal backtest outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
