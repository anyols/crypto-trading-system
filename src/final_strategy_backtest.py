from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_strategy_definitions import StrategyDefinition, get_final_strategies
from src.probability_table import ProbabilityTableConfig, build_probability_dataset
from src.strategy import load_raw_candles


PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]
BTC_PRODUCT_ID = "BTC-USD"


@dataclass(frozen=True)
class FinalBacktestConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    holdout_days: int
    initial_equity: float
    position_pct: float
    max_gross_leverage: float
    fee_rate: float
    slippage_rate: float
    percentile_window: int
    bucket_size: int
    threshold_pct: float


@dataclass
class Position:
    product_id: str
    side: str
    entry_time: pd.Timestamp
    exit_time_target: pd.Timestamp
    entry_price: float
    notional: float
    entry_fee: float


def range_mask(series: pd.Series, lower: float | None, upper: float | None) -> pd.Series:
    mask = pd.Series(True, index=series.index)

    if lower is not None:
        mask &= series >= lower
    if upper is not None:
        mask &= series < upper

    return mask.fillna(False)

'''
def filter_mask(dataset: pd.DataFrame, filter_name: str) -> pd.Series:
    if filter_name == "none":
        return pd.Series(True, index=dataset.index)

    if filter_name == "asset_above_ema200":
        return (dataset["price_vs_ema200_pct"] > 0).fillna(False)

    if filter_name == "asset_below_ema200":
        return (dataset["price_vs_ema200_pct"] < 0).fillna(False)

    if filter_name == "btc_above_ema200":
        return (dataset["btc_price_vs_ema200_pct"] > 0).fillna(False)

    if filter_name == "btc_below_ema200":
        return (dataset["btc_price_vs_ema200_pct"] < 0).fillna(False)

    raise ValueError(f"Unknown filter_name: {filter_name}")
'''

def filter_mask(dataset: pd.DataFrame, filter_name: str) -> pd.Series:
    if filter_name == "none":
        return pd.Series(True, index=dataset.index)

    if filter_name == "asset_above_ema200":
        return (dataset["price_vs_ema200_pct"] > 0).fillna(False)

    if filter_name == "asset_below_ema200":
        return (dataset["price_vs_ema200_pct"] < 0).fillna(False)

    if filter_name == "btc_above_ema200":
        return (dataset["btc_price_vs_ema200_pct"] > 0).fillna(False)

    if filter_name == "btc_below_ema200":
        return (dataset["btc_price_vs_ema200_pct"] < 0).fillna(False)

    if filter_name == "volume_z_gt_0":
        return (dataset["volume_zscore_20"] > 0).fillna(False)

    if filter_name == "volume_z_gt_1":
        return (dataset["volume_zscore_20"] > 1).fillna(False)

    raise ValueError(f"Unknown filter_name: {filter_name}")
    

def add_btc_context(dataset: pd.DataFrame) -> pd.DataFrame:
    data = dataset.copy()

    btc = data[data["product_id"] == BTC_PRODUCT_ID][
        ["time", "price_vs_ema200_pct", "return_lookback_pct"]
    ].copy()

    btc = btc.rename(
        columns={
            "price_vs_ema200_pct": "btc_price_vs_ema200_pct",
            "return_lookback_pct": "btc_return_lookback_pct",
        }
    )

    data = data.merge(btc, on="time", how="left")
    return data


def build_feature_dataset(
    candles: pd.DataFrame,
    config: FinalBacktestConfig,
    lookback_hours: float,
    horizon_minutes: int,
) -> pd.DataFrame:
    table_config = ProbabilityTableConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        timeframe=config.timeframe,
        lookback_hours=lookback_hours,
        horizon_minutes=horizon_minutes,
        threshold_pct=config.threshold_pct,
        percentile_window=config.percentile_window,
        min_samples_per_bucket=1,
        bucket_size=config.bucket_size,
    )

    dataset = build_probability_dataset(candles, table_config)
    dataset = add_btc_context(dataset)
    dataset = dataset[dataset["product_id"].isin(PRODUCTS)].copy()

    return dataset.sort_values(["product_id", "time"]).reset_index(drop=True)


def get_holdout_window(candles: pd.DataFrame, holdout_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_time = candles["time"].max()
    start_time = end_time - pd.Timedelta(days=holdout_days)
    return start_time, end_time


def build_signal_mask_for_strategy(
    dataset: pd.DataFrame,
    strategy: StrategyDefinition,
    signal_side: str,
) -> pd.Series:
    mask = pd.Series(True, index=dataset.index)

    if strategy.strategy_id == "A_v3_baseline":
        vol_mask = range_mask(dataset["realized_vol_percentile"], 40.0, 50.0)

        if signal_side == "long":
            mask &= dataset["product_id"].isin(["BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD"])
            mask &= dataset["return_lookback_pct"] < -4.0
            mask &= vol_mask
        elif signal_side == "short":
            mask &= dataset["product_id"].isin(["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"])
            mask &= range_mask(dataset["return_lookback_pct"], 0.25, 1.0)
            mask &= vol_mask
        else:
            raise ValueError("A_v3_baseline supports only long/short sub-signals.")

        return mask.fillna(False)

    # Generic strategies B-E
    if strategy.vol_lower is not None or strategy.vol_upper is not None:
        mask &= range_mask(dataset["realized_vol_percentile"], strategy.vol_lower, strategy.vol_upper)

    mask &= range_mask(dataset["return_lookback_pct"], strategy.momentum_lower, strategy.momentum_upper)
    mask &= filter_mask(dataset, strategy.filter_name)

    if strategy.allowed_products is not None:
        mask &= dataset["product_id"].isin(strategy.allowed_products)

    if strategy.excluded_products is not None:
        mask &= ~dataset["product_id"].isin(strategy.excluded_products)

    if signal_side == "long":
        pass
    elif signal_side == "short":
        pass
    else:
        raise ValueError(f"Unsupported signal_side: {signal_side}")

    return mask.fillna(False)


def build_signals_for_strategy(
    feature_dataset: pd.DataFrame,
    strategy: StrategyDefinition,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> pd.DataFrame:
    if strategy.lookback_hours is None or strategy.horizon_minutes is None:
        return pd.DataFrame()

    data = feature_dataset.copy()
    data = data[(data["time"] >= holdout_start) & (data["time"] <= holdout_end)].copy()

    signal_frames: list[pd.DataFrame] = []

    if strategy.side in ("long", "short"):
        side = strategy.side
        mask = build_signal_mask_for_strategy(data, strategy, side)
        part = data[mask].copy()
        if not part.empty:
            part["strategy_id"] = strategy.strategy_id
            part["strategy_name"] = strategy.name
            part["side"] = side
            part["exit_time_target"] = part["time"] + pd.to_timedelta(strategy.horizon_minutes, unit="m")
            signal_frames.append(part)

    elif strategy.side == "both":
        for side in ["long", "short"]:
            mask = build_signal_mask_for_strategy(data, strategy, side)
            part = data[mask].copy()
            if not part.empty:
                part["strategy_id"] = strategy.strategy_id
                part["strategy_name"] = strategy.name
                part["side"] = side
                part["exit_time_target"] = part["time"] + pd.to_timedelta(strategy.horizon_minutes, unit="m")
                signal_frames.append(part)

    else:
        raise ValueError(f"Unsupported strategy side: {strategy.side}")

    if not signal_frames:
        return pd.DataFrame()

    signals = pd.concat(signal_frames, ignore_index=True)

    signals = signals[
        [
            "time",
            "exit_time_target",
            "product_id",
            "strategy_id",
            "strategy_name",
            "side",
            "close",
            "return_lookback_pct",
            "realized_vol_percentile",
            "price_vs_ema200_pct",
            "btc_price_vs_ema200_pct",
            "volume_zscore_20",
        ]
    ].copy()

    signals = signals.sort_values(["time", "product_id", "side"]).reset_index(drop=True)
    return signals


def apply_entry_slippage(raw_price: float, side: str, slippage_rate: float) -> float:
    if side == "long":
        return raw_price * (1.0 + slippage_rate)
    if side == "short":
        return raw_price * (1.0 - slippage_rate)
    raise ValueError(f"Invalid side: {side}")


def apply_exit_slippage(raw_price: float, side: str, slippage_rate: float) -> float:
    if side == "long":
        return raw_price * (1.0 - slippage_rate)
    if side == "short":
        return raw_price * (1.0 + slippage_rate)
    raise ValueError(f"Invalid side: {side}")


def compute_unrealized_pnl(position: Position, current_price: float) -> float:
    if position.side == "long":
        return position.notional * ((current_price - position.entry_price) / position.entry_price)
    if position.side == "short":
        return position.notional * ((position.entry_price - current_price) / position.entry_price)
    raise ValueError(f"Invalid side: {position.side}")


def calculate_profit_factor(values: pd.Series) -> float:
    wins = values[values > 0]
    losses = values[values < 0]

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0

    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return float("inf")
    return 0.0


def calculate_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def calculate_annualized_metrics(equity_df: pd.DataFrame, timeframe: str) -> tuple[float, float, float]:
    if equity_df.empty or len(equity_df) < 3:
        return 0.0, 0.0, 0.0

    equity = equity_df["equity"].astype(float)
    returns = equity.pct_change().dropna()

    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0, 0.0, 0.0

    if timeframe == "15m":
        periods_per_year = 365 * 24 * 4
    elif timeframe == "1h":
        periods_per_year = 365 * 24
    else:
        periods_per_year = 365

    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year) * 100.0)
    sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(periods_per_year))

    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        sortino = 0.0
    else:
        sortino = float((returns.mean() / downside.std(ddof=1)) * np.sqrt(periods_per_year))

    return ann_vol, sharpe, sortino


def build_strategy_metrics(
    strategy: StrategyDefinition,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    asset_contrib_df: pd.DataFrame,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    config: FinalBacktestConfig,
) -> dict[str, object]:
    ending_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else config.initial_equity
    total_return_pct = ((ending_equity / config.initial_equity) - 1.0) * 100.0

    period_days = max((holdout_end - holdout_start).total_seconds() / 86400.0, 1e-9)
    if ending_equity > 0:
        annualized_return_pct = ((ending_equity / config.initial_equity) ** (365.0 / period_days) - 1.0) * 100.0
    else:
        annualized_return_pct = -100.0

    ann_vol_pct, sharpe_ratio, sortino_ratio = calculate_annualized_metrics(
        equity_df=equity_df,
        timeframe=config.timeframe,
    )

    max_drawdown_pct = calculate_max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0
    calmar_ratio = annualized_return_pct / abs(max_drawdown_pct) if max_drawdown_pct < 0 else 0.0

    if trades_df.empty:
        return {
            "strategy_id": strategy.strategy_id,
            "strategy_name": strategy.name,
            "category": strategy.category,
            "description": strategy.description,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "initial_equity": config.initial_equity,
            "ending_equity": ending_equity,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "annualized_volatility_pct": ann_vol_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "best_trade_return_pct": 0.0,
            "worst_trade_return_pct": 0.0,
            "profit_factor": 0.0,
            "fees_paid": 0.0,
            "avg_holding_hours": 0.0,
            "exposure_time_pct": 0.0,
            "avg_gross_exposure_pct": 0.0,
            "top_asset_by_pnl": "",
            "top_asset_pnl": 0.0,
        }

    net_pnl = trades_df["net_pnl"]
    trade_returns = trades_df["trade_return_pct"]

    top_asset_row = (
        asset_contrib_df.sort_values("net_pnl", ascending=False).iloc[0]
        if not asset_contrib_df.empty
        else None
    )

    return {
        "strategy_id": strategy.strategy_id,
        "strategy_name": strategy.name,
        "category": strategy.category,
        "description": strategy.description,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "initial_equity": config.initial_equity,
        "ending_equity": ending_equity,
        "total_return_pct": total_return_pct,
        "annualized_return_pct": annualized_return_pct,
        "annualized_volatility_pct": ann_vol_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": int(len(trades_df)),
        "win_rate_pct": float((net_pnl > 0).mean() * 100.0),
        "avg_trade_return_pct": float(trade_returns.mean()),
        "median_trade_return_pct": float(trade_returns.median()),
        "best_trade_return_pct": float(trade_returns.max()),
        "worst_trade_return_pct": float(trade_returns.min()),
        "profit_factor": calculate_profit_factor(net_pnl),
        "fees_paid": float(trades_df["total_fees"].sum()),
        "avg_holding_hours": float(trades_df["holding_hours"].mean()),
        "exposure_time_pct": float(equity_df["in_market"].mean() * 100.0) if "in_market" in equity_df else 0.0,
        "avg_gross_exposure_pct": float(equity_df["gross_exposure_pct"].mean()) if "gross_exposure_pct" in equity_df else 0.0,
        "top_asset_by_pnl": str(top_asset_row["product_id"]) if top_asset_row is not None else "",
        "top_asset_pnl": float(top_asset_row["net_pnl"]) if top_asset_row is not None else 0.0,
    }


'''
def run_rule_strategy_backtest(
    strategy: StrategyDefinition,
    signals_df: pd.DataFrame,
    holdout_candles: pd.DataFrame,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    config: FinalBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if signals_df.empty:
        times = pd.Index(sorted(holdout_candles["time"].unique()))
        equity_df = pd.DataFrame(
            {
                "time": times,
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "equity": config.initial_equity,
                "cash": config.initial_equity,
                "unrealized_pnl": 0.0,
                "in_market": 0,
                "gross_exposure": 0.0,
                "gross_exposure_pct": 0.0,
            }
        )
        return pd.DataFrame(), equity_df, pd.DataFrame(columns=["strategy_id", "strategy_name", "product_id", "net_pnl"])

    prices = holdout_candles.pivot(index="time", columns="product_id", values="close").sort_index()
    all_times = prices.index

    signals_map = {
        ts: grp.sort_values(["product_id", "side"]).to_dict("records")
        for ts, grp in signals_df.groupby("time")
    }

    positions: list[Position] = []
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    cash = float(config.initial_equity)

    for current_time in all_times:
        current_prices = prices.loc[current_time]

        # 1) Close matured positions
        remaining_positions: list[Position] = []
        for pos in positions:
            raw_price = current_prices.get(pos.product_id, np.nan)

            if pd.isna(raw_price):
                remaining_positions.append(pos)
                continue

            if current_time >= pos.exit_time_target:
                exit_price = apply_exit_slippage(float(raw_price), pos.side, config.slippage_rate)
                if pos.side == "long":
                    gross_pnl = pos.notional * ((exit_price - pos.entry_price) / pos.entry_price)
                else:
                    gross_pnl = pos.notional * ((pos.entry_price - exit_price) / pos.entry_price)

                exit_fee = pos.notional * config.fee_rate
                net_pnl_after_exit = gross_pnl - exit_fee
                cash += net_pnl_after_exit

                holding_hours = (current_time - pos.entry_time).total_seconds() / 3600.0
                total_fees = pos.entry_fee + exit_fee
                total_net_pnl = gross_pnl - total_fees
                trade_return_pct = (total_net_pnl / pos.notional) * 100.0 if pos.notional > 0 else 0.0

                trade_rows.append(
                    {
                        "strategy_id": strategy.strategy_id,
                        "strategy_name": strategy.name,
                        "category": strategy.category,
                        "product_id": pos.product_id,
                        "side": pos.side,
                        "entry_time": pos.entry_time,
                        "exit_time": current_time,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "notional": pos.notional,
                        "gross_pnl": gross_pnl,
                        "entry_fee": pos.entry_fee,
                        "exit_fee": exit_fee,
                        "total_fees": total_fees,
                        "net_pnl": total_net_pnl,
                        "trade_return_pct": trade_return_pct,
                        "holding_hours": holding_hours,
                    }
                )
            else:
                remaining_positions.append(pos)

        positions = remaining_positions

        # 2) Current equity after closes
        unrealized_pnl = 0.0
        gross_exposure = 0.0
        open_products = {p.product_id for p in positions}

        for pos in positions:
            raw_price = current_prices.get(pos.product_id, np.nan)
            if pd.notna(raw_price):
                unrealized_pnl += compute_unrealized_pnl(pos, float(raw_price))
                gross_exposure += pos.notional

        current_equity = cash + unrealized_pnl
        max_allowed_exposure = max(current_equity, 0.0) * config.max_gross_leverage

        # 3) Open new positions
        if current_time in signals_map:
            for signal in signals_map[current_time]:
                product_id = signal["product_id"]

                if product_id in open_products:
                    continue

                raw_price = current_prices.get(product_id, np.nan)
                if pd.isna(raw_price):
                    continue

                if current_equity <= 0:
                    continue

                notional = current_equity * config.position_pct

                if gross_exposure + notional > max_allowed_exposure:
                    continue

                entry_price = apply_entry_slippage(float(raw_price), signal["side"], config.slippage_rate)
                entry_fee = notional * config.fee_rate
                cash -= entry_fee

                pos = Position(
                    product_id=product_id,
                    side=signal["side"],
                    entry_time=current_time,
                    exit_time_target=pd.Timestamp(signal["exit_time_target"]),
                    entry_price=entry_price,
                    notional=notional,
                    entry_fee=entry_fee,
                )
                positions.append(pos)
                open_products.add(product_id)
                gross_exposure += notional

        # 4) Recompute equity after opens
        unrealized_pnl = 0.0
        gross_exposure = 0.0
        for pos in positions:
            raw_price = current_prices.get(pos.product_id, np.nan)
            if pd.notna(raw_price):
                unrealized_pnl += compute_unrealized_pnl(pos, float(raw_price))
                gross_exposure += pos.notional

        current_equity = cash + unrealized_pnl

        equity_rows.append(
            {
                "time": current_time,
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "category": strategy.category,
                "equity": current_equity,
                "cash": cash,
                "unrealized_pnl": unrealized_pnl,
                "in_market": 1 if positions else 0,
                "num_open_positions": len(positions),
                "gross_exposure": gross_exposure,
                "gross_exposure_pct": (gross_exposure / current_equity * 100.0) if current_equity > 0 else 0.0,
            }
        )

    # Close any leftover positions at last available price
    if positions:
        last_time = all_times[-1]
        last_prices = prices.loc[last_time]
        for pos in positions:
            raw_price = last_prices.get(pos.product_id, np.nan)
            if pd.isna(raw_price):
                continue

            exit_price = apply_exit_slippage(float(raw_price), pos.side, config.slippage_rate)
            if pos.side == "long":
                gross_pnl = pos.notional * ((exit_price - pos.entry_price) / pos.entry_price)
            else:
                gross_pnl = pos.notional * ((pos.entry_price - exit_price) / pos.entry_price)

            exit_fee = pos.notional * config.fee_rate
            total_fees = pos.entry_fee + exit_fee
            total_net_pnl = gross_pnl - total_fees
            trade_return_pct = (total_net_pnl / pos.notional) * 100.0 if pos.notional > 0 else 0.0
            holding_hours = (last_time - pos.entry_time).total_seconds() / 3600.0

            trade_rows.append(
                {
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.name,
                    "category": strategy.category,
                    "product_id": pos.product_id,
                    "side": pos.side,
                    "entry_time": pos.entry_time,
                    "exit_time": last_time,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "notional": pos.notional,
                    "gross_pnl": gross_pnl,
                    "entry_fee": pos.entry_fee,
                    "exit_fee": exit_fee,
                    "total_fees": total_fees,
                    "net_pnl": total_net_pnl,
                    "trade_return_pct": trade_return_pct,
                    "holding_hours": holding_hours,
                }
            )

    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_rows)

    asset_contrib_df = (
        trades_df.groupby(["strategy_id", "strategy_name", "product_id"], as_index=False)
        .agg(
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            fees_paid=("total_fees", "sum"),
            num_trades=("product_id", "count"),
            win_rate_pct=("net_pnl", lambda s: float((s > 0).mean() * 100.0)),
        )
        if not trades_df.empty
        else pd.DataFrame(columns=["strategy_id", "strategy_name", "product_id", "net_pnl"])
    )

    return trades_df, equity_df, asset_contrib_df
'''
def prepare_next_bar_signals(
    signals_df: pd.DataFrame,
    all_times: pd.Index,
    horizon_minutes: int,
) -> pd.DataFrame:
    """
    Convert signal timestamps into executable next-bar entry timestamps.

    Signal is known only after candle close at signal_time.
    Therefore, the earliest realistic execution is the next candle open.

    Example:
        signal at 10:00 candle close
        entry at 10:15 candle open
    """
    if signals_df.empty:
        return signals_df.copy()

    # all_times is the sorted index of available candle timestamps.
    next_time_map = {
        all_times[i]: all_times[i + 1]
        for i in range(len(all_times) - 1)
    }

    signals = signals_df.copy()

    signals["signal_time"] = signals["time"]
    signals["entry_time"] = signals["signal_time"].map(next_time_map)

    # Last candle has no next bar, so it cannot be executed realistically.
    signals = signals.dropna(subset=["entry_time"]).copy()

    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=True)
    signals["exit_time_target"] = signals["entry_time"] + pd.to_timedelta(
        horizon_minutes,
        unit="m",
    )

    return signals

def run_rule_strategy_backtest(
    strategy: StrategyDefinition,
    signals_df: pd.DataFrame,
    holdout_candles: pd.DataFrame,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    config: FinalBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realistic portfolio backtest.

    Execution model:
        - signal is detected at candle close
        - enter at next candle open
        - exit at open once target holding period has elapsed
        - mark unrealized PnL at candle close
    """
    if strategy.horizon_minutes is None:
        raise ValueError(f"Strategy {strategy.strategy_id} has no horizon_minutes.")

    open_prices = (
        holdout_candles
        .pivot(index="time", columns="product_id", values="open")
        .sort_index()
    )
    close_prices = (
        holdout_candles
        .pivot(index="time", columns="product_id", values="close")
        .sort_index()
    )

    all_times = close_prices.index

    if len(all_times) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if signals_df.empty:
        equity_df = pd.DataFrame(
            {
                "time": all_times,
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "category": strategy.category,
                "equity": config.initial_equity,
                "cash": config.initial_equity,
                "unrealized_pnl": 0.0,
                "in_market": 0,
                "num_open_positions": 0,
                "gross_exposure": 0.0,
                "gross_exposure_pct": 0.0,
            }
        )
        return (
            pd.DataFrame(),
            equity_df,
            pd.DataFrame(
                columns=[
                    "strategy_id",
                    "strategy_name",
                    "product_id",
                    "net_pnl",
                ]
            ),
        )

    executable_signals = prepare_next_bar_signals(
        signals_df=signals_df,
        all_times=all_times,
        horizon_minutes=strategy.horizon_minutes,
    )

    signals_map = {
        ts: grp.sort_values(["product_id", "side"]).to_dict("records")
        for ts, grp in executable_signals.groupby("entry_time")
    }

    positions: list[Position] = []
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    cash = float(config.initial_equity)

    for current_time in all_times:
        current_open_prices = open_prices.loc[current_time]
        current_close_prices = close_prices.loc[current_time]

        # 1) Close matured positions at current candle open.
        remaining_positions: list[Position] = []

        for pos in positions:
            raw_exit_price = current_open_prices.get(pos.product_id, np.nan)

            if pd.isna(raw_exit_price):
                remaining_positions.append(pos)
                continue

            if current_time >= pos.exit_time_target:
                exit_price = apply_exit_slippage(
                    raw_price=float(raw_exit_price),
                    side=pos.side,
                    slippage_rate=config.slippage_rate,
                )

                if pos.side == "long":
                    gross_pnl = pos.notional * (
                        (exit_price - pos.entry_price) / pos.entry_price
                    )
                else:
                    gross_pnl = pos.notional * (
                        (pos.entry_price - exit_price) / pos.entry_price
                    )

                exit_fee = pos.notional * config.fee_rate
                total_fees = pos.entry_fee + exit_fee
                total_net_pnl = gross_pnl - total_fees

                cash += gross_pnl - exit_fee

                holding_hours = (
                    current_time - pos.entry_time
                ).total_seconds() / 3600.0

                trade_return_pct = (
                    (total_net_pnl / pos.notional) * 100.0
                    if pos.notional > 0
                    else 0.0
                )

                trade_rows.append(
                    {
                        "strategy_id": strategy.strategy_id,
                        "strategy_name": strategy.name,
                        "category": strategy.category,
                        "product_id": pos.product_id,
                        "side": pos.side,
                        "entry_time": pos.entry_time,
                        "exit_time": current_time,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "notional": pos.notional,
                        "gross_pnl": gross_pnl,
                        "entry_fee": pos.entry_fee,
                        "exit_fee": exit_fee,
                        "total_fees": total_fees,
                        "net_pnl": total_net_pnl,
                        "trade_return_pct": trade_return_pct,
                        "holding_hours": holding_hours,
                    }
                )
            else:
                remaining_positions.append(pos)

        positions = remaining_positions

        # 2) Mark current account state before new entries.
        unrealized_pnl = 0.0
        gross_exposure = 0.0
        open_products = {p.product_id for p in positions}

        for pos in positions:
            raw_mark_price = current_close_prices.get(pos.product_id, np.nan)
            if pd.notna(raw_mark_price):
                unrealized_pnl += compute_unrealized_pnl(pos, float(raw_mark_price))
                gross_exposure += pos.notional

        current_equity = cash + unrealized_pnl
        max_allowed_exposure = max(current_equity, 0.0) * config.max_gross_leverage

        # 3) Open new next-bar signals at current candle open.
        if current_time in signals_map:
            for signal in signals_map[current_time]:
                product_id = signal["product_id"]

                # Avoid overlapping positions in the same asset.
                if product_id in open_products:
                    continue

                raw_entry_price = current_open_prices.get(product_id, np.nan)
                if pd.isna(raw_entry_price):
                    continue

                if current_equity <= 0:
                    continue

                notional = current_equity * config.position_pct

                if gross_exposure + notional > max_allowed_exposure:
                    continue

                entry_price = apply_entry_slippage(
                    raw_price=float(raw_entry_price),
                    side=signal["side"],
                    slippage_rate=config.slippage_rate,
                )

                entry_fee = notional * config.fee_rate
                cash -= entry_fee

                pos = Position(
                    product_id=product_id,
                    side=signal["side"],
                    entry_time=current_time,
                    exit_time_target=pd.Timestamp(signal["exit_time_target"]),
                    entry_price=entry_price,
                    notional=notional,
                    entry_fee=entry_fee,
                )

                positions.append(pos)
                open_products.add(product_id)
                gross_exposure += notional

        # 4) Mark-to-market after entries at current close.
        unrealized_pnl = 0.0
        gross_exposure = 0.0

        for pos in positions:
            raw_mark_price = current_close_prices.get(pos.product_id, np.nan)
            if pd.notna(raw_mark_price):
                unrealized_pnl += compute_unrealized_pnl(pos, float(raw_mark_price))
                gross_exposure += pos.notional

        current_equity = cash + unrealized_pnl

        equity_rows.append(
            {
                "time": current_time,
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "category": strategy.category,
                "equity": current_equity,
                "cash": cash,
                "unrealized_pnl": unrealized_pnl,
                "in_market": 1 if positions else 0,
                "num_open_positions": len(positions),
                "gross_exposure": gross_exposure,
                "gross_exposure_pct": (
                    gross_exposure / current_equity * 100.0
                    if current_equity > 0
                    else 0.0
                ),
            }
        )

    # 5) Force-close leftovers at final candle close.
    if positions:
        last_time = all_times[-1]
        last_close_prices = close_prices.loc[last_time]

        for pos in positions:
            raw_exit_price = last_close_prices.get(pos.product_id, np.nan)
            if pd.isna(raw_exit_price):
                continue

            exit_price = apply_exit_slippage(
                raw_price=float(raw_exit_price),
                side=pos.side,
                slippage_rate=config.slippage_rate,
            )

            if pos.side == "long":
                gross_pnl = pos.notional * (
                    (exit_price - pos.entry_price) / pos.entry_price
                )
            else:
                gross_pnl = pos.notional * (
                    (pos.entry_price - exit_price) / pos.entry_price
                )

            exit_fee = pos.notional * config.fee_rate
            total_fees = pos.entry_fee + exit_fee
            total_net_pnl = gross_pnl - total_fees

            holding_hours = (last_time - pos.entry_time).total_seconds() / 3600.0
            trade_return_pct = (
                (total_net_pnl / pos.notional) * 100.0
                if pos.notional > 0
                else 0.0
            )

            trade_rows.append(
                {
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.name,
                    "category": strategy.category,
                    "product_id": pos.product_id,
                    "side": pos.side,
                    "entry_time": pos.entry_time,
                    "exit_time": last_time,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "notional": pos.notional,
                    "gross_pnl": gross_pnl,
                    "entry_fee": pos.entry_fee,
                    "exit_fee": exit_fee,
                    "total_fees": total_fees,
                    "net_pnl": total_net_pnl,
                    "trade_return_pct": trade_return_pct,
                    "holding_hours": holding_hours,
                }
            )

    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_rows)

    asset_contrib_df = (
        trades_df.groupby(
            ["strategy_id", "strategy_name", "product_id"],
            as_index=False,
        )
        .agg(
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            fees_paid=("total_fees", "sum"),
            num_trades=("product_id", "count"),
            win_rate_pct=("net_pnl", lambda s: float((s > 0).mean() * 100.0)),
        )
        if not trades_df.empty
        else pd.DataFrame(
            columns=[
                "strategy_id",
                "strategy_name",
                "product_id",
                "net_pnl",
            ]
        )
    )

    return trades_df, equity_df, asset_contrib_df

def run_buy_and_hold_benchmark(
    strategy: StrategyDefinition,
    holdout_candles: pd.DataFrame,
    config: FinalBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = (
        holdout_candles[holdout_candles["product_id"].isin(PRODUCTS)]
        .pivot(index="time", columns="product_id", values="close")
        .sort_index()
        .dropna(how="any")
    )

    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    first_time = prices.index[0]
    last_time = prices.index[-1]

    per_asset_capital = config.initial_equity / len(PRODUCTS)
    quantities: dict[str, float] = {}
    trade_rows: list[dict[str, object]] = []

    entry_fees_total = 0.0
    for product_id in PRODUCTS:
        entry_price = float(prices.loc[first_time, product_id]) * (1.0 + config.slippage_rate)
        entry_fee = per_asset_capital * config.fee_rate
        qty = (per_asset_capital - entry_fee) / entry_price
        quantities[product_id] = qty
        entry_fees_total += entry_fee

    equity_rows: list[dict[str, object]] = []
    for current_time in prices.index:
        market_value = 0.0
        for product_id in PRODUCTS:
            market_value += quantities[product_id] * float(prices.loc[current_time, product_id])

        equity_rows.append(
            {
                "time": current_time,
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "category": strategy.category,
                "equity": market_value,
                "cash": 0.0,
                "unrealized_pnl": market_value - config.initial_equity,
                "in_market": 1,
                "num_open_positions": len(PRODUCTS),
                "gross_exposure": market_value,
                "gross_exposure_pct": 100.0,
            }
        )

    asset_contrib_rows: list[dict[str, object]] = []
    for product_id in PRODUCTS:
        entry_price = float(prices.loc[first_time, product_id]) * (1.0 + config.slippage_rate)
        exit_price = float(prices.loc[last_time, product_id]) * (1.0 - config.slippage_rate)
        qty = quantities[product_id]

        proceeds = qty * exit_price
        initial_spent = per_asset_capital
        exit_fee = proceeds * config.fee_rate
        net_pnl = proceeds - exit_fee - initial_spent

        trade_rows.append(
            {
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "category": strategy.category,
                "product_id": product_id,
                "side": "long",
                "entry_time": first_time,
                "exit_time": last_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": per_asset_capital,
                "gross_pnl": proceeds - initial_spent,
                "entry_fee": per_asset_capital * config.fee_rate,
                "exit_fee": exit_fee,
                "total_fees": per_asset_capital * config.fee_rate + exit_fee,
                "net_pnl": net_pnl,
                "trade_return_pct": (net_pnl / per_asset_capital) * 100.0,
                "holding_hours": (last_time - first_time).total_seconds() / 3600.0,
            }
        )

        asset_contrib_rows.append(
            {
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "product_id": product_id,
                "net_pnl": net_pnl,
                "gross_pnl": proceeds - initial_spent,
                "fees_paid": per_asset_capital * config.fee_rate + exit_fee,
                "num_trades": 1,
                "win_rate_pct": 100.0 if net_pnl > 0 else 0.0,
            }
        )

    return (
        pd.DataFrame(trade_rows),
        pd.DataFrame(equity_rows),
        pd.DataFrame(asset_contrib_rows),
    )


def save_outputs(
    metrics_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    asset_contrib_df: pd.DataFrame,
    strategy_defs_df: pd.DataFrame,
    config: FinalBacktestConfig,
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"{config.timeframe}_holdout_{config.holdout_days}d_"
        f"pos_{str(config.position_pct).replace('.', 'p')}x_"
        f"lev_{str(config.max_gross_leverage).replace('.', 'p')}x"
    )

    paths = {
        "metrics": config.output_dir / f"final_strategy_metrics_{suffix}.csv",
        "trades": config.output_dir / f"final_strategy_trades_{suffix}.csv",
        "equity": config.output_dir / f"final_strategy_equity_{suffix}.csv",
        "asset_contributions": config.output_dir / f"final_strategy_asset_contributions_{suffix}.csv",
        "definitions": config.output_dir / f"final_strategy_definitions_{suffix}.csv",
    }

    metrics_df.to_csv(paths["metrics"], index=False)
    trades_df.to_csv(paths["trades"], index=False)
    equity_df.to_csv(paths["equity"], index=False)
    asset_contrib_df.to_csv(paths["asset_contributions"], index=False)
    strategy_defs_df.to_csv(paths["definitions"], index=False)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final 1-year holdout comparison for frozen non-ML strategies.")

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_5y_15m"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_strategy_backtests"))
    parser.add_argument("--timeframe", type=str, default="15m")

    parser.add_argument("--holdout-days", type=int, default=365)
    parser.add_argument("--initial-equity", type=float, default=100000.0)
    parser.add_argument("--position-pct", type=float, default=0.20)
    parser.add_argument("--max-gross-leverage", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.0001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)

    parser.add_argument("--percentile-window", type=int, default=200)
    parser.add_argument("--bucket-size", type=int, default=10)
    parser.add_argument("--threshold-pct", type=float, default=0.30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FinalBacktestConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        holdout_days=args.holdout_days,
        initial_equity=args.initial_equity,
        position_pct=args.position_pct,
        max_gross_leverage=args.max_gross_leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        threshold_pct=args.threshold_pct,
    )

    candles = load_raw_candles(config.input_dir, config.timeframe)
    candles = candles[candles["product_id"].isin(PRODUCTS)].copy()
    candles = candles.sort_values(["product_id", "time"]).reset_index(drop=True)

    holdout_start, holdout_end = get_holdout_window(candles, config.holdout_days)
    holdout_candles = candles[candles["time"] >= holdout_start].copy()

    strategies = get_final_strategies()

    # Build feature datasets only for unique lookback/horizon pairs.
    combo_cache: dict[tuple[float, int], pd.DataFrame] = {}
    for strategy in strategies:
        if strategy.lookback_hours is None or strategy.horizon_minutes is None:
            continue
        key = (strategy.lookback_hours, strategy.horizon_minutes)
        if key not in combo_cache:
            print(f"Building feature dataset for lookback={key[0]}h horizon={key[1]}m")
            combo_cache[key] = build_feature_dataset(
                candles=candles,
                config=config,
                lookback_hours=key[0],
                horizon_minutes=key[1],
            )

    metrics_rows: list[dict[str, object]] = []
    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    all_asset_contrib: list[pd.DataFrame] = []

    for strategy in strategies:
        print(f"\nRunning strategy: {strategy.strategy_id} | {strategy.name}")

        if strategy.strategy_id == "F_buy_and_hold_equal_weight":
            trades_df, equity_df, asset_contrib_df = run_buy_and_hold_benchmark(
                strategy=strategy,
                holdout_candles=holdout_candles,
                config=config,
            )
        else:
            feature_dataset = combo_cache[(strategy.lookback_hours, strategy.horizon_minutes)]
            signals_df = build_signals_for_strategy(
                feature_dataset=feature_dataset,
                strategy=strategy,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            )

            print(f"Signals found: {len(signals_df):,}")

            trades_df, equity_df, asset_contrib_df = run_rule_strategy_backtest(
                strategy=strategy,
                signals_df=signals_df,
                holdout_candles=holdout_candles,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
                config=config,
            )

        metrics = build_strategy_metrics(
            strategy=strategy,
            equity_df=equity_df,
            trades_df=trades_df,
            asset_contrib_df=asset_contrib_df,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            config=config,
        )

        metrics_rows.append(metrics)
        if not trades_df.empty:
            all_trades.append(trades_df)
        if not equity_df.empty:
            all_equity.append(equity_df)
        if not asset_contrib_df.empty:
            all_asset_contrib.append(asset_contrib_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["sharpe_ratio", "total_return_pct"],
        ascending=False,
    ).reset_index(drop=True)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    asset_contrib_df = pd.concat(all_asset_contrib, ignore_index=True) if all_asset_contrib else pd.DataFrame()
    strategy_defs_df = pd.DataFrame([asdict(s) for s in strategies])

    paths = save_outputs(
        metrics_df=metrics_df,
        trades_df=trades_df,
        equity_df=equity_df,
        asset_contrib_df=asset_contrib_df,
        strategy_defs_df=strategy_defs_df,
        config=config,
    )

    print("\nFinal strategy comparison metrics:")
    display_cols = [
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
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[display_cols].to_string(index=False))

    print("\nSaved outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()