"""
Backtest engine for the volatility-compression breakout strategy.

Run from project root:

    python -m src.backtest --timeframe 1h

Example with ATR trailing exit:

    python -m src.backtest --timeframe 1h --exit-mode atr_trailing_stop --stop-atr-multiplier 3 --trailing-atr-multiplier 3

Design:
    - long-only
    - one open position per asset
    - signal detected on candle close
    - entry happens on the next candle open
    - fees and slippage included
    - stop-loss checked using OHLC candle data
    - if stop and take-profit are both touched in the same candle, stop-loss wins
    - supports fixed target, EMA exits, ATR trailing stop, and Donchian-low exit

This is a research-grade candle backtester, not a high-frequency execution simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.indicators import add_volatility_strategy_features
from src.strategy import (
    VolatilityBreakoutConfig,
    generate_volatility_breakout_signals,
    load_raw_candles,
)


ExitMode = Literal[
    "fixed_r_target",
    "ema50_exit",
    "ema200_exit",
    "atr_trailing_stop",
    "donchian_low_exit",
]

ExitReason = Literal[
    "stop_loss",
    "take_profit",
    "ema50_exit",
    "ema200_exit",
    "atr_trailing_stop",
    "donchian_low_exit",
    "end_of_data",
]


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.005
    max_position_pct: float = 1.00
    fee_rate: float = 0.006
    slippage_rate: float = 0.001
    stop_atr_multiplier: float = 2.0
    take_profit_r_multiple: float = 3.0
    exit_mode: ExitMode = "fixed_r_target"
    trailing_atr_multiplier: float = 3.0
    conservative_same_bar_exit: bool = True


@dataclass
class OpenPosition:
    product_id: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop_price: float
    take_profit_price: float
    entry_fee: float
    initial_risk_cash: float
    entry_index: int
    highest_close_since_entry: float


@dataclass
class BacktestResult:
    product_id: str
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float | int | str]


def periods_per_year_from_timeframe(timeframe: str) -> int:
    """
    Crypto trades 24/7, so annualization uses 365 days.
    """
    mapping = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "1h": 365 * 24,
        "6h": 365 * 4,
        "1d": 365,
    }
    return mapping.get(timeframe, 365)


def calculate_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calculate_sharpe_ratio(equity_curve: pd.DataFrame, periods_per_year: int) -> float:
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return 0.0

    returns = (
        equity_curve["equity"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def calculate_buy_and_hold_return(product_data: pd.DataFrame) -> float:
    clean = product_data.dropna(subset=["close"]).sort_values("time")

    if len(clean) < 2:
        return 0.0

    first_close = float(clean["close"].iloc[0])
    last_close = float(clean["close"].iloc[-1])

    if first_close <= 0:
        return 0.0

    return last_close / first_close - 1.0


def calculate_trade_metrics(
    product_id: str,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_equity: float,
    buy_hold_return: float,
    periods_per_year: int,
) -> dict[str, float | int | str]:
    ending_equity = (
        float(equity_curve["equity"].iloc[-1])
        if not equity_curve.empty
        else initial_equity
    )

    total_return = ending_equity / initial_equity - 1.0
    max_drawdown = (
        calculate_max_drawdown(equity_curve["equity"])
        if not equity_curve.empty
        else 0.0
    )
    sharpe = calculate_sharpe_ratio(equity_curve, periods_per_year)

    base_metrics: dict[str, float | int | str] = {
        "product_id": product_id,
        "initial_equity": initial_equity,
        "ending_equity": ending_equity,
        "total_return_pct": total_return * 100.0,
        "buy_hold_return_pct": buy_hold_return * 100.0,
        "excess_return_vs_buy_hold_pct": (total_return - buy_hold_return) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
    }

    if trades.empty:
        base_metrics.update(
            {
                "num_trades": 0,
                "win_rate_pct": 0.0,
                "avg_trade_return_pct": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "avg_holding_bars": 0.0,
            }
        )
        return base_metrics

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

    base_metrics.update(
        {
            "num_trades": int(len(trades)),
            "win_rate_pct": float((trades["net_pnl"] > 0).mean() * 100.0),
            "avg_trade_return_pct": float(trades["return_pct"].mean()),
            "avg_win": float(wins["net_pnl"].mean()) if not wins.empty else 0.0,
            "avg_loss": float(losses["net_pnl"].mean()) if not losses.empty else 0.0,
            "profit_factor": float(profit_factor),
            "best_trade": float(trades["net_pnl"].max()),
            "worst_trade": float(trades["net_pnl"].min()),
            "avg_holding_bars": float(trades["holding_bars"].mean()),
        }
    )

    return base_metrics


def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    config: BacktestConfig,
) -> float:
    """
    Risk-based sizing:

        risk_cash = equity * risk_per_trade
        qty = risk_cash / abs(entry_price - stop_price)

    Then cap notional exposure.
    """
    risk_per_unit = entry_price - stop_price

    if equity <= 0 or entry_price <= 0 or risk_per_unit <= 0:
        return 0.0

    risk_cash = equity * config.risk_per_trade
    raw_qty = risk_cash / risk_per_unit

    max_position_value = equity * config.max_position_pct
    max_qty = max_position_value / entry_price

    return max(min(raw_qty, max_qty), 0.0)


def create_position(
    product_id: str,
    entry_row: pd.Series,
    signal_row: pd.Series,
    entry_index: int,
    equity: float,
    config: BacktestConfig,
) -> tuple[OpenPosition | None, float]:
    """
    Create a long position at the next candle open.

    ATR comes from the signal candle.
    Entry price comes from the next candle open plus adverse slippage.
    """
    atr_value = float(signal_row.get("atr_14", np.nan))

    if np.isnan(atr_value) or atr_value <= 0:
        return None, equity

    raw_entry_price = float(entry_row["open"])
    entry_price = raw_entry_price * (1.0 + config.slippage_rate)

    stop_price = entry_price - (config.stop_atr_multiplier * atr_value)
    risk_per_unit = entry_price - stop_price

    if risk_per_unit <= 0:
        return None, equity

    take_profit_price = entry_price + (
        config.take_profit_r_multiple * risk_per_unit
    )

    qty = calculate_position_size(
        equity=equity,
        entry_price=entry_price,
        stop_price=stop_price,
        config=config,
    )

    if qty <= 0:
        return None, equity

    entry_fee = qty * entry_price * config.fee_rate
    equity_after_entry_fee = equity - entry_fee

    if equity_after_entry_fee <= 0:
        return None, equity

    position = OpenPosition(
        product_id=product_id,
        entry_time=entry_row["time"],
        entry_price=entry_price,
        qty=qty,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        entry_fee=entry_fee,
        initial_risk_cash=risk_per_unit * qty,
        entry_index=entry_index,
        highest_close_since_entry=entry_price,
    )

    return position, equity_after_entry_fee


def update_atr_trailing_stop(
    position: OpenPosition,
    row: pd.Series,
    config: BacktestConfig,
) -> None:
    """
    Update trailing stop after the candle closes.

    Important:
        The updated trailing stop is NOT allowed to trigger inside the same candle.
        It applies from the next candle onward.
    """
    atr_value = float(row.get("atr_14", np.nan))
    close = float(row["close"])

    if np.isnan(atr_value) or atr_value <= 0:
        return

    position.highest_close_since_entry = max(
        position.highest_close_since_entry,
        close,
    )

    proposed_stop = position.highest_close_since_entry - (
        config.trailing_atr_multiplier * atr_value
    )

    position.stop_price = max(position.stop_price, proposed_stop)


def check_exit(
    position: OpenPosition,
    row: pd.Series,
    config: BacktestConfig,
    is_last_row: bool = False,
) -> tuple[bool, float, ExitReason | None]:
    """
    Check whether a position exits on the current candle.

    Stop-loss always has priority.

    For fixed_r_target:
        - stop-loss and take-profit are both active.
        - if both are touched in the same candle, stop-loss wins by default.

    For trend-following exit modes:
        - protective stop remains active.
        - no fixed take-profit is used.
        - exit is based on EMA, Donchian structure, or trailing ATR.
    """
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])

    stop_hit = low <= position.stop_price

    if config.exit_mode == "fixed_r_target":
        target_hit = high >= position.take_profit_price

        if stop_hit and target_hit:
            if config.conservative_same_bar_exit:
                exit_price = position.stop_price * (1.0 - config.slippage_rate)
                return True, exit_price, "stop_loss"

            exit_price = position.take_profit_price * (1.0 - config.slippage_rate)
            return True, exit_price, "take_profit"

        if stop_hit:
            exit_price = position.stop_price * (1.0 - config.slippage_rate)
            return True, exit_price, "stop_loss"

        if target_hit:
            exit_price = position.take_profit_price * (1.0 - config.slippage_rate)
            return True, exit_price, "take_profit"

    elif config.exit_mode == "ema50_exit":
        if stop_hit:
            exit_price = position.stop_price * (1.0 - config.slippage_rate)
            return True, exit_price, "stop_loss"

        ema_50 = row.get("ema_50", np.nan)
        if not pd.isna(ema_50) and close < float(ema_50):
            exit_price = close * (1.0 - config.slippage_rate)
            return True, exit_price, "ema50_exit"

    elif config.exit_mode == "ema200_exit":
        if stop_hit:
            exit_price = position.stop_price * (1.0 - config.slippage_rate)
            return True, exit_price, "stop_loss"

        ema_200 = row.get("ema_200", np.nan)
        if not pd.isna(ema_200) and close < float(ema_200):
            exit_price = close * (1.0 - config.slippage_rate)
            return True, exit_price, "ema200_exit"

    elif config.exit_mode == "donchian_low_exit":
        if stop_hit:
            exit_price = position.stop_price * (1.0 - config.slippage_rate)
            return True, exit_price, "stop_loss"

        donchian_low_value = row.get("donchian_low_20", np.nan)
        if not pd.isna(donchian_low_value) and close < float(donchian_low_value):
            exit_price = close * (1.0 - config.slippage_rate)
            return True, exit_price, "donchian_low_exit"

    elif config.exit_mode == "atr_trailing_stop":
        if stop_hit:
            exit_price = position.stop_price * (1.0 - config.slippage_rate)
            return True, exit_price, "atr_trailing_stop"

        update_atr_trailing_stop(position, row, config)

    else:
        raise ValueError(f"Unknown exit mode: {config.exit_mode}")

    if is_last_row:
        exit_price = close * (1.0 - config.slippage_rate)
        return True, exit_price, "end_of_data"

    return False, np.nan, None


def close_position(
    position: OpenPosition,
    exit_row: pd.Series,
    exit_index: int,
    exit_price: float,
    exit_reason: ExitReason,
    equity: float,
    config: BacktestConfig,
) -> tuple[dict[str, float | str | pd.Timestamp], float]:
    """
    Close a position.

    Equity accounting:
        Entry fee is deducted when the position opens.
        Exit fee is deducted here.

    Trade-level net_pnl:
        Includes BOTH entry fee and exit fee.
    """
    exit_fee = position.qty * exit_price * config.fee_rate
    gross_pnl = (exit_price - position.entry_price) * position.qty
    net_pnl = gross_pnl - position.entry_fee - exit_fee

    ending_equity = equity + gross_pnl - exit_fee

    position_value = position.entry_price * position.qty
    return_pct = (net_pnl / position_value) * 100.0 if position_value > 0 else 0.0
    r_multiple = (
        net_pnl / position.initial_risk_cash
        if position.initial_risk_cash > 0
        else 0.0
    )

    trade = {
        "product_id": position.product_id,
        "entry_time": position.entry_time,
        "exit_time": exit_row["time"],
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "stop_price": position.stop_price,
        "take_profit_price": position.take_profit_price,
        "qty": position.qty,
        "entry_fee": position.entry_fee,
        "exit_fee": exit_fee,
        "total_fees": position.entry_fee + exit_fee,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "r_multiple": r_multiple,
        "exit_reason": exit_reason,
        "holding_bars": int(exit_index - position.entry_index + 1),
    }

    return trade, ending_equity


def run_backtest_for_product(
    signals: pd.DataFrame,
    product_id: str,
    config: BacktestConfig,
    timeframe: str = "1h",
) -> BacktestResult:
    data = (
        signals[signals["product_id"] == product_id]
        .copy()
        .sort_values("time")
        .reset_index(drop=True)
    )

    required_cols = {
        "time",
        "product_id",
        "open",
        "high",
        "low",
        "close",
        "long_signal",
        "atr_14",
    }
    missing = required_cols - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns for backtest: {sorted(missing)}")

    if len(data) < 2:
        trades = pd.DataFrame()
        equity_curve = pd.DataFrame()
        metrics = calculate_trade_metrics(
            product_id=product_id,
            trades=trades,
            equity_curve=equity_curve,
            initial_equity=config.initial_equity,
            buy_hold_return=0.0,
            periods_per_year=periods_per_year_from_timeframe(timeframe),
        )
        return BacktestResult(product_id, trades, equity_curve, metrics)

    equity = config.initial_equity
    position: OpenPosition | None = None
    pending_signal_row: pd.Series | None = None

    trades: list[dict[str, float | str | pd.Timestamp]] = []
    equity_records: list[dict[str, float | str | pd.Timestamp | bool]] = []

    for i, row in data.iterrows():
        # 1. Enter at this candle's open if the previous candle generated a signal.
        if position is None and pending_signal_row is not None:
            position, equity = create_position(
                product_id=product_id,
                entry_row=row,
                signal_row=pending_signal_row,
                entry_index=i,
                equity=equity,
                config=config,
            )
            pending_signal_row = None

        # 2. Manage open position using this candle's OHLC.
        if position is not None:
            should_exit, exit_price, exit_reason = check_exit(
                position=position,
                row=row,
                config=config,
                is_last_row=(i == len(data) - 1),
            )

            if should_exit and exit_reason is not None:
                trade, equity = close_position(
                    position=position,
                    exit_row=row,
                    exit_index=i,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    equity=equity,
                    config=config,
                )
                trades.append(trade)
                position = None

        # 3. Mark-to-market equity after managing the candle.
        if position is not None:
            unrealized_pnl = (float(row["close"]) - position.entry_price) * position.qty
            mark_to_market_equity = equity + unrealized_pnl
            in_position = True
        else:
            unrealized_pnl = 0.0
            mark_to_market_equity = equity
            in_position = False

        equity_records.append(
            {
                "time": row["time"],
                "product_id": product_id,
                "equity": mark_to_market_equity,
                "realized_equity": equity,
                "unrealized_pnl": unrealized_pnl,
                "close": float(row["close"]),
                "in_position": in_position,
            }
        )

        # 4. If flat, schedule next-bar entry based on current candle signal.
        #    This prevents cheating by entering on the same candle close.
        if (
            position is None
            and bool(row.get("long_signal", False))
            and i < len(data) - 1
        ):
            pending_signal_row = row

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_records)

    buy_hold_return = calculate_buy_and_hold_return(data)
    metrics = calculate_trade_metrics(
        product_id=product_id,
        trades=trades_df,
        equity_curve=equity_curve,
        initial_equity=config.initial_equity,
        buy_hold_return=buy_hold_return,
        periods_per_year=periods_per_year_from_timeframe(timeframe),
    )

    return BacktestResult(
        product_id=product_id,
        trades=trades_df,
        equity_curve=equity_curve,
        metrics=metrics,
    )


def run_backtests(
    signals: pd.DataFrame,
    config: BacktestConfig,
    timeframe: str = "1h",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trades: list[pd.DataFrame] = []
    all_equity_curves: list[pd.DataFrame] = []
    all_metrics: list[dict[str, float | int | str]] = []

    product_ids = sorted(signals["product_id"].dropna().unique())

    for product_id in product_ids:
        result = run_backtest_for_product(
            signals=signals,
            product_id=product_id,
            config=config,
            timeframe=timeframe,
        )

        if not result.trades.empty:
            all_trades.append(result.trades)

        if not result.equity_curve.empty:
            all_equity_curves.append(result.equity_curve)

        all_metrics.append(result.metrics)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = (
        pd.concat(all_equity_curves, ignore_index=True)
        if all_equity_curves
        else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(all_metrics)

    return trades_df, equity_df, metrics_df


def prepare_signals_from_raw_data(
    input_dir: Path,
    timeframe: str,
    max_vol_percentile: float,
    volume_multiplier: float,
    stop_atr_multiplier: float,
    take_profit_r_multiple: float,
) -> pd.DataFrame:
    candles = load_raw_candles(input_dir=input_dir, timeframe=timeframe)
    features = add_volatility_strategy_features(candles)

    strategy_config = VolatilityBreakoutConfig(
        max_vol_percentile=max_vol_percentile,
        volume_multiplier=volume_multiplier,
        stop_atr_multiplier=stop_atr_multiplier,
        take_profit_r_multiple=take_profit_r_multiple,
    )

    return generate_volatility_breakout_signals(
        features=features,
        config=strategy_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the volatility-compression breakout strategy."
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
        default=Path("data/backtests"),
        help="Directory where backtest outputs will be saved.",
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
        "--risk-per-trade",
        type=float,
        default=0.005,
        help="Fraction of current equity risked per trade. 0.005 = 0.5%.",
    )

    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=1.0,
        help="Maximum notional position size as a fraction of equity. 1.0 = 100%.",
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
        "--max-vol-percentile",
        type=float,
        default=20.0,
        help="Maximum realized volatility percentile allowed for compression.",
    )

    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=1.0,
        help="Volume must exceed volume_sma_20 times this multiplier.",
    )

    parser.add_argument(
        "--stop-atr-multiplier",
        type=float,
        default=2.0,
        help="Initial stop distance measured in ATR units.",
    )

    parser.add_argument(
        "--take-profit-r-multiple",
        type=float,
        default=3.0,
        help="Take-profit distance measured in initial risk units.",
    )

    parser.add_argument(
        "--exit-mode",
        type=str,
        default="fixed_r_target",
        choices=[
            "fixed_r_target",
            "ema50_exit",
            "ema200_exit",
            "atr_trailing_stop",
            "donchian_low_exit",
        ],
        help="Exit mode used by the backtester.",
    )

    parser.add_argument(
        "--trailing-atr-multiplier",
        type=float,
        default=3.0,
        help="ATR multiplier used for atr_trailing_stop mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    signals = prepare_signals_from_raw_data(
        input_dir=args.input_dir,
        timeframe=args.timeframe,
        max_vol_percentile=args.max_vol_percentile,
        volume_multiplier=args.volume_multiplier,
        stop_atr_multiplier=args.stop_atr_multiplier,
        take_profit_r_multiple=args.take_profit_r_multiple,
    )

    backtest_config = BacktestConfig(
        initial_equity=args.initial_equity,
        risk_per_trade=args.risk_per_trade,
        max_position_pct=args.max_position_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        stop_atr_multiplier=args.stop_atr_multiplier,
        take_profit_r_multiple=args.take_profit_r_multiple,
        exit_mode=args.exit_mode,
        trailing_atr_multiplier=args.trailing_atr_multiplier,
    )

    trades, equity_curves, metrics = run_backtests(
        signals=signals,
        config=backtest_config,
        timeframe=args.timeframe,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trades_path = args.output_dir / f"trades_{args.timeframe}.csv"
    equity_path = args.output_dir / f"equity_curves_{args.timeframe}.csv"
    metrics_path = args.output_dir / f"metrics_{args.timeframe}.csv"

    trades.to_csv(trades_path, index=False)
    equity_curves.to_csv(equity_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print("\nBacktest summary:")

    if metrics.empty:
        print("No metrics generated.")
    else:
        display_cols = [
            "product_id",
            "total_return_pct",
            "buy_hold_return_pct",
            "excess_return_vs_buy_hold_pct",
            "num_trades",
            "win_rate_pct",
            "profit_factor",
            "max_drawdown_pct",
            "sharpe_ratio",
        ]
        display_cols = [col for col in display_cols if col in metrics.columns]
        print(metrics[display_cols].to_string(index=False))

    print(f"\nSaved trades to: {trades_path}")
    print(f"Saved equity curves to: {equity_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()