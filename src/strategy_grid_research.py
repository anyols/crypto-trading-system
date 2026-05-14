"""
Strategy grid research for volatility-conditioned momentum/reversion signals.

Run from project root:

    python -m src.strategy_grid_research --timeframe 15m

Recommended first run:

    python -m src.strategy_grid_research \
        --timeframe 15m \
        --lookbacks 4 6 12 24 48 \
        --horizons 60 120 240 480 720 \
        --filters none asset_above_ema200 asset_below_ema200 btc_above_ema200 btc_below_ema200 volume_z_gt_0 \
        --output-dir data/strategy_grid_research

Purpose:
    V3 is promising but suspiciously overfit because it was manually refined after
    inspecting earlier results. This file explores a broader strategy family under
    a stricter research protocol.

Research question:
    Does realized-volatility regime improve short-term momentum/reversion signals
    in crypto markets beyond momentum alone?

This script tests combinations of:
    - lookback window: e.g. 4h, 6h, 12h, 24h, 48h
    - holding horizon: e.g. 1h, 2h, 4h, 8h, 12h
    - volatility regime: none, broad regimes, deciles
    - momentum bucket: negative / flat / positive recent returns
    - side: long or short
    - optional filters: trend, BTC trend, volume, weekday/weekend

Validation protocol:
    - oldest portion of data: train / discovery
    - middle portion of data: validation / ranking
    - newest portion of data: final holdout / sanity check

Important:
    The holdout should not be used for further rule tweaking. Once you inspect it,
    stop modifying the strategy based on that result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.probability_table import ProbabilityTableConfig, build_probability_dataset
from src.strategy import load_raw_candles


@dataclass(frozen=True)
class VolRegime:
    name: str
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class MomentumBucket:
    name: str
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    side: str
    lookback_hours: float
    horizon_minutes: int
    vol_regime: str
    vol_lower: float | None
    vol_upper: float | None
    momentum_bucket: str
    momentum_lower: float | None
    momentum_upper: float | None
    filter_name: str


@dataclass(frozen=True)
class GridResearchConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    lookbacks: list[float]
    horizons: list[int]
    filters: list[str]
    threshold_pct: float
    percentile_window: int
    bucket_size: int
    train_frac: float
    validation_frac: float
    fee_rate: float
    slippage_rate: float
    min_samples_train: int
    min_samples_validation: int
    top_n: int
    btc_product_id: str
    products: list[str] | None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy search space
# ─────────────────────────────────────────────────────────────────────────────

def build_vol_regimes() -> list[VolRegime]:
    """
    Volatility regimes based on realized-volatility percentile.

    We include both broad regimes and deciles.
    Broad regimes are less overfit-prone.
    Deciles are more precise but more fragile.
    """
    regimes = [
        VolRegime("vol_none", None, None),
        VolRegime("vol_low_0_30", 0.0, 30.0),
        VolRegime("vol_mid_30_70", 30.0, 70.0),
        VolRegime("vol_high_70_100", 70.0, 100.0),
    ]

    for lower in range(0, 100, 10):
        regimes.append(
            VolRegime(
                name=f"vol_{lower}_{lower + 10}",
                lower=float(lower),
                upper=float(lower + 10),
            )
        )

    return regimes


def build_momentum_buckets() -> list[MomentumBucket]:
    """
    Buckets for recent lookback return, in percent.

    These cover both mean-reversion and continuation hypotheses.
    The script tests both long and short sides on each bucket.
    """
    return [
        MomentumBucket("mom_lt_minus6", None, -6.0),
        MomentumBucket("mom_minus6_to_minus4", -6.0, -4.0),
        MomentumBucket("mom_minus4_to_minus2", -4.0, -2.0),
        MomentumBucket("mom_minus2_to_minus1", -2.0, -1.0),
        MomentumBucket("mom_minus1_to_minus0p25", -1.0, -0.25),
        MomentumBucket("mom_flat_minus0p25_to_0p25", -0.25, 0.25),
        MomentumBucket("mom_0p25_to_1", 0.25, 1.0),
        MomentumBucket("mom_1_to_2", 1.0, 2.0),
        MomentumBucket("mom_2_to_4", 2.0, 4.0),
        MomentumBucket("mom_4_to_6", 4.0, 6.0),
        MomentumBucket("mom_gt_6", 6.0, None),
    ]


def make_strategy_id(
    side: str,
    lookback_hours: float,
    horizon_minutes: int,
    vol_regime: str,
    momentum_bucket: str,
    filter_name: str,
) -> str:
    lookback = f"{lookback_hours:g}h".replace(".", "p")
    return (
        f"{side}__lb_{lookback}__h_{horizon_minutes}m__"
        f"{vol_regime}__{momentum_bucket}__filter_{filter_name}"
    )


def build_strategy_specs(
    config: GridResearchConfig,
    lookback_hours: float,
    horizon_minutes: int,
) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    vol_regimes = build_vol_regimes()
    momentum_buckets = build_momentum_buckets()

    for side in ["long", "short"]:
        for vol in vol_regimes:
            for mom in momentum_buckets:
                for filter_name in config.filters:
                    strategy_id = make_strategy_id(
                        side=side,
                        lookback_hours=lookback_hours,
                        horizon_minutes=horizon_minutes,
                        vol_regime=vol.name,
                        momentum_bucket=mom.name,
                        filter_name=filter_name,
                    )
                    specs.append(
                        StrategySpec(
                            strategy_id=strategy_id,
                            side=side,
                            lookback_hours=lookback_hours,
                            horizon_minutes=horizon_minutes,
                            vol_regime=vol.name,
                            vol_lower=vol.lower,
                            vol_upper=vol.upper,
                            momentum_bucket=mom.name,
                            momentum_lower=mom.lower,
                            momentum_upper=mom.upper,
                            filter_name=filter_name,
                        )
                    )

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def add_btc_context(dataset: pd.DataFrame, btc_product_id: str) -> pd.DataFrame:
    """
    Add BTC trend context to every asset row.

    Uses BTC price_vs_ema200_pct at the same timestamp.
    """
    data = dataset.copy()

    btc = data[data["product_id"] == btc_product_id][
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


def build_dataset_for_combo(
    candles: pd.DataFrame,
    config: GridResearchConfig,
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
    dataset = add_btc_context(dataset, config.btc_product_id)

    if config.products:
        dataset = dataset[dataset["product_id"].isin(config.products)].copy()

    return dataset.sort_values(["product_id", "time"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Split management
# ─────────────────────────────────────────────────────────────────────────────

def assign_periods(dataset: pd.DataFrame, train_frac: float, validation_frac: float) -> pd.DataFrame:
    """
    Chronological train / validation / holdout split.
    """
    data = dataset.copy()

    min_time = data["time"].min()
    max_time = data["time"].max()
    total_seconds = (max_time - min_time).total_seconds()

    train_end = min_time + pd.Timedelta(seconds=total_seconds * train_frac)
    validation_end = min_time + pd.Timedelta(seconds=total_seconds * (train_frac + validation_frac))

    data["period"] = "holdout"
    data.loc[data["time"] < validation_end, "period"] = "validation"
    data.loc[data["time"] < train_end, "period"] = "train"

    return data


def build_split_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset.groupby("period")
        .agg(
            start=("time", "min"),
            end=("time", "max"),
            rows=("time", "count"),
            products=("product_id", "nunique"),
        )
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Condition masks
# ─────────────────────────────────────────────────────────────────────────────

def range_mask(series: pd.Series, lower: float | None, upper: float | None) -> pd.Series:
    mask = pd.Series(True, index=series.index)

    if lower is not None:
        mask &= series >= lower
    if upper is not None:
        mask &= series < upper

    return mask.fillna(False)


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

    if filter_name == "weekday":
        return (dataset["day_of_week"] < 5).fillna(False)

    if filter_name == "weekend":
        return (dataset["day_of_week"] >= 5).fillna(False)

    raise ValueError(f"Unknown filter_name: {filter_name}")


def strategy_mask(dataset: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    mask = pd.Series(True, index=dataset.index)

    if spec.vol_regime != "vol_none":
        mask &= range_mask(dataset["realized_vol_percentile"], spec.vol_lower, spec.vol_upper)

    mask &= range_mask(dataset["return_lookback_pct"], spec.momentum_lower, spec.momentum_upper)
    mask &= filter_mask(dataset, spec.filter_name)

    return mask.fillna(False)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_profit_factor(net_returns_pct: pd.Series) -> float:
    wins = net_returns_pct[net_returns_pct > 0]
    losses = net_returns_pct[net_returns_pct < 0]

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0

    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return float("inf")
    return 0.0


def calculate_trade_path_drawdown(net_returns_pct: pd.Series) -> float:
    """
    Drawdown of a simple sequential trade equity curve.

    This is not a full portfolio backtest. It measures how rough the signal's
    sequence of trades is if each signal is treated as one unit trade.
    """
    if net_returns_pct.empty:
        return 0.0

    equity = (1.0 + net_returns_pct / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def evaluate_spec_on_period(
    dataset: pd.DataFrame,
    spec: StrategySpec,
    period: str,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, object]:
    data = dataset[dataset["period"] == period]

    if data.empty:
        return empty_result(spec, period)

    selected = data[strategy_mask(data, spec)].copy()

    if selected.empty:
        return empty_result(spec, period)

    side_multiplier = 1.0 if spec.side == "long" else -1.0
    gross_return_pct = side_multiplier * selected["future_return_pct"]

    # Approximate round-trip cost in percent.
    cost_pct = 2.0 * (fee_rate + slippage_rate) * 100.0
    net_return_pct = gross_return_pct - cost_pct

    if spec.side == "long":
        p_desired = float((selected["future_direction"] == "UP").mean())
        p_adverse = float((selected["future_direction"] == "DOWN").mean())
    else:
        p_desired = float((selected["future_direction"] == "DOWN").mean())
        p_adverse = float((selected["future_direction"] == "UP").mean())

    result = {
        **spec.__dict__,
        "period": period,
        "samples": int(len(selected)),
        "products": int(selected["product_id"].nunique()),
        "start": selected["time"].min(),
        "end": selected["time"].max(),
        "p_desired": p_desired,
        "p_adverse": p_adverse,
        "directional_edge": p_desired - p_adverse,
        "avg_gross_return_pct": float(gross_return_pct.mean()),
        "median_gross_return_pct": float(gross_return_pct.median()),
        "avg_net_return_pct": float(net_return_pct.mean()),
        "median_net_return_pct": float(net_return_pct.median()),
        "std_net_return_pct": float(net_return_pct.std(ddof=1)) if len(net_return_pct) > 1 else 0.0,
        "p10_net_return_pct": float(net_return_pct.quantile(0.10)),
        "p90_net_return_pct": float(net_return_pct.quantile(0.90)),
        "win_rate_pct": float((net_return_pct > 0).mean() * 100.0),
        "profit_factor": calculate_profit_factor(net_return_pct),
        "trade_path_max_drawdown_pct": calculate_trade_path_drawdown(net_return_pct),
        "total_net_return_units_pct": float(net_return_pct.sum()),
        "cost_pct_per_trade": cost_pct,
    }

    return result


def empty_result(spec: StrategySpec, period: str) -> dict[str, object]:
    return {
        **spec.__dict__,
        "period": period,
        "samples": 0,
        "products": 0,
        "start": pd.NaT,
        "end": pd.NaT,
        "p_desired": 0.0,
        "p_adverse": 0.0,
        "directional_edge": 0.0,
        "avg_gross_return_pct": 0.0,
        "median_gross_return_pct": 0.0,
        "avg_net_return_pct": 0.0,
        "median_net_return_pct": 0.0,
        "std_net_return_pct": 0.0,
        "p10_net_return_pct": 0.0,
        "p90_net_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "trade_path_max_drawdown_pct": 0.0,
        "total_net_return_units_pct": 0.0,
        "cost_pct_per_trade": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grid run
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_grid_for_combo(
    candles: pd.DataFrame,
    config: GridResearchConfig,
    lookback_hours: float,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Building dataset for lookback={lookback_hours:g}h horizon={horizon_minutes}m")

    dataset = build_dataset_for_combo(
        candles=candles,
        config=config,
        lookback_hours=lookback_hours,
        horizon_minutes=horizon_minutes,
    )
    dataset = assign_periods(dataset, config.train_frac, config.validation_frac)

    specs = build_strategy_specs(config, lookback_hours, horizon_minutes)
    rows: list[dict[str, object]] = []

    print(f"Evaluating {len(specs)} specs for lookback={lookback_hours:g}h horizon={horizon_minutes}m")

    for spec in specs:
        for period in ["train", "validation", "holdout"]:
            rows.append(
                evaluate_spec_on_period(
                    dataset=dataset,
                    spec=spec,
                    period=period,
                    fee_rate=config.fee_rate,
                    slippage_rate=config.slippage_rate,
                )
            )

    results = pd.DataFrame(rows)
    split_summary = build_split_summary(dataset)
    split_summary["lookback_hours"] = lookback_hours
    split_summary["horizon_minutes"] = horizon_minutes

    return results, split_summary


def build_validation_leaderboard(results: pd.DataFrame, config: GridResearchConfig) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    id_cols = [
        "strategy_id",
        "side",
        "lookback_hours",
        "horizon_minutes",
        "vol_regime",
        "vol_lower",
        "vol_upper",
        "momentum_bucket",
        "momentum_lower",
        "momentum_upper",
        "filter_name",
    ]

    metric_cols = [
        "samples",
        "products",
        "p_desired",
        "p_adverse",
        "directional_edge",
        "avg_net_return_pct",
        "median_net_return_pct",
        "win_rate_pct",
        "profit_factor",
        "trade_path_max_drawdown_pct",
        "total_net_return_units_pct",
    ]

    frames: list[pd.DataFrame] = []
    for period in ["train", "validation", "holdout"]:
        part = results[results["period"] == period][id_cols + metric_cols].copy()
        rename = {col: f"{period}_{col}" for col in metric_cols}
        part = part.rename(columns=rename)
        frames.append(part)

    wide = frames[0]
    for part in frames[1:]:
        wide = wide.merge(part, on=id_cols, how="outer")

    for col in wide.columns:
        if col.endswith("samples") or col.endswith("products"):
            wide[col] = wide[col].fillna(0)
        elif col not in id_cols:
            wide[col] = wide[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Candidate must have enough samples and positive train/validation behavior.
    candidate_mask = (
        (wide["train_samples"] >= config.min_samples_train)
        & (wide["validation_samples"] >= config.min_samples_validation)
        & (wide["train_avg_net_return_pct"] > 0)
        & (wide["validation_avg_net_return_pct"] > 0)
        & (wide["train_profit_factor"] > 1.0)
        & (wide["validation_profit_factor"] > 1.0)
    )

    candidates = wide[candidate_mask].copy()

    if candidates.empty:
        return candidates

    # Ranking score emphasizes validation performance but punishes tiny sample sizes.
    candidates["validation_score"] = (
        candidates["validation_avg_net_return_pct"]
        * np.sqrt(candidates["validation_samples"].clip(lower=1))
        * np.maximum(candidates["validation_profit_factor"] - 1.0, 0.0)
    )

    candidates["holdout_pass"] = (
        (candidates["holdout_samples"] >= max(50, config.min_samples_validation // 3))
        & (candidates["holdout_avg_net_return_pct"] > 0)
        & (candidates["holdout_profit_factor"] > 1.0)
    )

    candidates["robustness_label"] = np.where(
        candidates["holdout_pass"],
        "train_validation_holdout_positive",
        "train_validation_positive_holdout_failed_or_weak",
    )

    sort_cols = [
        "validation_score",
        "validation_avg_net_return_pct",
        "validation_profit_factor",
        "validation_samples",
    ]

    return candidates.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def run_strategy_grid(candles: pd.DataFrame, config: GridResearchConfig) -> dict[str, pd.DataFrame]:
    candles = candles.copy().sort_values(["product_id", "time"]).reset_index(drop=True)

    if config.products:
        candles = candles[candles["product_id"].isin(config.products)].copy()

    result_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []

    for lookback_hours in config.lookbacks:
        for horizon_minutes in config.horizons:
            result, split_summary = evaluate_grid_for_combo(
                candles=candles,
                config=config,
                lookback_hours=lookback_hours,
                horizon_minutes=horizon_minutes,
            )
            result_frames.append(result)
            split_frames.append(split_summary)

    results = pd.concat(result_frames, ignore_index=True)
    split_summary = pd.concat(split_frames, ignore_index=True)
    leaderboard = build_validation_leaderboard(results, config)

    return {
        "grid_results": results,
        "split_summary": split_summary,
        "validation_leaderboard": leaderboard,
        "holdout_passes": leaderboard[leaderboard["holdout_pass"]].copy() if not leaderboard.empty else pd.DataFrame(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save / CLI
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(results: dict[str, pd.DataFrame], config: GridResearchConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lookbacks = "_".join(f"{x:g}".replace(".", "p") for x in config.lookbacks)
    horizons = "_".join(str(x) for x in config.horizons)
    product_suffix = "all" if not config.products else "_".join(p.replace("-", "") for p in config.products)

    suffix = (
        f"{config.timeframe}_lb_{lookbacks}_h_{horizons}_"
        f"train_{config.train_frac:g}_val_{config.validation_frac:g}_{product_suffix}"
    ).replace(".", "p")

    paths = {
        "grid_results": config.output_dir / f"strategy_grid_results_{suffix}.csv",
        "split_summary": config.output_dir / f"strategy_grid_split_summary_{suffix}.csv",
        "validation_leaderboard": config.output_dir / f"strategy_grid_validation_leaderboard_{suffix}.csv",
        "holdout_passes": config.output_dir / f"strategy_grid_holdout_passes_{suffix}.csv",
    }

    for key, path in paths.items():
        results[key].to_csv(path, index=False)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore volatility/momentum strategy grid with train/validation/holdout splits.")

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/strategy_grid_research"))
    parser.add_argument("--timeframe", type=str, default="15m")

    parser.add_argument("--lookbacks", type=float, nargs="+", default=[4.0, 6.0, 12.0, 24.0, 48.0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[60, 120, 240, 480, 720])
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["none", "asset_above_ema200", "asset_below_ema200", "btc_above_ema200", "btc_below_ema200", "volume_z_gt_0"],
        choices=[
            "none",
            "asset_above_ema200",
            "asset_below_ema200",
            "btc_above_ema200",
            "btc_below_ema200",
            "volume_z_gt_0",
            "volume_z_gt_1",
            "weekday",
            "weekend",
        ],
    )

    parser.add_argument("--threshold-pct", type=float, default=0.30)
    parser.add_argument("--percentile-window", type=int, default=200)
    parser.add_argument("--bucket-size", type=int, default=10)

    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--validation-frac", type=float, default=0.20)

    parser.add_argument("--fee-rate", type=float, default=0.0001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)

    parser.add_argument("--min-samples-train", type=int, default=500)
    parser.add_argument("--min-samples-validation", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=100)

    parser.add_argument("--btc-product-id", type=str, default="BTC-USD")
    parser.add_argument("--products", nargs="+", default=None)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_frac <= 0 or args.validation_frac <= 0:
        raise ValueError("train_frac and validation_frac must be positive.")

    if args.train_frac + args.validation_frac >= 1.0:
        raise ValueError("train_frac + validation_frac must be less than 1.0 so holdout exists.")

    if any(x <= 0 for x in args.lookbacks):
        raise ValueError("All lookbacks must be positive.")

    if any(x <= 0 for x in args.horizons):
        raise ValueError("All horizons must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    config = GridResearchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        lookbacks=args.lookbacks,
        horizons=args.horizons,
        filters=args.filters,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        min_samples_train=args.min_samples_train,
        min_samples_validation=args.min_samples_validation,
        top_n=args.top_n,
        btc_product_id=args.btc_product_id,
        products=args.products,
    )

    candles = load_raw_candles(config.input_dir, config.timeframe)
    results = run_strategy_grid(candles, config)
    paths = save_outputs(results, config)

    print("\nTop validation-ranked strategies:")
    leaderboard = results["validation_leaderboard"]

    if leaderboard.empty:
        print("No strategies passed train/validation filters.")
    else:
        display_cols = [
            "strategy_id",
            "side",
            "lookback_hours",
            "horizon_minutes",
            "vol_regime",
            "momentum_bucket",
            "filter_name",
            "train_samples",
            "validation_samples",
            "validation_avg_net_return_pct",
            "validation_profit_factor",
            "validation_win_rate_pct",
            "holdout_samples",
            "holdout_avg_net_return_pct",
            "holdout_profit_factor",
            "holdout_pass",
        ]
        display_cols = [col for col in display_cols if col in leaderboard.columns]
        print(leaderboard[display_cols].head(config.top_n).to_string(index=False))

    print("\nHoldout-passing strategies:")
    holdout = results["holdout_passes"]

    if holdout.empty:
        print("No validation-ranked strategies also passed holdout.")
    else:
        display_cols = [
            "strategy_id",
            "side",
            "lookback_hours",
            "horizon_minutes",
            "vol_regime",
            "momentum_bucket",
            "filter_name",
            "validation_avg_net_return_pct",
            "holdout_avg_net_return_pct",
            "validation_profit_factor",
            "holdout_profit_factor",
            "holdout_samples",
        ]
        display_cols = [col for col in display_cols if col in holdout.columns]
        print(holdout[display_cols].head(config.top_n).to_string(index=False))

    print("\nSaved strategy grid outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
