"""
Short-term strategy rediscovery using development data only.

Run from project root:

    python -m src.short_term_rediscovery \
        --input-dir data/raw_5y_15m \
        --timeframe 15m \
        --holdout-days 365 \
        --lookbacks 4 6 12 24 48 \
        --horizons 60 120 240 480 720 \
        --filters none asset_above_ema200 asset_below_ema200 btc_above_ema200 btc_below_ema200 volume_z_gt_0 \
        --output-dir data/short_term_rediscovery

Purpose:
    Rebuild the short-term strategy discovery process without touching the final
    holdout year.

    The previous V3 strategy was useful but contaminated because it was refined
    after looking at intermediate test results. This file starts over and searches
    for short-horizon volatility/momentum strategies using only pre-holdout data.

Protocol:
    - Load full historical data.
    - Reserve the last N days as final holdout.
    - DO NOT evaluate or report performance on the holdout here.
    - Split the remaining development data into train and validation.
    - Search a grid of strategy conditions.
    - Rank candidates using train + validation only.
    - Export candidate rules for later frozen holdout testing.

Outputs:
    data/short_term_rediscovery/
        shortterm_split_summary_*.csv
        shortterm_all_results_*.csv
        shortterm_validation_leaderboard_*.csv
        shortterm_selected_candidates_*.csv
        shortterm_config_*.json

Important:
    This is a discovery/ranking tool, not the final portfolio backtest.
    The final one-year holdout should be run only after strategies are frozen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.probability_table import ProbabilityTableConfig, build_probability_dataset
from src.strategy import load_raw_candles


BTC_PRODUCT_ID = "BTC-USD"
DEFAULT_PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]


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
class CandidateSpec:
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
class RediscoveryConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    holdout_days: int
    train_frac_of_development: float
    lookbacks: list[float]
    horizons: list[int]
    filters: list[str]
    threshold_pct: float
    percentile_window: int
    bucket_size: int
    fee_rate: float
    slippage_rate: float
    min_samples_train: int
    min_samples_validation: int
    min_validation_10d_windows: int
    top_n: int
    products: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

def build_vol_regimes() -> list[VolRegime]:
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


def build_candidate_specs(
    config: RediscoveryConfig,
    lookback_hours: float,
    horizon_minutes: int,
) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []

    for side in ["long", "short"]:
        for vol in build_vol_regimes():
            for mom in build_momentum_buckets():
                for filter_name in config.filters:
                    specs.append(
                        CandidateSpec(
                            strategy_id=make_strategy_id(
                                side=side,
                                lookback_hours=lookback_hours,
                                horizon_minutes=horizon_minutes,
                                vol_regime=vol.name,
                                momentum_bucket=mom.name,
                                filter_name=filter_name,
                            ),
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
# Data preparation and split
# ─────────────────────────────────────────────────────────────────────────────

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


def build_dataset_for_combo(
    candles: pd.DataFrame,
    config: RediscoveryConfig,
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
    dataset = dataset[dataset["product_id"].isin(config.products)].copy()

    return dataset.sort_values(["product_id", "time"]).reset_index(drop=True)


def split_development_and_holdout(
    candles: pd.DataFrame,
    config: RediscoveryConfig,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Return development candles only and timestamps defining the final holdout.

    Holdout is intentionally excluded from all rediscovery calculations.
    """
    max_time = candles["time"].max()
    holdout_start = max_time - pd.Timedelta(days=config.holdout_days)
    holdout_end = max_time

    development = candles[candles["time"] < holdout_start].copy()

    if development.empty:
        raise ValueError("Development data is empty. Reduce holdout_days or download more history.")

    return development, holdout_start, holdout_end, max_time


def assign_train_validation_periods(
    dataset: pd.DataFrame,
    config: RediscoveryConfig,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    data = dataset.copy()

    min_time = data["time"].min()
    max_time = data["time"].max()
    total_seconds = (max_time - min_time).total_seconds()

    validation_start = min_time + pd.Timedelta(
        seconds=total_seconds * config.train_frac_of_development
    )

    data["period"] = np.where(data["time"] < validation_start, "train", "validation")

    return data, validation_start


def build_split_summary(
    dataset: pd.DataFrame,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    validation_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []

    for period, group in dataset.groupby("period"):
        rows.append(
            {
                "period": period,
                "start": group["time"].min(),
                "end": group["time"].max(),
                "rows": len(group),
                "products": group["product_id"].nunique(),
            }
        )

    rows.append(
        {
            "period": "reserved_final_holdout_not_used",
            "start": holdout_start,
            "end": holdout_end,
            "rows": np.nan,
            "products": np.nan,
        }
    )

    rows.append(
        {
            "period": "validation_start_marker",
            "start": validation_start,
            "end": validation_start,
            "rows": np.nan,
            "products": np.nan,
        }
    )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Masks and filters
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


def candidate_mask(dataset: pd.DataFrame, spec: CandidateSpec) -> pd.Series:
    mask = pd.Series(True, index=dataset.index)

    if spec.vol_regime != "vol_none":
        mask &= range_mask(dataset["realized_vol_percentile"], spec.vol_lower, spec.vol_upper)

    mask &= range_mask(dataset["return_lookback_pct"], spec.momentum_lower, spec.momentum_upper)
    mask &= filter_mask(dataset, spec.filter_name)

    return mask.fillna(False)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def profit_factor(pnl_like: pd.Series) -> float:
    wins = pnl_like[pnl_like > 0]
    losses = pnl_like[pnl_like < 0]

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0

    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return float("inf")
    return 0.0


def trade_path_max_drawdown(net_returns_pct: pd.Series) -> float:
    if net_returns_pct.empty:
        return 0.0

    equity = (1.0 + net_returns_pct / 100.0).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min() * 100.0)


def window_unit_return_metrics(
    selected: pd.DataFrame,
    net_return_pct: pd.Series,
    window_days: int,
) -> dict[str, float | int]:
    """
    Screening metric for repeated calendar windows.

    This is not a full portfolio result. It aggregates unit signal returns in
    non-overlapping windows inside train/validation data.

    These metrics are used only for strategy selection. Final results still come
    from portfolio backtesting with capital constraints and next-bar execution.
    """
    prefix = f"{window_days}d"

    if selected.empty:
        return {
            f"num_{prefix}_windows": 0,
            f"positive_{prefix}_window_rate_pct": 0.0,
            f"avg_{prefix}_units_return_pct": 0.0,
            f"median_{prefix}_units_return_pct": 0.0,
            f"worst_{prefix}_units_return_pct": 0.0,
            f"best_{prefix}_units_return_pct": 0.0,
        }

    temp = selected[["time"]].copy()
    temp["net_return_pct"] = net_return_pct.to_numpy()

    start = temp["time"].min()
    seconds = window_days * 86400
    temp["window_id"] = ((temp["time"] - start).dt.total_seconds() // seconds).astype(int)

    window_returns = temp.groupby("window_id")["net_return_pct"].sum()

    if window_returns.empty:
        return {
            f"num_{prefix}_windows": 0,
            f"positive_{prefix}_window_rate_pct": 0.0,
            f"avg_{prefix}_units_return_pct": 0.0,
            f"median_{prefix}_units_return_pct": 0.0,
            f"worst_{prefix}_units_return_pct": 0.0,
            f"best_{prefix}_units_return_pct": 0.0,
        }

    return {
        f"num_{prefix}_windows": int(len(window_returns)),
        f"positive_{prefix}_window_rate_pct": float((window_returns > 0).mean() * 100.0),
        f"avg_{prefix}_units_return_pct": float(window_returns.mean()),
        f"median_{prefix}_units_return_pct": float(window_returns.median()),
        f"worst_{prefix}_units_return_pct": float(window_returns.min()),
        f"best_{prefix}_units_return_pct": float(window_returns.max()),
    }


def multi_window_unit_return_metrics(
    selected: pd.DataFrame,
    net_return_pct: pd.Series,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    for window_days in [10, 30, 90]:
        metrics.update(
            window_unit_return_metrics(
                selected=selected,
                net_return_pct=net_return_pct,
                window_days=window_days,
            )
        )

    return metrics


def evaluate_candidate_on_period(
    dataset: pd.DataFrame,
    spec: CandidateSpec,
    period: str,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, object]:
    data = dataset[dataset["period"] == period]

    base = {
        **asdict(spec),
        "period": period,
    }

    if data.empty:
        return empty_metrics(base)

    selected = data[candidate_mask(data, spec)].copy()

    if selected.empty:
        return empty_metrics(base)

    side_multiplier = 1.0 if spec.side == "long" else -1.0
    gross_return_pct = side_multiplier * selected["future_return_pct"]

    # Round-trip percentage cost.
    cost_pct = 2.0 * (fee_rate + slippage_rate) * 100.0
    net_return_pct = gross_return_pct - cost_pct

    if spec.side == "long":
        p_desired = float((selected["future_direction"] == "UP").mean())
        p_adverse = float((selected["future_direction"] == "DOWN").mean())
    else:
        p_desired = float((selected["future_direction"] == "DOWN").mean())
        p_adverse = float((selected["future_direction"] == "UP").mean())

    window_metrics = multi_window_unit_return_metrics(selected, net_return_pct)

    return {
        **base,
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
        "win_rate_pct": float((net_return_pct > 0).mean() * 100.0),
        "profit_factor": profit_factor(net_return_pct),
        "trade_path_max_drawdown_pct": trade_path_max_drawdown(net_return_pct),
        "total_net_units_return_pct": float(net_return_pct.sum()),
        "cost_pct_per_trade": cost_pct,
        **window_metrics,
    }


def empty_metrics(base: dict[str, object]) -> dict[str, object]:
    return {
        **base,
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
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "trade_path_max_drawdown_pct": 0.0,
        "total_net_units_return_pct": 0.0,
        "cost_pct_per_trade": 0.0,
        "num_10d_windows": 0,
        "positive_10d_window_rate_pct": 0.0,
        "avg_10d_units_return_pct": 0.0,
        "median_10d_units_return_pct": 0.0,
        "worst_10d_units_return_pct": 0.0,
        "best_10d_units_return_pct": 0.0,
        "num_30d_windows": 0,
        "positive_30d_window_rate_pct": 0.0,
        "avg_30d_units_return_pct": 0.0,
        "median_30d_units_return_pct": 0.0,
        "worst_30d_units_return_pct": 0.0,
        "best_30d_units_return_pct": 0.0,
        "num_90d_windows": 0,
        "positive_90d_window_rate_pct": 0.0,
        "avg_90d_units_return_pct": 0.0,
        "median_90d_units_return_pct": 0.0,
        "worst_90d_units_return_pct": 0.0,
        "best_90d_units_return_pct": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grid evaluation and ranking
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_combo(
    development_candles: pd.DataFrame,
    config: RediscoveryConfig,
    lookback_hours: float,
    horizon_minutes: int,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Building development dataset: lookback={lookback_hours:g}h horizon={horizon_minutes}m")

    dataset = build_dataset_for_combo(
        candles=development_candles,
        config=config,
        lookback_hours=lookback_hours,
        horizon_minutes=horizon_minutes,
    )

    dataset, validation_start = assign_train_validation_periods(dataset, config)

    specs = build_candidate_specs(config, lookback_hours, horizon_minutes)
    print(f"Evaluating {len(specs)} candidate rules")

    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.append(
            evaluate_candidate_on_period(
                dataset=dataset,
                spec=spec,
                period="train",
                fee_rate=config.fee_rate,
                slippage_rate=config.slippage_rate,
            )
        )
        rows.append(
            evaluate_candidate_on_period(
                dataset=dataset,
                spec=spec,
                period="validation",
                fee_rate=config.fee_rate,
                slippage_rate=config.slippage_rate,
            )
        )

    result = pd.DataFrame(rows)

    split_summary = build_split_summary(
        dataset=dataset,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        validation_start=validation_start,
    )
    split_summary["lookback_hours"] = lookback_hours
    split_summary["horizon_minutes"] = horizon_minutes

    return result, split_summary


def build_leaderboard(results: pd.DataFrame, config: RediscoveryConfig) -> pd.DataFrame:
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
        "total_net_units_return_pct",
        "num_10d_windows",
        "positive_10d_window_rate_pct",
        "avg_10d_units_return_pct",
        "median_10d_units_return_pct",
        "worst_10d_units_return_pct",
        "best_10d_units_return_pct",
        "num_30d_windows",
        "positive_30d_window_rate_pct",
        "avg_30d_units_return_pct",
        "median_30d_units_return_pct",
        "worst_30d_units_return_pct",
        "best_30d_units_return_pct",
        "num_90d_windows",
        "positive_90d_window_rate_pct",
        "avg_90d_units_return_pct",
        "median_90d_units_return_pct",
        "worst_90d_units_return_pct",
        "best_90d_units_return_pct",
    ]

    train = results[results["period"] == "train"][id_cols + metric_cols].copy()
    validation = results[results["period"] == "validation"][id_cols + metric_cols].copy()

    train = train.rename(columns={col: f"train_{col}" for col in metric_cols})
    validation = validation.rename(columns={col: f"validation_{col}" for col in metric_cols})

    wide = train.merge(validation, on=id_cols, how="inner")

    for col in wide.columns:
        if col not in id_cols:
            wide[col] = wide[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    candidate_mask = (
        (wide["train_samples"] >= config.min_samples_train)
        & (wide["validation_samples"] >= config.min_samples_validation)
        & (wide["validation_num_10d_windows"] >= config.min_validation_10d_windows)
        & (wide["train_avg_net_return_pct"] > 0)
        & (wide["validation_avg_net_return_pct"] > 0)
        & (wide["train_profit_factor"] > 1.0)
        & (wide["validation_profit_factor"] > 1.0)
        & (wide["validation_positive_10d_window_rate_pct"] > 50.0)
    )

    candidates = wide[candidate_mask].copy()

    if candidates.empty:
        return candidates

    # Competition-style ranking: prioritize validation, but include 10d consistency.
    candidates["validation_score"] = (
        candidates["validation_avg_net_return_pct"]
        * np.sqrt(candidates["validation_samples"].clip(lower=1))
        * np.maximum(candidates["validation_profit_factor"] - 1.0, 0.0)
        * (candidates["validation_positive_10d_window_rate_pct"] / 100.0)
    )

    # Stability between train and validation. Lower is better.
    candidates["train_validation_avg_return_gap"] = (
        candidates["validation_avg_net_return_pct"] - candidates["train_avg_net_return_pct"]
    ).abs()
    candidates["train_validation_pf_gap"] = (
        candidates["validation_profit_factor"] - candidates["train_profit_factor"]
    ).abs()

    candidates["stability_score"] = 1.0 / (
        1.0
        + candidates["train_validation_avg_return_gap"]
        + candidates["train_validation_pf_gap"]
    )

    candidates["final_discovery_score"] = candidates["validation_score"] * candidates["stability_score"]

    return candidates.sort_values(
        [
            "final_discovery_score",
            "validation_avg_net_return_pct",
            "validation_profit_factor",
            "validation_positive_10d_window_rate_pct",
            "validation_samples",
        ],
        ascending=False,
    ).reset_index(drop=True)


def build_selected_candidates(leaderboard: pd.DataFrame, config: RediscoveryConfig) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard

    # Avoid selecting 100 versions of the same tiny idea.
    # Keep top candidates but preserve diversity across side/lookback/horizon/regime.
    selected_rows = []
    seen_keys = set()

    for _, row in leaderboard.iterrows():
        key = (
            row["side"],
            row["lookback_hours"],
            row["horizon_minutes"],
            row["vol_regime"],
            row["momentum_bucket"],
        )

        if key in seen_keys:
            continue

        selected_rows.append(row)
        seen_keys.add(key)

        if len(selected_rows) >= config.top_n:
            break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def run_rediscovery(candles: pd.DataFrame, config: RediscoveryConfig) -> dict[str, pd.DataFrame]:
    candles = candles.copy().sort_values(["product_id", "time"]).reset_index(drop=True)
    candles = candles[candles["product_id"].isin(config.products)].copy()

    development_candles, holdout_start, holdout_end, max_time = split_development_and_holdout(candles, config)

    print("Rediscovery split:")
    print(f"Development: {development_candles['time'].min()} -> {development_candles['time'].max()}")
    print(f"Reserved final holdout, NOT used here: {holdout_start} -> {holdout_end}")

    result_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []

    for lookback_hours in config.lookbacks:
        for horizon_minutes in config.horizons:
            result, split_summary = evaluate_combo(
                development_candles=development_candles,
                config=config,
                lookback_hours=lookback_hours,
                horizon_minutes=horizon_minutes,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            )
            result_frames.append(result)
            split_frames.append(split_summary)

    all_results = pd.concat(result_frames, ignore_index=True)
    split_summary = pd.concat(split_frames, ignore_index=True)
    leaderboard = build_leaderboard(all_results, config)
    selected_candidates = build_selected_candidates(leaderboard, config)

    return {
        "all_results": all_results,
        "split_summary": split_summary,
        "leaderboard": leaderboard,
        "selected_candidates": selected_candidates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save and CLI
# ─────────────────────────────────────────────────────────────────────────────

def make_suffix(config: RediscoveryConfig) -> str:
    lookbacks = "_".join(f"{x:g}".replace(".", "p") for x in config.lookbacks)
    horizons = "_".join(str(x) for x in config.horizons)

    return (
        f"{config.timeframe}_dev_only_holdout_{config.holdout_days}d_"
        f"lb_{lookbacks}_h_{horizons}"
    ).replace(".", "p")


def save_outputs(results: dict[str, pd.DataFrame], config: RediscoveryConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = make_suffix(config)

    paths = {
        "all_results": config.output_dir / f"shortterm_all_results_{suffix}.csv",
        "split_summary": config.output_dir / f"shortterm_split_summary_{suffix}.csv",
        "leaderboard": config.output_dir / f"shortterm_validation_leaderboard_{suffix}.csv",
        "selected_candidates": config.output_dir / f"shortterm_selected_candidates_{suffix}.csv",
        "config": config.output_dir / f"shortterm_config_{suffix}.json",
    }

    results["all_results"].to_csv(paths["all_results"], index=False)
    results["split_summary"].to_csv(paths["split_summary"], index=False)
    results["leaderboard"].to_csv(paths["leaderboard"], index=False)
    results["selected_candidates"].to_csv(paths["selected_candidates"], index=False)

    json_config = asdict(config)
    json_config["input_dir"] = str(config.input_dir)
    json_config["output_dir"] = str(config.output_dir)

    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rediscover short-term vol/momentum strategies using development data only."
    )

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_5y_15m"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/short_term_rediscovery"))
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--holdout-days", type=int, default=365)
    parser.add_argument("--train-frac-of-development", type=float, default=0.75)

    parser.add_argument("--lookbacks", type=float, nargs="+", default=[4.0, 6.0, 12.0, 24.0, 48.0])
    parser.add_argument("--horizons", type=int, nargs="+", default=[60, 120, 240, 480, 720])
    parser.add_argument(
        "--filters",
        nargs="+",
        default=[
            "none",
            "asset_above_ema200",
            "asset_below_ema200",
            "btc_above_ema200",
            "btc_below_ema200",
            "volume_z_gt_0",
        ],
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

    parser.add_argument("--fee-rate", type=float, default=0.0001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)

    parser.add_argument("--min-samples-train", type=int, default=500)
    parser.add_argument("--min-samples-validation", type=int, default=150)
    parser.add_argument("--min-validation-10d-windows", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=50)

    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.holdout_days <= 0:
        raise ValueError("holdout-days must be positive.")

    if not 0.1 <= args.train_frac_of_development <= 0.9:
        raise ValueError("train-frac-of-development should be between 0.1 and 0.9.")

    if any(x <= 0 for x in args.lookbacks):
        raise ValueError("All lookbacks must be positive.")

    if any(x <= 0 for x in args.horizons):
        raise ValueError("All horizons must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    config = RediscoveryConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        holdout_days=args.holdout_days,
        train_frac_of_development=args.train_frac_of_development,
        lookbacks=args.lookbacks,
        horizons=args.horizons,
        filters=args.filters,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        min_samples_train=args.min_samples_train,
        min_samples_validation=args.min_samples_validation,
        min_validation_10d_windows=args.min_validation_10d_windows,
        top_n=args.top_n,
        products=args.products,
    )

    candles = load_raw_candles(config.input_dir, config.timeframe)
    results = run_rediscovery(candles, config)
    paths = save_outputs(results, config)

    print("\nTop rediscovered short-term candidates:")
    leaderboard = results["leaderboard"]

    if leaderboard.empty:
        print("No candidates passed the train/validation filters.")
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
            "validation_positive_10d_window_rate_pct",
            "validation_median_10d_units_return_pct",
            "validation_worst_10d_units_return_pct",
            "final_discovery_score",
        ]
        display_cols = [col for col in display_cols if col in leaderboard.columns]
        print(leaderboard[display_cols].head(config.top_n).to_string(index=False))

    print("\nSelected candidates for human review:")
    selected = results["selected_candidates"]
    if selected.empty:
        print("No selected candidates.")
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
            "validation_profit_factor",
            "validation_positive_10d_window_rate_pct",
        ]
        display_cols = [col for col in display_cols if col in selected.columns]
        print(selected[display_cols].head(config.top_n).to_string(index=False))

    print("\nSaved short-term rediscovery outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
