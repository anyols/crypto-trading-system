"""
Walk-forward strategy research.

Run from project root.

Smoke test:

    py -m src.walkforward_strategy_research ^
      --input-dir data/raw_5y_15m ^
      --output-dir data/walkforward_strategy_research_smoke ^
      --first-test-year 2024 ^
      --last-test-year 2024 ^
      --lookbacks 6 24 ^
      --horizons 240 720 ^
      --filters none asset_below_ema200 btc_below_ema200 ^
      --top-n 3

Full run:

    py -m src.walkforward_strategy_research ^
      --input-dir data/raw_5y_15m ^
      --output-dir data/walkforward_strategy_research_full ^
      --first-test-year 2024 ^
      --last-test-year 2026 ^
      --lookbacks 4 6 12 24 48 ^
      --horizons 60 120 240 480 720 ^
      --filters none asset_above_ema200 asset_below_ema200 btc_above_ema200 btc_below_ema200 volume_z_gt_0 ^
      --top-n 5

Purpose:
    Test whether the strategy discovery PROCESS survives through time.

    For each fold:
        1. Use only data before the test year.
        2. Split that pre-test data into train/validation.
        3. Run the short-term strategy grid on train/validation only.
        4. Select top candidates using validation score only.
        5. Freeze those candidates.
        6. Test them on the next unseen year using next-bar execution and fees.

Methodological goal:
    This is stricter than simply testing one hand-picked rule. It asks whether a
    systematic rule-selection process would have worked when repeated through time.

Outputs:
    walkforward_fold_metrics_*.csv
    walkforward_selected_candidates_*.csv
    walkforward_rank_summary_*.csv
    walkforward_trades_*.csv
    walkforward_equity_*.csv
    walkforward_asset_contributions_*.csv
    walkforward_split_summary_*.csv

Important:
    This file does NOT train ML. ML should be added later as a benchmark using
    the same fold definitions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.final_strategy_backtest import (
    FinalBacktestConfig,
    build_feature_dataset,
    build_signals_for_strategy,
    build_strategy_metrics,
    run_buy_and_hold_benchmark,
    run_rule_strategy_backtest,
)
from src.final_strategy_definitions import StrategyDefinition
from src.short_term_rediscovery import (
    DEFAULT_PRODUCTS,
    RediscoveryConfig,
    build_leaderboard,
    evaluate_combo,
)
from src.strategy import load_raw_candles


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_last_time: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardConfig:
    input_dir: Path
    output_dir: Path
    timeframe: str
    products: list[str]
    first_test_year: int
    last_test_year: int | None
    min_train_days: int
    min_test_days: int
    train_frac_of_development: float
    lookbacks: list[float]
    horizons: list[int]
    filters: list[str]
    threshold_pct: float
    percentile_window: int
    bucket_size: int
    fee_rate: float
    slippage_rate: float
    initial_equity: float
    position_pct: float
    max_gross_leverage: float
    min_samples_train: int
    min_samples_validation: int
    min_validation_10d_windows: int
    top_n: int
    include_benchmark: bool
    embargo_hours: float | None


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return value
    return value


def safe_float_or_none(value: Any) -> float | None:
    value = nan_to_none(value)
    if value is None:
        return None
    return float(value)


def make_safe_id(text: str, max_len: int = 130) -> str:
    safe = str(text)
    for old, new in [
        (" ", "_"),
        ("<", "lt"),
        (">", "gt"),
        ("=", "eq"),
        ("+", "plus"),
        ("-", "minus"),
        (".", "p"),
        ("/", "_"),
        ("%", "pct"),
        (",", ""),
        ("[", ""),
        ("]", ""),
        ("(", ""),
        (")", ""),
    ]:
        safe = safe.replace(old, new)
    return safe[:max_len]


def make_suffix(config: WalkForwardConfig) -> str:
    lookbacks = "_".join(f"{x:g}".replace(".", "p") for x in config.lookbacks)
    horizons = "_".join(str(x) for x in config.horizons)
    return (
        f"{config.timeframe}_test_{config.first_test_year}_to_{config.last_test_year or 'auto'}_"
        f"lb_{lookbacks}_h_{horizons}_top{config.top_n}_"
        f"pos_{str(config.position_pct).replace('.', 'p')}x"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fold construction
# ─────────────────────────────────────────────────────────────────────────────

def build_yearly_folds(candles: pd.DataFrame, config: WalkForwardConfig) -> list[FoldDefinition]:
    candles = candles.copy().sort_values("time")
    min_time = candles["time"].min()
    max_time = candles["time"].max()

    last_year = config.last_test_year if config.last_test_year is not None else int(max_time.year)
    max_horizon_hours = max(config.horizons) / 60.0
    embargo_hours = config.embargo_hours if config.embargo_hours is not None else max_horizon_hours

    folds: list[FoldDefinition] = []

    for year in range(config.first_test_year, last_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

        if test_start > max_time:
            continue

        actual_test_end = min(test_end, max_time)

        test_candles = candles[(candles["time"] >= test_start) & (candles["time"] < actual_test_end)]
        if test_candles.empty:
            continue

        test_days = (test_candles["time"].max() - test_candles["time"].min()).total_seconds() / 86400.0
        if test_days < config.min_test_days:
            print(f"Skipping {year}: only {test_days:.1f} test days available")
            continue

        train_end = test_start - pd.Timedelta(hours=embargo_hours)
        train_candles = candles[candles["time"] < train_end]

        train_days = (train_end - min_time).total_seconds() / 86400.0
        if train_days < config.min_train_days:
            print(f"Skipping {year}: only {train_days:.1f} train days available")
            continue

        folds.append(
            FoldDefinition(
                fold_id=f"test_{year}",
                train_start=min_time,
                train_end=train_end,
                test_start=test_start,
                test_end=actual_test_end,
                test_last_time=test_candles["time"].max(),
            )
        )

    if not folds:
        raise RuntimeError("No valid walk-forward folds were created.")

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Selection stage
# ─────────────────────────────────────────────────────────────────────────────

def make_rediscovery_config(config: WalkForwardConfig, output_dir: Path) -> RediscoveryConfig:
    return RediscoveryConfig(
        input_dir=config.input_dir,
        output_dir=output_dir,
        timeframe=config.timeframe,
        holdout_days=365,  # unused here because folds are manually controlled
        train_frac_of_development=config.train_frac_of_development,
        lookbacks=config.lookbacks,
        horizons=config.horizons,
        filters=config.filters,
        threshold_pct=config.threshold_pct,
        percentile_window=config.percentile_window,
        bucket_size=config.bucket_size,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        min_samples_train=config.min_samples_train,
        min_samples_validation=config.min_samples_validation,
        min_validation_10d_windows=config.min_validation_10d_windows,
        top_n=config.top_n,
        products=config.products,
    )


def add_robust_walkforward_score(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """
    Re-rank candidates for walk-forward selection.

    The first version ranked mostly by validation strength. That produced useful
    candidates, but the out-of-sample results showed Rank 2/3 were more robust
    than Rank 1. This score is deliberately more conservative.

    It rewards:
        - positive validation edge
        - enough samples
        - positive 10-day window consistency
        - profit factor above 1, capped to avoid rewarding tiny-sample monsters
        - train/validation stability

    It penalizes:
        - bad validation trade-path drawdown
        - ugly worst 10-day validation windows
        - large train/validation gaps
        - extremely narrow/fragile validation-only behavior
    """
    if leaderboard.empty:
        return leaderboard.copy()

    df = leaderboard.copy()

    required = [
        "validation_avg_net_return_pct",
        "validation_profit_factor",
        "validation_samples",
        "validation_positive_10d_window_rate_pct",
        "validation_trade_path_max_drawdown_pct",
        "validation_worst_10d_units_return_pct",
        "train_avg_net_return_pct",
        "train_profit_factor",
        "train_samples",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Do not allow absurd validation PF to dominate the ranking. A PF of 12 from
    # a small sample is usually selection noise, not a reason to worship the rule.
    pf_capped = df["validation_profit_factor"].clip(lower=1.0, upper=4.0)
    train_pf_capped = df["train_profit_factor"].clip(lower=1.0, upper=4.0)

    return_component = df["validation_avg_net_return_pct"].clip(lower=0.0)
    pf_component = (pf_capped - 1.0).clip(lower=0.0)

    # Sample component grows slowly so 2,000 samples is not 10x better than 200.
    sample_component = np.log1p(df["validation_samples"].clip(lower=0.0))

    window_component = (df["validation_positive_10d_window_rate_pct"].clip(lower=0.0, upper=100.0) / 100.0) ** 1.5

    # Penalize bad 10-day downside and bad trade-sequence drawdown.
    worst_10d_penalty = 1.0 / (1.0 + df["validation_worst_10d_units_return_pct"].clip(upper=0.0).abs() / 10.0)
    dd_penalty = 1.0 / (1.0 + df["validation_trade_path_max_drawdown_pct"].clip(upper=0.0).abs() / 20.0)

    # Train/validation stability. If validation looks wildly better than train,
    # assume we are probably overfitting validation noise.
    avg_gap = (df["validation_avg_net_return_pct"] - df["train_avg_net_return_pct"]).abs()
    avg_scale = df["train_avg_net_return_pct"].abs() + 0.25
    avg_stability = 1.0 / (1.0 + avg_gap / avg_scale)

    pf_gap = (pf_capped - train_pf_capped).abs()
    pf_stability = 1.0 / (1.0 + pf_gap)

    train_sample_penalty = np.minimum(1.0, df["train_samples"].clip(lower=0.0) / 1000.0)
    validation_sample_penalty = np.minimum(1.0, df["validation_samples"].clip(lower=0.0) / 250.0)

    # Mild preference for simpler/broader regimes. Do not overdo this, because
    # some decile filters genuinely helped, but broad/no-vol rules are less fragile.
    vol_regime = df.get("vol_regime", pd.Series("", index=df.index)).astype(str)
    regime_simplicity = np.where(
        vol_regime.eq("vol_none"),
        1.08,
        np.where(vol_regime.isin(["vol_low_0_30", "vol_mid_30_70", "vol_high_70_100"]), 1.04, 0.96),
    )

    df["robust_walkforward_score"] = (
        return_component
        * pf_component
        * sample_component
        * window_component
        * worst_10d_penalty
        * dd_penalty
        * avg_stability
        * pf_stability
        * train_sample_penalty
        * validation_sample_penalty
        * regime_simplicity
    )

    df["robust_score_return_component"] = return_component
    df["robust_score_pf_component"] = pf_component
    df["robust_score_sample_component"] = sample_component
    df["robust_score_window_component"] = window_component
    df["robust_score_worst_10d_penalty"] = worst_10d_penalty
    df["robust_score_dd_penalty"] = dd_penalty
    df["robust_score_avg_stability"] = avg_stability
    df["robust_score_pf_stability"] = pf_stability
    df["robust_score_regime_simplicity"] = regime_simplicity

    return df.sort_values(
        [
            "robust_walkforward_score",
            "validation_avg_net_return_pct",
            "validation_profit_factor",
            "validation_positive_10d_window_rate_pct",
            "validation_samples",
        ],
        ascending=False,
    ).reset_index(drop=True)


def build_robust_selected_candidates(leaderboard: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Select a diverse set of robust candidates.

    Without diversity, the top-N list often becomes five near-duplicates of the
    same rule. That looks better in-sample and worse out-of-sample. We keep one
    candidate per broad rule key first, then fill remaining slots if needed.
    """
    if leaderboard.empty:
        return leaderboard.copy()

    selected_rows = []
    seen_keys = set()

    for _, row in leaderboard.iterrows():
        key = (
            row.get("side"),
            row.get("lookback_hours"),
            row.get("horizon_minutes"),
            row.get("momentum_bucket"),
            row.get("filter_name"),
        )

        # Notice vol_regime is intentionally NOT in this key. If the same rule
        # works with no-vol/broad-vol/decile variants, we prefer the highest
        # robust score among them rather than selecting all variants.
        if key in seen_keys:
            continue

        selected_rows.append(row)
        seen_keys.add(key)

        if len(selected_rows) >= top_n:
            break

    selected = pd.DataFrame(selected_rows)

    if len(selected) < top_n:
        already = set(selected.get("strategy_id", pd.Series(dtype=str)).astype(str))
        fill = leaderboard[~leaderboard["strategy_id"].astype(str).isin(already)].head(top_n - len(selected))
        selected = pd.concat([selected, fill], ignore_index=True)

    return selected.reset_index(drop=True)


def select_candidates_for_fold(
    candles: pd.DataFrame,
    fold: FoldDefinition,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run grid selection using only pre-test data for this fold.
    """
    development_candles = candles[(candles["time"] >= fold.train_start) & (candles["time"] < fold.train_end)].copy()

    if development_candles.empty:
        raise RuntimeError(f"No development candles for fold {fold.fold_id}")

    rediscovery_config = make_rediscovery_config(config, config.output_dir / "_internal_rediscovery")

    result_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []

    for lookback_hours in config.lookbacks:
        for horizon_minutes in config.horizons:
            result, split_summary = evaluate_combo(
                development_candles=development_candles,
                config=rediscovery_config,
                lookback_hours=lookback_hours,
                horizon_minutes=horizon_minutes,
                holdout_start=fold.test_start,
                holdout_end=fold.test_end,
            )
            result["fold_id"] = fold.fold_id
            split_summary["fold_id"] = fold.fold_id
            result_frames.append(result)
            split_frames.append(split_summary)

    all_results = pd.concat(result_frames, ignore_index=True)
    split_summary = pd.concat(split_frames, ignore_index=True)

    leaderboard = build_leaderboard(all_results.drop(columns=["fold_id"], errors="ignore"), rediscovery_config)
    leaderboard = add_robust_walkforward_score(leaderboard)
    leaderboard["fold_id"] = fold.fold_id

    selected = build_robust_selected_candidates(
        leaderboard.drop(columns=["fold_id"], errors="ignore"),
        top_n=config.top_n,
    )
    selected["fold_id"] = fold.fold_id
    selected["selected_rank"] = np.arange(1, len(selected) + 1)

    for col, value in [
        ("train_start", fold.train_start),
        ("train_end", fold.train_end),
        ("test_start", fold.test_start),
        ("test_end", fold.test_end),
    ]:
        selected[col] = value
        leaderboard[col] = value

    return selected, leaderboard, split_summary


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio testing stage
# ─────────────────────────────────────────────────────────────────────────────

def strategy_from_selected_row(row: pd.Series, fold_id: str) -> StrategyDefinition:
    original_id = str(row["strategy_id"])
    rank = int(row["selected_rank"])
    strategy_id = f"{fold_id}_rank{rank}_{make_safe_id(original_id)}"

    return StrategyDefinition(
        strategy_id=strategy_id,
        name=f"{fold_id} Rank {rank}: {original_id}",
        category="walkforward_selected_candidate",
        side=str(row["side"]),  # type: ignore[arg-type]
        lookback_hours=float(row["lookback_hours"]),
        horizon_minutes=int(row["horizon_minutes"]),
        momentum_lower=safe_float_or_none(row.get("momentum_lower")),
        momentum_upper=safe_float_or_none(row.get("momentum_upper")),
        vol_lower=safe_float_or_none(row.get("vol_lower")),
        vol_upper=safe_float_or_none(row.get("vol_upper")),
        filter_name=str(row["filter_name"]),  # type: ignore[arg-type]
        description=(
            f"Walk-forward selected candidate. Original rule id: {original_id}. "
            f"Selected using only data before {fold_id}."
        ),
    )


def make_benchmark_strategy(fold_id: str) -> StrategyDefinition:
    return StrategyDefinition(
        strategy_id=f"{fold_id}_benchmark_buy_hold_equal_weight",
        name=f"{fold_id} Buy & Hold Equal Weight",
        category="walkforward_benchmark",
        side="long",
        lookback_hours=None,
        horizon_minutes=None,
        momentum_lower=None,
        momentum_upper=None,
        vol_lower=None,
        vol_upper=None,
        filter_name="none",
        description="Equal-weight buy-and-hold benchmark for this walk-forward fold.",
    )


def make_final_backtest_config(config: WalkForwardConfig) -> FinalBacktestConfig:
    return FinalBacktestConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        timeframe=config.timeframe,
        holdout_days=365,  # unused for manual folds
        initial_equity=config.initial_equity,
        position_pct=config.position_pct,
        max_gross_leverage=config.max_gross_leverage,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        percentile_window=config.percentile_window,
        bucket_size=config.bucket_size,
        threshold_pct=config.threshold_pct,
    )


def test_selected_candidates_for_fold(
    candles: pd.DataFrame,
    fold: FoldDefinition,
    selected: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test frozen selected candidates on the fold test period using next-bar execution.
    """
    bt_config = make_final_backtest_config(config)

    # Include all history up to test end so rolling features have enough prior context.
    feature_candles = candles[(candles["time"] < fold.test_end)].copy()
    test_candles = candles[(candles["time"] >= fold.test_start) & (candles["time"] < fold.test_end)].copy()

    if test_candles.empty:
        raise RuntimeError(f"No test candles for {fold.fold_id}")

    strategies = [strategy_from_selected_row(row, fold.fold_id) for _, row in selected.iterrows()]

    feature_cache: dict[tuple[float, int], pd.DataFrame] = {}
    for strategy in strategies:
        if strategy.lookback_hours is None or strategy.horizon_minutes is None:
            continue
        key = (strategy.lookback_hours, strategy.horizon_minutes)
        if key not in feature_cache:
            print(f"{fold.fold_id}: building test features lookback={key[0]}h horizon={key[1]}m")
            feature_cache[key] = build_feature_dataset(
                candles=feature_candles,
                config=bt_config,
                lookback_hours=key[0],
                horizon_minutes=key[1],
            )

    metric_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    asset_frames: list[pd.DataFrame] = []

    for selected_index, strategy in enumerate(strategies, start=1):
        print(f"{fold.fold_id}: testing rank {selected_index} | {strategy.strategy_id}")

        feature_dataset = feature_cache[(strategy.lookback_hours, strategy.horizon_minutes)]
        signals = build_signals_for_strategy(
            feature_dataset=feature_dataset,
            strategy=strategy,
            holdout_start=fold.test_start,
            holdout_end=fold.test_last_time,
        )

        trades, equity, asset_contrib = run_rule_strategy_backtest(
            strategy=strategy,
            signals_df=signals,
            holdout_candles=test_candles,
            holdout_start=fold.test_start,
            holdout_end=fold.test_last_time,
            config=bt_config,
        )

        metrics = build_strategy_metrics(
            strategy=strategy,
            equity_df=equity,
            trades_df=trades,
            asset_contrib_df=asset_contrib,
            holdout_start=fold.test_start,
            holdout_end=fold.test_last_time,
            config=bt_config,
        )
        metrics["fold_id"] = fold.fold_id
        metrics["selected_rank"] = selected_index
        metrics["selection_original_strategy_id"] = selected.iloc[selected_index - 1]["strategy_id"]
        metrics["selection_validation_score"] = selected.iloc[selected_index - 1].get("final_discovery_score", np.nan)
        metrics["test_start"] = fold.test_start
        metrics["test_end"] = fold.test_last_time

        for frame in [trades, equity, asset_contrib]:
            if not frame.empty:
                frame["fold_id"] = fold.fold_id
                frame["selected_rank"] = selected_index
                frame["selection_original_strategy_id"] = metrics["selection_original_strategy_id"]

        metric_rows.append(metrics)
        if not trades.empty:
            trade_frames.append(trades)
        if not equity.empty:
            equity_frames.append(equity)
        if not asset_contrib.empty:
            asset_frames.append(asset_contrib)

    if config.include_benchmark:
        benchmark = make_benchmark_strategy(fold.fold_id)
        trades, equity, asset_contrib = run_buy_and_hold_benchmark(
            strategy=benchmark,
            holdout_candles=test_candles,
            config=bt_config,
        )
        metrics = build_strategy_metrics(
            strategy=benchmark,
            equity_df=equity,
            trades_df=trades,
            asset_contrib_df=asset_contrib,
            holdout_start=fold.test_start,
            holdout_end=fold.test_last_time,
            config=bt_config,
        )
        metrics["fold_id"] = fold.fold_id
        metrics["selected_rank"] = 0
        metrics["selection_original_strategy_id"] = "benchmark_buy_hold_equal_weight"
        metrics["selection_validation_score"] = np.nan
        metrics["test_start"] = fold.test_start
        metrics["test_end"] = fold.test_last_time

        for frame in [trades, equity, asset_contrib]:
            if not frame.empty:
                frame["fold_id"] = fold.fold_id
                frame["selected_rank"] = 0
                frame["selection_original_strategy_id"] = "benchmark_buy_hold_equal_weight"

        metric_rows.append(metrics)
        if not trades.empty:
            trade_frames.append(trades)
        if not equity.empty:
            equity_frames.append(equity)
        if not asset_contrib.empty:
            asset_frames.append(asset_contrib)

    fold_metrics = pd.DataFrame(metric_rows)
    fold_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    fold_equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    fold_asset = pd.concat(asset_frames, ignore_index=True) if asset_frames else pd.DataFrame()

    return fold_metrics, fold_trades, fold_equity, fold_asset


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summaries
# ─────────────────────────────────────────────────────────────────────────────

def build_rank_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    rows = []
    for rank, group in metrics.groupby("selected_rank"):
        returns = group["total_return_pct"].astype(float) / 100.0
        compounded_return_pct = (float((1.0 + returns).prod()) - 1.0) * 100.0

        rows.append(
            {
                "selected_rank": rank,
                "label": "benchmark" if rank == 0 else f"rank_{int(rank)}",
                "num_folds": int(group["fold_id"].nunique()),
                "compounded_return_pct": compounded_return_pct,
                "avg_fold_return_pct": float(group["total_return_pct"].mean()),
                "median_fold_return_pct": float(group["total_return_pct"].median()),
                "positive_fold_rate_pct": float((group["total_return_pct"] > 0).mean() * 100.0),
                "avg_sharpe_ratio": float(group["sharpe_ratio"].mean()),
                "median_sharpe_ratio": float(group["sharpe_ratio"].median()),
                "avg_max_drawdown_pct": float(group["max_drawdown_pct"].mean()),
                "worst_max_drawdown_pct": float(group["max_drawdown_pct"].min()),
                "avg_profit_factor": float(group["profit_factor"].replace([np.inf, -np.inf], np.nan).mean()),
                "total_trades": int(group["num_trades"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("selected_rank").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main run and outputs
# ─────────────────────────────────────────────────────────────────────────────

def run_walkforward(candles: pd.DataFrame, config: WalkForwardConfig) -> dict[str, pd.DataFrame]:
    candles = candles.copy()
    candles["time"] = pd.to_datetime(candles["time"], utc=True)
    candles = candles[candles["product_id"].isin(config.products)].copy()
    candles = candles.sort_values(["product_id", "time"]).reset_index(drop=True)

    folds = build_yearly_folds(candles, config)
    print("Walk-forward folds:")
    for fold in folds:
        print(
            f"{fold.fold_id}: train {fold.train_start} -> {fold.train_end}; "
            f"test {fold.test_start} -> {fold.test_last_time}"
        )

    selected_frames: list[pd.DataFrame] = []
    leaderboard_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    asset_frames: list[pd.DataFrame] = []

    for fold in folds:
        print("\n" + "=" * 100)
        print(f"Starting fold {fold.fold_id}")

        selected, leaderboard, split_summary = select_candidates_for_fold(candles, fold, config)

        if selected.empty:
            print(f"WARNING: no selected candidates for {fold.fold_id}")
            continue

        selected_frames.append(selected)
        leaderboard_frames.append(leaderboard)
        split_frames.append(split_summary)

        fold_metrics, fold_trades, fold_equity, fold_asset = test_selected_candidates_for_fold(
            candles=candles,
            fold=fold,
            selected=selected.head(config.top_n),
            config=config,
        )

        metric_frames.append(fold_metrics)
        if not fold_trades.empty:
            trade_frames.append(fold_trades)
        if not fold_equity.empty:
            equity_frames.append(fold_equity)
        if not fold_asset.empty:
            asset_frames.append(fold_asset)

    selected_candidates = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    leaderboards = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
    split_summary = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    asset = pd.concat(asset_frames, ignore_index=True) if asset_frames else pd.DataFrame()
    rank_summary = build_rank_summary(metrics)

    return {
        "selected_candidates": selected_candidates,
        "leaderboards": leaderboards,
        "split_summary": split_summary,
        "metrics": metrics,
        "trades": trades,
        "equity": equity,
        "asset_contributions": asset,
        "rank_summary": rank_summary,
    }


def save_outputs(results: dict[str, pd.DataFrame], config: WalkForwardConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = make_suffix(config)

    paths = {
        "selected_candidates": config.output_dir / f"walkforward_selected_candidates_{suffix}.csv",
        "leaderboards": config.output_dir / f"walkforward_leaderboards_{suffix}.csv",
        "split_summary": config.output_dir / f"walkforward_split_summary_{suffix}.csv",
        "metrics": config.output_dir / f"walkforward_fold_metrics_{suffix}.csv",
        "trades": config.output_dir / f"walkforward_trades_{suffix}.csv",
        "equity": config.output_dir / f"walkforward_equity_{suffix}.csv",
        "asset_contributions": config.output_dir / f"walkforward_asset_contributions_{suffix}.csv",
        "rank_summary": config.output_dir / f"walkforward_rank_summary_{suffix}.csv",
    }

    for key, path in paths.items():
        results[key].to_csv(path, index=False)

    config_path = config.output_dir / f"walkforward_config_{suffix}.txt"
    with open(config_path, "w", encoding="utf-8") as f:
        for key, value in asdict(config).items():
            f.write(f"{key}: {value}\n")
    paths["config"] = config_path

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward strategy research and testing.")

    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_5y_15m"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/walkforward_strategy_research_full"))
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)

    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--last-test-year", type=int, default=None)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--min-test-days", type=int, default=90)
    parser.add_argument("--train-frac-of-development", type=float, default=0.75)
    parser.add_argument("--embargo-hours", type=float, default=None)

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

    parser.add_argument("--fee-rate", type=float, default=0.0001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--initial-equity", type=float, default=100000.0)
    parser.add_argument("--position-pct", type=float, default=0.20)
    parser.add_argument("--max-gross-leverage", type=float, default=1.0)

    parser.add_argument("--min-samples-train", type=int, default=500)
    parser.add_argument("--min-samples-validation", type=int, default=150)
    parser.add_argument("--min-validation-10d-windows", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--no-benchmark", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = WalkForwardConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        products=args.products,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_train_days=args.min_train_days,
        min_test_days=args.min_test_days,
        train_frac_of_development=args.train_frac_of_development,
        lookbacks=args.lookbacks,
        horizons=args.horizons,
        filters=args.filters,
        threshold_pct=args.threshold_pct,
        percentile_window=args.percentile_window,
        bucket_size=args.bucket_size,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        initial_equity=args.initial_equity,
        position_pct=args.position_pct,
        max_gross_leverage=args.max_gross_leverage,
        min_samples_train=args.min_samples_train,
        min_samples_validation=args.min_samples_validation,
        min_validation_10d_windows=args.min_validation_10d_windows,
        top_n=args.top_n,
        include_benchmark=not args.no_benchmark,
        embargo_hours=args.embargo_hours,
    )

    candles = load_raw_candles(config.input_dir, config.timeframe)
    results = run_walkforward(candles, config)
    paths = save_outputs(results, config)

    print("\nWalk-forward rank summary:")
    rank_summary = results["rank_summary"]
    if rank_summary.empty:
        print("No rank summary generated.")
    else:
        print(rank_summary.to_string(index=False))

    print("\nFold metrics:")
    metrics = results["metrics"]
    if metrics.empty:
        print("No fold metrics generated.")
    else:
        display_cols = [
            "fold_id",
            "selected_rank",
            "selection_original_strategy_id",
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "profit_factor",
            "num_trades",
            "win_rate_pct",
        ]
        display_cols = [c for c in display_cols if c in metrics.columns]
        print(metrics[display_cols].to_string(index=False))

    print("\nSaved walk-forward outputs:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
