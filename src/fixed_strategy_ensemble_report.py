"""
Fixed strategy ensemble report.

Purpose:
    Build ensemble portfolios from the fixed, non-V3 strategy set.

    This script combines already-backtested strategy equity curves from:
        data/final_strategy_backtests_all_candidates_nextbar

    It intentionally excludes:
        - Competition V3 Strategy / A_v3_baseline
        - Buy & Hold benchmark
        - Ablation baselines by default

    Default included categories:
        - systematic_grid_candidate
        - rediscovered_clean_candidate

    Default ensembles:
        - Clean Equal-Weight All
        - Clean Long-Only Equal-Weight
        - Systematic Grid Equal-Weight
        - Rediscovered Equal-Weight
        - Momentum Continuation Ensemble
        - Selloff Rebound Ensemble

    These are portfolio-of-strategy ensembles built from standalone strategy
    equity curves. This is not a shared order-level execution simulator.

Run:

    py -m src.fixed_strategy_ensemble_report ^
      --backtest-dir data/final_strategy_backtests_all_candidates_nextbar ^
      --output-dir reports/fixed_strategy_ensemble_clean ^
      --initial-equity 100000

Outputs:
    reports/fixed_strategy_ensemble_clean/
        fixed_strategy_ensemble_metrics.csv
        fixed_strategy_ensemble_equity.csv
        fixed_strategy_ensemble_weights.csv
        fixed_strategy_ensemble_trade_summary.csv
        fixed_strategy_ensemble_equity.html
        fixed_strategy_ensemble_drawdown.html
        fixed_strategy_ensemble_return_drawdown.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Report-friendly names. Keep code IDs untouched.
STRATEGY_DISPLAY_NAMES = {
    "B_extreme_selloff_rebound_grid": "Extreme Selloff Rebound",
    "C_upside_continuation_below_trend": "Below-Trend Momentum Continuation",
    "D_low_vol_overextension_short_grid": "Low-Volatility Blowoff Short",
    "R1_shortterm_extreme_selloff_rebound": "BTC-Regime Selloff Rebound",
    "R2_highvol_upside_continuation_btc_below": "High-Volatility Momentum Burst",
    "R4_highvol_selloff_rebound_volume": "Volume-Confirmed Panic Rebound",
    "E_momentum_only_baseline_grid": "Momentum-Only Selloff Baseline",
    "R1_baseline_selloff_no_vol_filter": "Selloff Rebound Without Vol Filter",
    "R3_upside_continuation_no_vol_btc_below": "Momentum Burst Without Vol Filter",
    "F_buy_and_hold_equal_weight": "Equal-Weight Buy & Hold",
    "A_v3_baseline": "Competition V3 Strategy",
}


@dataclass(frozen=True)
class FixedEnsembleSpec:
    ensemble_id: str
    display_name: str
    strategy_ids: tuple[str, ...]
    description: str


DEFAULT_ENSEMBLES = [
    FixedEnsembleSpec(
        ensemble_id="clean_all_equal_weight",
        display_name="Clean Strategies Equal-Weight",
        strategy_ids=(
            "B_extreme_selloff_rebound_grid",
            "C_upside_continuation_below_trend",
            "D_low_vol_overextension_short_grid",
            "R1_shortterm_extreme_selloff_rebound",
            "R2_highvol_upside_continuation_btc_below",
            "R4_highvol_selloff_rebound_volume",
        ),
        description="Equal-weight ensemble of all non-V3 clean/systematic strategies.",
    ),
    FixedEnsembleSpec(
        ensemble_id="clean_long_only_equal_weight",
        display_name="Clean Long-Only Equal-Weight",
        strategy_ids=(
            "B_extreme_selloff_rebound_grid",
            "C_upside_continuation_below_trend",
            "R1_shortterm_extreme_selloff_rebound",
            "R2_highvol_upside_continuation_btc_below",
            "R4_highvol_selloff_rebound_volume",
        ),
        description="Equal-weight ensemble of clean long-side strategies only. Excludes V3 and the short-side D strategy.",
    ),
    FixedEnsembleSpec(
        ensemble_id="systematic_grid_equal_weight",
        display_name="Systematic Grid Equal-Weight",
        strategy_ids=(
            "B_extreme_selloff_rebound_grid",
            "C_upside_continuation_below_trend",
            "D_low_vol_overextension_short_grid",
        ),
        description="Equal-weight ensemble of systematic grid-derived candidates.",
    ),
    FixedEnsembleSpec(
        ensemble_id="rediscovered_equal_weight",
        display_name="Rediscovered Clean Equal-Weight",
        strategy_ids=(
            "R1_shortterm_extreme_selloff_rebound",
            "R2_highvol_upside_continuation_btc_below",
            "R4_highvol_selloff_rebound_volume",
        ),
        description="Equal-weight ensemble of development-only rediscovered candidates.",
    ),
    FixedEnsembleSpec(
        ensemble_id="momentum_continuation_equal_weight",
        display_name="Momentum Continuation Ensemble",
        strategy_ids=(
            "C_upside_continuation_below_trend",
            "R2_highvol_upside_continuation_btc_below",
        ),
        description="Equal-weight ensemble of upside momentum continuation strategies.",
    ),
    FixedEnsembleSpec(
        ensemble_id="selloff_rebound_equal_weight",
        display_name="Selloff Rebound Ensemble",
        strategy_ids=(
            "B_extreme_selloff_rebound_grid",
            "R1_shortterm_extreme_selloff_rebound",
            "R4_highvol_selloff_rebound_volume",
        ),
        description="Equal-weight ensemble of selloff/panic rebound strategies.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found in {folder} matching {pattern}")
    return files[-1]


def load_backtest_outputs(backtest_dir: Path) -> dict[str, pd.DataFrame]:
    equity_path = latest_file(backtest_dir, "final_strategy_equity_*.csv")
    trades_path = latest_file(backtest_dir, "final_strategy_trades_*.csv")
    metrics_path = latest_file(backtest_dir, "final_strategy_metrics_*.csv")
    definitions_path = latest_file(backtest_dir, "final_strategy_definitions_*.csv")

    equity = pd.read_csv(equity_path, parse_dates=["time"])
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    metrics = pd.read_csv(metrics_path)
    definitions = pd.read_csv(definitions_path)

    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "definitions": definitions,
        "paths": pd.DataFrame(
            [
                {"name": "equity", "path": str(equity_path)},
                {"name": "trades", "path": str(trades_path)},
                {"name": "metrics", "path": str(metrics_path)},
                {"name": "definitions", "path": str(definitions_path)},
            ]
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min() * 100.0)


def profit_factor(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0:
        return float("inf")
    return 0.0


def annualized_metrics(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    eq = equity.sort_values("time").copy()
    start_equity = float(eq["equity"].iloc[0])
    end_equity = float(eq["equity"].iloc[-1])

    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else 0.0

    start_time = pd.Timestamp(eq["time"].iloc[0])
    end_time = pd.Timestamp(eq["time"].iloc[-1])
    years = max((end_time - start_time).total_seconds() / (365.25 * 86400), 1 / 365.25)

    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0

    daily = eq.set_index("time")["equity"].resample("1D").last().dropna()
    daily_returns = daily.pct_change().dropna()

    if daily_returns.empty or float(daily_returns.std()) == 0.0:
        annualized_vol = 0.0
        sharpe = 0.0
        sortino = 0.0
    else:
        annualized_vol = float(daily_returns.std() * np.sqrt(365.25))
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365.25))
        downside = daily_returns[daily_returns < 0]
        if downside.empty or float(downside.std()) == 0.0:
            sortino = 0.0
        else:
            sortino = float(daily_returns.mean() / downside.std() * np.sqrt(365.25))

    mdd = max_drawdown_pct(eq["equity"])
    calmar = float((annualized_return * 100.0) / abs(mdd)) if mdd < 0 else 0.0

    return {
        "total_return_pct": float(total_return * 100.0),
        "annualized_return_pct": float(annualized_return * 100.0),
        "annualized_volatility_pct": float(annualized_vol * 100.0),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": mdd,
        "calmar_ratio": calmar,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble construction
# ─────────────────────────────────────────────────────────────────────────────

def available_strategy_ids(equity: pd.DataFrame) -> set[str]:
    return set(equity["strategy_id"].dropna().astype(str).unique())


def validate_ensemble_specs(specs: list[FixedEnsembleSpec], equity: pd.DataFrame) -> list[FixedEnsembleSpec]:
    available = available_strategy_ids(equity)
    valid_specs = []

    for spec in specs:
        missing = [sid for sid in spec.strategy_ids if sid not in available]
        if missing:
            print(f"WARNING: skipping {spec.ensemble_id}; missing strategies: {missing}")
            continue
        valid_specs.append(spec)

    if not valid_specs:
        raise RuntimeError("No valid ensemble specs. Check strategy IDs and backtest output folder.")

    return valid_specs


def build_equal_weights(spec: FixedEnsembleSpec) -> pd.DataFrame:
    weight = 1.0 / len(spec.strategy_ids)
    return pd.DataFrame(
        [
            {
                "ensemble_id": spec.ensemble_id,
                "ensemble_name": spec.display_name,
                "strategy_id": sid,
                "strategy_name": STRATEGY_DISPLAY_NAMES.get(sid, sid),
                "weight": weight,
                "description": spec.description,
            }
            for sid in spec.strategy_ids
        ]
    )


def build_ensemble_equity(
    equity: pd.DataFrame,
    weights: pd.DataFrame,
    spec: FixedEnsembleSpec,
    initial_equity: float,
) -> pd.DataFrame:
    strategy_ids = weights["strategy_id"].tolist()
    data = equity[equity["strategy_id"].isin(strategy_ids)].copy()

    if data.empty:
        return pd.DataFrame()

    pivot = data.pivot_table(index="time", columns="strategy_id", values="equity", aggfunc="last").sort_index()
    pivot = pivot.ffill().dropna(how="all")

    normalized = pivot.copy()
    for sid in normalized.columns:
        first_valid = normalized[sid].dropna()
        if first_valid.empty:
            normalized[sid] = np.nan
        else:
            normalized[sid] = normalized[sid] / float(first_valid.iloc[0])

    normalized = normalized.ffill().fillna(1.0)

    weight_map = weights.set_index("strategy_id")["weight"].to_dict()
    ensemble_norm = pd.Series(0.0, index=normalized.index)

    for sid, weight in weight_map.items():
        if sid in normalized.columns:
            ensemble_norm += float(weight) * normalized[sid]

    out = pd.DataFrame(
        {
            "time": ensemble_norm.index,
            "ensemble_id": spec.ensemble_id,
            "display_name": spec.display_name,
            "equity": initial_equity * ensemble_norm.values,
        }
    )
    out["running_max"] = out["equity"].cummax()
    out["drawdown_pct"] = (out["equity"] / out["running_max"] - 1.0) * 100.0
    return out


def build_scaled_trade_summary(
    trades: pd.DataFrame,
    weights: pd.DataFrame,
    spec: FixedEnsembleSpec,
) -> dict[str, float | int | str]:
    if trades.empty:
        return {
            "ensemble_id": spec.ensemble_id,
            "display_name": spec.display_name,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
        }

    weight_map = weights.set_index("strategy_id")["weight"].to_dict()
    data = trades[trades["strategy_id"].isin(weight_map)].copy()

    if data.empty:
        return {
            "ensemble_id": spec.ensemble_id,
            "display_name": spec.display_name,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
        }

    data["weight"] = data["strategy_id"].map(weight_map).fillna(0.0)
    data["scaled_net_pnl"] = data["net_pnl"] * data["weight"]

    if "total_fees" in data.columns:
        data["scaled_fees"] = data["total_fees"] * data["weight"]
    else:
        data["scaled_fees"] = 0.0

    return {
        "ensemble_id": spec.ensemble_id,
        "display_name": spec.display_name,
        "num_trades": int(len(data)),
        "win_rate_pct": float((data["scaled_net_pnl"] > 0).mean() * 100.0),
        "profit_factor": profit_factor(data["scaled_net_pnl"]),
        "net_pnl": float(data["scaled_net_pnl"].sum()),
        "fees_paid": float(data["scaled_fees"].sum()),
    }


def build_ensemble_metrics(
    ensemble_equity: pd.DataFrame,
    trade_summary: dict[str, float | int | str],
    spec: FixedEnsembleSpec,
) -> dict[str, float | int | str]:
    metrics = annualized_metrics(ensemble_equity[["time", "equity"]])
    return {
        "ensemble_id": spec.ensemble_id,
        "display_name": spec.display_name,
        "description": spec.description,
        **metrics,
        "num_trades": int(trade_summary.get("num_trades", 0)),
        "win_rate_pct": float(trade_summary.get("win_rate_pct", 0.0)),
        "profit_factor": float(trade_summary.get("profit_factor", 0.0)),
        "fees_paid": float(trade_summary.get("fees_paid", 0.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(equity: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    if equity.empty:
        return

    fig = px.line(
        equity,
        x="time",
        y="equity",
        color="display_name",
        title="Fixed Clean Strategy Ensembles: Equity Curves",
        height=650,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Equity",
        legend_title="Ensemble",
        hovermode="x unified",
    )
    fig.write_html(output_dir / "fixed_strategy_ensemble_equity.html")

    fig = px.line(
        equity,
        x="time",
        y="drawdown_pct",
        color="display_name",
        title="Fixed Clean Strategy Ensembles: Drawdowns",
        height=650,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend_title="Ensemble",
        hovermode="x unified",
    )
    fig.write_html(output_dir / "fixed_strategy_ensemble_drawdown.html")

    fig = px.scatter(
        metrics,
        x="max_drawdown_pct",
        y="total_return_pct",
        color="display_name",
        size=metrics["sharpe_ratio"].clip(lower=0.1),
        hover_data=["sharpe_ratio", "sortino_ratio", "calmar_ratio", "num_trades", "profit_factor"],
        title="Fixed Clean Strategy Ensembles: Return vs Drawdown",
        height=650,
    )
    fig.update_layout(
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Total Return (%)",
        legend_title="Ensemble",
    )
    fig.write_html(output_dir / "fixed_strategy_ensemble_return_drawdown.html")

    bar_data = metrics.sort_values("total_return_pct", ascending=False).copy()
    fig = px.bar(
        bar_data,
        x="display_name",
        y="total_return_pct",
        color="sharpe_ratio",
        text=bar_data["total_return_pct"].round(1),
        color_continuous_scale="Viridis",
        title="Fixed Clean Strategy Ensembles: Total Return",
        height=600,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Ensemble",
        yaxis_title="Total Return (%)",
        showlegend=False,
    )
    fig.write_html(output_dir / "fixed_strategy_ensemble_total_return.html")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_strategy_ensembles(
    backtest_dir: Path,
    output_dir: Path,
    initial_equity: float,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_backtest_outputs(backtest_dir)
    equity = outputs["equity"]
    trades = outputs["trades"]

    specs = validate_ensemble_specs(DEFAULT_ENSEMBLES, equity)

    all_equity = []
    all_weights = []
    metric_rows = []
    trade_summary_rows = []

    for spec in specs:
        print(f"Building ensemble: {spec.display_name}")
        weights = build_equal_weights(spec)
        ensemble_equity = build_ensemble_equity(
            equity=equity,
            weights=weights,
            spec=spec,
            initial_equity=initial_equity,
        )

        if ensemble_equity.empty:
            print(f"WARNING: empty equity for {spec.ensemble_id}")
            continue

        trade_summary = build_scaled_trade_summary(trades, weights, spec)
        metrics = build_ensemble_metrics(ensemble_equity, trade_summary, spec)

        all_weights.append(weights)
        all_equity.append(ensemble_equity)
        trade_summary_rows.append(trade_summary)
        metric_rows.append(metrics)

    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    weights_df = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    trade_summary_df = pd.DataFrame(trade_summary_rows)

    metrics_df = metrics_df.sort_values(["sharpe_ratio", "total_return_pct"], ascending=False).reset_index(drop=True)

    equity_df.to_csv(output_dir / "fixed_strategy_ensemble_equity.csv", index=False)
    weights_df.to_csv(output_dir / "fixed_strategy_ensemble_weights.csv", index=False)
    metrics_df.to_csv(output_dir / "fixed_strategy_ensemble_metrics.csv", index=False)
    trade_summary_df.to_csv(output_dir / "fixed_strategy_ensemble_trade_summary.csv", index=False)
    outputs["paths"].to_csv(output_dir / "fixed_strategy_ensemble_input_files.csv", index=False)

    save_plots(equity_df, metrics_df, output_dir)

    return {
        "equity": equity_df,
        "weights": weights_df,
        "metrics": metrics_df,
        "trade_summary": trade_summary_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean fixed-strategy ensemble portfolios.")
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=Path("data/final_strategy_backtests_all_candidates_nextbar"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/fixed_strategy_ensemble_clean"),
    )
    parser.add_argument("--initial-equity", type=float, default=100000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_fixed_strategy_ensembles(
        backtest_dir=args.backtest_dir,
        output_dir=args.output_dir,
        initial_equity=args.initial_equity,
    )

    print("\nFixed clean strategy ensemble metrics:")
    metrics = results["metrics"]
    if metrics.empty:
        print("No metrics generated.")
    else:
        print(metrics.to_string(index=False))

    print("\nSaved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
