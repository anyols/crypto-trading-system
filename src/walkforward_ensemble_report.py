"""
Walk-forward ensemble report.

Purpose:
    Combine the walk-forward selected candidates into ensemble portfolios.

    This does NOT pick strategies using future information. It uses the selected
    candidates that were already selected inside each walk-forward fold using
    only prior train/validation data.

    For each year/fold, it builds:
        - Equal Weight Top 3
        - Equal Weight Top 5
        - Score Weighted Top 3
        - Score Weighted Top 5
        - Rank 1 standalone
        - Buy & Hold benchmark, if present

    The ensemble equity is calculated as a capital allocation across standalone
    selected-strategy equity curves:

        ensemble_equity_t = initial_equity * sum_i(weight_i * equity_i_t / equity_i_0)

    This is a clean portfolio-of-strategies approximation. It is not a shared
    order-level execution simulator, but it is the right next layer for comparing
    strategy selection + allocation through time.

Example:

    py -m src.walkforward_ensemble_report ^
      --input-root data ^
      --folder-template walkforward_{year}_v3 ^
      --years 2024 2025 2026 ^
      --output-dir reports/walkforward_ensemble_v3 ^
      --initial-equity 100000

Outputs:
    reports/walkforward_ensemble_v3/
        walkforward_ensemble_year_metrics.csv
        walkforward_ensemble_aggregate_summary.csv
        walkforward_ensemble_equity.csv
        walkforward_ensemble_weights.csv
        walkforward_ensemble_trade_summary.csv
        walkforward_ensemble_equity.html
        walkforward_ensemble_drawdown.html
        walkforward_ensemble_return_sharpe.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass(frozen=True)
class EnsembleSpec:
    ensemble_id: str
    display_name: str
    top_n: int
    weighting: str  # "equal" or "score"


ENSEMBLES = [
    EnsembleSpec(
        ensemble_id="rank1_standalone",
        display_name="WF Rank 1 Standalone",
        top_n=1,
        weighting="equal",
    ),
    EnsembleSpec(
        ensemble_id="ew_top3",
        display_name="WF Equal-Weight Top 3",
        top_n=3,
        weighting="equal",
    ),
    EnsembleSpec(
        ensemble_id="ew_top5",
        display_name="WF Equal-Weight Top 5",
        top_n=5,
        weighting="equal",
    ),
    EnsembleSpec(
        ensemble_id="score_top3",
        display_name="WF Score-Weighted Top 3",
        top_n=3,
        weighting="score",
    ),
    EnsembleSpec(
        ensemble_id="score_top5",
        display_name="WF Score-Weighted Top 5",
        top_n=5,
        weighting="score",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────────────────────────────────────

def latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern))
    if not files:
        return None
    return files[-1]


def require_latest(folder: Path, pattern: str) -> Path:
    path = latest_file(folder, pattern)
    if path is None:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return path


def load_year_outputs(folder: Path) -> dict[str, pd.DataFrame]:
    equity_path = require_latest(folder, "walkforward_equity_*.csv")
    selected_path = require_latest(folder, "walkforward_selected_candidates_*.csv")
    metrics_path = require_latest(folder, "walkforward_fold_metrics_*.csv")

    trades_path = latest_file(folder, "walkforward_trades_*.csv")

    equity = pd.read_csv(equity_path, parse_dates=["time"])
    selected = pd.read_csv(selected_path)
    metrics = pd.read_csv(metrics_path)
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"]) if trades_path else pd.DataFrame()

    return {
        "equity": equity,
        "selected": selected,
        "metrics": metrics,
        "trades": trades,
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


def annualized_metrics(equity: pd.DataFrame, initial_equity: float) -> dict[str, float]:
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
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0

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

def score_column(selected: pd.DataFrame) -> str | None:
    candidates = [
        "robust_walkforward_score",
        "final_discovery_score",
        "validation_score",
        "selection_validation_score",
    ]
    for col in candidates:
        if col in selected.columns:
            return col
    return None


def compute_weights(selected: pd.DataFrame, ranks: list[int], weighting: str) -> pd.DataFrame:
    rows = selected[selected["selected_rank"].isin(ranks)].copy()
    rows = rows.sort_values("selected_rank")

    if rows.empty:
        return pd.DataFrame(columns=["selected_rank", "weight", "weighting", "score_used"])

    if weighting == "equal" or len(rows) == 1:
        rows["weight"] = 1.0 / len(rows)
        rows["score_used"] = np.nan
    elif weighting == "score":
        col = score_column(rows)
        if col is None:
            rows["weight"] = 1.0 / len(rows)
            rows["score_used"] = np.nan
        else:
            scores = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            scores = scores.clip(lower=0.0)
            if float(scores.sum()) <= 0:
                rows["weight"] = 1.0 / len(rows)
            else:
                rows["weight"] = scores / scores.sum()
            rows["score_used"] = scores
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    rows["weighting"] = weighting
    keep = ["selected_rank", "weight", "weighting", "score_used"]
    for col in ["strategy_id", "side", "lookback_hours", "horizon_minutes", "vol_regime", "momentum_bucket", "filter_name"]:
        if col in rows.columns:
            keep.append(col)

    return rows[keep].reset_index(drop=True)


def build_ensemble_equity_for_year(
    equity: pd.DataFrame,
    weights: pd.DataFrame,
    year: int,
    ensemble: EnsembleSpec,
    initial_equity: float,
) -> pd.DataFrame:
    selected_ranks = weights["selected_rank"].astype(int).tolist()

    data = equity[equity["selected_rank"].isin(selected_ranks)].copy()
    data = data[data["selected_rank"] != 0].copy()

    if data.empty:
        return pd.DataFrame()

    pivot = data.pivot_table(index="time", columns="selected_rank", values="equity", aggfunc="last").sort_index()
    pivot = pivot.ffill().dropna(how="all")

    normalized = pivot.copy()
    for col in normalized.columns:
        first_valid = normalized[col].dropna()
        if first_valid.empty:
            normalized[col] = np.nan
        else:
            normalized[col] = normalized[col] / float(first_valid.iloc[0])

    normalized = normalized.ffill().fillna(1.0)

    weight_map = weights.set_index("selected_rank")["weight"].to_dict()
    ensemble_norm = pd.Series(0.0, index=normalized.index)
    for rank, weight in weight_map.items():
        if rank in normalized.columns:
            ensemble_norm += float(weight) * normalized[rank]

    out = pd.DataFrame(
        {
            "time": ensemble_norm.index,
            "year": year,
            "ensemble_id": ensemble.ensemble_id,
            "display_name": ensemble.display_name,
            "equity": initial_equity * ensemble_norm.values,
        }
    )

    out["running_max"] = out["equity"].cummax()
    out["drawdown_pct"] = (out["equity"] / out["running_max"] - 1.0) * 100.0
    return out


def build_benchmark_equity_for_year(
    equity: pd.DataFrame,
    year: int,
    initial_equity: float,
) -> pd.DataFrame:
    benchmark = equity[equity["selected_rank"] == 0].copy()
    if benchmark.empty:
        return pd.DataFrame()

    benchmark = benchmark.sort_values("time")
    first = float(benchmark["equity"].iloc[0])
    benchmark["equity"] = initial_equity * benchmark["equity"] / first
    benchmark["year"] = year
    benchmark["ensemble_id"] = "benchmark_buy_hold"
    benchmark["display_name"] = "Equal-Weight Buy & Hold"
    benchmark["running_max"] = benchmark["equity"].cummax()
    benchmark["drawdown_pct"] = (benchmark["equity"] / benchmark["running_max"] - 1.0) * 100.0
    return benchmark[["time", "year", "ensemble_id", "display_name", "equity", "running_max", "drawdown_pct"]]


def build_scaled_trade_summary(
    trades: pd.DataFrame,
    weights: pd.DataFrame,
    year: int,
    ensemble: EnsembleSpec,
) -> dict[str, float | int | str]:
    if trades.empty or weights.empty:
        return {
            "year": year,
            "ensemble_id": ensemble.ensemble_id,
            "display_name": ensemble.display_name,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
        }

    selected_ranks = weights["selected_rank"].astype(int).tolist()
    weight_map = weights.set_index("selected_rank")["weight"].to_dict()

    data = trades[trades["selected_rank"].isin(selected_ranks)].copy()
    if data.empty:
        return {
            "year": year,
            "ensemble_id": ensemble.ensemble_id,
            "display_name": ensemble.display_name,
            "num_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
        }

    data["weight"] = data["selected_rank"].map(weight_map).fillna(0.0)
    data["scaled_net_pnl"] = data["net_pnl"] * data["weight"]
    data["scaled_fees"] = data.get("total_fees", 0.0) * data["weight"]

    return {
        "year": year,
        "ensemble_id": ensemble.ensemble_id,
        "display_name": ensemble.display_name,
        "num_trades": int(len(data)),
        "win_rate_pct": float((data["scaled_net_pnl"] > 0).mean() * 100.0),
        "profit_factor": profit_factor(data["scaled_net_pnl"]),
        "net_pnl": float(data["scaled_net_pnl"].sum()),
        "fees_paid": float(data["scaled_fees"].sum()),
    }


def build_year_metrics(
    ensemble_equity: pd.DataFrame,
    trade_summary: dict[str, float | int | str],
    initial_equity: float,
) -> dict[str, float | int | str]:
    metrics = annualized_metrics(ensemble_equity[["time", "equity"]], initial_equity)
    out = {
        "year": int(ensemble_equity["year"].iloc[0]),
        "ensemble_id": str(ensemble_equity["ensemble_id"].iloc[0]),
        "display_name": str(ensemble_equity["display_name"].iloc[0]),
        **metrics,
        "num_trades": int(trade_summary.get("num_trades", 0)),
        "win_rate_pct": float(trade_summary.get("win_rate_pct", 0.0)),
        "profit_factor": float(trade_summary.get("profit_factor", 0.0)),
        "fees_paid": float(trade_summary.get("fees_paid", 0.0)),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation and plots
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregate_summary(year_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for ensemble_id, group in year_metrics.groupby("ensemble_id"):
        returns = group["total_return_pct"].astype(float) / 100.0
        compounded = (float((1.0 + returns).prod()) - 1.0) * 100.0
        rows.append(
            {
                "ensemble_id": ensemble_id,
                "display_name": group["display_name"].iloc[0],
                "num_years": int(group["year"].nunique()),
                "compounded_return_pct": compounded,
                "avg_year_return_pct": float(group["total_return_pct"].mean()),
                "median_year_return_pct": float(group["total_return_pct"].median()),
                "positive_year_rate_pct": float((group["total_return_pct"] > 0).mean() * 100.0),
                "avg_sharpe_ratio": float(group["sharpe_ratio"].mean()),
                "median_sharpe_ratio": float(group["sharpe_ratio"].median()),
                "worst_year_return_pct": float(group["total_return_pct"].min()),
                "worst_max_drawdown_pct": float(group["max_drawdown_pct"].min()),
                "avg_max_drawdown_pct": float(group["max_drawdown_pct"].mean()),
                "total_trades": int(group["num_trades"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["avg_sharpe_ratio", "compounded_return_pct"],
        ascending=False,
    ).reset_index(drop=True)


def save_plots(equity: pd.DataFrame, year_metrics: pd.DataFrame, output_dir: Path) -> None:
    if equity.empty:
        return

    active = equity[equity["ensemble_id"] != "benchmark_buy_hold"].copy()

    fig = px.line(
        active,
        x="time",
        y="equity",
        color="display_name",
        facet_col="year",
        facet_col_wrap=2,
        title="Walk-Forward Ensemble Equity Curves",
        height=750,
    )
    fig.update_yaxes(matches=None)
    fig.write_html(output_dir / "walkforward_ensemble_equity.html")

    fig = px.line(
        active,
        x="time",
        y="drawdown_pct",
        color="display_name",
        facet_col="year",
        facet_col_wrap=2,
        title="Walk-Forward Ensemble Drawdowns",
        height=750,
    )
    fig.update_yaxes(matches=None)
    fig.write_html(output_dir / "walkforward_ensemble_drawdown.html")

    active_metrics = year_metrics[year_metrics["ensemble_id"] != "benchmark_buy_hold"].copy()
    fig = px.scatter(
        active_metrics,
        x="max_drawdown_pct",
        y="total_return_pct",
        color="display_name",
        symbol="year",
        size=active_metrics["sharpe_ratio"].clip(lower=0.1),
        hover_data=["year", "sharpe_ratio", "num_trades", "profit_factor"],
        title="Walk-Forward Ensemble Return vs Drawdown",
        height=650,
    )
    fig.write_html(output_dir / "walkforward_ensemble_return_drawdown.html")

    heatmap_data = active_metrics.pivot(index="display_name", columns="year", values="total_return_pct")
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.astype(str),
            y=heatmap_data.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(heatmap_data.values, 2),
            texttemplate="%{text}%",
            hovertemplate="Strategy=%{y}<br>Year=%{x}<br>Return=%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(title="Walk-Forward Ensemble Yearly Returns", height=500)
    fig.write_html(output_dir / "walkforward_ensemble_yearly_return_heatmap.html")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ensembles(
    input_root: Path,
    folder_template: str,
    years: Iterable[int],
    output_dir: Path,
    initial_equity: float,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_equity = []
    all_weights = []
    year_metric_rows = []
    trade_summary_rows = []

    for year in years:
        folder = input_root / folder_template.format(year=year)
        print(f"Loading {year}: {folder}")
        outputs = load_year_outputs(folder)
        equity = outputs["equity"]
        selected = outputs["selected"]
        trades = outputs["trades"]

        if "selected_rank" not in equity.columns:
            raise ValueError(f"Equity file for {year} has no selected_rank column.")
        if "selected_rank" not in selected.columns:
            raise ValueError(f"Selected candidates file for {year} has no selected_rank column.")

        for ensemble in ENSEMBLES:
            ranks = list(range(1, ensemble.top_n + 1))
            weights = compute_weights(selected, ranks, ensemble.weighting)
            if weights.empty:
                print(f"WARNING: no weights for {year} {ensemble.ensemble_id}")
                continue

            weights["year"] = year
            weights["ensemble_id"] = ensemble.ensemble_id
            weights["display_name"] = ensemble.display_name
            all_weights.append(weights)

            ens_equity = build_ensemble_equity_for_year(
                equity=equity,
                weights=weights,
                year=year,
                ensemble=ensemble,
                initial_equity=initial_equity,
            )
            if ens_equity.empty:
                print(f"WARNING: empty equity for {year} {ensemble.ensemble_id}")
                continue

            trade_summary = build_scaled_trade_summary(
                trades=trades,
                weights=weights,
                year=year,
                ensemble=ensemble,
            )
            metrics = build_year_metrics(ens_equity, trade_summary, initial_equity)

            all_equity.append(ens_equity)
            trade_summary_rows.append(trade_summary)
            year_metric_rows.append(metrics)

        benchmark_equity = build_benchmark_equity_for_year(equity, year, initial_equity)
        if not benchmark_equity.empty:
            trade_summary = {
                "year": year,
                "ensemble_id": "benchmark_buy_hold",
                "display_name": "Equal-Weight Buy & Hold",
                "num_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "fees_paid": 0.0,
            }
            all_equity.append(benchmark_equity)
            trade_summary_rows.append(trade_summary)
            year_metric_rows.append(build_year_metrics(benchmark_equity, trade_summary, initial_equity))

    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    weights_df = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    year_metrics_df = pd.DataFrame(year_metric_rows)
    trade_summary_df = pd.DataFrame(trade_summary_rows)
    aggregate_df = build_aggregate_summary(year_metrics_df) if not year_metrics_df.empty else pd.DataFrame()

    equity_df.to_csv(output_dir / "walkforward_ensemble_equity.csv", index=False)
    weights_df.to_csv(output_dir / "walkforward_ensemble_weights.csv", index=False)
    year_metrics_df.to_csv(output_dir / "walkforward_ensemble_year_metrics.csv", index=False)
    trade_summary_df.to_csv(output_dir / "walkforward_ensemble_trade_summary.csv", index=False)
    aggregate_df.to_csv(output_dir / "walkforward_ensemble_aggregate_summary.csv", index=False)

    save_plots(equity_df, year_metrics_df, output_dir)

    return {
        "equity": equity_df,
        "weights": weights_df,
        "year_metrics": year_metrics_df,
        "trade_summary": trade_summary_df,
        "aggregate": aggregate_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build walk-forward ensemble portfolios from selected candidates.")
    parser.add_argument("--input-root", type=Path, default=Path("data"))
    parser.add_argument("--folder-template", type=str, default="walkforward_{year}_v3")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/walkforward_ensemble_v3"))
    parser.add_argument("--initial-equity", type=float, default=100000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_ensembles(
        input_root=args.input_root,
        folder_template=args.folder_template,
        years=args.years,
        output_dir=args.output_dir,
        initial_equity=args.initial_equity,
    )

    print("\nWalk-forward ensemble aggregate summary:")
    aggregate = results["aggregate"]
    if aggregate.empty:
        print("No aggregate summary generated.")
    else:
        print(aggregate.to_string(index=False))

    print("\nSaved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
