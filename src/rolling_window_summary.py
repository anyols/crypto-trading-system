from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern} in {folder}")
    return files[-1]


def calculate_rolling_stats(equity: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []

    equity = equity.copy()
    equity["time"] = pd.to_datetime(equity["time"], utc=True)

    for strategy_id, group in equity.groupby("strategy_id"):
        group = group.sort_values("time").set_index("time")
        daily = group["equity"].resample("1D").last().dropna()

        if daily.empty:
            continue

        for window in windows:
            rolling_return = (daily / daily.shift(window) - 1.0).dropna() * 100.0

            if rolling_return.empty:
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "window_days": window,
                        "num_windows": 0,
                        "avg_return_pct": 0.0,
                        "median_return_pct": 0.0,
                        "positive_window_rate_pct": 0.0,
                        "best_return_pct": 0.0,
                        "worst_return_pct": 0.0,
                    }
                )
                continue

            rows.append(
                {
                    "strategy_id": strategy_id,
                    "window_days": window,
                    "num_windows": int(len(rolling_return)),
                    "avg_return_pct": float(rolling_return.mean()),
                    "median_return_pct": float(rolling_return.median()),
                    "positive_window_rate_pct": float((rolling_return > 0).mean() * 100.0),
                    "best_return_pct": float(rolling_return.max()),
                    "worst_return_pct": float(rolling_return.min()),
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rolling return summary from final strategy equity file."
    )

    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=Path("data/final_strategy_backtests_all_candidates_nextbar"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/final_strategy_report_all_candidates_nextbar"),
    )

    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[10, 30, 90],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = find_latest_file(args.backtest_dir, "final_strategy_equity_*.csv")
    equity = pd.read_csv(equity_path)

    summary = calculate_rolling_stats(equity, args.windows)

    output_path = args.output_dir / "rolling_window_summary.csv"
    summary.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()