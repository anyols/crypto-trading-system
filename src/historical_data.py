"""
Download historical OHLCV candles from Coinbase Exchange public API.

Run from project root:

    python -m src.historical_data --days 30 --granularity 3600

Granularity options:
    60      = 1 minute
    300     = 5 minutes
    900     = 15 minutes
    3600    = 1 hour
    21600   = 6 hours
    86400   = 1 day
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.exchange.coinbase.com"
PRODUCT_IDS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]

VALID_GRANULARITIES = {
    60: "1m",
    300: "5m",
    900: "15m",
    3600: "1h",
    21600: "6h",
    86400: "1d",
}

MAX_CANDLES_PER_REQUEST = 300


@dataclass(frozen=True)
class DownloadConfig:
    product_ids: list[str]
    start: datetime
    end: datetime
    granularity: int
    output_dir: Path
    request_sleep_seconds: float = 0.25


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_coinbase_timestamp(dt: datetime) -> str:
    """
    Coinbase Exchange accepts ISO-8601 timestamps.

    Example:
        2026-05-04T12:00:00Z
    """
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware.")

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_granularity(granularity: int) -> None:
    if granularity not in VALID_GRANULARITIES:
        valid = ", ".join(str(x) for x in VALID_GRANULARITIES)
        raise ValueError(f"Invalid granularity: {granularity}. Valid options: {valid}")


def fetch_candle_chunk(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
    session: requests.Session,
) -> pd.DataFrame:
    """
    Fetch one chunk of candles for a single product.

    Coinbase response rows are:
        [time, low, high, open, close, volume]
    """
    url = f"{BASE_URL}/products/{product_id}/candles"

    params = {
        "start": to_coinbase_timestamp(start),
        "end": to_coinbase_timestamp(end),
        "granularity": granularity,
    }

    response = session.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Coinbase request failed for {product_id}. "
            f"Status={response.status_code}. Body={response.text[:500]}"
        )

    rows = response.json()

    if not rows:
        return pd.DataFrame(
            columns=["time", "low", "high", "open", "close", "volume", "product_id"]
        )

    df = pd.DataFrame(
        rows,
        columns=["time", "low", "high", "open", "close", "volume"],
    )

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["product_id"] = product_id

    numeric_cols = ["low", "high", "open", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df[["time", "product_id", "open", "high", "low", "close", "volume"]]


def fetch_product_candles(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
    session: requests.Session,
    request_sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch candles over a larger date range by splitting into <=300-candle chunks.
    """
    validate_granularity(granularity)

    if start >= end:
        raise ValueError("start must be before end.")

    chunk_seconds = granularity * MAX_CANDLES_PER_REQUEST
    chunk_delta = timedelta(seconds=chunk_seconds)

    all_chunks: list[pd.DataFrame] = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_delta, end)

        print(
            f"Fetching {product_id}: "
            f"{to_coinbase_timestamp(chunk_start)} -> {to_coinbase_timestamp(chunk_end)}"
        )

        chunk_df = fetch_candle_chunk(
            product_id=product_id,
            start=chunk_start,
            end=chunk_end,
            granularity=granularity,
            session=session,
        )

        if not chunk_df.empty:
            all_chunks.append(chunk_df)

        chunk_start = chunk_end
        time.sleep(request_sleep_seconds)

    if not all_chunks:
        return pd.DataFrame(
            columns=["time", "product_id", "open", "high", "low", "close", "volume"]
        )

    df = pd.concat(all_chunks, ignore_index=True)

    # Coinbase often returns candles newest-first. We want clean chronological data.
    df = (
        df.drop_duplicates(subset=["time", "product_id"])
        .sort_values(["product_id", "time"])
        .reset_index(drop=True)
    )

    return df


def save_product_candles(
    df: pd.DataFrame,
    product_id: str,
    granularity: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timeframe = VALID_GRANULARITIES[granularity]

    if df.empty:
        filename = f"{product_id}_{timeframe}_empty.csv"
    else:
        start_date = df["time"].min().strftime("%Y%m%d")
        end_date = df["time"].max().strftime("%Y%m%d")
        filename = f"{product_id}_{timeframe}_{start_date}_{end_date}.csv"

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)

    return output_path


def download_all(config: DownloadConfig) -> dict[str, Path]:
    saved_files: dict[str, Path] = {}

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "crypto-trading-system/0.1",
                "Accept": "application/json",
            }
        )

        for product_id in config.product_ids:
            df = fetch_product_candles(
                product_id=product_id,
                start=config.start,
                end=config.end,
                granularity=config.granularity,
                session=session,
                request_sleep_seconds=config.request_sleep_seconds,
            )

            output_path = save_product_candles(
                df=df,
                product_id=product_id,
                granularity=config.granularity,
                output_dir=config.output_dir,
            )

            saved_files[product_id] = output_path
            print(f"Saved {len(df):,} rows for {product_id} -> {output_path}")

    return saved_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV candles from Coinbase Exchange."
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of past days to download.",
    )

    parser.add_argument(
        "--granularity",
        type=int,
        default=3600,
        choices=list(VALID_GRANULARITIES.keys()),
        help="Candle granularity in seconds.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where CSV files will be saved.",
    )

    parser.add_argument(
        "--products",
        nargs="+",
        default=PRODUCT_IDS,
        help="Coinbase product IDs to download.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    end = utc_now()
    start = end - timedelta(days=args.days)

    config = DownloadConfig(
        product_ids=args.products,
        start=start,
        end=end,
        granularity=args.granularity,
        output_dir=args.output_dir,
    )

    download_all(config)


if __name__ == "__main__":
    main()