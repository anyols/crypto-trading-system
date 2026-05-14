"""
Download historical OHLCV candles from Coinbase Exchange public API.

Run from project root:

    python -m src.historical_data --days 30 --granularity 3600

For 5 years of 15m data:

    python -m src.historical_data --days 1825 --granularity 900

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
import random
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

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class DownloadConfig:
    product_ids: list[str]
    start: datetime
    end: datetime
    granularity: int
    output_dir: Path
    request_sleep_seconds: float = 0.35
    max_retries: int = 6
    skip_failed_chunks: bool = False


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


def empty_candle_frame(product_id: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=["time", "product_id", "open", "high", "low", "close", "volume"]
    )

    if product_id is not None:
        df["product_id"] = product_id

    return df


def parse_coinbase_rows(rows: list, product_id: str) -> pd.DataFrame:
    """
    Coinbase response rows are:
        [time, low, high, open, close, volume]
    """
    if not rows:
        return empty_candle_frame(product_id)

    df = pd.DataFrame(
        rows,
        columns=["time", "low", "high", "open", "close", "volume"],
    )

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["product_id"] = product_id

    numeric_cols = ["low", "high", "open", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df = df[["time", "product_id", "open", "high", "low", "close", "volume"]]

    return df.sort_values("time").reset_index(drop=True)


def sleep_before_retry(attempt: int, retry_after_header: str | None = None) -> None:
    """
    Sleep with exponential backoff.

    If Coinbase sends Retry-After, respect it when possible.
    """
    if retry_after_header:
        try:
            retry_after_seconds = float(retry_after_header)
            sleep_seconds = min(max(retry_after_seconds, 1.0), 90.0)
        except ValueError:
            sleep_seconds = min(2**attempt, 90.0)
    else:
        sleep_seconds = min(2**attempt, 90.0)

    jitter = random.uniform(0.0, 0.75)
    total_sleep = sleep_seconds + jitter

    print(f"Sleeping {total_sleep:.2f}s before retry...")
    time.sleep(total_sleep)


def fetch_candle_chunk(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
    session: requests.Session,
    max_retries: int = 6,
) -> pd.DataFrame:
    """
    Fetch one chunk of candles for a single product.

    This function is retry-safe because Coinbase sometimes returns temporary
    500/502/503/504 errors or 429 rate limits during long downloads.
    """
    url = f"{BASE_URL}/products/{product_id}/candles"

    params = {
        "start": to_coinbase_timestamp(start),
        "end": to_coinbase_timestamp(end),
        "granularity": granularity,
    }

    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                rows = response.json()
                return parse_coinbase_rows(rows, product_id)

            last_error = (
                f"Status={response.status_code}. Body={response.text[:500]}"
            )

            if response.status_code in RETRY_STATUS_CODES:
                print(
                    f"Temporary Coinbase error for {product_id}. "
                    f"Attempt {attempt}/{max_retries}. "
                    f"{to_coinbase_timestamp(start)} -> {to_coinbase_timestamp(end)}. "
                    f"{last_error}"
                )
                sleep_before_retry(
                    attempt=attempt,
                    retry_after_header=response.headers.get("Retry-After"),
                )
                continue

            raise RuntimeError(
                f"Coinbase request failed for {product_id}. "
                f"{to_coinbase_timestamp(start)} -> {to_coinbase_timestamp(end)}. "
                f"{last_error}"
            )

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            print(
                f"Network error for {product_id}. "
                f"Attempt {attempt}/{max_retries}. "
                f"{to_coinbase_timestamp(start)} -> {to_coinbase_timestamp(end)}. "
                f"Error={last_error}"
            )
            sleep_before_retry(attempt=attempt)

    raise RuntimeError(
        f"Coinbase request failed for {product_id} after {max_retries} retries. "
        f"{to_coinbase_timestamp(start)} -> {to_coinbase_timestamp(end)}. "
        f"Last error: {last_error}"
    )


def fetch_product_candles(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int,
    session: requests.Session,
    request_sleep_seconds: float = 0.35,
    max_retries: int = 6,
    skip_failed_chunks: bool = False,
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
    failed_chunks: list[dict[str, str]] = []

    chunk_start = start
    chunk_number = 1

    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_delta, end)

        print(
            f"Fetching {product_id} chunk {chunk_number}: "
            f"{to_coinbase_timestamp(chunk_start)} -> {to_coinbase_timestamp(chunk_end)}"
        )

        try:
            chunk_df = fetch_candle_chunk(
                product_id=product_id,
                start=chunk_start,
                end=chunk_end,
                granularity=granularity,
                session=session,
                max_retries=max_retries,
            )

            if not chunk_df.empty:
                all_chunks.append(chunk_df)

        except RuntimeError as exc:
            if not skip_failed_chunks:
                raise

            print(f"WARNING: skipping failed chunk for {product_id}: {exc}")
            failed_chunks.append(
                {
                    "product_id": product_id,
                    "start": to_coinbase_timestamp(chunk_start),
                    "end": to_coinbase_timestamp(chunk_end),
                    "error": str(exc),
                }
            )

        chunk_start = chunk_end
        chunk_number += 1
        time.sleep(request_sleep_seconds)

    if failed_chunks:
        print(f"WARNING: {len(failed_chunks)} chunks failed for {product_id}.")

    if not all_chunks:
        return empty_candle_frame(product_id)

    df = pd.concat(all_chunks, ignore_index=True)

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

    # Write atomically-ish: save temp then replace.
    temp_path = output_path.with_suffix(".tmp.csv")
    df.to_csv(temp_path, index=False)
    temp_path.replace(output_path)

    return output_path


def save_failed_product_log(
    failures: list[dict[str, str]],
    output_dir: Path,
    granularity: int,
) -> Path | None:
    if not failures:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timeframe = VALID_GRANULARITIES[granularity]
    path = output_dir / f"failed_downloads_{timeframe}_{utc_now().strftime('%Y%m%d_%H%M%S')}.csv"

    pd.DataFrame(failures).to_csv(path, index=False)
    return path


def download_all(config: DownloadConfig) -> dict[str, Path]:
    saved_files: dict[str, Path] = {}
    product_failures: list[dict[str, str]] = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "crypto-trading-system/0.1",
                "Accept": "application/json",
            }
        )

        for product_id in config.product_ids:
            try:
                df = fetch_product_candles(
                    product_id=product_id,
                    start=config.start,
                    end=config.end,
                    granularity=config.granularity,
                    session=session,
                    request_sleep_seconds=config.request_sleep_seconds,
                    max_retries=config.max_retries,
                    skip_failed_chunks=config.skip_failed_chunks,
                )

                output_path = save_product_candles(
                    df=df,
                    product_id=product_id,
                    granularity=config.granularity,
                    output_dir=config.output_dir,
                )

                saved_files[product_id] = output_path
                print(f"Saved {len(df):,} rows for {product_id} -> {output_path}")

            except Exception as exc:
                print(f"ERROR: failed to download {product_id}: {exc}")
                product_failures.append(
                    {
                        "product_id": product_id,
                        "start": to_coinbase_timestamp(config.start),
                        "end": to_coinbase_timestamp(config.end),
                        "granularity": str(config.granularity),
                        "error": str(exc),
                    }
                )

                # Continue with other products. This is better for long downloads.
                continue

    failure_log = save_failed_product_log(
        failures=product_failures,
        output_dir=config.output_dir,
        granularity=config.granularity,
    )

    if failure_log is not None:
        print(f"Saved product failure log -> {failure_log}")

    if not saved_files:
        raise RuntimeError("No products were downloaded successfully.")

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

    parser.add_argument(
        "--request-sleep-seconds",
        type=float,
        default=0.35,
        help="Sleep time between successful chunk requests.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries per chunk for temporary Coinbase/network errors.",
    )

    parser.add_argument(
        "--skip-failed-chunks",
        action="store_true",
        help=(
            "Skip chunks that fail after all retries. "
            "Use only if you prefer partial data over crashing."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.days <= 0:
        raise ValueError("--days must be positive.")

    end = utc_now()
    start = end - timedelta(days=args.days)

    config = DownloadConfig(
        product_ids=args.products,
        start=start,
        end=end,
        granularity=args.granularity,
        output_dir=args.output_dir,
        request_sleep_seconds=args.request_sleep_seconds,
        max_retries=args.max_retries,
        skip_failed_chunks=args.skip_failed_chunks,
    )

    print("Download configuration:")
    print(f"Products: {config.product_ids}")
    print(f"Start: {to_coinbase_timestamp(config.start)}")
    print(f"End: {to_coinbase_timestamp(config.end)}")
    print(f"Granularity: {config.granularity} ({VALID_GRANULARITIES[config.granularity]})")
    print(f"Output dir: {config.output_dir}")
    print(f"Max retries per chunk: {config.max_retries}")
    print(f"Sleep between chunks: {config.request_sleep_seconds}s")
    print(f"Skip failed chunks: {config.skip_failed_chunks}")

    download_all(config)


if __name__ == "__main__":
    main()