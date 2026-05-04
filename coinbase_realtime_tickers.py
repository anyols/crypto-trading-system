"""
Coinbase real-time ticker stream for BTC, SOL, DOGE, ETH, and XRP.

Run:
    pip install websockets
    python coinbase_realtime_tickers.py

This script only listens to public market data. It does NOT trade.
"""

from __future__ import annotations

import asyncio
import json
import signal
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

PRODUCT_IDS = [
    "BTC-USD",
    "SOL-USD",
    "DOGE-USD",
    "ETH-USD",
    "XRP-USD",
]


@dataclass
class Ticker:
    product_id: str
    price: float
    best_bid: float | None
    best_ask: float | None
    volume_24h: float | None
    low_24h: float | None
    high_24h: float | None
    price_percent_change_24h: float | None
    exchange_timestamp: str
    received_at_utc: str


def to_float(value: Any) -> float | None:
    """Safely convert Coinbase string fields to float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ticker_message(message: str) -> list[Ticker]:
    """
    Parse Coinbase Advanced Trade ticker messages.

    Expected shape:
    {
        "channel": "ticker",
        "timestamp": "...",
        "events": [
            {
                "type": "snapshot" | "update",
                "tickers": [ ... ]
            }
        ]
    }
    """
    data = json.loads(message)

    if data.get("channel") != "ticker":
        return []

    exchange_timestamp = data.get("timestamp", "")
    received_at_utc = datetime.now(timezone.utc).isoformat()

    parsed: list[Ticker] = []

    for event in data.get("events", []):
        for raw in event.get("tickers", []):
            product_id = raw.get("product_id")
            price = to_float(raw.get("price"))

            if product_id is None or price is None:
                continue

            parsed.append(
                Ticker(
                    product_id=product_id,
                    price=price,
                    best_bid=to_float(raw.get("best_bid")),
                    best_ask=to_float(raw.get("best_ask")),
                    volume_24h=to_float(raw.get("volume_24_h")),
                    low_24h=to_float(raw.get("low_24_h")),
                    high_24h=to_float(raw.get("high_24_h")),
                    price_percent_change_24h=to_float(raw.get("price_percent_chg_24_h")),
                    exchange_timestamp=exchange_timestamp,
                    received_at_utc=received_at_utc,
                )
            )

    return parsed


async def subscribe(ws: websockets.WebSocketClientProtocol) -> None:
    """Subscribe to heartbeat and ticker channels."""
    # Heartbeats help keep subscriptions open during quiet periods.
    await ws.send(
        json.dumps(
            {
                "type": "subscribe",
                "channel": "heartbeats",
            }
        )
    )

    await ws.send(
        json.dumps(
            {
                "type": "subscribe",
                "product_ids": PRODUCT_IDS,
                "channel": "ticker",
            }
        )
    )


async def stream_tickers(stop_event: asyncio.Event) -> None:
    """
    Connect, subscribe, parse ticker messages, and reconnect if needed.
    """
    latest: dict[str, Ticker] = {}
    reconnect_delay_seconds = 1
    max_reconnect_delay_seconds = 30

    while not stop_event.is_set():
        try:
            print(f"Connecting to Coinbase Advanced Trade WebSocket: {COINBASE_WS_URL}")

            async with websockets.connect(
                COINBASE_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_queue=2048,
            ) as ws:
                await subscribe(ws)
                print(f"Subscribed to: {', '.join(PRODUCT_IDS)}")
                reconnect_delay_seconds = 1

                async for message in ws:
                    tickers = parse_ticker_message(message)

                    for ticker in tickers:
                        latest[ticker.product_id] = ticker

                        # Keep the first version simple: print clean normalized rows.
                        print(
                            f"{ticker.received_at_utc} | "
                            f"{ticker.product_id:<8} | "
                            f"price={ticker.price:<14,.6f} | "
                            f"bid={ticker.best_bid} | "
                            f"ask={ticker.best_ask} | "
                            f"24h_change={ticker.price_percent_change_24h}"
                        )

                    if stop_event.is_set():
                        break

        except (ConnectionClosed, WebSocketException, OSError) as exc:
            print(f"WebSocket problem: {exc}. Reconnecting in {reconnect_delay_seconds}s...")
            await asyncio.sleep(reconnect_delay_seconds)
            reconnect_delay_seconds = min(
                reconnect_delay_seconds * 2,
                max_reconnect_delay_seconds,
            )

        except json.JSONDecodeError as exc:
            # Bad messages should not kill the collector.
            print(f"JSON parsing error: {exc}")

        except Exception as exc:
            # Do not hide unknown bugs forever. For now we log and reconnect.
            print(f"Unexpected error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(reconnect_delay_seconds)

    print("Stopping ticker stream.")


def install_shutdown_handlers(stop_event: asyncio.Event) -> None:
    """Handle Ctrl+C / SIGTERM cleanly."""
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows fallback. Ctrl+C still works via KeyboardInterrupt.
            pass


async def main() -> None:
    stop_event = asyncio.Event()
    install_shutdown_handlers(stop_event)
    await stream_tickers(stop_event)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped by user.")
