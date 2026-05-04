"""
Interactive dashboard for the crypto volatility strategy project.

Run from the project root:

    streamlit run dashboard.py

Install dependencies:

    pip install streamlit plotly pandas numpy requests

What this dashboard does in v1:
    1. Downloads historical Coinbase OHLCV candles
    2. Generates volatility-compression breakout signals
    3. Visualizes candles, indicators, volatility, volume, and signals
    4. Explains the full research pipeline and parameter reasoning

What it does NOT do yet:
    - It does not trade
    - It does not place orders
    - It does not run the live ticker stream
    - It does not yet backtest strategy profitability
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.historical_data import (
    DownloadConfig,
    PRODUCT_IDS,
    VALID_GRANULARITIES,
    download_all,
    utc_now,
)
from src.indicators import add_volatility_strategy_features
from src.strategy import (
    VolatilityBreakoutConfig,
    generate_volatility_breakout_signals,
    load_raw_candles,
    summarize_signals,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Volatility Strategy Dashboard",
    page_icon="📈",
    layout="wide",
)

RAW_DATA_DIR = Path("data/raw")
SIGNAL_DATA_DIR = Path("data/signals")

TIMEFRAME_TO_GRANULARITY = {label: seconds for seconds, label in VALID_GRANULARITIES.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_raw_candles(input_dir: str, timeframe: str) -> pd.DataFrame:
    return load_raw_candles(Path(input_dir), timeframe)


@st.cache_data(show_spinner=False)
def cached_features(candles: pd.DataFrame) -> pd.DataFrame:
    return add_volatility_strategy_features(candles)


@st.cache_data(show_spinner=False)
def cached_signals(
    features: pd.DataFrame,
    max_vol_percentile: float,
    volume_multiplier: float,
    stop_atr_multiplier: float,
    take_profit_r_multiple: float,
) -> pd.DataFrame:
    config = VolatilityBreakoutConfig(
        max_vol_percentile=max_vol_percentile,
        volume_multiplier=volume_multiplier,
        stop_atr_multiplier=stop_atr_multiplier,
        take_profit_r_multiple=take_profit_r_multiple,
    )
    return generate_volatility_breakout_signals(features, config)


def clear_data_caches() -> None:
    cached_load_raw_candles.clear()
    cached_features.clear()
    cached_signals.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_price_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = df[df["product_id"] == product_id].copy()
    product = product.sort_values("time")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=product["time"],
            open=product["open"],
            high=product["high"],
            low=product["low"],
            close=product["close"],
            name="OHLC",
        )
    )

    optional_lines = [
        ("ema_50", "EMA 50"),
        ("ema_200", "EMA 200"),
        ("donchian_high_20", "Donchian High 20"),
        ("stop_price", "Signal Stop"),
        ("take_profit_price", "Signal Take Profit"),
    ]

    for column, label in optional_lines:
        if column in product.columns:
            if column in {"stop_price", "take_profit_price"}:
                y_values = product[column].where(product.get("long_signal", False))
            else:
                y_values = product[column]

            fig.add_trace(
                go.Scatter(
                    x=product["time"],
                    y=y_values,
                    mode="lines",
                    name=label,
                )
            )

    if "long_signal" in product.columns:
        signal_rows = product[product["long_signal"]]

        fig.add_trace(
            go.Scatter(
                x=signal_rows["time"],
                y=signal_rows["close"],
                mode="markers",
                name="Long Signal",
                marker={"size": 10, "symbol": "triangle-up"},
                customdata=signal_rows[
                    [
                        "realized_vol_percentile_200",
                        "entry_price",
                        "stop_price",
                        "take_profit_price",
                    ]
                ],
                hovertemplate=(
                    "Long Signal<br>"
                    "Time=%{x}<br>"
                    "Close=%{y}<br>"
                    "Vol Percentile=%{customdata[0]:.2f}<br>"
                    "Entry=%{customdata[1]:.4f}<br>"
                    "Stop=%{customdata[2]:.4f}<br>"
                    "Take Profit=%{customdata[3]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{product_id} Price, Trend Lines, Breakouts, and Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    return fig


def make_volatility_chart(
    df: pd.DataFrame,
    product_id: str,
    max_vol_percentile: float,
) -> go.Figure:
    product = df[df["product_id"] == product_id].copy()
    product = product.sort_values("time")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=product["time"],
            y=product["realized_vol_percentile_200"],
            mode="lines",
            name="Realized Vol Percentile 200",
        )
    )

    fig.add_hline(
        y=max_vol_percentile,
        line_dash="dash",
        annotation_text="Compression threshold",
    )

    if "long_signal" in product.columns:
        signal_rows = product[product["long_signal"]]
        fig.add_trace(
            go.Scatter(
                x=signal_rows["time"],
                y=signal_rows["realized_vol_percentile_200"],
                mode="markers",
                name="Signal During Compression",
                marker={"size": 9},
            )
        )

    fig.update_layout(
        title=f"{product_id} Realized Volatility Percentile",
        xaxis_title="Time",
        yaxis_title="Percentile",
        height=420,
        hovermode="x unified",
    )

    return fig


def make_realized_vol_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = df[df["product_id"] == product_id].copy()
    product = product.sort_values("time")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=product["time"],
            y=product["realized_vol_20"],
            mode="lines",
            name="Realized Vol 20",
        )
    )

    fig.update_layout(
        title=f"{product_id} Rolling Realized Volatility",
        xaxis_title="Time",
        yaxis_title="Volatility",
        height=380,
        hovermode="x unified",
    )

    return fig


def make_volume_chart(df: pd.DataFrame, product_id: str) -> go.Figure:
    product = df[df["product_id"] == product_id].copy()
    product = product.sort_values("time")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=product["time"],
            y=product["volume"],
            name="Volume",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=product["time"],
            y=product["volume_sma_20"],
            mode="lines",
            name="Volume SMA 20",
        )
    )

    fig.update_layout(
        title=f"{product_id} Volume Confirmation",
        xaxis_title="Time",
        yaxis_title="Volume",
        height=380,
        hovermode="x unified",
    )

    return fig


def show_signal_diagnostics(df: pd.DataFrame, product_id: str) -> None:
    product = df[df["product_id"] == product_id].copy()
    product = product.sort_values("time")

    latest = product.tail(1)

    if latest.empty:
        st.warning("No rows available for this product.")
        return

    row = latest.iloc[0]

    checks = pd.DataFrame(
        [
            {
                "Condition": "Low realized volatility",
                "Pass": bool(row.get("low_volatility", False)),
                "Value": row.get("realized_vol_percentile_200"),
                "Rule": "vol percentile <= threshold",
            },
            {
                "Condition": "Price breakout",
                "Pass": bool(row.get("price_breakout", False)),
                "Value": row.get("close"),
                "Rule": "close > previous Donchian high",
            },
            {
                "Condition": "Volume confirmed",
                "Pass": bool(row.get("volume_confirmed", False)),
                "Value": row.get("volume"),
                "Rule": "volume > volume SMA × multiplier",
            },
            {
                "Condition": "Asset bullish regime",
                "Pass": bool(row.get("asset_bull_regime", False)),
                "Value": row.get("close"),
                "Rule": "close > EMA 200 and EMA 50 > EMA 200",
            },
            {
                "Condition": "BTC bullish regime",
                "Pass": bool(row.get("btc_bull_regime", False)),
                "Value": row.get("btc_close"),
                "Rule": "BTC close > BTC EMA 200",
            },
        ]
    )

    st.dataframe(checks, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Explainer tab
# ─────────────────────────────────────────────────────────────────────────────

def render_how_it_works_tab() -> None:
    st.subheader("How the program works")
    st.caption(
        "This tab explains the full pipeline, from user inputs to downloaded data, "
        "indicator calculation, signal generation, and visualization."
    )

    st.markdown(
        """
        <style>
        .flowchart-wrapper {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        .flow-box {
            border: 1px solid rgba(120, 120, 120, 0.35);
            border-radius: 12px;
            padding: 14px 16px;
            background: rgba(250, 250, 250, 0.03);
        }
        .flow-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 6px;
        }
        .flow-body {
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .flow-arrow {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            opacity: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "1. User chooses research inputs",
            "From the sidebar, you select the assets, timeframe, amount of history, "
            "and strategy parameters such as volatility threshold, volume confirmation, "
            "ATR stop distance, and take-profit multiple.",
        ),
        (
            "2. Historical OHLCV candles are downloaded from Coinbase",
            "The dashboard calls the Coinbase historical candles downloader and stores "
            "the data locally in data/raw. Each row contains time, open, high, low, close, and volume.",
        ),
        (
            "3. Indicators are calculated",
            "The app computes the research features required by the strategy: log returns, "
            "realized volatility, volatility percentile, EMA 50, EMA 200, ATR 14, "
            "Donchian highs/lows, and volume SMA 20.",
        ),
        (
            "4. BTC market regime is added",
            "BTC is used as a market filter. For altcoin long signals, the system requires "
            "BTC to be in a bullish regime, which reduces the chance of buying weak altcoin breakouts "
            "during broad crypto market weakness.",
        ),
        (
            "5. Strategy rules are applied",
            "A long signal appears only when the market is quiet, price breaks out "
            "above its recent range, volume confirms the move, the asset trend is bullish, and BTC is bullish.",
        ),
        (
            "6. Risk levels are computed",
            "For each signal, the dashboard calculates a suggested entry price, ATR-based stop price, "
            "risk per unit, and take-profit level. These are preparation steps for the future backtest engine.",
        ),
        (
            "7. Everything is visualized",
            "The app renders interactive charts for price, trend lines, breakout levels, volatility, "
            "volume, and signal markers. You can inspect the signal table, diagnostics, and raw feature data.",
        ),
        (
            "8. Outputs are saved",
            "Generated signals and signal summaries are saved in data/signals so the research process stays reproducible.",
        ),
        (
            "9. Future step: backtesting",
            "The dashboard currently explains and visualizes signals. The next major upgrade is a backtest engine "
            "that turns signals into simulated trades with fees, slippage, equity curves, and drawdown analysis.",
        ),
    ]

    html = '<div class="flowchart-wrapper">'
    for i, (title, body) in enumerate(steps):
        html += f"""
        <div class="flow-box">
            <div class="flow-title">{title}</div>
            <div class="flow-body">{body}</div>
        </div>
        """
        if i < len(steps) - 1:
            html += '<div class="flow-arrow">↓</div>'
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Signal logic in plain English")
    st.markdown(
        """
A **long signal** is generated only when **all** of the following are true:

1. **Volatility compression:** current realized volatility is low relative to its recent history  
2. **Breakout:** price closes above the previous 20-bar high  
3. **Volume confirmation:** current volume is above the 20-bar average volume  
4. **Asset trend filter:** the asset is above EMA 200 and EMA 50 is above EMA 200  
5. **BTC market filter:** BTC is above its own EMA 200  

This avoids the common beginner mistake of buying low volatility blindly.  
**Low volatility alone has no direction.** The breakout tells you *which way* the move is trying to happen.
        """
    )

    st.code(
        """long_signal = (
    (realized_vol_percentile_200 <= max_vol_percentile)
    and (close > donchian_high_20)
    and (volume > volume_sma_20 * volume_multiplier)
    and (close > ema_200)
    and (ema_50 > ema_200)
    and (btc_close > btc_ema_200)
)""",
        language="python",
    )

    st.subheader("Parameter guide")
    parameter_reference = pd.DataFrame(
        [
            {
                "Parameter": "Timeframe",
                "Why it matters": "Controls how noisy or smooth the data is.",
                "Typical starting range": "1h to 4h",
                "Interpretation": "Lower timeframes = more signals and more noise. Higher timeframes = fewer but cleaner signals.",
            },
            {
                "Parameter": "Historical days",
                "Why it matters": "Determines how much history is available for indicators and testing.",
                "Typical starting range": "180 to 730 days",
                "Interpretation": "Too little history makes the results fragile and can starve long-window indicators.",
            },
            {
                "Parameter": "Max volatility percentile",
                "Why it matters": "Defines what counts as compression.",
                "Typical starting range": "10 to 30",
                "Interpretation": "Lower = stricter compression. Higher = more signals but lower quality.",
            },
            {
                "Parameter": "Volume multiplier",
                "Why it matters": "Confirms whether the breakout has enough participation.",
                "Typical starting range": "0.8 to 1.5",
                "Interpretation": "Higher = stronger confirmation but fewer trades.",
            },
            {
                "Parameter": "Stop ATR multiplier",
                "Why it matters": "Sets stop distance in a volatility-adjusted way.",
                "Typical starting range": "1.5 to 3.0",
                "Interpretation": "Lower = tighter stops and more stop-outs. Higher = looser stops and smaller position sizes later.",
            },
            {
                "Parameter": "Take-profit R multiple",
                "Why it matters": "Defines reward target relative to initial risk.",
                "Typical starting range": "1.5 to 4.0",
                "Interpretation": "Higher targets can improve payoff per winner but reduce hit rate.",
            },
        ]
    )
    st.dataframe(parameter_reference, use_container_width=True, hide_index=True)

    st.subheader("Suggested research workflow")
    st.markdown(
        """
1. Start with **BTC-USD, ETH-USD, SOL-USD**  
2. Use **1h timeframe** and **365 days** of history  
3. Start with:
   - max volatility percentile = **20**
   - volume multiplier = **1.0**
   - stop ATR multiplier = **2.0**
   - take-profit R multiple = **3.0**
4. Inspect where signals appear on the chart  
5. Only after that, build the backtest and check whether the idea actually makes money  
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────

st.title("Crypto Volatility Strategy Dashboard")
st.caption(
    "Research dashboard for a realized-volatility compression breakout strategy. "
    "This is a research tool, not a trading bot."
)

with st.sidebar:
    st.header("Controls")

    selected_products = st.multiselect(
        "Products",
        options=PRODUCT_IDS,
        default=PRODUCT_IDS,
        help=(
            "Assets included in the research universe. "
            "BTC-USD must remain selected because BTC is used as the market regime filter "
            "for the other assets."
        ),
    )

    if "BTC-USD" not in selected_products:
        st.warning("BTC-USD is required because the strategy uses BTC as a market filter.")

    timeframe = st.selectbox(
        "Timeframe",
        options=list(TIMEFRAME_TO_GRANULARITY.keys()),
        index=list(TIMEFRAME_TO_GRANULARITY.keys()).index("1h"),
        help=(
            "The candle duration used for analysis. "
            "Typical starting range: 1h to 4h. "
            "Shorter timeframes create more signals but also more noise."
        ),
    )

    granularity = TIMEFRAME_TO_GRANULARITY[timeframe]

    days = st.number_input(
        "Historical days to download",
        min_value=1,
        max_value=2000,
        value=365,
        step=30,
        help=(
            "How far back to download historical candles. "
            "Typical starting range: 180 to 730 days. "
            "More history improves research quality, but also increases load time."
        ),
    )

    st.divider()
    st.subheader("Strategy Parameters")

    max_vol_percentile = st.slider(
        "Max realized volatility percentile",
        min_value=1.0,
        max_value=80.0,
        value=20.0,
        step=1.0,
        help=(
            "Defines what counts as low-volatility compression. "
            "A value of 20 means current volatility must be in the lowest 20% of its recent history. "
            "Typical starting range: 10 to 30. Lower values are stricter."
        ),
    )

    volume_multiplier = st.slider(
        "Volume multiplier",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help=(
            "Breakout volume must be above volume_sma_20 × this multiplier. "
            "Typical starting range: 0.8 to 1.5. "
            "Higher values demand stronger confirmation and reduce false breakouts."
        ),
    )

    stop_atr_multiplier = st.slider(
        "Stop ATR multiplier",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.25,
        help=(
            "ATR-based stop distance. A value of 2.0 means the initial stop is 2 × ATR below entry. "
            "Typical starting range: 1.5 to 3.0. Lower = tighter stops, higher = looser stops."
        ),
    )

    take_profit_r_multiple = st.slider(
        "Take-profit R multiple",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.25,
        help=(
            "Reward target measured in units of initial risk. "
            "A value of 3.0 means take-profit = entry + 3 × risk_per_unit. "
            "Typical starting range: 1.5 to 4.0."
        ),
    )

    st.divider()
    st.subheader("Actions")

    download_button = st.button(
        "Download / refresh historical candles",
        type="primary",
        help=(
            "Downloads Coinbase historical OHLCV data and saves it into data/raw. "
            "Use this whenever you change assets, timeframe, or history length."
        ),
    )

    generate_button = st.button(
        "Generate signals",
        help=(
            "Runs the feature engineering and strategy logic on the currently available candle data, "
            "then saves the signal tables into data/signals."
        ),
    )

    clear_cache_button = st.button(
        "Clear dashboard cache",
        help=(
            "Clears Streamlit's cached data. Useful when you downloaded fresh data "
            "but the dashboard still appears to show old results."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────

if clear_cache_button:
    clear_data_caches()
    st.success("Dashboard cache cleared.")

if download_button:
    if not selected_products:
        st.error("Select at least one product.")
    elif "BTC-USD" not in selected_products:
        st.error("BTC-USD must be selected because the strategy uses BTC as a market filter.")
    else:
        end = utc_now()
        start = end - timedelta(days=int(days))

        config = DownloadConfig(
            product_ids=selected_products,
            start=start,
            end=end,
            granularity=granularity,
            output_dir=RAW_DATA_DIR,
        )

        with st.spinner("Downloading Coinbase historical candles..."):
            saved_files = download_all(config)
            clear_data_caches()

        st.success("Download complete.")
        st.write(saved_files)

if generate_button:
    try:
        with st.spinner("Loading candles and generating signals..."):
            candles_for_generation = cached_load_raw_candles(str(RAW_DATA_DIR), timeframe)
            candles_for_generation = candles_for_generation[
                candles_for_generation["product_id"].isin(selected_products)
            ].copy()

            if "BTC-USD" not in candles_for_generation["product_id"].unique():
                st.error("BTC-USD data is missing. Download BTC-USD candles first.")
            else:
                features_for_generation = cached_features(candles_for_generation)
                signal_data = cached_signals(
                    features_for_generation,
                    max_vol_percentile,
                    volume_multiplier,
                    stop_atr_multiplier,
                    take_profit_r_multiple,
                )

                SIGNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                signals_path = SIGNAL_DATA_DIR / f"volatility_breakout_signals_{timeframe}.csv"
                summary_path = SIGNAL_DATA_DIR / f"volatility_breakout_summary_{timeframe}.csv"

                signal_data.to_csv(signals_path, index=False)
                summarize_signals(signal_data).to_csv(summary_path, index=False)

                st.success("Signals generated.")
                st.write(f"Saved signals to `{signals_path}`")
                st.write(f"Saved summary to `{summary_path}`")

    except Exception as exc:
        st.error(f"Signal generation failed: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────────────────────

try:
    candles = cached_load_raw_candles(str(RAW_DATA_DIR), timeframe)
    candles = candles[candles["product_id"].isin(selected_products)].copy()

    if candles.empty:
        st.warning("No candles found for the selected products. Download data first.")
        st.stop()

    features = cached_features(candles)

    signals = cached_signals(
        features,
        max_vol_percentile,
        volume_multiplier,
        stop_atr_multiplier,
        take_profit_r_multiple,
    )

except FileNotFoundError:
    st.info(
        "No historical candle files found yet. Use the sidebar button to download candles."
    )
    st.stop()
except Exception as exc:
    st.error(f"Dashboard failed to load data: {type(exc).__name__}: {exc}")
    st.stop()

available_products = sorted(signals["product_id"].unique())

if not available_products:
    st.warning("No products available after filtering.")
    st.stop()

selected_chart_product = st.selectbox(
    "Asset to visualize",
    options=available_products,
    index=0,
    help=(
        "Choose which asset to inspect in the charts and data tables. "
        "This does not change the signal generation logic itself; it only changes what you are looking at."
    ),
)

summary = summarize_signals(signals)
product_data = signals[signals["product_id"] == selected_chart_product].copy()
product_signals = product_data[product_data["long_signal"]].copy()

latest_row = product_data.sort_values("time").tail(1).iloc[0]

metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{len(product_data):,}")
metric_cols[1].metric("Long signals", f"{int(product_data['long_signal'].sum()):,}")
metric_cols[2].metric("Latest close", f"{latest_row['close']:,.4f}")
metric_cols[3].metric(
    "Latest vol percentile",
    "N/A" if pd.isna(latest_row["realized_vol_percentile_200"]) else f"{latest_row['realized_vol_percentile_200']:.2f}",
)
metric_cols[4].metric(
    "BTC bull regime",
    str(bool(latest_row.get("btc_bull_regime", False))),
)

tab_price, tab_volatility, tab_signals, tab_data, tab_explainer = st.tabs(
    ["Price & Signals", "Volatility", "Signals", "Data", "How It Works"]
)

with tab_price:
    st.caption(
        "This chart shows price candles, trend filters, breakout levels, and long signal markers. "
        "Use it to visually inspect whether the strategy is firing in sensible places."
    )

    st.plotly_chart(
        make_price_chart(signals, selected_chart_product),
        use_container_width=True,
    )

    st.subheader("Current Signal Diagnostics")
    st.write(
        "This shows why the latest candle does or does not qualify for a long signal."
    )
    show_signal_diagnostics(signals, selected_chart_product)

with tab_volatility:
    st.caption(
        "These charts explain the volatility-compression part of the strategy. "
        "The percentile chart is usually more useful than raw volatility because it compares each asset against its own history."
    )

    st.plotly_chart(
        make_volatility_chart(signals, selected_chart_product, max_vol_percentile),
        use_container_width=True,
    )

    st.plotly_chart(
        make_realized_vol_chart(signals, selected_chart_product),
        use_container_width=True,
    )

    st.plotly_chart(
        make_volume_chart(signals, selected_chart_product),
        use_container_width=True,
    )

with tab_signals:
    st.caption(
        "This tab lists the actual signal rows generated by the strategy. "
        "It helps you check how often the model fires and whether the signals cluster in sensible regimes."
    )

    st.subheader("Signal Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader(f"{selected_chart_product} Long Signals")

    if product_signals.empty:
        st.warning("No long signals found for this asset with the current parameters.")
    else:
        columns_to_show = [
            "time",
            "product_id",
            "close",
            "realized_vol_percentile_200",
            "donchian_high_20",
            "volume",
            "volume_sma_20",
            "entry_price",
            "stop_price",
            "take_profit_price",
            "btc_bull_regime",
        ]
        st.dataframe(
            product_signals[columns_to_show].sort_values("time", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = product_signals.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download this asset's signals as CSV",
            data=csv_bytes,
            file_name=f"{selected_chart_product}_signals_{timeframe}.csv",
            mime="text/csv",
            help="Downloads only the signal rows for the currently selected asset.",
        )

with tab_data:
    st.caption(
        "This tab is for debugging and verification. "
        "Use it to inspect raw candles and the feature columns used by the strategy."
    )

    st.subheader("Raw Candles Preview")
    st.dataframe(
        candles[candles["product_id"] == selected_chart_product]
        .sort_values("time", ascending=False)
        .head(500),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Feature Data Preview")
    preview_cols = [
        "time",
        "product_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "realized_vol_20",
        "realized_vol_percentile_200",
        "ema_50",
        "ema_200",
        "atr_14",
        "donchian_high_20",
        "volume_sma_20",
        "long_signal",
    ]

    existing_preview_cols = [c for c in preview_cols if c in signals.columns]

    st.dataframe(
        signals[signals["product_id"] == selected_chart_product]
        [existing_preview_cols]
        .sort_values("time", ascending=False)
        .head(500),
        use_container_width=True,
        hide_index=True,
    )

with tab_explainer:
    render_how_it_works_tab()

st.divider()
st.caption(
    "Next serious upgrade: add a backtest engine so each signal becomes a simulated trade "
    "with entry, stop, trailing stop, fees, slippage, equity curve, and drawdown."
)
