# Experiment Log

This file records strategy experiments, results, diagnosis, and decisions.  
The goal is to avoid cherry-picking and preserve the research process.

---

## Experiment Family 1 — Volatility Compression Breakout

### Hypothesis

A crypto asset that is in a bullish trend, while BTC is also bullish, may produce profitable continuation moves after a period of realized-volatility compression followed by an upside breakout.

The original model attempted to capture:

- volatility compression
- upside breakout
- volume confirmation
- asset bullish trend
- BTC bullish trend
- ATR-based risk management

### Data

- Source: Coinbase historical OHLCV candles
- Assets: BTC-USD, ETH-USD, SOL-USD, DOGE-USD, XRP-USD
- Timeframe: 1h
- Approximate test window: 365 days
- Initial equity: 10,000 per independent asset backtest
- Fee assumption: 0.6% per side
- Slippage assumption: 0.1% per side
- Position sizing: risk-based sizing, 0.5% equity risk per trade

### Main Strategy Versions Tested

#### A — Full Original Model

Entry:

- realized volatility percentile <= 30
- close > previous 20-bar Donchian high
- volume > 0.8 × volume SMA 20
- close > EMA 200
- EMA 50 > EMA 200
- BTC close > BTC EMA 200

Exit:

- 2 ATR initial stop
- fixed 3R take-profit

Result:

- Average return: -30.11%
- Average buy-and-hold return: +125.82%
- Average excess return: -155.94%
- Total trades: 349
- Average profit factor: 0.32
- Average max drawdown: -30.63%

Decision:

Rejected. The full original model lost heavily and massively underperformed buy-and-hold.

---

#### B — Breakout Only

Entry:

- close > previous 20-bar Donchian high

Exit:

- 2 ATR initial stop
- fixed 3R take-profit

Result:

- Average return: -77.68%
- Average buy-and-hold return: +125.82%
- Average excess return: -203.50%
- Total trades: 1,955
- Average profit factor: 0.41
- Average max drawdown: -77.84%

Diagnosis:

The raw Donchian breakout trigger has strongly negative performance under these assumptions. It creates excessive churn and fee drag.

Decision:

Rejected. Donchian breakout alone is not usable as the main entry trigger.

---

#### L — Full Model With Wider Stop

Entry:

- same as full original model

Exit:

- 3 ATR initial stop
- fixed 3R take-profit

Result:

- Average return: -15.82%
- Average buy-and-hold return: +125.82%
- Average excess return: -141.65%
- Total trades: 298
- Average profit factor: 0.57
- Average max drawdown: -18.36%

Diagnosis:

Wider stops helped, suggesting the 2 ATR stop was too tight for crypto noise. However, the strategy still lost money and underperformed heavily.

Decision:

Rejected as final model. Useful finding: wider ATR stops are less damaging than tight stops.

---

#### O — Full Model With EMA200 Exit

Entry:

- volatility compression
- Donchian breakout
- volume confirmation
- asset bullish trend
- BTC bullish trend

Exit:

- 3 ATR initial stop
- no fixed take-profit
- exit when close < EMA 200

Result:

- Average return: +4.39%
- Median return: -5.88%
- Average buy-and-hold return: +125.82%
- Average excess return: -121.43%
- Total trades: 258
- Average win rate: 16.65%
- Average profit factor: 1.09
- Average max drawdown: -21.43%
- Average Sharpe ratio: -0.16

Diagnosis:

The EMA200 exit materially improved performance compared with fixed take-profit exits. This suggests fixed R-multiple targets cut off upside too early. However, the result is still not strong enough and remains far below buy-and-hold.

Decision:

Rejected as final model, but useful finding: slow trend-following exits are better than fixed take-profit exits for this setup.

---

### Final Entry Iteration

After discovering that EMA200 exits helped, the next test kept the 3 ATR initial stop + EMA200 exit and tested alternative entries.

#### T — Low Volatility + Trend + BTC Trend

Entry:

- low realized-volatility percentile
- asset bullish trend
- BTC bullish trend
- no Donchian breakout

Exit:

- 3 ATR initial stop
- close < EMA 200

Result:

- Average return: -18.46%
- Average excess return: -144.29%
- Total trades: 729
- Average profit factor: 0.73
- Average max drawdown: -32.02%

Decision:

Rejected.

---

#### U — Trend Only + BTC Trend

Entry:

- asset bullish trend
- BTC bullish trend
- no volatility filter
- no breakout filter
- no volume filter

Exit:

- 3 ATR initial stop
- close < EMA 200

Result:

- Average return: -33.08%
- Average excess return: -158.90%
- Total trades: 1,202
- Average profit factor: 0.62
- Average max drawdown: -39.92%

Decision:

Rejected.

---

#### V — Pullback Reclaim + EMA200 Exit

Entry:

- asset bullish trend
- BTC bullish trend
- RSI reclaim above 50
- no Donchian breakout

Exit:

- 3 ATR initial stop
- close < EMA 200

Result:

- Average return: -19.00%
- Average excess return: -144.82%
- Total trades: 720
- Average profit factor: 0.71
- Average max drawdown: -30.33%

Decision:

Rejected.

---

#### W — EMA50 Breakout + EMA200 Exit

Entry:

- Donchian breakout
- close > EMA 50
- BTC bullish trend

Exit:

- 3 ATR initial stop
- close < EMA 200

Result:

- Average return: -31.21%
- Average excess return: -157.03%
- Total trades: 1,190
- Average profit factor: 0.66
- Average max drawdown: -43.74%

Decision:

Rejected.

---

## Overall Diagnosis

The volatility-compression breakout strategy family failed.

Main findings:

1. Plain Donchian breakout was extremely weak.
2. Volatility compression helped reduce some damage but did not create enough edge.
3. Wider ATR stops improved results but did not make the strategy profitable.
4. Fixed R-multiple take-profits were inferior to slower trend-following exits.
5. EMA200 exit was the best exit tested, but still massively underperformed buy-and-hold.
6. Alternative entries did not rescue the strategy.
7. Frequent trading created heavy fee drag.
8. The strategy failed to capture major crypto upside.

## Final Decision

Reject this strategy family.

The next research direction is volatility-regime allocation:

- hold exposure during favorable trend/volatility regimes
- reduce exposure or move to cash during unfavorable regimes
- prioritize fewer trades, lower fee drag, and capturing large crypto trends
