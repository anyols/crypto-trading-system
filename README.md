# Crypto Volatility & Momentum Strategy Research

Systematic research framework for testing short-horizon crypto momentum, reversal, volatility-regime, trend, and volume signals.

This project started as a short-window crypto trading challenge, but evolved into a broader systematic research project focused on one question:

> Do short-horizon crypto momentum and reversal signals become more robust when conditioned on volatility, trend, volume, and market regime?

The project is not presented as a finished trading bot. It is a research framework demonstrating data ingestion, feature engineering, strategy testing, overfitting control, ablation analysis, ensemble construction, benchmark comparison, and walk-forward validation.

---

## Key Results

In the final research run:

| Result                        | Summary                                                                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Best exploratory strategy     | **Competition V3 Strategy**, strong result but explicitly labeled as selection-biased                                            |
| Best systematic grid strategy | **Below-Trend Momentum Continuation**, around **+19% return**, Sharpe above **2**, max drawdown around **-3%**                   |
| Best fixed ensemble Sharpe    | **Clean Strategies Equal-Weight**, Sharpe around **2.4**, controlled drawdown                                                    |
| Best walk-forward portfolio   | **Walk-Forward Rank 1**, around **+87% compounded return** across 2024–2026, 100% positive years, average Sharpe around **2.56** |
| Benchmark comparison          | Buy-and-hold had much larger drawdowns and weaker consistency                                                                    |

The main conclusion is that short-horizon crypto extreme-move strategies showed persistent signal, especially on the long side. Volatility, trend, and volume filters improved robustness in several cases.

---

## Methodology

The project separates results into several research layers:

1. **Exploratory strategy discovery**  
   Initial strategy search, including the Competition V3 Strategy. Strong results are shown, but selection-bias risk is explicitly acknowledged.

2. **Systematic grid-derived strategies**  
   Rule-based strategy candidates generated from structured combinations of momentum, volatility, trend, and volume filters.

3. **Clean rediscovered strategies**  
   Candidates selected using only pre-holdout data, then tested on the final holdout.

4. **Ablation baselines**  
   Simpler versions of strategies used to test whether volatility or trend filters actually add value.

5. **Fixed clean ensembles**  
   Equal-weight portfolios of non-V3 clean strategies. No holdout-period weight optimization is used.

6. **Walk-forward validation**  
   Each test year uses only prior data for strategy selection, then tests selected candidates on the next unseen year.

7. **Walk-forward ensembles**  
   Combines the top walk-forward-selected candidates into equal-weight and score-weighted portfolios.

---

## Strategy Families Tested

The research evaluates:

- Selloff rebound strategies
- Upside momentum burst strategies
- Below-trend continuation strategies
- Low-volatility short overextension strategies
- Momentum-only baselines
- No-volatility-filter ablations
- Fixed clean strategy ensembles
- Walk-forward selected strategies
- Equal-weight buy-and-hold benchmark

---

## Repository Structure

```text
src/
    Python modules for data download, feature engineering, strategy research,
    backtesting, reporting, walk-forward validation, and ensemble analysis.

notebooks/
    Main technical research notebook.

reports/
    Generated reports and summaries. Large rendered HTML/PDF files are not
    committed by default.

data/
    Local data and generated backtest outputs. Raw data and large generated
    outputs are ignored by Git.
```

---

## Main Notebook

The main research report is:

```text
notebooks/crypto_volatility_momentum_research_v2_professional.ipynb
```

The notebook includes:

- strategy definitions,
- final holdout results,
- equity and drawdown charts,
- rolling return analysis,
- ablation tests,
- fixed clean ensembles,
- walk-forward validation,
- walk-forward ensemble analysis,
- limitations,
- final conclusion.

The committed notebook is stripped of outputs to keep the repository lightweight.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Example Commands

### Download historical data

```bash
python -m src.historical_data --days 1825 --granularity 900 --output-dir data/raw_5y_15m
```

### Run final strategy backtest

```bash
python -m src.final_strategy_backtest \
  --input-dir data/raw_5y_15m \
  --timeframe 15m \
  --holdout-days 365 \
  --initial-equity 100000 \
  --position-pct 0.20 \
  --max-gross-leverage 1.0 \
  --fee-rate 0.0001 \
  --slippage-rate 0 \
  --output-dir data/final_strategy_backtests_all_candidates_nextbar
```

PowerShell version:

```powershell
py -m src.final_strategy_backtest `
  --input-dir data\raw_5y_15m `
  --timeframe 15m `
  --holdout-days 365 `
  --initial-equity 100000 `
  --position-pct 0.20 `
  --max-gross-leverage 1.0 `
  --fee-rate 0.0001 `
  --slippage-rate 0 `
  --output-dir data\final_strategy_backtests_all_candidates_nextbar
```

### Run walk-forward strategy research

```powershell
py -m src.walkforward_strategy_research `
  --input-dir data\raw_5y_15m `
  --output-dir data\walkforward_2024_v3 `
  --first-test-year 2024 `
  --last-test-year 2024 `
  --lookbacks 4 6 12 24 48 `
  --horizons 60 120 240 480 720 `
  --filters none asset_above_ema200 asset_below_ema200 btc_above_ema200 btc_below_ema200 volume_z_gt_0 `
  --top-n 5
```

### Run walk-forward ensemble report

```powershell
py -m src.walkforward_ensemble_report `
  --input-root data `
  --folder-template "walkforward_{year}_v3" `
  --years 2024 2025 2026 `
  --output-dir reports\walkforward_ensemble_v3 `
  --initial-equity 100000
```

### Run fixed clean strategy ensemble report

```powershell
py -m src.fixed_strategy_ensemble_report `
  --backtest-dir data\final_strategy_backtests_all_candidates_nextbar `
  --output-dir reports\fixed_strategy_ensemble_clean `
  --initial-equity 100000
```

---

## Validation Assumptions

The backtests use:

- chronological splits,
- next-bar execution,
- transaction costs,
- fixed position sizing,
- max gross leverage constraint,
- no random train/test shuffling,
- separate treatment of exploratory and cleaner results.

---

## Limitations

This is historical research, not a production trading system.

Important limitations:

- OHLCV-only data,
- no order book simulation,
- limited slippage modeling,
- crypto regime dependence,
- some strategies have low trade counts,
- exploratory V3 strategy has selection-bias risk,
- live execution may differ from candle-based backtests.

---

## Disclaimer

This project is for research and educational purposes only. It is not financial advice or investment advice.
