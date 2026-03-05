# Trading Strategy Backtesting Methodology

> Classification: CONFIDENTIAL — Prepared by Quantitative Research Division

## 1. Overview

Comprehensive quantitative analysis of a systematic trading strategy on Bitcoin 15-minute price prediction markets. This methodology documents the data specifications, cost structure, validation framework, and risk management approach used throughout the backtesting process.

## 2. Market Structure

| Parameter | Value |
|---|---|
| **Asset** | Bitcoin/USDT |
| **Timeframe** | 15-minute candles |
| **Market Type** | Binary prediction (UP/DOWN) |
| **Position Duration** | 15 minutes (single candle) |
| **Market Hours** | 24/7 continuous |
| **Exchange** | Polymarket (prediction market) |

## 3. Data Specifications

| Parameter | Value |
|---|---|
| **Period** | 3 years (February 2023 – February 2026) |
| **Total Candles** | ~105,000 |
| **Data Format** | OHLCV (Open, High, Low, Close, Volume) |
| **Source** | Exchange historical API |
| **Missing Data** | <0.1% (handled via forward-fill) |
| **Data Quality** | Exchange-grade, point-in-time accurate |

## 4. Signal Generation

> ⚠️ **PROPRIETARY**: Strategy logic intentionally omitted to protect intellectual property.

- Entry criteria based on technical indicators (details confidential)
- Exit determined by next candle close (non-discretionary, fully systematic)
- Filters applied to avoid unfavorable market conditions
- No discretionary overlay — 100% rule-based

## 5. Position Sizing

- **Model**: Fixed fractional (non-compounding)
- **Size**: $1.00 per trade
- **Capital Base**: $100
- **Risk per Trade**: 1.02% of baseline capital
- **Max Concurrent**: 1 position
- **Leverage**: None

## 6. Cost Structure

| Cost Component | Value |
|---|---|
| Entry Spread | ~2% (market structure) |
| Platform Fee | 2% on winning payouts |
| Win Payout | +$0.98 per $1.00 bet |
| Loss Payout | -$1.02 per $1.00 bet |
| Breakeven WR | ~50.5% |

## 7. Execution Assumptions

- Signal timing: At candle close (no intra-bar execution)
- Order fill: At close price (conservative assumption)
- Slippage: Included in spread (no additional slippage modeled)
- No latency advantage assumed
- No lookahead bias — signals use only historical data

## 8. Validation Framework

The following institutional-grade statistical tests are performed:

| Test | Purpose |
|---|---|
| Z-Test / T-Test | Statistical significance of edge |
| Monte Carlo (10,000 runs) | Sequence-independent robustness |
| Bootstrap (10,000 samples) | Confidence intervals for all metrics |
| Permutation Tests | Strategy vs random baseline |
| Walk-Forward Analysis | 6-month rolling window stability |
| Out-of-Sample Testing | 3-month holdout validation |
| Deflated Sharpe Ratio | Overfitting adjustment |
| Ljung-Box Test | Return independence / autocorrelation |
| Tail Risk (VaR/CVaR) | Extreme loss characterization |

## 9. Regime Analysis

Market conditions are classified by two dimensions:

- **Volatility**: Low / Medium / High (ATR percentile-based)
- **Trend Strength**: Weak / Medium / Strong (ADX-based)

Performance is analyzed across all 9 regime combinations to ensure robustness.

## 10. Risk Management

- Maximum position size: $1.00 (fixed)
- Stop-loss: Not applicable (binary structure — max loss = $1.02)
- Monthly profit withdrawal strategy available
- Risk of ruin analysis: <0.001% (with $100 capital)
- Kelly Criterion position sizing evaluated

## 11. Limitations & Disclaimers

- Past performance does not guarantee future results
- Execution assumptions may not hold in live trading
- Market microstructure changes not modeled
- Extreme events (black swans) under-represented in 3-year sample
- Tax implications not considered
- Regulatory landscape for prediction markets may change
- Single-market dependency (BTC/Polymarket only)

## 12. Audit Trail

All analysis scripts, data, and results are version-controlled. The complete analysis is reproducible via:

```bash
python scripts/00_run_complete_analysis.py
```

---

**Classification:** CONFIDENTIAL  
**Prepared by:** Quantitative Research Division  
**Last Updated:** March 2026
