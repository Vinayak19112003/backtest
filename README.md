# Polymarket BTC 15m Prediction Market — Quantitative Analysis Suite

> Institutional-grade backtesting and statistical validation framework

## 📊 Overview

Comprehensive quantitative analysis of a systematic trading strategy on Polymarket's Bitcoin 15-minute price prediction markets. This repository contains professional-grade backtesting, forensic statistical validation, and risk analysis tools meeting institutional research standards.

**Target Audience:** Quantitative analysts, institutional investors, hedge fund researchers

## 🎯 Strategy Summary

| Parameter | Value |
|---|---|
| **Market** | Binary options (15-minute BTC price prediction) |
| **Approach** | Systematic technical strategy (proprietary signals) |
| **Sample Period** | 3 years (2023–2026) |
| **Total Trades** | 40,000+ historical executions |
| **Position Sizing** | Fixed fractional ($100 base capital) |

## 📁 Repository Structure

```
backtest/
├── data/                          # Market data (OHLCV)
│   └── BTCUSDT_15m_3_years.csv
├── scripts/                       # Analysis suite
│   ├── 00_run_complete_analysis.py    # Master orchestrator
│   ├── 01_generate_performance_report.py
│   ├── 02_generate_fixed_capital_analysis.py
│   ├── 03_generate_forensic_validation.py
│   └── 04_generate_visualizations.py
├── reports/                       # Generated analysis (auto)
│   ├── performance/
│   ├── fixed_capital/
│   ├── forensic_validation/
│   └── MASTER_SUMMARY.txt
├── visualizations/                # 26 professional charts (auto)
├── results/                       # Legacy reports
├── requirements.txt
├── METHODOLOGY.md
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Complete Analysis
```bash
python scripts/00_run_complete_analysis.py
```

### Command-Line Options
```bash
python scripts/00_run_complete_analysis.py --quick         # Skip charts (faster)
python scripts/00_run_complete_analysis.py --report-only   # Reports 1 & 2 only
python scripts/00_run_complete_analysis.py --forensic-only # Forensic validation only
python scripts/00_run_complete_analysis.py --help          # Show usage
```

### Individual Reports
```bash
python scripts/01_generate_performance_report.py     # Performance metrics
python scripts/02_generate_fixed_capital_analysis.py  # Capital management
python scripts/03_generate_forensic_validation.py     # Statistical tests
python scripts/04_generate_visualizations.py          # 26 charts
python scripts/05_generate_q3_filter_analysis.py      # Q3 filtered metrics
python scripts/06_generate_q3_filter_visualizations.py # 6 Q3 charts
```

## 📈 Analysis Components

### 1. Performance Report
- 40,000+ trade analysis with risk-adjusted returns (Sharpe, Sortino, Calmar)
- Temporal patterns (hourly, daily, monthly breakdown)
- Regime-based performance (volatility × trend strength)
- Statistical significance (Z-test, p-values)

### 2. Fixed Capital Analysis
- $100 fixed capital simulation with monthly profit withdrawals
- Risk of ruin calculation (Kelly criterion)
- 4 alternative withdrawal strategies compared
- Compounding vs. fixed capital trade-offs

### 3. Forensic Validation ⭐
Institutional-grade statistical validation with **8-test scorecard**:

| Test | Method |
|---|---|
| Statistical Significance | Z-test, t-test (p < 0.001 threshold) |
| Monte Carlo Simulation | 10,000 path randomizations |
| Bootstrap Resampling | 10,000 samples with replacement |
| Permutation Tests | Strategy vs random baseline (>3σ) |
| Walk-Forward Analysis | 6-month rolling windows |
| Out-of-Sample Testing | 3-month holdout validation |
| Overfitting Diagnostics | Deflated Sharpe Ratio (DSR > 2.0) |
| Autocorrelation Tests | Ljung-Box (return independence) |

### 4. Visualization Suite
26 professional charts with institutional navy/gold palette:
- Equity curves, drawdown analysis, underwater periods
- Daily P&L distribution with normal overlay
- Monthly returns heatmap, rolling Sharpe ratio
- Regime performance heatmaps (9 vol×trend combos)
- Monte Carlo paths, bootstrap distributions, Q-Q plot
- Win/loss streak distributions, hourly/weekly patterns

### 5. Q3 Filter Analysis 🔬
An optional standalone track evaluating the removal of the 30–44 minute block from trading:
- Dual-simulation (baseline vs Q3-filtered)
- Full metrics comparison and statistical testing (Z-test, Bootstrap, KS-test)
- 8 unique reports in `reports/q3_filter/`
- 6 comparative charts in `visualizations/q3_filter/`
- Run with `--include-q3` opt-in flag

## 🔒 Confidentiality

⚠️ **PROPRIETARY STRATEGY:** Entry/exit signals, indicator parameters, and filter logic are intentionally omitted to protect intellectual property.

**Public:** Data processing, statistical validation framework, reporting infrastructure  
**Confidential:** Signal generation logic, filter parameters, thresholds

## 🛡️ Risk Disclosure

- Past performance does not guarantee future results
- Backtest assumes execution at candle close (no additional slippage beyond modeled spread)
- Market microstructure may change over time
- This analysis is for research purposes only

## 📚 Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for detailed documentation of data specifications, cost structure, validation framework, and risk management approach.

## 📄 License

Proprietary — All Rights Reserved

---

**Classification:** Confidential  
**Last Updated:** March 2026  
**Version:** 2.0
