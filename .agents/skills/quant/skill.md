Build a professional-grade backtesting framework for Polymarket BTC 15-minute UP/DOWN prediction markets with institutional validation layers.

═══════════════════════════════════════════════════════════════════════
CONTEXT: Polymarket BTC 15-Minute Markets
═══════════════════════════════════════════════════════════════════════

MARKET STRUCTURE:
- New market every 15 minutes asking: "Will BTC price go UP or DOWN?"
- YES token = predicting price UP in next 15 minutes
- NO token = predicting price DOWN in next 15 minutes
- Settlement: Compare BTC price at market close vs market open
- Trading costs: 2% bid-ask spread + 2% Polymarket fee on wins

DATA REQUIREMENTS:
- BTC/USDT 15-minute OHLCV candles (3+ years historical)
- Each candle represents one complete market cycle
- Signal generated at candle close (time T)
- Outcome determined at next candle close (time T+15min)

═══════════════════════════════════════════════════════════════════════
TIER 1: CORE BACKTESTING ENGINE (BASELINE)
═══════════════════════════════════════════════════════════════════════

1. EVENT-DRIVEN BACKTEST ENGINE
   - Load data chronologically (no lookahead bias)
   - Process flow: Load candle → Calculate indicators → Generate signal → Execute trade → Record outcome
   - Track each 15-min market as separate trade
   - Record: entry_time, signal (YES/NO), entry_price, exit_price, outcome, pnl

2. STRATEGY SIGNAL GENERATION
   Support multiple strategy types:
   - **Technical:** RSI(14), MACD, Bollinger Bands, ADX, ATR
   - **Momentum:** 30m/1h/90m returns, price acceleration
   - **Volume:** OFI (Order Flow Imbalance), VPIN, volume surges
   - **ML Models:** XGBoost, LightGBM with feature engineering
   
   Signal logic:
   - BUY YES: Predict BTC will go UP in next 15 minutes
   - BUY NO: Predict BTC will go DOWN in next 15 minutes
   - Hold to settlement (no early exit for simplicity)

3. REALISTIC EXECUTION SIMULATION
   - Entry cost: 2% spread (buy YES at $0.51 instead of $0.50)
   - Win payout: $1.00 per share minus 2% Polymarket fee = $0.98
   - Loss: Lose entire bet amount
   - Example: Bet $100 on YES → Win: +$94 net | Loss: -$102 net
   - Breakeven win rate: 52% (to cover spread + fees)

4. CORE PERFORMANCE METRICS
   Calculate and display:
   - Total trades, wins, losses, win rate
   - Total PnL, ROI, final capital
   - Average win, average loss, profit factor
   - Maximum drawdown (USD and %), drawdown duration (days)
   - Sharpe ratio, Sortino ratio, Calmar ratio
   - Monthly breakdown: trades per month, monthly PnL, monthly win rate

═══════════════════════════════════════════════════════════════════════
TIER 2: STATISTICAL EDGE VALIDATION (CRITICAL)
═══════════════════════════════════════════════════════════════════════

5. STATISTICAL SIGNIFICANCE TESTING
   Validate that strategy edge is real, not random luck:
   
   **Bootstrap Confidence Intervals:**
   - Resample trades 10,000 times with replacement
   - Calculate 95% confidence interval for mean PnL
   - If CI includes zero, strategy is not statistically significant
   - Require CI lower bound > 0 for validation
   
   **T-Test for Mean Returns:**
   - Null hypothesis: Mean return = 0 (strategy is random)
   - Calculate t-statistic and p-value
   - Reject null if p < 0.05 (95% confidence)
   - Report: "Strategy has {p_value:.4f} probability of being random"
   
   **Minimum Sample Size:**
   - Require at least 100 trades for statistical validity
   - Flag if sample size < 100: "Insufficient data for validation"
   - Calculate required trades for 95% confidence given observed Sharpe
   
   **Randomization Test:**
   - Shuffle trade outcomes 1,000 times randomly
   - Compare actual Sharpe ratio vs distribution of random Sharpes
   - Strategy must beat >95% of random permutations
   - Visual: Histogram of random Sharpes with actual Sharpe marked

6. EDGE DECAY ANALYSIS
   Track if strategy edge is degrading over time:
   
   **Rolling Metrics:**
   - Calculate 30-day rolling win rate
   - Calculate 90-day rolling Sharpe ratio
   - Plot both over time to detect trends
   
   **Time-Bucketed Performance:**
   - Split data into quarterly buckets (Q1, Q2, Q3, Q4)
   - Calculate Sharpe for each quarter
   - Flag if recent quarter Sharpe < historical average by >30%
   
   **Edge Stability Score:**
   - Calculate coefficient of variation (CV) of quarterly Sharpes
   - CV < 0.3 = stable edge
   - CV > 0.5 = unstable/degrading edge
   
   **Regime Shift Detection:**
   - Use rolling t-test to detect structural breaks in performance
   - Alert if edge significantly changed in recent 100 trades

7. SIGNAL QUALITY METRICS
   Measure predictive power of individual signals:
   
   **Information Coefficient (IC):**
   - Calculate correlation between predicted direction and actual outcome
   - IC = corr(predicted_return, actual_return)
   - IC > 0.05 = valuable signal, IC > 0.10 = strong signal
   
   **Feature Importance:**
   - For each indicator (RSI, MACD, OFI, etc.), calculate separate win rate
   - Identify which features contribute most to edge
   - Disable low-IC features to reduce noise
   
   **Calibration Plot:**
   - If using probability predictions (ML models):
   - Bucket predictions: 50-55%, 55-60%, 60-65%, etc.
   - For each bucket, calculate actual win rate
   - Perfect calibration: 60% confidence → 60% actual wins
   - Plot predicted vs actual to detect over/under-confidence

═══════════════════════════════════════════════════════════════════════
TIER 3: REGIME-AWARE BACKTESTING (ADVANCED)
═══════════════════════════════════════════════════════════════════════

8. MARKET REGIME CLASSIFICATION
   Identify different market conditions:
   
   **Volatility Regimes:**
   - Calculate ATR(14) percentile over rolling 96-period window
   - High Vol: ATR > 80th percentile
   - Medium Vol: 20th < ATR < 80th percentile
   - Low Vol: ATR < 20th percentile
   
   **Trend Regimes:**
   - Use ADX(14) to measure trend strength
   - Strong Trend: ADX > 25
   - Weak Trend: 15 < ADX < 25
   - Ranging: ADX < 15
   - Combine with +DI/-DI for direction (up/down/neutral)
   
   **Time-Based Regimes:**
   - US Market Hours (9:30 AM - 4 PM EST)
   - Asia Hours (8 PM - 2 AM EST)
   - Weekend (Saturday-Sunday)
   - Weekday (Monday-Friday)
   
   **Liquidity Regimes (if orderbook data available):**
   - Deep: Spread < 2%, Depth > median
   - Normal: 2% < Spread < 5%
   - Thin: Spread > 5%, Depth < 25th percentile

9. REGIME-SPECIFIC PERFORMANCE ANALYSIS
   Calculate separate metrics for each regime:
   
   **Performance Table by Regime:**
   ```
   Regime            | Trades | Win Rate | Sharpe | Max DD | Profit Factor
   ------------------|--------|----------|--------|--------|---------------
   High Vol / Strong |   234  |   48%    |  0.8   | -18%   |      1.2
   High Vol / Weak   |   456  |   62%    |  2.4   |  -8%   |      2.8
   Low Vol / Ranging |   789  |   55%    |  1.6   | -12%   |      1.9
   Weekend           |   123  |   67%    |  3.1   |  -5%   |      3.5
   ```
   
   **Regime Filtering Rules:**
   - Identify best-performing regimes (Sharpe > 2.0)
   - Identify worst-performing regimes (Sharpe < 0.5 or negative)
   - Create filtered strategy: Only trade in favorable regimes
   - Report improvement: "Filtered Sharpe: 2.8 vs Unfiltered: 1.4"
   
   **Adaptive Position Sizing by Regime:**
   - Increase position size in high-Sharpe regimes
   - Reduce position size in high-volatility regimes
   - Example: 2x size in Weekend + Low Vol, 0.5x size in High Vol + Strong Trend

10. WALK-FORWARD OPTIMIZATION
    Prevent overfitting through proper train/test splits:
    
    **Data Splitting:**
    - Train (In-Sample): First 60% of data
    - Validation: Next 20% of data
    - Test (Out-of-Sample): Final 20% of data
    
    **Rolling Window Validation:**
    - Window size: Train on 60 days, test on next 14 days
    - Roll forward: Shift window by 14 days, repeat
    - Track performance degradation: In-sample Sharpe vs Out-of-sample Sharpe
    
    **Overfitting Detection:**
    - Calculate degradation ratio: Test_Sharpe / Train_Sharpe
    - Healthy: Ratio > 0.7 (test performance is 70%+ of train)
    - Overfit: Ratio < 0.5 (strategy doesn't generalize)
    - Report: "Strategy passes overfitting test: Test Sharpe = 1.8, Train Sharpe = 2.1"

═══════════════════════════════════════════════════════════════════════
TIER 4: EXPECTED VALUE & EDGE DETECTION (ELITE)
═══════════════════════════════════════════════════════════════════════

11. MARKET MISPRICING DETECTION
    Calculate true probability vs market-implied probability:
    
    **Strategy Probability Estimation:**
    - For technical strategies: Use signal strength to estimate probability
    - For ML models: Use model's probability output directly
    - Example: RSI = 25 → P(UP) ≈ 65% based on historical calibration
    
    **Market-Implied Probability:**
    - If YES token = $0.55, NO token = $0.45
    - Market thinks P(UP) = 55%, P(DOWN) = 45%
    
    **Edge Calculation:**
    - Edge = Strategy_P(UP) - Market_P(UP)
    - Edge = 0.65 - 0.55 = 0.10 (10% probability edge)
    - Only trade when |Edge| > threshold (e.g., 5% minimum edge)
    
    **Expected Value Formula:**
    ```
    EV = [P(WIN) × Payout_if_WIN] - [P(LOSE) × Loss_if_LOSE] - Fees
    
    Example:
    Bet $100 on YES when P(UP) = 65%, entry price = $0.51
    Shares = $100 / $0.51 = 196 shares
    
    If WIN: Payout = 196 × $1.00 = $196, minus 2% fee = $192, PnL = $92
    If LOSE: Loss = $100
    
    EV = (0.65 × $92) - (0.35 × $100) = $59.80 - $35.00 = $24.80
    
    EV% = $24.80 / $100 = 24.8% expected return per trade
    ```
    
    **Trade Filter:**
    - Only execute trades where EV > 5%
    - Log rejected trades: "Skipped: EV = 2.3% < threshold"

12. KELLY CRITERION POSITION SIZING
    Optimize bet size based on edge:
    
    **Kelly Formula:**
    ```
    f* = (p × b - q) / b
    
    Where:
    - p = win probability (from strategy)
    - q = 1 - p (loss probability)
    - b = odds received (payout / bet)
    
    Example:
    p = 0.65, q = 0.35
    Win: +$0.92 per $1 bet → b = 0.92
    
    f* = (0.65 × 0.92 - 0.35) / 0.92
    f* = (0.598 - 0.35) / 0.92 = 0.270 = 27% of bankroll
    ```
    
    **Fractional Kelly for Safety:**
    - Full Kelly is aggressive (high volatility)
    - Use Quarter Kelly: f = f* / 4 = 6.75% of bankroll
    - Use Half Kelly: f = f* / 2 = 13.5% of bankroll
    
    **Position Size Calculation:**
    ```python
    kelly_fraction = calculate_kelly(win_prob, avg_win, avg_loss)
    quarter_kelly = kelly_fraction / 4
    
    # Cap at 5% maximum per trade
    position_size = min(quarter_kelly * capital, 0.05 * capital)
    
    # Minimum $1 bet
    position_size = max(position_size, 1.0)
    ```
    
    **Dynamic Sizing by Edge:**
    - Edge 5-8%: Use 0.25x Kelly (conservative)
    - Edge 8-12%: Use 0.5x Kelly (moderate)
    - Edge >12%: Use 1.0x Kelly (aggressive, capped at 5%)

13. CALIBRATION & MODEL VALIDATION
    Ensure probability estimates are accurate:
    
    **Calibration Curve:**
    - Bucket all trades by predicted probability: [50-55%, 55-60%, ..., 90-95%]
    - For each bucket, calculate actual win rate
    - Plot: X-axis = predicted probability, Y-axis = actual win rate
    - Perfect calibration = diagonal line (y = x)
    - Overconfident = curve below diagonal
    - Underconfident = curve above diagonal
    
    **Brier Score:**
    - Measures probability forecast accuracy
    - BS = mean((predicted_prob - actual_outcome)²)
    - Lower is better, perfect = 0
    - BS < 0.2 = well-calibrated, BS > 0.3 = poorly calibrated
    
    **Log Loss:**
    - LL = -mean(actual × log(predicted) + (1-actual) × log(1-predicted))
    - Penalizes confident wrong predictions heavily
    - Use for model selection and hyperparameter tuning

═══════════════════════════════════════════════════════════════════════
TIER 5: CAPACITY & SCALABILITY TESTING
═══════════════════════════════════════════════════════════════════════

14. POSITION SIZE STRESS TESTING
    Test strategy at different capital levels:
    
    **Test Sizes:** [$50, $100, $500, $1K, $5K, $10K]
    
    **Slippage Model:**
    - Base slippage: 2% (bid-ask spread)
    - Additional slippage for large orders:
      ```
      additional_slippage = 0.5% × (order_size / $1000)
      total_slippage = 2% + additional_slippage
      ```
    
    **Efficiency Curve:**
    - Plot: X-axis = position size, Y-axis = Sharpe ratio
    - Identify inflection point where Sharpe starts declining
    - Maximum efficient size = size where Sharpe drops by >20%
    
    **Capacity Report:**
    ```
    Position Size | Sharpe | ROI  | Max DD | Total Slippage
    --------------|--------|------|--------|----------------
    $100          |  2.4   | 180% |  -8%   |      2.0%
    $500          |  2.2   | 165% |  -9%   |      2.1%
    $1000         |  1.9   | 142% | -11%   |      2.3%
    $5000         |  1.3   |  98% | -15%   |      3.2%
    $10000        |  0.8   |  64% | -22%   |      4.5%
    
    Recommended max size: $1000 (before Sharpe degradation)
    ```

15. CONCURRENT POSITION ANALYSIS
    Test strategy with multiple open positions:
    
    **Correlation Risk:**
    - If holding 3 positions within 45 minutes, they're highly correlated
    - All positions likely win/lose together (not diversified)
    - Calculate portfolio-level drawdown vs single-position drawdown
    
    **Optimal Concurrency:**
    - Test max concurrent positions: 1, 3, 5, 10
    - Compare: Total return, Sharpe, max drawdown for each
    - Often optimal = 1-2 positions (due to correlation)
    
    **Position Timing Filter:**
    - Don't open new position if existing position opened < 30 min ago
    - Reduces correlation, improves risk-adjusted returns

═══════════════════════════════════════════════════════════════════════
OUTPUT: COMPREHENSIVE BACKTEST REPORT
═══════════════════════════════════════════════════════════════════════

Generate a professional report with these sections:

1. EXECUTIVE SUMMARY
   - Strategy name and description
   - Data period tested (start date, end date, total days)
   - Overall results: Win rate, Sharpe ratio, Total PnL, Max drawdown
   - Statistical significance: p-value, confidence intervals
   - Verdict: "Strategy passes validation" or "Strategy fails validation"

2. CORE PERFORMANCE METRICS
   - Total trades, wins, losses
   - Win rate, profit factor
   - Average win, average loss
   - Largest win, largest loss
   - Maximum consecutive wins/losses
   - Total PnL, ROI, final capital
   - Sharpe ratio, Sortino ratio, Calmar ratio
   - Maximum drawdown (USD and %), drawdown duration
   - Monthly breakdown table (trades, wins, PnL per month)

3. EDGE VALIDATION RESULTS
   - Bootstrap 95% CI for mean PnL: [$X.XX, $Y.YY]
   - T-test p-value: 0.0234 → "Statistically significant at 95% confidence"
   - Randomization test: "Strategy beats 97.8% of random permutations"
   - Information Coefficient: 0.12 → "Strong predictive signal"
   - Brier score: 0.18 → "Well-calibrated probability estimates"

4. REGIME PERFORMANCE ANALYSIS
   - Table: Performance by volatility regime
   - Table: Performance by trend regime
   - Table: Performance by time of day
   - Table: Performance by day of week
   - Recommendation: "Trade only in [Low Vol + Ranging] for Sharpe 3.2"
   - Regime filter impact: "Filtered Sharpe: 2.8 vs Unfiltered: 1.4 (+100%)"

5. WALK-FORWARD VALIDATION
   - In-sample Sharpe: 2.3
   - Out-of-sample Sharpe: 1.9
   - Degradation ratio: 0.83 → "Strategy generalizes well"
   - Rolling window results: Table of 12 test periods with performance
   - Verdict: "Passes overfitting test"

6. EXPECTED VALUE & EDGE ANALYSIS
   - Average edge per trade: 8.2%
   - Average EV per trade: 14.5%
   - Calibration curve plot (predicted vs actual)
   - High-conviction trades (EV > 20%): 87 trades, 74% win rate, Sharpe 4.1
   - Recommendation: "Focus on trades with edge > 10% for best results"

7. CAPACITY & SCALABILITY
   - Efficiency curve plot (position size vs Sharpe)
   - Recommended max position size: $1000
   - Estimated daily capacity: $5000 (5 trades × $1000)
   - Slippage impact at scale: +0.8% at $1000 size
   - Verdict: "Strategy scales to $1000 per trade without significant degradation"

8. VISUALIZATIONS
   - Equity curve (cumulative PnL over time)
   - Drawdown chart with regime overlays
   - Win/loss distribution histogram
   - Monthly returns heatmap
   - Regime performance heatmap
   - Calibration curve
   - Edge decay plot (rolling Sharpe over time)
   - Efficiency curve (size vs Sharpe)

9. TRADE LOG (CSV Export)
   Columns: timestamp, signal, entry_price, shares, bet_amount, outcome, pnl, capital, regime_vol, regime_trend, edge, ev, win_rate_recent, kelly_fraction

10. STRATEGY IMPROVEMENT RECOMMENDATIONS
    Based on validation results, suggest:
    - Parameter adjustments
    - Regime filters to add
    - Position sizing modifications
    - Additional signals to consider
    - Risk management enhancements

═══════════════════════════════════════════════════════════════════════
IMPLEMENTATION REQUIREMENTS
═══════════════════════════════════════════════════════════════════════

CODE STRUCTURE:
```
backtest/
├── data_loader.py          # Load BTC OHLCV data
├── indicators.py           # RSI, MACD, ATR, ADX, OFI, VPIN
├── strategy.py             # Signal generation logic
├── backtester.py           # Event-driven engine
├── metrics.py              # Performance calculations
├── edge_validator.py       # Statistical tests, bootstrap, IC
├── regime_detector.py      # Volatility/trend classification
├── ev_analyzer.py          # Expected value, Kelly sizing
├── capacity_tester.py      # Position size stress tests
├── visualizer.py           # Charts and plots
├── reporting.py            # Generate final report
└── run_backtest.py         # Main execution script
```

CONFIGURATION:
- Initial capital: $10,000
- Position size: Dynamic (Kelly Criterion) or Fixed ($100)
- Entry slippage: 2% base + market impact
- Fees: 2% on wins
- Min edge threshold: 5%
- Regime lookback: 96 periods
- Bootstrap iterations: 10,000
- Walk-forward: 60-day train, 14-day test

CRITICAL RULES:
- No lookahead bias (only use data available at signal time)
- All indicators must warm up properly (drop NaN values)
- Kelly sizing requires minimum 20-trade history
- Regime filters must be backtested separately (not optimized on full data)
- Statistical significance required (p < 0.05) for validation

═══════════════════════════════════════════════════════════════════════

Use this framework to build a production-ready backtesting system. Prioritize:
1. Accurate execution simulation (spread + fees)
2. Statistical edge validation (bootstrap, t-test, IC)
3. Regime-aware performance analysis
4. Walk-forward testing to prevent overfitting
5. Expected value filtering for trade selection
6. Dynamic position sizing with Kelly Criterion

Target metrics for validation:
- Sharpe ratio > 2.0
- Win rate > 55%
- p-value < 0.05
- Out-of-sample degradation < 30%
- Positive EV on >80% of trades

Generate clean, modular Python code optimized for Antigravity IDE. Include comprehensive logging and error handling. Export all results to CSV and JSON for further analysis.