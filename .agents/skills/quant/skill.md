Create a complete backtesting framework for Polymarket BTC 15-minute UP/DOWN prediction market strategies with the following requirements:

## Core Requirements

1. **Data Structure**
   - Load historical BTC OHLCV data (1-minute granularity minimum)
   - Load Polymarket orderbook data: YES/NO token prices, bid/ask spreads, volumes
   - Synchronize timestamps between BTC price data and Polymarket market data
   - Store market settlement outcomes (actual UP/DOWN results every 15 minutes)

2. **Strategy Implementation**
   - Support multiple strategy types: technical indicators (RSI, MA, Bollinger Bands), ML models (XGBoost, LightGBM), combined approaches
   - Generate trading signals based on BTC price action
   - Implement entry logic: when to buy YES (predicting UP) or NO (predicting DOWN)
   - Implement exit logic: hold until settlement or early exit based on profit targets

3. **Event-Driven Backtesting Engine**
   - Simulate tick-by-tick execution (no look-ahead bias)
   - Process data chronologically: data arrives → calculate indicators → generate signal → place order → execute → update portfolio
   - Track each 15-minute market cycle separately
   - Record entry price, exit price, position size, and outcome for each trade

4. **Realistic Execution Simulation**
   - Apply bid-ask spread costs (use historical spreads from orderbook data)
   - Implement slippage model based on order size vs available liquidity
   - Add Polymarket fees (2% on winning positions)
   - Respect position size limits based on orderbook depth
   - Simulate partial fills if liquidity is insufficient

5. **Risk Management**
   - Maximum position size per trade (e.g., $100, $500, $1000)
   - Maximum concurrent positions limit
   - Stop-loss mechanisms (optional: exit if token price moves against position by X%)
   - Bankroll management: Kelly Criterion or fixed fractional sizing

6. **Performance Metrics Calculation**
   **Returns:**
   - Total PnL, ROI, win rate, profit factor
   - Sharpe ratio, Sortino ratio, Calmar ratio
   - Maximum drawdown, average drawdown duration
   
   **Trade Statistics:**
   - Number of trades, average trade duration
   - Average win size, average loss size
   - Largest winning trade, largest losing trade
   - Consecutive wins/losses streaks
   
   **Execution Quality:**
   - Fill rate (% of orders successfully executed)
   - Average slippage per trade
   - Strategy capacity (max capital deployable given liquidity)

7. **Walk-Forward Analysis**
   - Split data into train/validation/test sets (60/20/20)
   - Rolling window optimization: train on 30 days, test on next 7 days, roll forward
   - Track performance degradation between in-sample and out-of-sample periods
   - Flag overfitting if out-of-sample performance drops significantly

## Advanced Validation Layers

8. **Edge Validation Layer**
   **Statistical Significance Testing:**
   - Bootstrap analysis: resample trades 10,000 times to calculate confidence intervals
   - T-test for mean returns vs zero (is strategy genuinely profitable?)
   - Calculate p-values: reject null hypothesis that returns are random
   - Minimum sample size check: require at least 100 trades for statistical validity
   
   **Edge Decay Analysis:**
   - Split backtest into time buckets (monthly or quarterly)
   - Track Sharpe ratio evolution over time
   - Detect if edge is strengthening, stable, or degrading
   - Calculate rolling 30-day win rate and profit factor
   - Flag strategies where recent performance < historical average by >20%
   
   **Signal Quality Metrics:**
   - Information Coefficient (IC): correlation between predicted direction and actual outcome
   - Calculate IC for each signal component separately
   - Rank signals by predictive power
   - Identify which indicators contribute most to edge
   
   **Randomization Tests:**
   - Shuffle entry/exit times randomly and re-run backtest
   - Compare actual strategy PnL vs 1,000 random permutations
   - Strategy must outperform >95% of random strategies to validate edge
   - Monte Carlo simulation: randomize entry timing ±2 minutes to test robustness

9. **Regime Detection Layer**
   **Market State Classification:**
   - **Volatility Regimes:** High (ATR > 80th percentile), Medium, Low (ATR < 20th percentile)
   - **Trend Regimes:** Strong Up (ADX > 25, +DI > -DI), Strong Down, Ranging (ADX < 20)
   - **Liquidity Regimes:** Deep (orderbook depth > median), Thin (depth < 25th percentile)
   - **Time-Based Regimes:** US market hours, Asia hours, weekend, weekday
   
   **Regime-Specific Performance:**
   - Calculate separate metrics for each regime
   - Sharpe ratio by volatility level (does strategy work in high vol?)
   - Win rate by trend direction (does strategy fail in trending markets?)
   - Drawdowns by regime (which conditions cause worst losses?)
   
   **Adaptive Filtering:**
   - Only trade in favorable regimes (e.g., disable strategy when ADX < 15)
   - Adjust position sizing by regime (reduce size in high volatility)
   - Create regime-switching rules (use Strategy A in trending, Strategy B in ranging)
   
   **Regime Transition Analysis:**
   - Detect regime changes in real-time using Hidden Markov Models or threshold rules
   - Measure strategy performance during regime transitions
   - Test if edge exists during stable regimes but disappears during transitions

10. **Capacity Stress Test**
    **Scalability Analysis:**
    - Test strategy with position sizes: $100, $500, $1K, $5K, $10K, $50K
    - Calculate slippage impact at each size level
    - Identify maximum profitable position size (capacity limit)
    - Plot efficiency curve: ROI vs position size
    
    **Liquidity Consumption:**
    - Calculate % of available orderbook depth consumed per trade
    - Flag trades that would move market by >5%
    - Estimate realistic daily volume capacity (don't exceed 10% of market volume)
    - Test with aggressive execution (immediate fills) vs passive (limit orders)
    
    **Concurrent Position Stress:**
    - Simulate 1, 3, 5, 10 concurrent positions
    - Test if correlation between positions increases risk
    - Calculate portfolio-level drawdown vs single-position drawdown
    - Identify optimal max concurrent positions before returns diminish
    
    **Market Impact Modeling:**
    - Implement square-root market impact model: impact ∝ √(order_size / avg_volume)
    - Add temporary impact (price reverts after trade) and permanent impact
    - Recalculate strategy returns with market impact included
    - Determine break-even point where impact costs exceed edge
    
    **Frequency Scaling:**
    - Test strategy at different trading frequencies (every market, every 2nd market, hourly)
    - Calculate optimal trading frequency that maximizes Sharpe ratio
    - Identify if overtrading erodes returns due to costs

11. **Expected Value vs Market Price Edge Analysis**
    **True Probability Estimation:**
    - Calculate strategy's predicted probability of UP: P(UP) from model or signal strength
    - Compare against market-implied probability from token prices
    - Market implied: P(UP)_market = YES_price / (YES_price + NO_price)
    
    **Edge Calculation:**
    - Edge = P(UP)_strategy - P(UP)_market
    - Only trade when |Edge| > threshold (e.g., 5% probability difference)
    - Calculate expected value: EV = (P(UP) × Payout_if_UP) - (P(DOWN) × Loss_if_DOWN) - Fees
    
    **Kelly Criterion Position Sizing:**
    - Optimal bet size: f* = (p × b - q) / b
    - Where p = win probability, q = 1-p, b = odds received
    - Calculate Kelly fraction for each trade based on edge
    - Implement fractional Kelly (e.g., 0.5× Kelly) for safety
    
    **Mispricing Detection:**
    - Identify trades where strategy confidence is highest AND market price is most wrong
    - Create "high conviction" subset: trades with EV > 10%
    - Backtest high-conviction trades separately (should have higher Sharpe)
    - Track calibration: does a 70% model prediction win 70% of the time?
    
    **Market Efficiency Analysis:**
    - Calculate average edge size over time (is market getting more efficient?)
    - Identify which market conditions produce largest mispricings
    - Test if edge disappears as markets mature or during high-liquidity periods
    - Compare edge on first market of the day vs later markets
    
    **Arbitrage Opportunity Detection:**
    - Check if YES_price + NO_price ≠ 1.0 (accounting for fees)
    - Identify locked-in profit opportunities
    - Calculate frequency and size of arbitrage vs directional edge
    
    **Fair Value Tracking:**
    - Calculate theoretical fair value based on strategy model
    - Track deviation: Market_Price - Fair_Value over time
    - Enter when deviation exceeds 2 standard deviations
    - Create mean-reversion overlay strategy

## Comprehensive Reporting

12. **Visualization & Reporting**
    - Equity curve (cumulative PnL over time)
    - Drawdown chart with regime annotations
    - Trade distribution histogram (win/loss sizes)
    - Performance by market regime (heatmap)
    - Performance by time of day and day of week
    - Confusion matrix (predicted UP/DOWN vs actual outcome)
    - Edge validation dashboard (statistical significance tests)
    - Regime performance comparison table
    - Capacity scaling curve (ROI vs position size)
    - EV vs actual returns scatter plot (calibration check)

13. **Data Export**
    - Export all trades to CSV: timestamp, signal, entry_price, exit_price, position_size, pnl, outcome, regime, edge, EV
    - Export performance metrics summary to JSON (overall + by regime)
    - Export edge validation results (p-values, IC, bootstrap CI)
    - Export capacity analysis results
    - Export equity curve data for external analysis

14. **Code Structure**
    ```
    ├── data_loader.py          # Load and preprocess historical data
    ├── indicators.py           # Technical indicators and features
    ├── strategy.py             # Strategy logic and signal generation
    ├── backtester.py           # Event-driven backtest engine
    ├── execution.py            # Order execution and slippage simulation
    ├── portfolio.py            # Position tracking and PnL calculation
    ├── metrics.py              # Performance metrics calculation
    ├── edge_validator.py       # Statistical significance and edge tests
    ├── regime_detector.py      # Market regime classification
    ├── capacity_tester.py      # Scalability and stress testing
    ├── ev_analyzer.py          # Expected value and mispricing detection
    ├── visualizer.py           # Charts and reports generation
    ├── walk_forward.py         # Walk-forward optimization
    └── main.py                 # Run backtest with configuration
    ```

## Strategy Examples to Test

**Strategy 1: RSI Mean Reversion**
- Buy YES if RSI(14) < 30 and price dropped in last 5 minutes
- Buy NO if RSI(14) > 70 and price increased in last 5 minutes
- Hold until settlement
- Apply regime filter: only trade when ADX < 20 (ranging market)

**Strategy 2: Momentum Breakout**
- Buy YES if price breaks above 5-minute high with volume surge
- Buy NO if price breaks below 5-minute low with volume surge
- Exit early if profit reaches 10% or loss exceeds 5%
- Apply regime filter: only trade when ADX > 25 (trending market)

**Strategy 3: ML Prediction Model with EV Filter**
- Train XGBoost on features: RSI, MACD, Bollinger %B, volume, time-of-day
- Predict probability of UP vs DOWN
- Calculate EV: only trade when EV > 5% after fees
- Position size using Kelly Criterion based on predicted edge
- Only trade in medium/high liquidity regimes

## Configuration Parameters
- Initial capital: $10,000
- Position size: Dynamic (Kelly Criterion) or Fixed ($100 per trade)
- Max concurrent positions: 3
- Slippage: 0.5% base + market impact model
- Fees: 2% on winning positions
- Data period: Last 6 months of historical data
- Minimum edge threshold: 5% probability advantage
- Regime detection lookback: 50 periods for regime classification
- Bootstrap iterations: 10,000 for edge validation
- Capacity test sizes: [100, 500, 1000, 5000, 10000, 50000]

## Output Format
Generate a comprehensive backtest report including:
1. Executive summary: overall Sharpe, win rate, max drawdown
2. Edge validation results: statistical significance, p-values, confidence intervals
3. Regime performance breakdown: table showing metrics by regime type
4. Capacity analysis: maximum scalable capital before efficiency degrades
5. EV calibration: scatter plot of predicted edge vs realized returns
6. Best and worst trades analysis
7. Parameter sensitivity analysis
8. Strategy improvement recommendations based on regime/capacity/edge findings

Use clean, modular, well-commented Python code optimized for Antigravity IDE. Include error handling and logging for debugging. Ensure all validation layers can be toggled on/off via configuration.
