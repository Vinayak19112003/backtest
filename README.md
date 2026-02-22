# Polymarket BTC 15m UP/DOWN Backtest

Backtesting framework for RSI + Bollinger Bands strategy on Polymarket 15-minute BTC prediction markets.

## Strategy
- **Data:** 15-minute OHLCV candles (each candle = one market)
- **Signal timing:** Checked at candle close
- **BUY YES:** RSI(14) < 43 AND close < BB lower band (20-period, 1.1 SD)
- **BUY NO:** RSI(14) > 58 AND close > BB upper band (20-period, 1.1 SD)
- **Outcome:** Next candle close (15 minutes later)
- **Position:** $100 fixed per trade

## Logic Flow
```text
Candle i (10:00-10:15):
  Close at 10:15 → Calculate RSI/BB → Generate signal
  
Candle i+1 (10:15-10:30):
  Close at 10:30 → Determine outcome (UP/DOWN)
  
Example:
  10:15: BTC=$60,000, RSI=42, close < BB_lower → BUY YES
  10:30: BTC=$60,100 → UP → WIN +$94
```

## Data
- 3 years BTC/USDT 15-minute candles
- ~105,000 candles → ~105,000 potential markets
- File: `data/BTCUSDT_15m_3_years.csv`

## Costs
- Entry spread/slippage: 2%
- Polymarket fee: 2% on winning payouts
- Win: +$94 | Loss: -$102
- Breakeven: 52% win rate

## Usage
```bash
python run_backtest.py
```

## Output
- Train/Validation/Test split performance
- Regime analysis (volatility/trend)
- Statistical validation (p-value, bootstrap CI)
- Parameter sensitivity analysis
- CSV export of all trades
