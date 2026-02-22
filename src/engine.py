import pandas as pd
import numpy as np
from src.indicators import calculate_rsi, calculate_bollinger_bands

def run_backtest(df, initial_capital=10000, position_size=100, 
                 rsi_lower=43, rsi_upper=58, bb_period=20, bb_std=1.1):
    """
    Run backtest using 15-minute candles
    Signal at candle i close → Outcome at candle i+1 close
    """
    
    # Calculate indicators on 15m close prices
    df['rsi'] = calculate_rsi(df['close'], period=14)
    sma, upper, lower = calculate_bollinger_bands(df['close'], period=bb_period, num_std=bb_std)
    df['bb_upper'] = upper
    df['bb_middle'] = sma
    df['bb_lower'] = lower
    df = df.dropna()
    
    trades = []
    capital = initial_capital
    
    # Loop through candles, use i+1 for outcome
    for i in range(len(df) - 1):
        signal_candle = df.iloc[i]     # Check signal at this candle close
        outcome_candle = df.iloc[i + 1] # Outcome at next candle close (15 min later)
        
        # Entry price = signal candle close (market boundary)
        entry_price = signal_candle['close']
        
        # Exit price = outcome candle close (settlement)
        exit_price = outcome_candle['close']
        
        # Generate signals based on signal candle indicators
        buy_yes = (signal_candle['rsi'] < rsi_lower) and \
                  (signal_candle['close'] < signal_candle['bb_lower'])
        buy_no = (signal_candle['rsi'] > rsi_upper) and \
                 (signal_candle['close'] > signal_candle['bb_upper'])
        
        if not (buy_yes or buy_no):
            continue
        
        # Polymarket binary outcome
        actual_outcome = 'UP' if exit_price > entry_price else 'DOWN'
        predicted_outcome = 'UP' if buy_yes else 'DOWN'
        win = (predicted_outcome == actual_outcome)
        
        # Polymarket PnL model
        SPREAD_COST = 0.02      # 2% entry cost
        WIN_FEE = 0.02          # 2% fee on winning payouts
        INVESTMENT = position_size
        COST = INVESTMENT * SPREAD_COST
        TOTAL_INVESTED = INVESTMENT + COST  # $102 for $100 position
        
        if win:
            gross_payout = INVESTMENT * 2    # $200
            fee = gross_payout * WIN_FEE     # $4
            net_payout = gross_payout - fee  # $196
            pnl = net_payout - TOTAL_INVESTED  # +$94
        else:
            pnl = -TOTAL_INVESTED  # -$102
        
        capital += pnl
        
        # Store trade details
        trades.append({
            'timestamp': signal_candle.name,  # Signal time
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': signal_candle.name,
            'exit_time': outcome_candle.name,
            'rsi': signal_candle['rsi'],
            'bb_position': 'lower' if buy_yes else 'upper',
            'signal': 'BUY_YES' if buy_yes else 'BUY_NO',
            'predicted': predicted_outcome,
            'actual': actual_outcome,
            'win': win,
            'outcome': 'WIN' if win else 'LOSS', # Add just in case it's helpful
            'pnl': pnl,
            'capital_after': capital,
            'regime_vol': signal_candle.get('regime_vol', 'Unknown'),
            'regime_trend': signal_candle.get('regime_trend', 'Unknown'),
            'hour': signal_candle.name.hour,
            'weekday': signal_candle.name.weekday(),
            'day_name': signal_candle.name.day_name()
        })
    
    trades_df = pd.DataFrame(trades)
    return trades_df, capital
