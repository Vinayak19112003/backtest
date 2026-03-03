import pandas as pd
import numpy as np
from src.indicators import calculate_rsi, calculate_bollinger_bands

def run_backtest(df, initial_capital=100.0, risk_per_trade=1.0, 
                 rsi_lower=43, rsi_upper=58, bb_period=20, bb_std=1.1,
                 volatility_limit=None, rsi_1h_lower=None, rsi_1h_upper=None,
                 skip_high_vol_strong_trend=False, block_us_open=False, require_confirmation=False,
                 skip_no_wick=False, wick_body_ratio=0.90, min_wick_points=0,
                 trade_start_date=None):
    """
    Run backtest using 15-minute candles
    Signal at candle i close → Outcome at candle i+1 close
    
    trade_start_date: If set, indicators are computed on the full df (for warm-up)
                      but trades are only recorded from this date onward.
                      Pass the full dataset and set this to your target start date.
    """
    
    # Calculate indicators on 15m close prices
    df['rsi'] = calculate_rsi(df['close'], period=14)
    if bb_period is not None and bb_std is not None:
        sma, upper, lower = calculate_bollinger_bands(df['close'], period=bb_period, num_std=bb_std)
        df['bb_upper'] = upper
        df['bb_middle'] = sma
        df['bb_lower'] = lower
    else:
        # Dummy values or don't set them, but for uniformity let's assign None or 0 and dropna carefully
        df['bb_upper'] = np.nan
        df['bb_middle'] = np.nan
        df['bb_lower'] = np.nan
        
    # Calculate 1h RSI if needed
    if rsi_1h_lower is not None or rsi_1h_upper is not None:
        # Resample close prices to 1H, taking the last close in the hour
        df_1h = df['close'].resample('1h').last().to_frame()
        df_1h['rsi_1h'] = calculate_rsi(df_1h['close'], period=14)
        
        # Merge back to 15m timeframe, using forward fill
        # By forward filling, the 10:00 1h-candle RSI (which includes 09:00-09:45 data and finishes at 10:00) 
        # applies to 10:00, 10:15, 10:30, 10:45 candles.
        df['rsi_1h'] = df_1h['rsi_1h'].reindex(df.index, method='ffill')
    else:
        df['rsi_1h'] = np.nan
        
    # Calculate 24h volatility: (24h High - 24h Low) / 24h Low * 100
    df['vol_24h'] = (df['high'].rolling(window=96).max() - df['low'].rolling(window=96).min()) / df['low'].rolling(window=96).min() * 100
    df = df.dropna(subset=['rsi', 'vol_24h'])
    if bb_period is not None and bb_std is not None:
        df = df.dropna(subset=['bb_lower'])
    
    trades = []
    capital = initial_capital
    
    # Loop through candles, use i+1 for outcome. Start at 1 to allow checking i-1 (previous candle)
    for i in range(1, len(df) - 1):
        prev_candle = df.iloc[i - 1]   # Previous candle (used for confirmation logic)
        signal_candle = df.iloc[i]     # Check signal at this candle close
        outcome_candle = df.iloc[i + 1] # Outcome at next candle close (15 min later)
        
        # Pre-fetch warm-up filter: skip trades before the target start date
        if trade_start_date is not None and signal_candle.name < trade_start_date:
            continue
        
        # Entry price = signal candle close (market boundary)
        entry_price = signal_candle['close']
        
        # Exit price = outcome candle close (settlement)
        exit_price = outcome_candle['close']
        
        # Volatility filter
        if volatility_limit is not None and signal_candle['vol_24h'] > volatility_limit:
            continue
            
        # Regime filter
        if skip_high_vol_strong_trend:
            if signal_candle.get('regime_vol') == 'High' and signal_candle.get('regime_trend') == 'Strong':
                continue
                
        # Time-of-day filter (Option 3: Block US Open 13:00-15:00 UTC)
        if block_us_open:
            hour = signal_candle.name.hour
            if hour >= 13 and hour <= 15:
                continue
        
        # Generate signals based on signal candle indicators
        if require_confirmation:
            # Option 2: Triple Confirmation
            # buy_yes: Previous candle RSI was oversold AND current candle is GREEN (close > open)
            buy_yes = (prev_candle['rsi'] < rsi_lower) and (signal_candle['close'] > signal_candle['open'])
            # buy_no: Previous candle RSI was overbought AND current candle is RED (close < open)
            buy_no = (prev_candle['rsi'] > rsi_upper) and (signal_candle['close'] < signal_candle['open'])
        else:
            if bb_period is None or bb_std is None:
                # Pure RSI strategy
                buy_yes = (signal_candle['rsi'] < rsi_lower)
                buy_no = (signal_candle['rsi'] > rsi_upper)
            else:
                # RSI + BB Strategy
                buy_yes = (signal_candle['rsi'] < rsi_lower) and \
                          (signal_candle['close'] < signal_candle['bb_lower'])
                buy_no = (signal_candle['rsi'] > rsi_upper) and \
                         (signal_candle['close'] > signal_candle['bb_upper'])
                         
        # Apply 1h RSI filter if requested
        if rsi_1h_lower is not None and buy_yes:
            if pd.isna(signal_candle['rsi_1h']) or signal_candle['rsi_1h'] >= rsi_1h_lower:
                buy_yes = False
                
        if rsi_1h_upper is not None and buy_no:
            if pd.isna(signal_candle['rsi_1h']) or signal_candle['rsi_1h'] <= rsi_1h_upper:
                buy_no = False
        
        if not (buy_yes or buy_no):
            continue
        
        # Wick filter: Skip outcome candles that moved one-directionally
        # In Polymarket, if a candle has no wick (body = entire range), the odds
        # never touch 0.50, so you can't enter at a fair price in live trading.
        # We skip trades where the outcome candle body covers > wick_body_ratio (90%) of the range.
        if skip_no_wick:
            oc_range = outcome_candle['high'] - outcome_candle['low']
            oc_body = abs(outcome_candle['close'] - outcome_candle['open'])
            if oc_range > 0:
                body_pct = oc_body / oc_range
                if body_pct >= wick_body_ratio:
                    continue  # No wick = can't enter at 0.50 in live
            else:
                continue  # Flat candle (high == low), no movement at all
        
        # Minimum wick size filter (directional)
        # For BUY YES (UP): lower wick must sweep >= min_wick_points below open
        #   so odds dipped near 0.50 in live Polymarket for a realistic entry
        # For BUY NO (DOWN): upper wick must sweep >= min_wick_points above open
        if min_wick_points > 0:
            if buy_yes:
                lower_wick = outcome_candle['open'] - outcome_candle['low']
                if lower_wick < min_wick_points:
                    continue  # Not enough downward sweep to enter at 0.50
            elif buy_no:
                upper_wick = outcome_candle['high'] - outcome_candle['open']
                if upper_wick < min_wick_points:
                    continue  # Not enough upward sweep to enter at 0.50
        
        # Polymarket binary outcome
        actual_outcome = 'UP' if exit_price > entry_price else 'DOWN'
        predicted_outcome = 'UP' if buy_yes else 'DOWN'
        win = (predicted_outcome == actual_outcome)
        
        # Polymarket PnL model
        SPREAD_COST = 0.02      # 2% entry cost
        WIN_FEE = 0.02          # 2% fee on winning payouts
        
        # Adjust investment so that exactly risk_per_trade is risked
        INVESTMENT = risk_per_trade / (1 + SPREAD_COST)
        
        if win:
            gross_payout = INVESTMENT * 2
            fee = gross_payout * WIN_FEE
            net_payout = gross_payout - fee
            pnl = net_payout - risk_per_trade
        else:
            pnl = -risk_per_trade
        
        capital += pnl
        
        # Store trade details
        trades.append({
            'timestamp': signal_candle.name,  # Signal time
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': signal_candle.name,
            'exit_time': outcome_candle.name,
            'rsi': signal_candle['rsi'],
            'bb_position': 'none' if bb_period is None else ('lower' if buy_yes else 'upper'),
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
