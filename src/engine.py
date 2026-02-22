import pandas as pd
import numpy as np
from tqdm import tqdm
from .strategy import generate_signal, calculate_ev

def simulate_slippage(position_size, base_slippage=0.005):
    """
    Calculate variable slippage based on position size.
    0.5% base + 0.3% * sqrt(size/100)
    """
    return base_slippage + (0.003 * np.sqrt(position_size / 100))

def run_backtest(df: pd.DataFrame, 
                 initial_capital=10000, 
                 position_size=100, 
                 max_concurrent=3,
                 rsi_lower=43,
                 rsi_upper=58,
                 min_ev=5.0):
    
    """Run event-driven backtest tick-by-tick."""
    # We need to track: portfolio capital, active positions, trade history
    trades = []
    capital = initial_capital
    
    active_positions = []
    
    # Pre-calculate rolling dictionary lookup avoiding slow dataframe dot access if possible
    # We will use itertuples, which is fast enough.
    
    df_records = list(df.itertuples(index=False))
    
    # Create dict to easily get closing price at settlement
    # Settlement price is the close price at min 15.
    # Group by window_start to get settlement prices
    # Since we mapped window_start, settlement of window takes place at the closing price of the 14th minute or the 15th's open?
    # Requirement: "Market settlement price = close price at minute 15". i.e., minute_in_window = 15... wait, 15m windows are 0-14. 
    # Minute 15 is the start of the next window.
    # We can just look ahead to the row where timestamp == window_start + 15 mins.
    
    settlement_prices = {}
    for row in df_records:
        if row.minute % 15 == 14:
            # We treat the close of minute 14 (e.g. 00:14:59) as settlement price.
            # Alternately, we can treat the open of 00:15
            pass
            
    # Simpler: just loop and resolve trades when they expire
    
    # State tracking
    pending_orders = [] # Orders placed but not yet executed (execute next minute)
    
    for i, row in tqdm(enumerate(df_records), total=len(df_records), desc="Backtesting"):
        current_time = row.timestamp
        current_window = row.window_start
        
        # 1. Execute pending orders (using current row's OPEN or CLOSE? Requirement: "close price of the minute AFTER signal triggers")
        # So if signal at min 1, we enter at CLOSE of min 2.
        executed_this_bar = []
        for order in pending_orders:
            # Check if this row is exactly 1 minute after the signal
            if current_time > order['signal_time']:
                # Execute order
                slippage = simulate_slippage(position_size)
                # For simplicity, calculate total cost = investment + entry_fee_equivalent
                # Polymarket fee is on win. Slippage increases entry price effectively, 
                # but in binary options, we just pay the cost of the token.
                # Assume 50c token. $100 buys 200 tokens. 
                # Total investment per trade logic:
                # Investment = $100. Spread/slippage = 2% total. Let's just use the fixed $102 structure with dynamic slippage.
                # Slippage cost = investment * slippage.
                # So if slippage is 2%, cost is $2. Total = 102.
                total_investment = position_size * (1 + slippage)
                
                pos = {
                    'entry_time': current_time,
                    'entry_price': row.close,
                    'signal_type': order['signal_type'],
                    'investment': total_investment,
                    'position_size': position_size,
                    'window_start': current_window,
                    'confidence': order['confidence'],
                    'ev': order['ev'],
                    'rsi_value': order['rsi'],
                    'bb_position': order['bb_position'],
                    'regime_vol': row.atr,     # Will map this later
                    'regime_trend': row.adx    # Will map this later
                }
                active_positions.append(pos)
                capital -= total_investment
                executed_this_bar.append(order)
                
        for order in executed_this_bar:
            pending_orders.remove(order)
            
        # 2. Resolve expired positions
        # Expire if current_time >= window_start + 15 minutes
        # We resolve them at the close price of the settlement minute
        active_remaining = []
        for pos in active_positions:
            expiry_time = pos['window_start'] + pd.Timedelta(minutes=15)
            if current_time >= expiry_time:
                # Resolve
                settlement_price = row.close
                won = False
                if pos['signal_type'] == "YES" and settlement_price > pos['entry_price']:
                    won = True
                elif pos['signal_type'] == "NO" and settlement_price <= pos['entry_price']:
                    won = True
                    
                gross_payout = pos['position_size'] * 2 if won else 0
                fee = gross_payout * 0.02 if won else 0
                net_payout = gross_payout - fee
                
                pnl = net_payout - pos['investment']
                capital += net_payout
                
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price_btc': pos['entry_price'],
                    'exit_price_btc': settlement_price,
                    'signal_type': pos['signal_type'],
                    'investment': pos['investment'],
                    'payout': net_payout,
                    'pnl': pnl,
                    'outcome': 'WIN' if won else 'LOSS',
                    'confidence': pos['confidence'],
                    'ev': pos['ev'],
                    'rsi_value': pos['rsi_value'],
                    'bb_position': pos['bb_position'],
                    'regime_vol': pos['regime_vol'],
                    'regime_trend': pos['regime_trend'],
                    'capital_after': capital
                }
                trades.append(trade_record)
            else:
                active_remaining.append(pos)
        active_positions = active_remaining
        
        # 3. Generate new signals
        if len(active_positions) + len(pending_orders) < max_concurrent:
            # Only 1 position per window. Check if we already have one for this window
            has_window_pos = any(p['window_start'] == current_window for p in active_positions + pending_orders)
            
            if not has_window_pos:
                signal, conf = generate_signal(row, rsi_lower, rsi_upper)
                if signal:
                    # check EV
                    ev = calculate_ev(conf, investment=(position_size*1.02), payout=(position_size*2))
                    if ev >= min_ev:
                        bb_pos = (row.close - row.bb_sma) / (row.bb_upper - row.bb_sma) if row.bb_upper != row.bb_sma else 0
                        
                        pending_orders.append({
                            'signal_time': current_time,
                            'signal_type': signal,
                            'confidence': conf,
                            'ev': ev,
                            'rsi': row.rsi,
                            'bb_position': bb_pos
                        })

    return pd.DataFrame(trades), capital
