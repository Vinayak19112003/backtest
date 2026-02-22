def calculate_confidence(rsi, close, bb_sma, bb_upper, bb_lower, signal_type, rsi_lower=43, rsi_upper=58):
    """Calculate strategy confidence based on indicator extremity."""
    # RSI confidence component
    if signal_type == "YES":
        rsi_conf = 0.50 + ((rsi_lower - rsi) / 100)
    elif signal_type == "NO":
        rsi_conf = 0.50 + ((rsi - rsi_upper) / 100)
    else:
        return 0.50
        
    # BB confidence component
    if bb_upper != bb_sma:
        bb_deviation = (close - bb_sma) / (bb_upper - bb_sma)
    else:
        bb_deviation = 0
        
    bb_conf_adj = abs(bb_deviation) * 0.10
    
    # Combined, capped at 0.70 (70%)
    combined = rsi_conf + bb_conf_adj
    return min(combined, 0.70)

def generate_signal(row, rsi_lower=43, rsi_upper=58):
    """
    Check if current row generates a trading signal.
    Must be first 5 minutes of 15-minute window (minutes 0,1,2,3,4).
    """
    minute_in_window = row.minute % 15
    if minute_in_window > 4:
        # Too late in the 15-minute window
        return None, 0.0
        
    signal = None
    if row.rsi < rsi_lower and row.close < row.bb_lower:
        signal = "YES"
    elif row.rsi > rsi_upper and row.close > row.bb_upper:
        signal = "NO"
        
    if signal:
        conf = calculate_confidence(row.rsi, row.close, row.bb_sma, row.bb_upper, row.bb_lower, signal, rsi_lower, rsi_upper)
        return signal, conf
        
    return None, 0.0

def calculate_ev(confidence, investment=102, payout=200, fee=4):
    """Calculate expected value for a trade."""
    net_win = payout - fee - investment # 196 - 102 = 94
    net_loss = -investment # -102
    
    ev = (confidence * net_win) + ((1 - confidence) * net_loss)
    return ev
