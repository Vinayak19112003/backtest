import pandas as pd
import numpy as np

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    # Wilder's smoothing (RMA)
    alpha = 1 / period
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 1.2):
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 50) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Calculate Average Directional Index (ADX) and Directional Indicators."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Avoid division by zero
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    atr = atr.replace(0, np.nan)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx, plus_di, minus_di
