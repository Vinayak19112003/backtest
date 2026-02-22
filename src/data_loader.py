import pandas as pd
import logging
from .indicators import calculate_rsi, calculate_bollinger_bands, calculate_atr, calculate_adx

logger = logging.getLogger(__name__)

def load_and_prep_data(filepath: str, rsi_period=14, bb_period=20, bb_sd=1.2, atr_period=50, adx_period=14) -> pd.DataFrame:
    """Load historical minutely data, add technical indicators."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Parsing dates
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in cols_to_keep):
        df.columns = [c.lower() for c in df.columns]
    
    df = df[cols_to_keep].copy()
    
    logger.info("Cleaning data...")
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.dropna()
    
    logger.info("Computing indicators...")
    # Add RSI
    df['rsi'] = calculate_rsi(df['close'], period=rsi_period)
    
    # Add Bollinger Bands
    sma, upper, lower = calculate_bollinger_bands(df['close'], period=bb_period, num_std=bb_sd)
    df['bb_sma'] = sma
    df['bb_upper'] = upper
    df['bb_lower'] = lower
    
    # Add ATR (for volatility regime)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=atr_period)
    
    # Add ADX (for trend regime)
    adx, plus_di, minus_di = calculate_adx(df['high'], df['low'], df['close'], period=adx_period)
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # Useful time components
    df['minute'] = df['timestamp'].dt.minute
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Market Window ID
    # Polymarket windows are 00:00 - 00:15, 00:15 - 00:30, etc.
    # Group by flooring timestamp to 15T (15 min)
    df['window_start'] = df['timestamp'].dt.floor('15min')
    
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Data ready: {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df
