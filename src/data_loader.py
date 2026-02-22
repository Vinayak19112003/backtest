import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(file_path='data/BTCUSDT_15m_3_years.csv'):
    """Load 15-minute BTC data - each row is one Polymarket market"""
    df = pd.read_csv(file_path)
    
    # Parsing dates dynamically based on file format
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df.set_index('timestamp', inplace=True)
    
    # Verify 15-minute intervals
    time_diffs = df.index.to_series().diff()
    assert time_diffs.mode()[0] == pd.Timedelta('15min'), "Data must be 15-minute intervals"
    
    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    print(f"Loaded {len(df)} 15-minute candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Expected markets: {len(df)-1} (need next candle for outcome)")
    
    return df
