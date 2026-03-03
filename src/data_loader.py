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
    
    # Detect interval from data
    time_diffs = df.index.to_series().diff()
    detected_interval = time_diffs.mode()[0]
    
    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    interval_str = str(detected_interval).split(' ')[-1] if 'day' not in str(detected_interval) else str(detected_interval)
    print(f"Loaded {len(df)} candles (interval: {detected_interval})")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Expected markets: {len(df)-1} (need next candle for outcome)")
    
    return df
