"""
Real-Time Feature Engine - RSI Mean Reversion Strategy
Pure quant strategy: RSI < 43 = BUY YES, RSI > 58 = BUY NO
"""

import numpy as np
import pandas as pd
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# RSI Thresholds (Optimized for 15m markets)
RSI_OVERSOLD = 43   # BUY YES when RSI < 43
RSI_OVERBOUGHT = 58 # BUY NO when RSI > 58


@dataclass
class TradingFeatures:
    """Container for trading features."""
    timestamp: pd.Timestamp
    close: float
    rsi_14: float
    atr_15m: float
    rsi_14_prev: float  # Previous closed candle RSI for reference
    open: float        # Open price of current candle


class RealtimeFeatureEngine:
    """Compute RSI-based trading features."""
    
    REQUIRED_CANDLES = 1000  # Reduced requirement since we don't need 50 EMA
    
    def __init__(self):
        print("Loaded Strategy: RSI Mean Reversion (Quant) - Cleaned")
        self.candles = []
        self._lock = threading.Lock()
    
    def add_candle(self, candle):
        """Add a new candle to the internal buffer (thread-safe)."""
        with self._lock:
            if not self.candles:
                self.candles.append(candle)
                return

            # RACE CONDITION FIX:
            # sometimes 'Live' candle (T+1) arrives before 'Close' candle (T).
            # We must support updating/inserting past candles.
            
            # 1. Check if candle exists (update)
            # Scan backwards (optimization)
            for i in range(len(self.candles) - 1, -1, -1):
                if self.candles[i]['timestamp'] == candle['timestamp']:
                    self.candles[i] = candle
                    return
                
                # Optimization: If we go back too far (e.g. 2 hours), stop
                if i < len(self.candles) - 10: 
                    break
            
            # 2. Not found - insert and sort
            self.candles.append(candle)
            self.candles.sort(key=lambda x: x['timestamp'])
            
            # Maintain buffer size
            if len(self.candles) > self.REQUIRED_CANDLES + 50:
                self.candles = self.candles[-self.REQUIRED_CANDLES:]
    
    def compute_features(self, df: Optional[pd.DataFrame] = None) -> Optional[TradingFeatures]:
        """Compute RSI and trend features from 15m candles."""
        if df is None:
            with self._lock:
                if len(self.candles) < 20: # Minimal requirement for RSI
                    return None
                candles_snapshot = list(self.candles)  # Thread-safe copy
            df = pd.DataFrame(candles_snapshot)
        
        if len(df) < 20:
            return None
            
        # Ensure timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Resample to 15m candles
        df_15m = df.set_index('timestamp').resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(df_15m) < 20:
            return None
        
        # RSI 14 (Wilder's smoothing)
        delta = df_15m['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_14 = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50.0
        
        current = df_15m.iloc[-1]
            
        # ATR
        tr = pd.concat([
            df_15m['high'] - df_15m['low'],
            abs(df_15m['high'] - df_15m['close'].shift(1)),
            abs(df_15m['low'] - df_15m['close'].shift(1))
        ], axis=1).max(axis=1)
        atr_15m = tr.rolling(14).mean().iloc[-1]
        
        # Handle NaNs
        if pd.isna(rsi_14): rsi_14 = 50.0
        
        # Calculate RSI of previous closed candle
        rsi_14_prev = 50.0
        if len(rsi_series) > 1:
            rsi_14_prev = rsi_series.iloc[-2]
            
        if pd.isna(atr_15m): atr_15m = 0.0
        
        return TradingFeatures(
            timestamp=df.iloc[-1]['timestamp'],
            close=current['close'],
            rsi_14=rsi_14,
            atr_15m=atr_15m,
            rsi_14_prev=rsi_14_prev,
            open=current['open']
        )
    
    def predict_probability(self, features: TradingFeatures) -> float:
        """Return base probability (0.5) - no ML model."""
        return 0.5
    
    def calculate_rolling_sharpe(self, trades_df: pd.DataFrame, window: int = 500) -> Optional[float]:
        """Calculate annualized Sharpe ratio for the last N trades."""
        if len(trades_df) < window:
            return None
            
        recent = trades_df.tail(window)
        # Assuming returns are PnL / Start Balance (normative)
        returns = recent['pnl'] / 100.0 
        if returns.std() == 0: return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def get_market_regime(self, rolling_sharpe: Optional[float]) -> str:
        """Classify market regime based on Sharpe performance."""
        if rolling_sharpe is None: return 'WARMUP'
        if rolling_sharpe > 1.0: return 'HIGH_SHARPE'
        if rolling_sharpe > 0.3: return 'MEDIUM_SHARPE'
        return 'LOW_SHARPE'

    def calculate_signal_confidence(self, features: TradingFeatures, side: str, hour: int, day: int) -> int:
        """Score signal confidence from 0-100 based on confluence."""
        confidence = 50
        
        # RSI Extremity
        rsi = features.rsi_14
        if side == 'YES':
            if rsi < 40: confidence += 20
            elif rsi < 43: confidence += 10
        elif side == 'NO':
            if rsi > 60: confidence += 20
            elif rsi > 58: confidence += 10
            
        # Bonus for high-performance hours (from audit)
        if hour in [19, 20, 21, 22]: confidence += 10
        if day in [5, 6]: confidence += 15
        
        return min(100, max(0, confidence))

    def check_signal(self, features: TradingFeatures, probability: float) -> Tuple[Optional[str], float]:
        """
        Check for RSI mean reversion signal.
        
        - RSI < 43: BUY YES (oversold, expect bounce)
        - RSI > 58: BUY NO (overbought, expect pullback)
        """
        if probability < 0.0: # probability logic disabled for now
            pass
            
        rsi = features.rsi_14
        
        # CANDLE COLOR FILTER (Avoid falling knives)
        # If candle body > 1.5 * ATR, it is HUGE.
        # Don't fade a HUGE candle.
        body = abs(features.close - features.open)
        is_huge = body > (1.5 * features.atr_15m)
        is_green = features.close > features.open

        
        # Oversold - expect bounce UP
        if rsi < RSI_OVERSOLD:
            edge = (RSI_OVERSOLD - rsi) / RSI_OVERSOLD
            print(f"[DEBUG] SIGNAL YES! RSI={rsi:.2f} < {RSI_OVERSOLD} (Edge: {edge:.2f})")
            return ('YES', min(0.5, edge))
        
        # Overbought - expect pullback DOWN
        elif rsi > RSI_OVERBOUGHT:
            edge = (rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
            print(f"[DEBUG] SIGNAL NO! RSI={rsi:.2f} > {RSI_OVERBOUGHT} (Edge: {edge:.2f})")
            return ('NO', min(0.5, edge))
        
        return (None, 0.0)


# Backward compatibility aliases
RealtimeFeatureEngineV2 = RealtimeFeatureEngine
TradingFeaturesV2 = TradingFeatures
