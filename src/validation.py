import pandas as pd
import numpy as np
from scipy import stats

def calculate_ic(trades_df: pd.DataFrame):
    """Calculate Information Coefficient (IC) - correlation between signal confidence and actual outcome (1 for WIN, -1 for loss)."""
    if len(trades_df) == 0:
        return 0.0
    
    outcomes = np.where(trades_df['outcome'] == 'WIN', 1.0, -1.0)
    confidences = trades_df['confidence'].values
    
    # rank correlation
    ic, p_val = stats.spearmanr(confidences, outcomes)
    return ic if not np.isnan(ic) else 0.0

def bootstrap_test(trades_df: pd.DataFrame, iterations: int = 10000):
    """Calculate 95% confidence interval for mean PnL using bootstrapping."""
    if len(trades_df) < 2:
        return 0.0, 0.0
        
    pnls = trades_df['pnl'].values
    bootstrapped_means = []
    
    # Vectorized bootstrap is faster
    # Generate random indices with replacement
    n = len(pnls)
    indices = np.random.randint(0, n, size=(iterations, n))
    bootstrapped_means = np.mean(pnls[indices], axis=1)
    
    lower_bound = np.percentile(bootstrapped_means, 2.5)
    upper_bound = np.percentile(bootstrapped_means, 97.5)
    
    return lower_bound, upper_bound

def run_t_test(trades_df: pd.DataFrame):
    """T-test for statistical significance of returns vs zero."""
    if len(trades_df) < 2:
        return 1.0
        
    pnls = trades_df['pnl'].values
    t_stat, p_value = stats.ttest_1samp(pnls, 0.0, alternative='greater')
    return p_value

def calculate_sharpe(returns: pd.Series, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio from a series of PnLs or returns. 
       Usually Sharpe relies on returns. For simplicity if provided with trade PnL we treat it as return per period.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(len(returns)) # Simplistic scaling

def calculate_max_drawdown(equity_curve: pd.Series):
    """Calculate maximum drawdown from an equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def assign_regimes(df: pd.DataFrame):
    """Assign volatility and trend regimes to 15m candles"""
    from src.indicators import calculate_atr, calculate_adx
    import numpy as np
    
    # ATR for volatility on 15m bars
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # ADX for trend strength on 15m bars
    adx, _, _ = calculate_adx(df['high'], df['low'], df['close'], period=14)
    df['adx'] = adx
    
    # Calculate percentiles for volatility regime
    atr_20 = df['atr'].quantile(0.20)
    atr_80 = df['atr'].quantile(0.80)
    
    df['regime_vol'] = pd.cut(df['atr'], 
                              bins=[0, atr_20, atr_80, np.inf],
                              labels=['Low', 'Medium', 'High'])
    
    # Trend regime based on ADX
    df['regime_trend'] = df['adx'].apply(
        lambda x: 'Strong' if x > 25 else 'Weak'
    )
    
    return df

def edge_decay_analysis(trades_df: pd.DataFrame):
    """Split into quarterly buckets and calculate Sharpe"""
    if len(trades_df) == 0:
        return pd.DataFrame()
        
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['quarter'] = trades_df['entry_time'].dt.to_period('Q')
    
    quarters = []
    sharpes = []
    
    for q, grp in trades_df.groupby('quarter'):
        sharpe = calculate_sharpe(grp['pnl'])
        quarters.append(str(q))
        sharpes.append(sharpe)
        
    return pd.DataFrame({'Quarter': quarters, 'Sharpe': sharpes})
