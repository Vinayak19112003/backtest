import pandas as pd
import numpy as np

def generate_performance_summary(trades_df, final_capital, initial_capital=10000):
    """Calculate metrics for binary prediction markets"""
    
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_days': 0,
            'p_value': 1.0
        }
    
    # Core metrics
    win_rate = trades_df['win'].mean()
    total_pnl = trades_df['pnl'].sum()
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Returns per trade for Sharpe
    avg_investment = abs(trades_df[~trades_df['win']]['pnl'].mean()) if (~trades_df['win']).any() else 1.0
    returns = trades_df['pnl'] / avg_investment
    
    # Sharpe ratio (annualized for 96 markets/day)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 96)
    else:
        sharpe = 0
    
    # Maximum drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown / initial_capital * 100
    
    # Drawdown Duration (Peak to Trough)
    if not drawdown.empty and max_drawdown < 0:
        trough_idx = drawdown.idxmin()
        peak_idx = cumulative_pnl.loc[:trough_idx].idxmax()
        
        # Ensure exit_time exists as datetime
        if 'exit_time' in trades_df.columns:
            exit_times = pd.to_datetime(trades_df['exit_time'])
            peak_time = exit_times.loc[peak_idx]
            trough_time = exit_times.loc[trough_idx]
            max_dd_days = (trough_time - peak_time).days
        else:
            # Fallback to index difference if exit_time is missing
            max_dd_days = int(trough_idx - peak_idx)
    else:
        max_dd_days = 0
    
    # Statistical significance (t-test)
    from scipy import stats
    if len(returns) >= 2:
        t_stat, p_value = stats.ttest_1samp(returns, 0)
    else:
        p_value = 1.0
    
    # Profit factor
    gross_profit = trades_df[trades_df['win']]['pnl'].sum()
    gross_loss = abs(trades_df[~trades_df['win']]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_days': max_dd_days,
        'profit_factor': profit_factor,
        'p_value': p_value,
        'avg_win': trades_df[trades_df['win']]['pnl'].mean() if trades_df['win'].any() else 0,
        'avg_loss': trades_df[~trades_df['win']]['pnl'].mean() if (~trades_df['win']).any() else 0
    }

def print_executive_summary(summary, title="BACKTEST RESULTS"):
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)
    print(f"Total Trades : {summary.get('total_trades', 0)}")
    print(f"Win Rate     : {summary.get('win_rate', 0)*100:.2f}%")
    print(f"Total PnL    : ${summary.get('total_pnl', 0):.2f}")
    if 'total_return_pct' in summary:
        print(f"Total Return : {summary['total_return_pct']:.2f}%")
    print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio : {summary.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown : {summary.get('max_drawdown', 0):.2f} ({summary.get('max_drawdown_pct', 0):.2f}%)")
    print(f"Max DD Days  : {summary.get('max_drawdown_days', 0)} days")
    print("-" * 60)
    print("Edge Validation:")
    print(f"T-test p-val : {summary.get('p_value', 1.0):.5f}")
    print(f"Avg Win      : ${summary.get('avg_win', 0):.2f}")
    print(f"Avg Loss     : ${summary.get('avg_loss', 0):.2f}")
    print("=" * 60)
