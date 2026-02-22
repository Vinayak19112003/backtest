import pandas as pd
import json
import logging
from .validation import (calculate_sharpe, calculate_max_drawdown, 
                        run_t_test, bootstrap_test, calculate_ic, edge_decay_analysis)

logger = logging.getLogger(__name__)

def generate_performance_summary(trades_df: pd.DataFrame, capital_series: pd.Series, cost_assumption=0.04):
    """Calculate overall metrics for the backtest."""
    if len(trades_df) == 0:
        return {"error": "No trades executed."}
        
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['outcome'] == 'WIN'])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    sharpe = calculate_sharpe(trades_df['pnl'])
    max_dd = calculate_max_drawdown(capital_series)
    
    p_value = run_t_test(trades_df)
    ci_lower, ci_upper = bootstrap_test(trades_df)
    ic = calculate_ic(trades_df)
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "t_test_p_value": p_value,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "information_coefficient": ic
    }

def print_executive_summary(summary, title="BACKTEST RESULTS"):
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)
    print(f"Total Trades : {summary.get('total_trades', 0)}")
    print(f"Win Rate     : {summary.get('win_rate', 0)*100:.2f}%")
    print(f"Total PnL    : ${summary.get('total_pnl', 0):.2f}")
    print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio : {summary.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown : {summary.get('max_drawdown', 0)*100:.2f}%")
    print("-" * 60)
    print("Edge Validation:")
    print(f"T-test p-val : {summary.get('t_test_p_value', 1.0):.5f}")
    ci = summary.get('bootstrap_ci_95', [0,0])
    print(f"95% CI (PnL) : ${ci[0]:.2f} to ${ci[1]:.2f} per trade")
    print(f"Info Coeff   : {summary.get('information_coefficient', 0):.4f}")
    print("=" * 60)
