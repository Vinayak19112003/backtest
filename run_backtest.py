import pandas as pd
import numpy as np
import logging
import json
import os
import sys

# Ensure src in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_prep_data
from src.engine import run_backtest
from src.validation import assign_regimes, edge_decay_analysis
from src.reporting import generate_performance_summary, print_executive_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    data_path = 'data/BTCUSDT_15m_3_years.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data not found at {data_path}")
        return
        
    # 1. Data load and preprocessing
    df = load_and_prep_data(data_path)
    df = assign_regimes(df)
    
    # Generate splits
    total_len = len(df)
    train_end = int(total_len * 0.6)
    val_end = int(total_len * 0.8)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # 2. Run Backtest
    logger.info("Running baseline backtest on ALL DATA...")
    trades_df, final_capital = run_backtest(df, initial_capital=10000)
    
    if len(trades_df) == 0:
        logger.warning("No trades executed with baseline parameters.")
        return
        
    trades_df.to_csv('trades.csv', index=False)
    summary = generate_performance_summary(trades_df, trades_df['capital_after'])
    
    with open('performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print_executive_summary(summary, title="FULL 3-YEAR BACKTEST (ALL DATA)")
    
    # 3. Train / Val / Test Splits
    print("\n=== Train / Val / Test Performance ===")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        t, _ = run_backtest(split_df)
        if len(t) > 0:
            s_split = generate_performance_summary(t, t['capital_after'])
            print(f"[{name}] Trades: {len(t):4d} | Win Rate: {s_split['win_rate']*100:.2f}% | Sharpe: {s_split['sharpe_ratio']:.2f} | PnL: ${s_split['total_pnl']:.2f}")
        else:
            print(f"[{name}] No trades generated.")

    # 4. Breakdown by Regime
    print("\n=== Performance by Regime (All Data) ===")
    print("\nVolatility Regimes:")
    vol_stats = trades_df.groupby('regime_vol').apply(
        lambda x: pd.Series({
            'Trades': len(x),
            'WinRate': len(x[x['outcome']=='WIN'])/len(x) if len(x)>0 else 0,
            'TotalPnL': x['pnl'].sum()
        })
    )
    print(vol_stats)
    
    print("\nTrend Regimes:")
    trend_stats = trades_df.groupby('regime_trend').apply(
        lambda x: pd.Series({
            'Trades': len(x),
            'WinRate': len(x[x['outcome']=='WIN'])/len(x) if len(x)>0 else 0,
            'TotalPnL': x['pnl'].sum()
        })
    )
    print(trend_stats)

    # 5. Capacity Stress Test
    print("\n=== Capacity Stress Test ===")
    sizes = [100, 250, 500, 1000, 2500, 5000, 10000]
    for size in sizes:
        t_df, _ = run_backtest(df, position_size=size)
        if len(t_df) > 0:
            s = generate_performance_summary(t_df, t_df['capital_after'])
            roi = (s['total_pnl'] / 10000) * 100
            print(f"Position: ${size:5d} | Trades: {len(t_df):4d} | ROI: {roi:6.2f}% | Sharpe: {s.get('sharpe_ratio', 0):.2f}")

    # 6. Parameter Sensitivity
    print("\n=== Parameter Sensitivity: RSI Thresholds ===")
    bounds = [(40, 60), (43, 58), (45, 55), (48, 52)]
    for lower, upper in bounds:
        t_df, _ = run_backtest(df, rsi_lower=lower, rsi_upper=upper, position_size=100)
        s = generate_performance_summary(t_df, t_df['capital_after']) if len(t_df) > 0 else {"sharpe_ratio": 0, "total_trades": 0}
        print(f"RSI ({lower}/{upper}) | Trades: {s.get('total_trades', 0):4d} | Sharpe: {s.get('sharpe_ratio', 0):.2f}")

if __name__ == '__main__':
    main()
