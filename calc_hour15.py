import pandas as pd
import numpy as np

# Load the Q3-filtered trade log
df = pd.read_csv('reports/q3_filter/q3_filter_trade_log.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Original stats
orig_trades = len(df)
orig_wins = df['win'].sum()
orig_wr = orig_wins / orig_trades * 100
orig_pnl = df['pnl'].sum()
orig_pf = df[df['pnl'] > 0]['pnl'].sum() / abs(df[df['pnl'] < 0]['pnl'].sum())

# Remove hour 15
df_no_15 = df[df.index.hour != 15]

# New stats
new_trades = len(df_no_15)
new_wins = df_no_15['win'].sum()
new_wr = new_wins / new_trades * 100
new_pnl = df_no_15['pnl'].sum()
new_pf = df_no_15[df_no_15['pnl'] > 0]['pnl'].sum() / abs(df_no_15[df_no_15['pnl'] < 0]['pnl'].sum())

print("--- CURRENT Q3 FILTERED ---")
print(f"Trades: {orig_trades:,}")
print(f"Win Rate: {orig_wr:.2f}%")
print(f"Total PnL: ${orig_pnl:,.2f}")
print(f"Profit Factor: {orig_pf:.2f}")

print("\n--- IF WE ALSO REMOVE HOUR 15 ---")
print(f"Trades: {new_trades:,}")
print(f"Win Rate: {new_wr:.2f}%")
print(f"Total PnL: ${new_pnl:,.2f}")
print(f"Profit Factor: {new_pf:.2f}")

print("\n--- IMPROVEMENT ---")
print(f"Trades Removed: {orig_trades - new_trades:,} (Hour 15 trades)")
print(f"Win Rate Change: {new_wr - orig_wr:+.2f}%")
print(f"PnL Change: ${(new_pnl - orig_pnl):+,.2f}")
print(f"Profit Factor Change: {new_pf - orig_pf:+.2f}")
