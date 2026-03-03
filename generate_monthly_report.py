import pandas as pd
import os

# Load the trades
csv_path = 'results/rsi_43_58_adx25_atr80_baseline/trades.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit()

df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Group by Year and Month
df['year_month'] = df['timestamp'].dt.to_period('M')

monthly_stats = []

for period, group in df.groupby('year_month'):
    trades = len(group)
    wins = group['win'].sum()
    win_rate = wins / trades if trades > 0 else 0
    pnl = group['pnl'].sum()
    
    # Calculate Profit Factor
    gross_profit = group[group['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(group[group['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    monthly_stats.append({
        'Month': str(period),
        'Trades': trades,
        'Win Rate': f"{win_rate*100:.2f}%",
        'Profit Factor': f"{profit_factor:.2f}" if gross_loss > 0 else "N/A",
        'PnL': pnl
    })

# Convert to DataFrame for nice printing
report_df = pd.DataFrame(monthly_stats)

print("="*65)
print(f"{'MONTHLY PERFORMANCE REPORT':^65}")
print("="*65)
print(f"{'Month':<10} | {'Trades':<8} | {'Win Rate':<10} | {'PF':<6} | {'PnL':<10}")
print("-" * 65)

total_pnl = 0
for idx, row in report_df.iterrows():
    pnl_str = f"${row['PnL']:.2f}"
    print(f"{row['Month']:<10} | {row['Trades']:<8} | {row['Win Rate']:<10} | {row['Profit Factor']:<6} | {pnl_str:>10}")
    total_pnl += row['PnL']

print("-" * 65)
print(f"Total PnL: ${total_pnl:.2f}")
print("="*65)

# Save to file
report_path = 'results/rsi_43_58_adx25_atr80_baseline/monthly_report.txt'
with open(report_path, 'w') as f:
    f.write("="*65 + "\n")
    f.write(f"{'MONTHLY PERFORMANCE REPORT':^65}\n")
    f.write("="*65 + "\n")
    f.write(f"{'Month':<10} | {'Trades':<8} | {'Win Rate':<10} | {'PF':<6} | {'PnL':<10}\n")
    f.write("-" * 65 + "\n")
    for idx, row in report_df.iterrows():
        pnl_str = f"${row['PnL']:.2f}"
        f.write(f"{row['Month']:<10} | {row['Trades']:<8} | {row['Win Rate']:<10} | {row['Profit Factor']:<6} | {pnl_str:>10}\n")
    f.write("-" * 65 + "\n")
    f.write(f"Total PnL: ${total_pnl:.2f}\n")
    f.write("="*65 + "\n")

print(f"\nReport saved to: {report_path}")
