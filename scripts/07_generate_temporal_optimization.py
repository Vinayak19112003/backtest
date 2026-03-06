"""
═══════════════════════════════════════════════════════════════════════
07_generate_temporal_optimization.py
Temporal Optimization Analysis — Standalone Analysis Track
Strategy Code: ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════
Identifies the best and worst performing time periods across multiple
dimensions (hour, day, quarter of hour, and their combinations).

Outputs (all in reports/temporal_optimization/):
  - TEMPORAL_OPTIMIZATION_SUMMARY.txt
  - hourly_performance.csv
  - daily_performance.csv
  - quarter_performance.csv
  - hour_day_matrix.csv
  - hour_quarter_matrix.csv
  - day_quarter_matrix.csv
═══════════════════════════════════════════════════════════════════════
"""
import sys, os, time
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "temporal_optimization")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEE_RATE = 0.01
INITIAL_CAPITAL = 100.0
SIM_ENTRY_PRICE = 0.50
MIN_TRADES = 50  # Minimum sample size for identifying best/worst

# ═══════════════════════════════════════════════════════════════
# DATA LOADING & INDICATORS (baseline logic)
# ═══════════════════════════════════════════════════════════════
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    if 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))

def compute_indicators(df):
    df['rsi'] = calculate_rsi_wilder(df['close'], 14)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    
    up_m, dn_m = df['high'].diff(), -df['low'].diff()
    plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_s)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_s)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    atr_14_arr = atr_14.values
    n = len(df)
    atr_pct = np.zeros(n)
    for i in range(28, n):
        lb = min(96, i)
        rec = atr_14_arr[i-lb:i+1]
        if pd.isna(rec).all(): continue
        valid = rec[~np.isnan(rec)]
        if len(valid) < 5: continue
        atr_pct[i] = (valid < atr_14_arr[i]).sum() / len(valid) * 100
    df['atr_pct'] = atr_pct
    df = df.dropna(subset=['rsi', 'adx_14']).copy()
    return df

# ═══════════════════════════════════════════════════════════════
# SIMULATION ENGINE (extracts temporal features)
# ═══════════════════════════════════════════════════════════════
def run_temporal_simulation(df):
    rsi_arr = df['rsi'].values
    adx_arr = df['adx_14'].values
    atr_pct_arr = df['atr_pct'].values
    c_arr = df['close'].values
    ts_arr = df.index
    
    trades = []
    capital = INITIAL_CAPITAL
    
    for i in range(1, len(df) - 1):
        rsi = rsi_arr[i]
        buy_yes = rsi < 43
        buy_no = rsi > 57
        if not (buy_yes or buy_no): continue
        if adx_arr[i] > 25 and atr_pct_arr[i] > 80: continue

        signal = 'YES' if buy_yes else 'NO'
        target_shares = int(1.0 / SIM_ENTRY_PRICE)
        shares = max(1, target_shares)
        bet_amount = shares * SIM_ENTRY_PRICE
        if bet_amount > capital: continue

        settle_c = c_arr[i + 1]
        won = (signal == 'YES' and settle_c > c_arr[i]) or (signal == 'NO' and settle_c < c_arr[i])
        fees = bet_amount * FEE_RATE * 2
        pnl = (shares * 1.0) - bet_amount - fees if won else -bet_amount - fees
        capital += pnl
        
        ts = ts_arr[i]
        minute = ts.minute
        quarter = 'Q1' if minute < 15 else 'Q2' if minute < 30 else 'Q3' if minute < 45 else 'Q4'

        trades.append({
            'timestamp': ts,
            'signal': signal,
            'win': won,
            'pnl': pnl,
            'hour': ts.hour,
            'day': ts.dayofweek,
            'minute': minute,
            'quarter': quarter
        })

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        tdf.set_index('timestamp', inplace=True)
    return tdf

# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
print("═" * 70)
print("  TEMPORAL OPTIMIZATION ANALYSIS")
print("  Strategy Code: ALPHA-BTC-15M-v2.0")
print(f"  Started: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
print("═" * 70)

t0 = time.time()

print("\n[1/4] Loading data and computing indicators...")
df = load_data(os.path.join(BASE_DIR, 'data', 'BTCUSDT_15m_3_years.csv'))
df = compute_indicators(df)

print("\n[2/4] Running baseline simulation with temporal tracking...")
tdf = run_temporal_simulation(df)
total_trades = len(tdf)
baseline_wr = tdf['win'].mean() * 100
baseline_pnl = tdf['pnl'].sum()
print(f"  Processed {total_trades:,} trades.")

print("\n[3/4] Generating temporal matrices...")

# Maps integer dayofweek to string for readability
day_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
tdf['day_name'] = tdf['day'].map(day_map)

# Helper function to aggregate and format
def aggregate_dimension(group_cols):
    res = tdf.groupby(group_cols).agg(
        trades=('win', 'count'),
        wins=('win', 'sum'),
        pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    ).reset_index()
    res['win_rate'] = (res['wins'] / res['trades']) * 100
    
    # Calculate Sharpe approximation per slot if trades >= MIN_TRADES
    # (Using grouped daily PNL to approximate STD per slot, simplified to trade PnL std)
    std_devs = tdf.groupby(group_cols)['pnl'].std().reset_index(name='std_pnl')
    res = res.merge(std_devs, on=group_cols, how='left')
    res['sharpe'] = np.where((res['std_pnl'] > 0) & (res['trades'] >= MIN_TRADES), 
                             (res['avg_pnl'] / res['std_pnl']) * np.sqrt(365), 0)
    
    # Reorder columns
    cols = list(group_cols) + ['trades', 'win_rate', 'pnl', 'avg_pnl', 'sharpe']
    return res[cols]

# 1. Hourly
hourly_df = aggregate_dimension(['hour'])
hourly_df.sort_values(['win_rate', 'pnl'], ascending=[False, False], inplace=True)
hourly_df.to_csv(os.path.join(OUTPUT_DIR, "hourly_performance.csv"), index=False)
valid_hours = hourly_df[hourly_df['trades'] >= MIN_TRADES]
best_hours = valid_hours.head(5)
worst_hours = valid_hours.tail(5)

# 2. Daily
daily_df = aggregate_dimension(['day'])
daily_df['day_name'] = daily_df['day'].map(day_map)
daily_df.sort_values(['win_rate', 'pnl'], ascending=[False, False], inplace=True)
daily_df = daily_df[['day_name', 'day', 'trades', 'win_rate', 'pnl', 'avg_pnl', 'sharpe']]
daily_df.to_csv(os.path.join(OUTPUT_DIR, "daily_performance.csv"), index=False)
valid_days = daily_df[daily_df['trades'] >= MIN_TRADES]
best_days = valid_days.head(2)
worst_days = valid_days.tail(2)

# 3. Quarterly
quarter_df = aggregate_dimension(['quarter'])
quarter_df.sort_values('win_rate', ascending=False, inplace=True)
quarter_df.to_csv(os.path.join(OUTPUT_DIR, "quarter_performance.csv"), index=False)
valid_quarters = quarter_df[quarter_df['trades'] >= MIN_TRADES]
best_quarter = valid_quarters.head(1)
worst_quarter = valid_quarters.tail(1)

# 4. Hour x Day (168 slots)
hd_df = aggregate_dimension(['hour', 'day'])
hd_df['day_name'] = hd_df['day'].map(day_map)
hd_df.sort_values(['win_rate', 'pnl'], ascending=[False, False], inplace=True)
hd_df = hd_df[['hour', 'day_name', 'day', 'trades', 'win_rate', 'pnl', 'avg_pnl', 'sharpe']]
hd_df.to_csv(os.path.join(OUTPUT_DIR, "hour_day_matrix.csv"), index=False)
valid_hd = hd_df[hd_df['trades'] >= MIN_TRADES]
best_hd = valid_hd.head(10)
worst_hd = valid_hd.tail(10)

# 5. Hour x Quarter (96 slots)
hq_df = aggregate_dimension(['hour', 'quarter'])
hq_df.sort_values(['win_rate', 'pnl'], ascending=[False, False], inplace=True)
hq_df.to_csv(os.path.join(OUTPUT_DIR, "hour_quarter_matrix.csv"), index=False)
valid_hq = hq_df[hq_df['trades'] >= MIN_TRADES]
best_hq = valid_hq.head(10)
worst_hq = valid_hq.tail(10)

# 6. Day x Quarter (28 slots)
dq_df = aggregate_dimension(['day', 'quarter'])
dq_df['day_name'] = dq_df['day'].map(day_map)
dq_df.sort_values(['win_rate', 'pnl'], ascending=[False, False], inplace=True)
dq_df = dq_df[['day_name', 'day', 'quarter', 'trades', 'win_rate', 'pnl', 'avg_pnl', 'sharpe']]
dq_df.to_csv(os.path.join(OUTPUT_DIR, "day_quarter_matrix.csv"), index=False)
valid_dq = dq_df[dq_df['trades'] >= MIN_TRADES]
best_dq = valid_dq.head(5)
worst_dq = valid_dq.tail(5)


print("\n[4/4] Generating executive summary report...")

# Impact Analysis helper
def evaluate_removal(df_to_remove, label):
    # Determine the total metrics if we remove specific trades
    # For speed, we just subtract aggregate metrics since it's disjoint
    rem_trades = df_to_remove['trades'].sum()
    rem_wins = (df_to_remove['win_rate']/100 * df_to_remove['trades']).sum()
    rem_pnl = df_to_remove['pnl'].sum()
    
    keep_trades = total_trades - rem_trades
    keep_wins = (tdf['win'].sum()) - rem_wins
    keep_wr = (keep_wins / keep_trades) * 100 if keep_trades > 0 else 0
    keep_pnl = baseline_pnl - rem_pnl
    
    del_wr = keep_wr - baseline_wr
    del_pnl = keep_pnl - baseline_pnl
    
    return f"""  Scenario: Remove {label}
    Trades Removed: {rem_trades:,.0f} ({(rem_trades/total_trades)*100:.1f}%)
    New Win Rate:   {keep_wr:.2f}% ({del_wr:+.2f}%)
    New Total P&L:  ${keep_pnl:,.2f} (${del_pnl:+,.2f})"""

def evaluate_retention(df_to_keep, label):
    keep_trades = df_to_keep['trades'].sum()
    keep_wins = (df_to_keep['win_rate']/100 * df_to_keep['trades']).sum()
    keep_wr = (keep_wins / keep_trades) * 100 if keep_trades > 0 else 0
    keep_pnl = df_to_keep['pnl'].sum()
    
    del_wr = keep_wr - baseline_wr
    del_pnl = keep_pnl - baseline_pnl
    
    rem_trades = total_trades - keep_trades
    return f"""  Scenario: Trade ONLY {label}
    Trades Kept:    {keep_trades:,.0f} (Dropped {rem_trades:,.0f})
    New Win Rate:   {keep_wr:.2f}% ({del_wr:+.2f}%)
    New Total P&L:  ${keep_pnl:,.2f} (${del_pnl:+,.2f})"""

report = f"""═══════════════════════════════════════════════════════════════════════
TEMPORAL OPTIMIZATION ANALYSIS — SUMMARY REPORT
ALGORITHMIC TRADING STRATEGY — ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════

Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}
Classification: CONFIDENTIAL

Baseline Total Trades: {total_trades:,}
Baseline Win Rate:     {baseline_wr:.2f}%
Baseline Total P&L:    ${baseline_pnl:,.2f}

Only dimensions with >={MIN_TRADES} trades are considered for best/worst rankings.

═══════════════════════════════════════════════════════════════════════
1. HOURLY PERFORMANCE (Best & Worst)
═══════════════════════════════════════════════════════════════════════
BEST 5 HOURS (UTC):
Hour | Trades | Win Rate | Total P&L | Avg P&L | Sharpe
-----|--------|----------|-----------|---------|-------
"""
for _, r in best_hours.iterrows():
    report += f"{int(r['hour']):02d}   | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f} | ${r['avg_pnl']:>6.3f} | {r['sharpe']:>5.2f}\n"

report += "\nWORST 5 HOURS (UTC):\nHour | Trades | Win Rate | Total P&L | Avg P&L | Sharpe\n-----|--------|----------|-----------|---------|-------\n"
for _, r in worst_hours.iterrows():
    report += f"{int(r['hour']):02d}   | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f} | ${r['avg_pnl']:>6.3f} | {r['sharpe']:>5.2f}\n"

report += "\n" + evaluate_removal(worst_hours, "Worst 5 Hours") + "\n"

report += f"""
═══════════════════════════════════════════════════════════════════════
2. DAILY PERFORMANCE
═══════════════════════════════════════════════════════════════════════
BEST 2 DAYS:
Day       | Trades | Win Rate | Total P&L | Avg P&L | Sharpe
----------|--------|----------|-----------|---------|-------
"""
for _, r in best_days.iterrows():
    report += f"{r['day_name']:<9} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f} | ${r['avg_pnl']:>6.3f} | {r['sharpe']:>5.2f}\n"

report += "\nWORST 2 DAYS:\nDay       | Trades | Win Rate | Total P&L | Avg P&L | Sharpe\n----------|--------|----------|-----------|---------|-------\n"
for _, r in worst_days.iterrows():
    report += f"{r['day_name']:<9} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f} | ${r['avg_pnl']:>6.3f} | {r['sharpe']:>5.2f}\n"

report += "\n" + evaluate_removal(worst_days, "Worst 2 Days") + "\n"

report += f"""
═══════════════════════════════════════════════════════════════════════
3. QUARTER OF HOUR PERFORMANCE
═══════════════════════════════════════════════════════════════════════
Qtr | Trades | Win Rate | Total P&L | Avg P&L | Sharpe
----|--------|----------|-----------|---------|-------
"""
for _, r in quarter_df.iterrows():
    report += f"{r['quarter']:<3} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f} | ${r['avg_pnl']:>6.3f} | {r['sharpe']:>5.2f}\n"

report += "\n" + evaluate_removal(worst_quarter, f"Worst Quarter ({worst_quarter.iloc[0]['quarter']})") + "\n"

report += f"""
═══════════════════════════════════════════════════════════════════════
4. GRANULAR MATRICES — TOP & BOTTOM 10
═══════════════════════════════════════════════════════════════════════
ABSOLUTE BEST 10 HOUR × DAY SLOTS:
Hour | Day       | Trades | Win Rate | Total P&L
-----|-----------|--------|----------|----------
"""
for _, r in best_hd.iterrows():
    report += f"{int(r['hour']):02d}   | {r['day_name']:<9} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f}\n"

report += "\nABSOLUTE WORST 10 HOUR × DAY SLOTS:\nHour | Day       | Trades | Win Rate | Total P&L\n-----|-----------|--------|----------|----------\n"
for _, r in worst_hd.iterrows():
    report += f"{int(r['hour']):02d}   | {r['day_name']:<9} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f}\n"

report += "\n" + evaluate_removal(worst_hd, "Worst 10 Hour×Day Combinations") + "\n"
report += "\n" + evaluate_retention(best_hd, "Best 10 Hour×Day Combinations") + "\n"

report += f"""
═══════════════════════════════════════════════════════════════════════
5. HOUR × QUARTER SLOTS
═══════════════════════════════════════════════════════════════════════
BEST 10 HOUR × QUARTER SLOTS:
Hour | Qtr | Trades | Win Rate | Total P&L
-----|-----|--------|----------|----------
"""
for _, r in best_hq.iterrows():
    report += f"{int(r['hour']):02d}   | {r['quarter']:<3} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f}\n"

report += "\nWORST 10 HOUR × QUARTER SLOTS:\nHour | Qtr | Trades | Win Rate | Total P&L\n-----|-----|--------|----------|----------\n"
for _, r in worst_hq.iterrows():
    report += f"{int(r['hour']):02d}   | {r['quarter']:<3} | {int(r['trades']):<6d} | {r['win_rate']:>6.2f}%   | ${r['pnl']:>8.2f}\n"

report += "\n" + evaluate_removal(worst_hq, "Worst 10 Hour×Quarter Combinations") + "\n"

report += f"""
═══════════════════════════════════════════════════════════════════════
RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════
1. Review the performance of {int(worst_hours.iloc[0]['hour']):02d}:00 UTC and {worst_days.iloc[0]['day_name']}s as they show consistent underperformance.
2. Consider compound filters (e.g., skip {worst_hd.iloc[0]['day_name']} {int(worst_hd.iloc[0]['hour']):02d}:00-15m UTC) rather than blanket single-dimension filters if trade count is too severely impacted.
3. The worst quarter is '{worst_quarter.iloc[0]['quarter']}'. Check `hour_quarter_matrix.csv` to see if this is universally bad or skewed by specific hours.
"""

with open(os.path.join(OUTPUT_DIR, "TEMPORAL_OPTIMIZATION_SUMMARY.txt"), "w", encoding='utf-8') as f:
    f.write(report)

elapsed = time.time() - t0

print(f"\n{'═'*70}")
print(f"  TEMPORAL OPTIMIZATION COMPLETE")
print(f"{'═'*70}")
print(f"  Output Directory:   {OUTPUT_DIR}")
print(f"  Duration:           {elapsed:.1f}s")
print(f"  Files Generated:    7")
print(f"{'═'*70}")
